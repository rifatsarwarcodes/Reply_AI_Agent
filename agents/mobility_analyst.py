"""Mobility Analyst — cross-references transaction locations with GPS pings
to detect impossible-travel and far-from-home anomalies."""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from datetime import timedelta

from langfuse import observe

import config
from agents.base import BaseAgent
from models.schemas import (
    Dataset,
    LocationPing,
    RiskSignal,
    TransactionRisk,
    User,
)

log = logging.getLogger(__name__)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    la1, lo1, la2, lo2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = la2 - la1
    dlon = lo2 - lo1
    a = math.sin(dlat / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


class MobilityAnalyst(BaseAgent):
    name = "mobility"

    @observe(name="mobility_analyst.analyze")
    def analyze(self, dataset: Dataset, biotag_map: dict[str, User]) -> dict[str, TransactionRisk]:
        pings_by_user = self._index_pings(dataset.locations)
        city_coords = self._build_city_coords(dataset)

        risks: dict[str, TransactionRisk] = {}

        for tx in dataset.transactions:
            if not tx.location:
                continue

            user = biotag_map.get(tx.sender_id) or biotag_map.get(tx.recipient_id)
            if not user or not user.biotag:
                continue

            tx_coords = self._resolve_tx_location(tx.location, city_coords)
            if tx_coords is None:
                continue

            tr = TransactionRisk(transaction_id=tx.transaction_id)

            # -- Impossible / suspicious travel speed ---------------------
            window = timedelta(hours=config.LOCATION_TIME_WINDOW_H)
            nearby = [
                p for p in pings_by_user.get(user.biotag, [])
                if abs(p.timestamp - tx.timestamp) <= window
            ]

            if nearby:
                best_speed = float("inf")
                best_dist = 0.0
                for p in nearby:
                    dist = haversine_km(p.lat, p.lng, tx_coords[0], tx_coords[1])
                    dt_h = abs((tx.timestamp - p.timestamp).total_seconds()) / 3600.0
                    speed = dist / dt_h if dt_h > 0.01 else 0.0
                    if speed < best_speed:
                        best_speed = speed
                        best_dist = dist

                if best_speed >= config.IMPOSSIBLE_SPEED_KMH and best_dist > 50:
                    tr.signals.append(RiskSignal(
                        score=config.IMPOSSIBLE_TRAVEL_SCORE,
                        reason=f"Impossible travel: {best_dist:.0f} km at {best_speed:.0f} km/h",
                        agent=self.name,
                    ))
                elif best_speed >= config.SUSPICIOUS_SPEED_KMH and best_dist > 50:
                    tr.signals.append(RiskSignal(
                        score=config.SUSPICIOUS_TRAVEL_SCORE,
                        reason=f"Suspicious travel: {best_dist:.0f} km at {best_speed:.0f} km/h",
                        agent=self.name,
                    ))

            # -- Far from home (no nearby ping needed) --------------------
            home_dist = haversine_km(user.lat, user.lng, tx_coords[0], tx_coords[1])
            if home_dist >= config.FAR_FROM_HOME_KM:
                already_flagged = any(s.agent == self.name and "travel" in s.reason.lower()
                                      for s in tr.signals)
                if not already_flagged:
                    tr.signals.append(RiskSignal(
                        score=config.FAR_FROM_HOME_SCORE,
                        reason=f"Transaction {home_dist:.0f} km from home ({user.city})",
                        agent=self.name,
                    ))

            if tr.signals:
                risks[tx.transaction_id] = tr

        log.info("MobilityAnalyst flagged %d / %d transactions",
                 len(risks), len(dataset.transactions))
        return risks

    # ------------------------------------------------------------------
    @staticmethod
    def _index_pings(locations: list[LocationPing]) -> dict[str, list[LocationPing]]:
        idx: dict[str, list[LocationPing]] = defaultdict(list)
        for p in locations:
            idx[p.biotag].append(p)
        for v in idx.values():
            v.sort(key=lambda p: p.timestamp)
        return idx

    @staticmethod
    def _build_city_coords(dataset: Dataset) -> dict[str, tuple[float, float]]:
        """Map city name → (lat, lng) from user residences + location pings."""
        coords: dict[str, tuple[float, float]] = {}
        for u in dataset.users:
            if u.city:
                coords[u.city.lower()] = (u.lat, u.lng)
        for p in dataset.locations:
            if p.city and p.city.lower() not in coords:
                coords[p.city.lower()] = (p.lat, p.lng)
        return coords

    @staticmethod
    def _resolve_tx_location(
        location_str: str,
        city_coords: dict[str, tuple[float, float]],
    ) -> tuple[float, float] | None:
        """Try to map a transaction location string to coordinates."""
        loc = location_str.lower().strip()
        if not loc or "online" in loc:
            return None

        for city, coord in city_coords.items():
            if city in loc:
                return coord

        parts = loc.replace(",", " ").split()
        for token in reversed(parts):
            clean = token.strip("()- ")
            if clean in city_coords:
                return city_coords[clean]

        return None
