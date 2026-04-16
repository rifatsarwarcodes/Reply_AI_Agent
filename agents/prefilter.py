"""Pre-Filter Agent (Tier 1) — Gemini 2.5 Flash Lite.

Ultra-cheap first pass that removes ONLY obviously legitimate transactions.
The logic is inverted: we look for "definitely safe" signals and only
filter those out.  Everything else (including anything remotely ambiguous)
goes to Tier 2+ for deeper analysis.

A transaction is "obviously safe" ONLY if ALL of the following hold:
  - Known user with matching IBAN
  - Daytime hours (6-23)
  - Amount within 2 std deviations of the user's history for that type
  - To a previously-seen recipient
  - Type is a routine category (salary, utility, subscription)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from statistics import mean, stdev

from langfuse import observe

import config
from agents.base import BaseAgent
from models.schemas import Dataset, Transaction, User

log = logging.getLogger(__name__)

_ROUTINE_TYPES = {"salary", "utility", "subscription", "rent", "insurance"}


class PreFilterAgent(BaseAgent):
    name = "prefilter"

    @observe(name="prefilter.run")
    def run(
        self,
        dataset: Dataset,
        biotag_map: dict[str, User],
    ) -> tuple[list[Transaction], list[Transaction]]:
        """Return (needs_analysis, obviously_safe) transaction lists."""
        baselines = self._build_baselines(dataset, biotag_map)
        recipient_history = self._build_recipient_history(dataset)

        needs_analysis: list[Transaction] = []
        obviously_safe: list[Transaction] = []

        for tx in dataset.transactions:
            if self._is_obviously_safe(tx, biotag_map, baselines, recipient_history):
                obviously_safe.append(tx)
            else:
                needs_analysis.append(tx)

        log.info(
            "PreFilter: %d need analysis, %d obviously safe (from %d total)",
            len(needs_analysis), len(obviously_safe), len(dataset.transactions),
        )
        return needs_analysis, obviously_safe

    # ------------------------------------------------------------------

    @staticmethod
    def _is_obviously_safe(
        tx: Transaction,
        biotag_map: dict[str, User],
        baselines: dict[tuple[str, str], dict],
        recipient_history: dict[str, set[str]],
    ) -> bool:
        """Conservative check — only returns True if the transaction
        is clearly routine/legitimate."""
        sender = biotag_map.get(tx.sender_id)

        # Unknown sender → not safe
        if not sender:
            return False

        # IBAN mismatch → definitely not safe
        if tx.sender_iban and tx.sender_iban != sender.iban:
            return False

        # Night-time → not safe
        if tx.timestamp.hour in config.NIGHT_HOURS:
            return False

        # Large amount → not safe
        if tx.amount > 5000:
            return False

        # Anomalous amount for this user+type → not safe
        if sender.biotag:
            key = (sender.biotag, tx.transaction_type)
            bl = baselines.get(key)
            if bl and bl["std"] > 0:
                z = abs(tx.amount - bl["mean"]) / bl["std"]
                if z >= 2.0:
                    return False

        # New recipient → not safe
        if sender.biotag and tx.recipient_id:
            hist = recipient_history.get(sender.biotag, set())
            if tx.recipient_id not in hist:
                return False

        # Must be a routine type AND known recipient to be "obviously safe"
        if tx.transaction_type.lower() not in _ROUTINE_TYPES:
            return False

        return True

    # ------------------------------------------------------------------

    @staticmethod
    def _build_baselines(dataset: Dataset, biotag_map: dict[str, User]):
        buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
        for tx in dataset.transactions:
            user = biotag_map.get(tx.sender_id) or biotag_map.get(tx.recipient_id)
            if user and user.biotag:
                buckets[(user.biotag, tx.transaction_type)].append(tx.amount)
        baselines = {}
        for key, amounts in buckets.items():
            if len(amounts) >= 2:
                baselines[key] = {"mean": mean(amounts), "std": stdev(amounts)}
        return baselines

    @staticmethod
    def _build_recipient_history(dataset: Dataset) -> dict[str, set[str]]:
        history: dict[str, set[str]] = defaultdict(set)
        sorted_txs = sorted(dataset.transactions, key=lambda t: t.timestamp)
        for tx in sorted_txs:
            if tx.sender_id and tx.recipient_id:
                history[tx.sender_id].add(tx.recipient_id)
        return dict(history)
