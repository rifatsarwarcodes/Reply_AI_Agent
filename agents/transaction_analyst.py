"""Transaction Analyst — detects IBAN mismatches, amount anomalies, and
temporal anomalies by comparing each transaction against user profiles and
historical baselines."""

from __future__ import annotations

import logging
from collections import defaultdict
from statistics import mean, stdev

from langfuse import observe

import config
from agents.base import BaseAgent
from models.schemas import (
    Dataset,
    RiskSignal,
    TransactionRisk,
    User,
)

log = logging.getLogger(__name__)


class TransactionAnalyst(BaseAgent):
    name = "transaction"

    @observe(name="transaction_analyst.analyze")
    def analyze(self, dataset: Dataset, biotag_map: dict[str, User]) -> dict[str, TransactionRisk]:
        iban_to_user = {u.iban: u for u in dataset.users}
        biotag_to_user = biotag_map

        baselines = self._build_baselines(dataset, biotag_to_user)
        recipient_history = self._build_recipient_history(dataset)

        risks: dict[str, TransactionRisk] = {}
        for tx in dataset.transactions:
            tr = TransactionRisk(transaction_id=tx.transaction_id)

            sender_user = biotag_to_user.get(tx.sender_id)
            recipient_user = biotag_to_user.get(tx.recipient_id)

            # ---- 1. IBAN mismatch (strongest signal) ---------------------
            if sender_user and tx.sender_iban and tx.sender_iban != sender_user.iban:
                tr.signals.append(RiskSignal(
                    score=config.IBAN_MISMATCH_SCORE,
                    reason=f"Sender IBAN {tx.sender_iban[:12]}… ≠ profile {sender_user.iban[:12]}…",
                    agent=self.name,
                ))
            if recipient_user and tx.recipient_iban and tx.recipient_iban != recipient_user.iban:
                tr.signals.append(RiskSignal(
                    score=config.IBAN_MISMATCH_SCORE * 0.85,
                    reason=f"Recipient IBAN mismatch for {recipient_user.first_name}",
                    agent=self.name,
                ))

            # ---- 2. Amount anomaly (z-score) ----------------------------
            active_user = sender_user or recipient_user
            if active_user and active_user.biotag:
                key = (active_user.biotag, tx.transaction_type)
                bl = baselines.get(key)
                if bl and bl["std"] > 0:
                    z = abs(tx.amount - bl["mean"]) / bl["std"]
                    if z >= config.AMOUNT_ZSCORE_HIGH:
                        tr.signals.append(RiskSignal(
                            score=config.AMOUNT_HIGH_SCORE,
                            reason=f"Amount z-score {z:.1f} for type '{tx.transaction_type}'",
                            agent=self.name,
                        ))
                    elif z >= config.AMOUNT_ZSCORE_MED:
                        tr.signals.append(RiskSignal(
                            score=config.AMOUNT_MED_SCORE,
                            reason=f"Amount z-score {z:.1f} for type '{tx.transaction_type}'",
                            agent=self.name,
                        ))

            # ---- 3. Temporal anomaly (night-time) -----------------------
            if tx.timestamp.hour in config.NIGHT_HOURS:
                tr.signals.append(RiskSignal(
                    score=config.NIGHT_SCORE,
                    reason=f"Transaction at {tx.timestamp.strftime('%H:%M')} (night)",
                    agent=self.name,
                ))

            # ---- 4. New / rare recipient --------------------------------
            if sender_user and sender_user.biotag:
                hist = recipient_history.get(sender_user.biotag, set())
                if tx.recipient_id and tx.recipient_id not in hist:
                    tr.signals.append(RiskSignal(
                        score=config.NEW_RECIPIENT_SCORE,
                        reason=f"First transaction to recipient {tx.recipient_id}",
                        agent=self.name,
                    ))

            if tr.signals:
                risks[tx.transaction_id] = tr

        log.info("TransactionAnalyst flagged %d / %d transactions",
                 len(risks), len(dataset.transactions))
        return risks

    # ------------------------------------------------------------------
    @staticmethod
    def _build_baselines(dataset: Dataset, biotag_map: dict[str, User]):
        """Mean & std of amount grouped by (biotag, transaction_type)."""
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
        """For each sender biotag, collect all recipients seen so far
        (iterating chronologically) — the *first* time a recipient appears
        it is considered 'new'."""
        history: dict[str, set[str]] = defaultdict(set)
        sorted_txs = sorted(dataset.transactions, key=lambda t: t.timestamp)
        result: dict[str, set[str]] = {}
        for tx in sorted_txs:
            sid = tx.sender_id
            if sid:
                result[sid] = set(history[sid])
                if tx.recipient_id:
                    history[sid].add(tx.recipient_id)
        return result
