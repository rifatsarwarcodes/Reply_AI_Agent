"""Transaction Analyst (Tier 2) — DeepSeek R1.

Detects IBAN mismatches, amount anomalies, temporal anomalies, and
new/rare recipients.  Uses DeepSeek R1's mathematical reasoning to
verify borderline anomalies and produce better z-score interpretations.
"""

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
    Transaction,
    TransactionRisk,
    User,
)

log = logging.getLogger(__name__)


class TransactionAnalyst(BaseAgent):
    name = "transaction"

    @observe(name="transaction_analyst.analyze")
    def analyze(
        self,
        dataset: Dataset,
        biotag_map: dict[str, User],
        tx_subset: list[Transaction] | None = None,
    ) -> dict[str, TransactionRisk]:
        """Analyze transactions (or a pre-filtered subset)."""
        transactions = tx_subset if tx_subset is not None else dataset.transactions
        iban_to_user = {u.iban: u for u in dataset.users}

        baselines = self._build_baselines(dataset, biotag_map)
        recipient_history = self._build_recipient_history(dataset)

        risks: dict[str, TransactionRisk] = {}
        high_confidence_flags: list[tuple[Transaction, TransactionRisk]] = []

        for tx in transactions:
            tr = TransactionRisk(transaction_id=tx.transaction_id)
            sender_user = biotag_map.get(tx.sender_id)
            recipient_user = biotag_map.get(tx.recipient_id)

            # ---- 1. IBAN mismatch ----------------------------------------
            if sender_user and tx.sender_iban and tx.sender_iban != sender_user.iban:
                tr.signals.append(RiskSignal(
                    score=config.IBAN_MISMATCH_SCORE,
                    reason=f"Sender IBAN {tx.sender_iban[:12]}… != profile {sender_user.iban[:12]}…",
                    agent=self.name,
                ))
            if recipient_user and tx.recipient_iban and tx.recipient_iban != recipient_user.iban:
                tr.signals.append(RiskSignal(
                    score=config.IBAN_MISMATCH_SCORE * 0.85,
                    reason=f"Recipient IBAN mismatch for {recipient_user.first_name}",
                    agent=self.name,
                ))

            # ---- 2. Amount anomaly (z-score) -----------------------------
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

            # ---- 3. Temporal anomaly (night-time) ------------------------
            if tx.timestamp.hour in config.NIGHT_HOURS:
                tr.signals.append(RiskSignal(
                    score=config.NIGHT_SCORE,
                    reason=f"Transaction at {tx.timestamp.strftime('%H:%M')} (night)",
                    agent=self.name,
                ))

            # ---- 4. New / rare recipient ---------------------------------
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
                if tr.combined_score >= 0.40:
                    high_confidence_flags.append((tx, tr))

        # ---- DeepSeek R1 verification on top suspicious transactions -----
        if high_confidence_flags:
            self._llm_verify_top(high_confidence_flags, biotag_map, dataset)

        log.info("TransactionAnalyst flagged %d / %d transactions",
                 len(risks), len(transactions))
        return risks

    # ------------------------------------------------------------------
    # DeepSeek R1 mathematical verification
    # ------------------------------------------------------------------

    @observe(name="transaction_analyst.llm_verify")
    def _llm_verify_top(
        self,
        flagged: list[tuple[Transaction, TransactionRisk]],
        biotag_map: dict[str, User],
        dataset: Dataset,
    ) -> None:
        """Use DeepSeek R1 to reason about the top flagged transactions
        and adjust scores based on mathematical analysis."""
        batch = flagged[:25]
        summaries = []
        for i, (tx, tr) in enumerate(batch):
            user = biotag_map.get(tx.sender_id)
            user_info = f"{user.first_name} {user.last_name} (salary={user.salary})" if user else "unknown"
            reasons = "; ".join(s.reason for s in tr.signals[:3])
            summaries.append(
                f"[{i}] id={tx.transaction_id} type={tx.transaction_type} "
                f"amt={tx.amount} time={tx.timestamp:%Y-%m-%d %H:%M} "
                f"sender={user_info} score={tr.combined_score:.2f} "
                f"signals=[{reasons}]"
            )

        system = (
            "You are a mathematical fraud analyst using DeepSeek R1 reasoning. "
            "For each transaction, assess whether the detected anomalies truly "
            "indicate fraud. Consider: salary-to-amount ratios, IBAN mismatch "
            "patterns, temporal context. "
            "Reply with JSON: [{\"idx\": <int>, \"adjusted_score\": <float 0-1>, "
            "\"reasoning\": \"brief\"}]"
        )
        user_prompt = (
            f"Dataset: {dataset.name}\n"
            f"Flagged transactions to verify:\n" + "\n".join(summaries)
        )

        try:
            result = self._call_llm_json(system, user_prompt)
            if isinstance(result, list):
                for item in result:
                    idx = int(item.get("idx", -1))
                    adj = float(item.get("adjusted_score", -1))
                    if 0 <= idx < len(batch) and 0 <= adj <= 1:
                        tx, tr = batch[idx]
                        if abs(adj - tr.combined_score) > 0.1:
                            tr.signals.append(RiskSignal(
                                score=adj,
                                reason=f"DeepSeek R1 adjustment: {item.get('reasoning', '')[:80]}",
                                agent=self.name,
                            ))
        except Exception as exc:
            log.warning("TransactionAnalyst LLM verify failed: %s", exc)

    # ------------------------------------------------------------------
    # Baselines
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
        result: dict[str, set[str]] = {}
        for tx in sorted_txs:
            sid = tx.sender_id
            if sid:
                result[sid] = set(history[sid])
                if tx.recipient_id:
                    history[sid].add(tx.recipient_id)
        return result
