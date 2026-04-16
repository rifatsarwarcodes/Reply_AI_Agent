"""Memory Agent — DeepSeek R1.

Tracks evolving fraud patterns across the pipeline run and dynamically
adjusts detection thresholds.  Implements the "continuous learning"
requirement by:

1. Analyzing temporal / amount / type distributions of flagged vs safe txs
2. Detecting pattern shifts (e.g. hackers moving to night hours)
3. Recommending threshold adjustments to the Orchestrator
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from langfuse import observe

import config
from agents.base import BaseAgent
from models.schemas import Dataset, Transaction, TransactionRisk, User

log = logging.getLogger(__name__)


@dataclass
class PatternSnapshot:
    """Statistical snapshot of observed fraud patterns."""
    night_fraud_ratio: float = 0.0
    high_amount_fraud_ratio: float = 0.0
    avg_fraud_amount: float = 0.0
    top_fraud_types: list[str] = field(default_factory=list)
    top_fraud_locations: list[str] = field(default_factory=list)
    recommended_threshold: float = config.FRAUD_THRESHOLD


class MemoryAgent(BaseAgent):
    """Adaptive agent that tracks patterns and suggests threshold shifts."""

    name = "memory"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._history: list[PatternSnapshot] = []

    @property
    def current_threshold(self) -> float:
        if self._history:
            return self._history[-1].recommended_threshold
        return config.FRAUD_THRESHOLD

    @observe(name="memory_agent.analyze_patterns")
    def analyze_patterns(
        self,
        dataset: Dataset,
        merged_risks: dict[str, TransactionRisk],
        biotag_map: dict[str, User],
    ) -> PatternSnapshot:
        """Analyze the current state of detected fraud and recommend
        threshold adjustments."""
        tx_lookup = {tx.transaction_id: tx for tx in dataset.transactions}

        flagged_txs = [
            tx_lookup[tid]
            for tid, tr in merged_risks.items()
            if tr.combined_score >= config.FRAUD_THRESHOLD and tid in tx_lookup
        ]
        all_txs = dataset.transactions

        snapshot = self._compute_snapshot(flagged_txs, all_txs)

        try:
            llm_adjustment = self._llm_threshold_recommendation(
                dataset, snapshot, len(flagged_txs), len(all_txs)
            )
            if llm_adjustment is not None:
                snapshot.recommended_threshold = max(
                    config.FRAUD_THRESHOLD_MIN,
                    min(config.FRAUD_THRESHOLD_MAX, llm_adjustment),
                )
        except Exception as exc:
            log.warning("Memory agent LLM call failed: %s — using heuristic threshold", exc)

        self._history.append(snapshot)

        log.info(
            "MemoryAgent: night_ratio=%.2f, high_amt_ratio=%.2f, "
            "recommended_threshold=%.3f, top_types=%s",
            snapshot.night_fraud_ratio,
            snapshot.high_amount_fraud_ratio,
            snapshot.recommended_threshold,
            snapshot.top_fraud_types[:3],
        )
        return snapshot

    # ------------------------------------------------------------------

    @staticmethod
    def _compute_snapshot(
        flagged: list[Transaction],
        all_txs: list[Transaction],
    ) -> PatternSnapshot:
        if not flagged:
            return PatternSnapshot()

        night_count = sum(1 for tx in flagged if tx.timestamp.hour in config.NIGHT_HOURS)
        high_amount = sum(1 for tx in flagged if tx.amount > 5000)

        type_counts = Counter(tx.transaction_type for tx in flagged)
        location_counts = Counter(tx.location for tx in flagged if tx.location)

        flag_ratio = len(flagged) / max(len(all_txs), 1)

        # Heuristic: keep flagging ratio between 10-20%
        if flag_ratio > 0.25:
            recommended = config.FRAUD_THRESHOLD + 0.08
        elif flag_ratio > 0.20:
            recommended = config.FRAUD_THRESHOLD + 0.04
        elif flag_ratio < 0.05:
            recommended = config.FRAUD_THRESHOLD - 0.04
        else:
            recommended = config.FRAUD_THRESHOLD

        return PatternSnapshot(
            night_fraud_ratio=night_count / len(flagged),
            high_amount_fraud_ratio=high_amount / len(flagged),
            avg_fraud_amount=sum(tx.amount for tx in flagged) / len(flagged),
            top_fraud_types=[t for t, _ in type_counts.most_common(5)],
            top_fraud_locations=[l for l, _ in location_counts.most_common(5)],
            recommended_threshold=recommended,
        )

    @observe(name="memory_agent.llm_threshold")
    def _llm_threshold_recommendation(
        self,
        dataset: Dataset,
        snapshot: PatternSnapshot,
        n_flagged: int,
        n_total: int,
    ) -> float | None:
        """Ask DeepSeek R1 to reason about optimal threshold given patterns."""
        history_str = ""
        if self._history:
            prev = self._history[-1]
            history_str = (
                f"\nPrevious run: threshold={prev.recommended_threshold:.3f}, "
                f"night_ratio={prev.night_fraud_ratio:.2f}, "
                f"types={prev.top_fraud_types[:3]}"
            )

        system = (
            "You are a fraud detection tuning agent. Given the current detection "
            "statistics, recommend an optimal fraud threshold (float between "
            f"{config.FRAUD_THRESHOLD_MIN} and {config.FRAUD_THRESHOLD_MAX}). "
            "CRITICAL: The system is scored on BOTH fraud detection AND economic "
            "sustainability. Too many false positives hurt the score just like "
            "missed fraud. Aim for precision: flag 10-20% of transactions max. "
            "If the current flagging ratio is already high (>20%), RAISE the "
            "threshold. Only lower it if very few transactions are flagged (<5%). "
            "Reply with ONLY a JSON object: {\"threshold\": <float>, \"reasoning\": \"...\"}"
        )
        user_prompt = (
            f"Dataset: {dataset.name} ({n_total} transactions)\n"
            f"Currently flagged: {n_flagged} ({100*n_flagged/max(n_total,1):.1f}%)\n"
            f"Night-time fraud ratio: {snapshot.night_fraud_ratio:.2f}\n"
            f"High-amount (>5k) fraud ratio: {snapshot.high_amount_fraud_ratio:.2f}\n"
            f"Average fraud amount: {snapshot.avg_fraud_amount:.0f}\n"
            f"Top fraud types: {snapshot.top_fraud_types}\n"
            f"Top fraud locations: {snapshot.top_fraud_locations[:5]}"
            f"{history_str}"
        )

        result = self._call_llm_json(system, user_prompt)
        if isinstance(result, dict) and "threshold" in result:
            val = float(result["threshold"])
            log.info("Memory LLM recommends threshold=%.3f: %s",
                     val, result.get("reasoning", "")[:100])
            return val
        return None
