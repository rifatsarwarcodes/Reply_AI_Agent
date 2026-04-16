"""Orchestrator (Tier 4) — GPT-5.2 Codex.

Waterfall coordination:
  Tier 1  PreFilter (Gemini 2.5 Flash Lite)   — remove obvious legit
  Tier 2  TransactionAnalyst + MobilityAnalyst (DeepSeek R1) — logic verification
  Tier 3  CommsAnalyst (Claude 4.5) + AudioAnalyst (Gemini 3.1 Pro) — deep analysis
  Tier 4  Orchestrator (GPT-5.2 Codex) — final fusion + LLM review
  +       MemoryAgent (DeepSeek R1) — adaptive threshold tuning
"""

from __future__ import annotations

import logging
from pathlib import Path

from langfuse import observe

import config
from agents.base import BaseAgent, build_llm
from agents.prefilter import PreFilterAgent
from agents.transaction_analyst import TransactionAnalyst
from agents.mobility_analyst import MobilityAnalyst
from agents.comms_analyst import CommsAnalyst
from agents.audio_analyst import AudioAnalyst
from agents.memory_agent import MemoryAgent
from loaders.data_loader import load_dataset, resolve_biotags
from models.schemas import (
    Dataset,
    RiskSignal,
    TransactionRisk,
    User,
)

log = logging.getLogger(__name__)


class Orchestrator(BaseAgent):
    name = "orchestrator"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Each agent gets its own LLM, built for its specific role
        self.prefilter = PreFilterAgent()
        self.tx_agent = TransactionAnalyst()
        self.mob_agent = MobilityAnalyst()
        self.comms_agent = CommsAnalyst()
        self.audio_agent = AudioAnalyst()
        self.memory_agent = MemoryAgent()

        log.info(
            "Orchestrator initialized with heterogeneous models:\n"
            "  PreFilter:    %s\n"
            "  Transaction:  %s\n"
            "  Mobility:     %s\n"
            "  Comms:        %s\n"
            "  Audio:        %s\n"
            "  Memory:       %s\n"
            "  Orchestrator: %s",
            config.MODELS["prefilter"]["id"],
            config.MODELS["transaction"]["id"],
            config.MODELS["mobility"]["id"],
            config.MODELS["comms"]["id"],
            config.MODELS["audio"]["id"],
            config.MODELS["memory"]["id"],
            config.MODELS["orchestrator"]["id"],
        )

    @observe(name="orchestrator.run")
    def run(self, dataset_path: str) -> list[str]:
        dataset = load_dataset(dataset_path)
        biotag_map = resolve_biotags(
            dataset.users, dataset.transactions, dataset.locations
        )

        log.info("=" * 60)
        log.info("WATERFALL PIPELINE on '%s' (%d txs, %d users)",
                 dataset.name, len(dataset.transactions), len(dataset.users))
        log.info("=" * 60)

        # ────────────────────────────────────────────────────────────────
        # TIER 1: Pre-Filter (Gemini 2.5 Flash Lite — cheapest)
        # ────────────────────────────────────────────────────────────────
        log.info("▸ Tier 1: Pre-filtering with %s", config.MODELS["prefilter"]["id"])
        suspicious_txs, safe_txs = self.prefilter.run(dataset, biotag_map)
        log.info("  → %d suspicious, %d safe", len(suspicious_txs), len(safe_txs))

        # ────────────────────────────────────────────────────────────────
        # TIER 2: Logic Verification (DeepSeek R1 — mathematical reasoning)
        # ────────────────────────────────────────────────────────────────
        log.info("▸ Tier 2: Transaction + Mobility analysis with %s",
                 config.MODELS["transaction"]["id"])
        tx_risks = self.tx_agent.analyze(dataset, biotag_map, tx_subset=suspicious_txs)
        mob_risks = self.mob_agent.analyze(dataset, biotag_map, tx_subset=suspicious_txs)

        # ────────────────────────────────────────────────────────────────
        # TIER 3: Deep Content Analysis (Claude 4.5 + Gemini 3.1 Pro)
        # Comms/Audio run on ALL transactions because phishing events
        # propagate at user-level — a "safe-looking" transaction from a
        # compromised user is still suspect.
        # ────────────────────────────────────────────────────────────────
        log.info("▸ Tier 3: Comms analysis with %s (all txs)", config.MODELS["comms"]["id"])
        comms_risks = self.comms_agent.analyze(dataset, biotag_map)

        log.info("▸ Tier 3: Audio analysis with %s (all txs)", config.MODELS["audio"]["id"])
        audio_risks = self.audio_agent.analyze(dataset, biotag_map)

        # ────────────────────────────────────────────────────────────────
        # FUSION: Merge all risk signals
        # ────────────────────────────────────────────────────────────────
        merged = self._merge_risks(
            dataset, tx_risks, mob_risks, comms_risks, audio_risks
        )

        # ────────────────────────────────────────────────────────────────
        # MEMORY AGENT: Adaptive threshold tuning (DeepSeek R1)
        # ────────────────────────────────────────────────────────────────
        log.info("▸ Memory Agent: Analyzing patterns with %s",
                 config.MODELS["memory"]["id"])
        snapshot = self.memory_agent.analyze_patterns(dataset, merged, biotag_map)
        effective_threshold = snapshot.recommended_threshold
        log.info("  → Effective threshold: %.3f (base: %.3f)",
                 effective_threshold, config.FRAUD_THRESHOLD)

        # ────────────────────────────────────────────────────────────────
        # TIER 4: Final Decision (GPT-5.2 Codex)
        # ────────────────────────────────────────────────────────────────
        flagged = self._apply_threshold(merged, dataset, threshold=effective_threshold)

        if flagged:
            log.info("▸ Tier 4: Final review with %s (%d candidates)",
                     config.MODELS["orchestrator"]["id"], len(flagged))
            flagged = self._llm_final_review(flagged, merged, dataset, biotag_map)

        output_path = self._write_output(dataset.name, flagged)
        log.info(
            "OUTPUT: %s  (%d flagged / %d total = %.1f%%)",
            output_path, len(flagged), len(dataset.transactions),
            100.0 * len(flagged) / max(len(dataset.transactions), 1),
        )
        return flagged

    # ------------------------------------------------------------------
    # Score fusion
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_risks(
        dataset: Dataset,
        *risk_dicts: dict[str, TransactionRisk],
    ) -> dict[str, TransactionRisk]:
        merged: dict[str, TransactionRisk] = {}
        for rd in risk_dicts:
            for tx_id, tr in rd.items():
                if tx_id not in merged:
                    merged[tx_id] = TransactionRisk(transaction_id=tx_id)
                merged[tx_id].signals.extend(tr.signals)
        return merged

    @staticmethod
    def _apply_threshold(
        merged: dict[str, TransactionRisk],
        dataset: Dataset,
        threshold: float | None = None,
    ) -> list[str]:
        threshold = threshold or config.FRAUD_THRESHOLD
        scored = [
            (tx_id, tr.combined_score, tr)
            for tx_id, tr in merged.items()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        max_flags = int(len(dataset.transactions) * config.MAX_FLAG_RATIO)

        flagged: list[str] = []
        for tx_id, score, tr in scored:
            if score < threshold:
                break
            flagged.append(tx_id)
            if len(flagged) >= max_flags:
                break

        return flagged

    # ------------------------------------------------------------------
    # GPT-5.2 Codex final review
    # ------------------------------------------------------------------

    @observe(name="orchestrator.llm_final_review")
    def _llm_final_review(
        self,
        flagged: list[str],
        merged: dict[str, TransactionRisk],
        dataset: Dataset,
        biotag_map: dict[str, User],
    ) -> list[str]:
        """GPT-5.2 Codex final sanity check on flagged transactions."""
        tx_lookup = {tx.transaction_id: tx for tx in dataset.transactions}

        summaries: list[str] = []
        for tx_id in flagged[:40]:
            tx = tx_lookup.get(tx_id)
            tr = merged.get(tx_id)
            if not tx or not tr:
                continue
            user = biotag_map.get(tx.sender_id)
            user_info = f"{user.first_name} {user.last_name}" if user else "unknown"
            reasons = "; ".join(s.reason for s in tr.signals[:4])
            agents_involved = ", ".join(sorted({s.agent for s in tr.signals}))
            summaries.append(
                f"  ID: {tx_id}\n"
                f"  Type: {tx.transaction_type}, Amount: {tx.amount}, "
                f"Time: {tx.timestamp}, Sender: {user_info}\n"
                f"  Score: {tr.combined_score:.2f}  Agents: [{agents_involved}]\n"
                f"  Signals: {reasons}\n"
            )

        if not summaries:
            return flagged

        system = (
            "You are the chief fraud analyst using GPT-5.2 Codex reasoning. "
            "Review these flagged transactions from the MirrorPay platform. "
            "Multiple specialist agents (Transaction/DeepSeek-R1, Mobility/DeepSeek-R1, "
            "Comms/Claude-4.5, Audio/Gemini-3.1-Pro) have already analyzed them. "
            "For each transaction, confirm FRAUD or override to LEGIT. "
            "CRITICAL: A false negative (missing real fraud) costs 10x more than "
            "a false positive. When in doubt, say FRAUD. "
            "Reply with JSON: [{\"id\": \"<transaction_id>\", \"verdict\": \"FRAUD\"|\"LEGIT\", "
            "\"confidence\": 0.0-1.0}]"
        )
        user_prompt = (
            f"Dataset: {dataset.name} — {len(dataset.transactions)} total transactions.\n"
            f"Pre-filtered and analyzed by multi-model pipeline.\n\n"
            + "\n".join(summaries)
            + "\nRespond ONLY with the JSON array."
        )

        try:
            result = self._call_llm_json(system, user_prompt)
            if isinstance(result, list):
                overrides = {
                    item["id"]: item["verdict"].upper()
                    for item in result
                    if isinstance(item, dict) and "id" in item and "verdict" in item
                }
                revised = [
                    tx_id for tx_id in flagged
                    if overrides.get(tx_id, "FRAUD") == "FRAUD"
                ]
                removed = len(flagged) - len(revised)
                if removed:
                    log.info("GPT-5.2 Codex review removed %d false positives", removed)
                return revised
        except Exception as exc:
            log.warning("GPT-5.2 Codex final review failed: %s — keeping original flags", exc)

        return flagged

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    @staticmethod
    def _write_output(dataset_name: str, flagged: list[str]) -> Path:
        safe_name = dataset_name.replace(" ", "_").replace("/", "_")
        out_path = config.OUTPUT_DIR / f"{safe_name}.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            for tx_id in flagged:
                f.write(tx_id + "\n")
        return out_path
