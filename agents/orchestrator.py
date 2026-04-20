"""Orchestrator (Tier 4) — GPT-5.2 Codex.

Waterfall coordination with surgical precision:
  1. Hard-exclude obviously legitimate transactions (salary, etc.)
  2. Run Tier 2/3 agents only on non-excluded transactions
  3. Require multi-signal corroboration before flagging
  4. GPT-5.2 Codex final review with full context per transaction
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
    Transaction,
    TransactionRisk,
    User,
)

log = logging.getLogger(__name__)


def _is_hard_excluded(tx: Transaction) -> bool:
    """Transactions that should NEVER be flagged as fraud."""
    # Salary payments from employer codes
    if tx.sender_id.startswith(tuple(config.LEGIT_SENDER_PREFIXES)):
        return True

    desc = (tx.description or "").lower()
    for kw in config.LEGIT_DESCRIPTION_KEYWORDS:
        if kw in desc:
            return True

    return False


class Orchestrator(BaseAgent):
    name = "orchestrator"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prefilter = PreFilterAgent()
        self.tx_agent = TransactionAnalyst()
        self.mob_agent = MobilityAnalyst()
        self.comms_agent = CommsAnalyst()
        self.audio_agent = AudioAnalyst()
        self.memory_agent = MemoryAgent()

        log.info(
            "Orchestrator initialized:\n"
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
        log.info("PIPELINE on '%s' (%d txs, %d users)",
                 dataset.name, len(dataset.transactions), len(dataset.users))
        log.info("=" * 60)

        # ── STEP 0: Hard exclude obviously legitimate transactions ──────
        excluded_ids: set[str] = set()
        analyzable: list[Transaction] = []
        for tx in dataset.transactions:
            if _is_hard_excluded(tx):
                excluded_ids.add(tx.transaction_id)
            else:
                analyzable.append(tx)
        log.info("Hard excluded %d/%d transactions (salary/legit)",
                 len(excluded_ids), len(dataset.transactions))

        # ── TIER 1: Pre-filter ──────────────────────────────────────────
        log.info("▸ Tier 1: Pre-filtering (%s)", config.MODELS["prefilter"]["id"])
        suspicious_txs, safe_txs = self.prefilter.run(dataset, biotag_map)
        # Remove hard-excluded from suspicious
        suspicious_txs = [tx for tx in suspicious_txs if tx.transaction_id not in excluded_ids]
        log.info("  → %d suspicious, %d safe, %d excluded",
                 len(suspicious_txs), len(safe_txs), len(excluded_ids))

        # ── TIER 2: Logic verification ──────────────────────────────────
        log.info("▸ Tier 2: Transaction + Mobility (%s)", config.MODELS["transaction"]["id"])
        tx_risks = self.tx_agent.analyze(dataset, biotag_map, tx_subset=suspicious_txs)
        mob_risks = self.mob_agent.analyze(dataset, biotag_map, tx_subset=suspicious_txs)

        # ── TIER 3: Deep content analysis ───────────────────────────────
        # Comms/Audio analyze ALL non-excluded txs (user-level signals)
        non_excluded = [tx for tx in dataset.transactions if tx.transaction_id not in excluded_ids]
        log.info("▸ Tier 3: Comms (%s) + Audio (%s)",
                 config.MODELS["comms"]["id"], config.MODELS["audio"]["id"])
        comms_risks = self.comms_agent.analyze(dataset, biotag_map, tx_subset=non_excluded)
        audio_risks = self.audio_agent.analyze(dataset, biotag_map, tx_subset=non_excluded)

        # ── FUSION ──────────────────────────────────────────────────────
        merged = self._merge_risks(dataset, tx_risks, mob_risks, comms_risks, audio_risks)

        # Remove any hard-excluded that slipped through
        for tid in excluded_ids:
            merged.pop(tid, None)

        # ── MEMORY AGENT ────────────────────────────────────────────────
        log.info("▸ Memory Agent: Analyzing patterns (%s)", config.MODELS["memory"]["id"])
        snapshot = self.memory_agent.analyze_patterns(dataset, merged, biotag_map)
        effective_threshold = snapshot.recommended_threshold
        log.info("  → Effective threshold: %.3f", effective_threshold)

        # ── THRESHOLD + CORROBORATION ───────────────────────────────────
        flagged = self._apply_threshold(merged, dataset, threshold=effective_threshold)

        # ── TIER 4: GPT-5.2 Codex holistic review ──────────────────────
        if flagged:
            log.info("▸ Tier 4: GPT-5.2 Codex review (%d candidates)", len(flagged))
            flagged = self._llm_holistic_review(flagged, merged, dataset, biotag_map)

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
    # GPT-5.2 Codex holistic review — per-transaction context
    # ------------------------------------------------------------------

    @observe(name="orchestrator.llm_holistic_review")
    def _llm_holistic_review(
        self,
        flagged: list[str],
        merged: dict[str, TransactionRisk],
        dataset: Dataset,
        biotag_map: dict[str, User],
    ) -> list[str]:
        """Gemini 2.5 Pro reviews each flagged transaction with full context."""
        tx_lookup = {tx.transaction_id: tx for tx in dataset.transactions}

        summaries: list[str] = []
        for tx_id in flagged[:40]:
            tx = tx_lookup.get(tx_id)
            tr = merged.get(tx_id)
            if not tx or not tr:
                continue
            user = biotag_map.get(tx.sender_id)
            user_info = (
                f"{user.first_name} {user.last_name}, {user.job}, "
                f"salary={user.salary}, city={user.city}, "
                f"phishing_suscept={user.phishing_susceptibility:.0%}"
                if user else "unknown user"
            )
            reasons = "; ".join(s.reason for s in tr.signals[:5])
            agents_involved = sorted({s.agent for s in tr.signals})
            summaries.append(
                f"  ID: {tx_id}\n"
                f"  Type: {tx.transaction_type}, Amount: {tx.amount}, "
                f"Time: {tx.timestamp}, Location: {tx.location or 'none'}\n"
                f"  Sender: {user_info}\n"
                f"  Recipient: {tx.recipient_id}, Desc: {tx.description or 'none'}\n"
                f"  Score: {tr.combined_score:.2f}  Agents: {agents_involved}\n"
                f"  Signals: {reasons}\n"
            )

        if not summaries:
            return flagged

        system = (
            "You are the chief fraud analyst for MirrorPay. Review these flagged "
            "transactions. For each, decide FRAUD or LEGIT.\n\n"
            "KEY PRINCIPLES:\n"
            "- IBAN mismatches (sender IBAN != profile IBAN) are VERY strong fraud signals — always FRAUD\n"
            "- Post-phishing transactions to NEW recipients with high amounts are likely FRAUD\n"
            "- Post-phishing transactions to known recipients with normal amounts are PROBABLY LEGIT\n"
            "- Impossible travel is strong fraud evidence\n"
            "- FALSE NEGATIVES cost 10x more than false positives — when uncertain, say FRAUD\n"
            "- Only mark LEGIT if you are confident the transaction is normal behavior\n\n"
            "Reply with JSON: [{\"id\": \"<tx_id>\", \"verdict\": \"FRAUD\"|\"LEGIT\"}]"
        )
        user_prompt = (
            f"Dataset: {dataset.name} — {len(dataset.transactions)} total transactions.\n\n"
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
                    log.info("LLM review removed %d false positives", removed)
                return revised
        except Exception as exc:
            log.warning("LLM final review failed: %s — keeping original flags", exc)

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
