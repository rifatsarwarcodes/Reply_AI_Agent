"""Orchestrator — loads a dataset, runs every specialist agent, fuses their
risk scores, applies a cost-sensitive threshold, and writes the final
output file of suspected fraudulent transaction IDs."""

from __future__ import annotations

import logging
from pathlib import Path

from langfuse import observe

import config
from agents.base import BaseAgent
from agents.transaction_analyst import TransactionAnalyst
from agents.mobility_analyst import MobilityAnalyst
from agents.comms_analyst import CommsAnalyst
from agents.audio_analyst import AudioAnalyst
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
        self.tx_agent = TransactionAnalyst(llm=self.llm)
        self.mob_agent = MobilityAnalyst(llm=self.llm)
        self.comms_agent = CommsAnalyst(llm=self.llm)
        self.audio_agent = AudioAnalyst(llm=self.llm)

    @observe(name="orchestrator.run")
    def run(self, dataset_path: str) -> list[str]:
        dataset = load_dataset(dataset_path)
        biotag_map = resolve_biotags(
            dataset.users, dataset.transactions, dataset.locations
        )

        log.info("=== Running agents on '%s' (%d txs, %d users) ===",
                 dataset.name, len(dataset.transactions), len(dataset.users))

        tx_risks = self.tx_agent.analyze(dataset, biotag_map)
        mob_risks = self.mob_agent.analyze(dataset, biotag_map)
        comms_risks = self.comms_agent.analyze(dataset, biotag_map)
        audio_risks = self.audio_agent.analyze(dataset, biotag_map)

        merged = self._merge_risks(
            dataset, tx_risks, mob_risks, comms_risks, audio_risks
        )

        flagged = self._apply_threshold(merged, dataset)

        if flagged:
            flagged = self._llm_final_review(flagged, merged, dataset, biotag_map)

        output_path = self._write_output(dataset.name, flagged)
        log.info("Output written to %s  (%d flagged / %d total = %.1f%%)",
                 output_path, len(flagged), len(dataset.transactions),
                 100.0 * len(flagged) / max(len(dataset.transactions), 1))
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
    ) -> list[str]:
        scored = [
            (tx_id, tr.combined_score, tr)
            for tx_id, tr in merged.items()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        max_flags = int(len(dataset.transactions) * config.MAX_FLAG_RATIO)

        flagged: list[str] = []
        for tx_id, score, tr in scored:
            if score < config.FRAUD_THRESHOLD:
                break
            flagged.append(tx_id)
            if len(flagged) >= max_flags:
                break

        return flagged

    # ------------------------------------------------------------------
    # LLM final review on borderline cases
    # ------------------------------------------------------------------

    @observe(name="orchestrator.llm_final_review")
    def _llm_final_review(
        self,
        flagged: list[str],
        merged: dict[str, TransactionRisk],
        dataset: Dataset,
        biotag_map: dict[str, User],
    ) -> list[str]:
        """Quick LLM sanity-check: present the top flagged transactions and
        ask the model to confirm or override."""
        tx_lookup = {tx.transaction_id: tx for tx in dataset.transactions}

        summaries: list[str] = []
        for tx_id in flagged[:30]:
            tx = tx_lookup.get(tx_id)
            tr = merged.get(tx_id)
            if not tx or not tr:
                continue
            reasons = "; ".join(s.reason for s in tr.signals[:4])
            summaries.append(
                f"  ID: {tx_id}\n"
                f"  Type: {tx.transaction_type}, Amount: {tx.amount}, "
                f"Time: {tx.timestamp}, Sender: {tx.sender_id}\n"
                f"  Score: {tr.combined_score:.2f}  Signals: {reasons}\n"
            )

        if not summaries:
            return flagged

        system = (
            "You are a senior fraud analyst. Review these flagged transactions. "
            "For each, reply FRAUD or LEGIT. A false negative (missing real fraud) "
            "costs much more than a false positive. When in doubt, say FRAUD. "
            "Reply with JSON: [{\"id\": \"<transaction_id>\", \"verdict\": \"FRAUD\"|\"LEGIT\"}]"
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
