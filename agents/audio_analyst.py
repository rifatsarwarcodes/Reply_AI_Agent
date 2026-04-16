"""Audio Analyst (Tier 3) — Gemini 3.1 Pro Preview.

Processes MP3 files natively using Gemini's multimodal capabilities.
Sends audio directly to Gemini 3.1 Pro for vishing / social-engineering
detection, eliminating the need for a separate transcription step.
"""

from __future__ import annotations

import json as _json
import logging
import os
import re
from datetime import datetime, timedelta

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

_FNAME_RE = re.compile(r"(\d{8})_(\d{6})-(.+)\.mp3$")


def _parse_audio_filename(filepath: str):
    """Return (datetime, user_name_key) or None."""
    fname = os.path.basename(filepath)
    m = _FNAME_RE.search(fname)
    if not m:
        return None
    date_str, time_str, name_raw = m.group(1), m.group(2), m.group(3)
    ts = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
    name_key = name_raw.replace("_", " ").lower()
    return ts, name_key


class AudioAnalyst(BaseAgent):
    name = "audio"

    @observe(name="audio_analyst.analyze")
    def analyze(
        self,
        dataset: Dataset,
        biotag_map: dict[str, User],
        tx_subset: list[Transaction] | None = None,
    ) -> dict[str, TransactionRisk]:
        if not dataset.audio_files:
            log.info("AudioAnalyst: no audio files — skipping")
            return {}

        transactions = tx_subset if tx_subset is not None else dataset.transactions

        name_to_user: dict[str, User] = {}
        for u in dataset.users:
            key = f"{u.first_name} {u.last_name}".lower()
            name_to_user[key] = u

        audio_events: list[tuple[datetime, User, dict]] = []

        for fpath in dataset.audio_files:
            parsed = _parse_audio_filename(fpath)
            if not parsed:
                continue
            ts, name_key = parsed

            user = name_to_user.get(name_key)
            if not user:
                for full_name, u in name_to_user.items():
                    if name_key.split()[0] in full_name:
                        user = u
                        break
            if not user:
                continue

            analysis = self._gemini_native_audio_analysis(fpath, user)
            audio_events.append((ts, user, analysis))

        risks = self._propagate_to_transactions(audio_events, transactions, biotag_map)
        log.info("AudioAnalyst: %d audio files -> %d transactions flagged",
                 len(dataset.audio_files), len(risks))
        return risks

    # ------------------------------------------------------------------
    # Native Gemini audio analysis
    # ------------------------------------------------------------------

    @observe(name="audio_analyst.gemini_native")
    def _gemini_native_audio_analysis(self, filepath: str, user: User) -> dict:
        """Send MP3 directly to Gemini 3.1 Pro via OpenRouter multimodal API."""
        try:
            audio_b64 = self._encode_audio_b64(filepath)

            model = config.MODELS["audio"]["id"]
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a fraud analyst specializing in voice phishing (vishing) "
                        "detection. Analyze this phone call audio for signs of social "
                        "engineering, impersonation, urgency tactics, or attempts to "
                        "extract personal/financial information. "
                        "Respond with JSON: {\"is_vishing\": true/false, "
                        "\"confidence\": 0.0-1.0, \"tactics\": [\"list of tactics\"], "
                        "\"summary\": \"brief description\"}"
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"Caller/recipient: {user.first_name} {user.last_name}, "
                                f"{user.job} in {user.city}. "
                                f"Phishing susceptibility: {user.phishing_susceptibility:.0%}. "
                                f"Analyze this call for vishing indicators."
                            ),
                        },
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio_b64,
                                "format": "mp3",
                            },
                        },
                    ],
                },
            ]

            raw = self._call_openrouter_raw(
                model=model,
                messages=messages,
                temperature=config.MODELS["audio"]["temperature"],
                max_tokens=config.MODELS["audio"]["max_tokens"],
            )

            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
            result = _json.loads(raw)
            if isinstance(result, dict):
                log.info("Gemini audio analysis for %s: vishing=%s conf=%.2f",
                         os.path.basename(filepath),
                         result.get("is_vishing"), result.get("confidence", 0))
                return result

        except Exception as exc:
            log.warning("Gemini native audio failed for %s: %s — falling back to filename heuristic",
                        os.path.basename(filepath), exc)

        return {"is_vishing": False, "confidence": 0.2, "summary": "analysis unavailable"}

    # ------------------------------------------------------------------
    # Propagation
    # ------------------------------------------------------------------

    @staticmethod
    def _propagate_to_transactions(
        audio_events: list[tuple[datetime, User, dict]],
        transactions: list[Transaction],
        biotag_map: dict[str, User],
    ) -> dict[str, TransactionRisk]:
        risks: dict[str, TransactionRisk] = {}
        window = timedelta(days=config.POST_PHISHING_WINDOW_DAYS)

        for ts, user, analysis in audio_events:
            if not user.biotag:
                continue

            is_vishing = analysis.get("is_vishing", False)
            confidence = float(analysis.get("confidence", 0.2))
            base_score = 0.70 * confidence if is_vishing else 0.15 * confidence

            for tx in transactions:
                if tx.sender_id != user.biotag:
                    continue
                if ts <= tx.timestamp <= ts + window:
                    days_after = (tx.timestamp - ts).days + 1
                    decay = max(0.3, 1.0 - (days_after / config.POST_PHISHING_WINDOW_DAYS))
                    score = base_score * (0.5 + 0.5 * user.phishing_susceptibility) * decay
                    score = min(score, 0.80)
                    if score < 0.15:
                        continue

                    tr = risks.setdefault(
                        tx.transaction_id,
                        TransactionRisk(transaction_id=tx.transaction_id),
                    )
                    tactics = ", ".join(analysis.get("tactics", [])[:3])
                    tr.signals.append(RiskSignal(
                        score=score,
                        reason=(
                            f"Post-vishing ({ts:%Y-%m-%d %H:%M}, conf={confidence:.0%}): "
                            f"{tactics or analysis.get('summary', '')[:60]}"
                        ),
                        agent="audio",
                    ))
        return risks
