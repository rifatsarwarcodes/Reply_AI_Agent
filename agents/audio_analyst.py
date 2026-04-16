"""Audio Analyst — processes MP3 files from the Deus Ex level.
Extracts metadata from filenames and optionally transcribes audio
for vishing / social-engineering detection."""

from __future__ import annotations

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
    TransactionRisk,
    User,
)

log = logging.getLogger(__name__)

_FNAME_RE = re.compile(
    r"(\d{8})_(\d{6})-(.+)\.mp3$"
)


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


def _transcribe(filepath: str) -> str | None:
    """Attempt transcription via whisper (if installed)."""
    try:
        import whisper  # type: ignore
        model = whisper.load_model("base")
        result = model.transcribe(filepath, language="en")
        return result.get("text", "")
    except ImportError:
        return None
    except Exception as exc:
        log.warning("Whisper failed on %s: %s", filepath, exc)
        return None


class AudioAnalyst(BaseAgent):
    name = "audio"

    @observe(name="audio_analyst.analyze")
    def analyze(self, dataset: Dataset, biotag_map: dict[str, User]) -> dict[str, TransactionRisk]:
        if not dataset.audio_files:
            log.info("AudioAnalyst: no audio files — skipping")
            return {}

        name_to_user: dict[str, User] = {}
        for u in dataset.users:
            key = f"{u.first_name} {u.last_name}".lower()
            name_to_user[key] = u

        audio_events: list[tuple[datetime, User, str]] = []

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

            transcript = _transcribe(fpath)
            analysis = ""
            if transcript:
                analysis = self._llm_analyze_transcript(transcript, user)

            audio_events.append((ts, user, analysis))

        risks = self._propagate_to_transactions(audio_events, dataset, biotag_map)
        log.info("AudioAnalyst: %d audio files → %d transactions flagged",
                 len(dataset.audio_files), len(risks))
        return risks

    # ------------------------------------------------------------------

    @observe(name="audio_analyst.llm_analyze")
    def _llm_analyze_transcript(self, transcript: str, user: User) -> str:
        system = (
            "You are a fraud analyst reviewing a phone call transcript. "
            "Determine if this is a legitimate call or a vishing / social "
            "engineering attack. Consider: urgency, requests for personal "
            "info, impersonation of banks/authorities, pressure tactics. "
            "Reply with JSON: {\"is_vishing\": true/false, \"confidence\": 0-1, "
            "\"reason\": \"brief explanation\"}"
        )
        user_prompt = (
            f"Caller / recipient: {user.first_name} {user.last_name}, "
            f"{user.job} in {user.city}.\n"
            f"Phishing susceptibility: {user.phishing_susceptibility:.0%}.\n\n"
            f"Transcript:\n{transcript[:3000]}"
        )
        try:
            result = self._call_llm_json(system, user_prompt)
            if isinstance(result, dict):
                return f"vishing={result.get('is_vishing')}, conf={result.get('confidence')}"
        except Exception as exc:
            log.warning("Audio LLM analysis failed: %s", exc)
        return ""

    # ------------------------------------------------------------------

    @staticmethod
    def _propagate_to_transactions(
        audio_events: list[tuple[datetime, User, str]],
        dataset: Dataset,
        biotag_map: dict[str, User],
    ) -> dict[str, TransactionRisk]:
        """Flag transactions within a window after each audio event for
        the relevant user (analogous to post-phishing window)."""
        risks: dict[str, TransactionRisk] = {}
        window = timedelta(days=config.POST_PHISHING_WINDOW_DAYS)

        for ts, user, analysis in audio_events:
            if not user.biotag:
                continue
            is_vishing = "vishing=True" in analysis or "vishing=true" in analysis
            base_score = 0.60 if is_vishing else 0.25

            for tx in dataset.transactions:
                if tx.sender_id != user.biotag:
                    continue
                if ts <= tx.timestamp <= ts + window:
                    score = base_score * user.phishing_susceptibility
                    score = min(score, 0.80)
                    if score < 0.15:
                        continue
                    tr = risks.setdefault(
                        tx.transaction_id,
                        TransactionRisk(transaction_id=tx.transaction_id),
                    )
                    tr.signals.append(RiskSignal(
                        score=score,
                        reason=f"Post-audio event ({ts:%Y-%m-%d %H:%M}): {analysis[:80]}",
                        agent="audio",
                    ))
        return risks
