"""Comms Analyst (Tier 3) — Claude 4.5 Sonnet.

Deep content analysis of SMS and email for phishing / social engineering.
Claude's 1M-token context lets us feed entire conversation histories
and detect evolving social engineering tactics with high nuance.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from html.parser import HTMLParser

from langfuse import observe

import config
from agents.base import BaseAgent
from models.schemas import (
    Dataset,
    PhishingEvent,
    RiskSignal,
    Transaction,
    TransactionRisk,
    User,
)

log = logging.getLogger(__name__)


class _HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str):
        self._parts.append(data)

    def get_text(self) -> str:
        return " ".join(self._parts)


def strip_html(html: str) -> str:
    s = _HTMLStripper()
    s.feed(html)
    return s.get_text()


_DOMAIN_RX = re.compile(r"https?://([^\s/\"'>]+)", re.IGNORECASE)
_DATE_RX = re.compile(r"Date:\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})")
_EMAIL_DATE_RX = re.compile(
    r"Date:\s*\w{3},\s+(\d{1,2}\s+\w{3}\s+\d{4}\s+\d{2}:\d{2}:\d{2})"
)


class CommsAnalyst(BaseAgent):
    name = "comms"

    @observe(name="comms_analyst.analyze")
    def analyze(
        self,
        dataset: Dataset,
        biotag_map: dict[str, User],
        tx_subset: list[Transaction] | None = None,
    ) -> dict[str, TransactionRisk]:
        transactions = tx_subset if tx_subset is not None else dataset.transactions

        phishing_events = self._detect_all_phishing(dataset)

        user_events: dict[str, list[PhishingEvent]] = {}
        for ev in phishing_events:
            user_events.setdefault(ev.user_biotag, []).append(ev)

        # Claude 4.5 Sonnet deep analysis on detected phishing
        if phishing_events:
            self._claude_deep_analysis(dataset, phishing_events, biotag_map)

        risks: dict[str, TransactionRisk] = {}
        window = timedelta(days=config.POST_PHISHING_WINDOW_DAYS)

        for tx in transactions:
            user = biotag_map.get(tx.sender_id)
            if not user or not user.biotag:
                continue

            if tx.transaction_type == "transfer" and tx.description and "salary" in tx.description.lower():
                continue

            events = user_events.get(user.biotag, [])
            if not events:
                continue

            for ev in events:
                if ev.timestamp <= tx.timestamp <= ev.timestamp + window:
                    suscept = user.phishing_susceptibility
                    days_after = (tx.timestamp - ev.timestamp).days + 1
                    decay = max(0.3, 1.0 - (days_after / config.POST_PHISHING_WINDOW_DAYS))
                    score = ev.severity * (0.5 + 0.5 * suscept) * decay
                    score = min(score, 0.85)
                    if score < 0.25:
                        continue

                    tr = risks.setdefault(
                        tx.transaction_id,
                        TransactionRisk(transaction_id=tx.transaction_id),
                    )
                    tr.signals.append(RiskSignal(
                        score=score,
                        reason=(
                            f"Post-phishing ({ev.source}, {days_after}d after): "
                            f"{ev.message_snippet[:60]}… | "
                            f"susceptibility={suscept:.0%}"
                        ),
                        agent=self.name,
                    ))
                    break

        log.info("CommsAnalyst: %d phishing events -> %d transactions flagged",
                 len(phishing_events), len(risks))
        return risks

    # ------------------------------------------------------------------
    # Heuristic phishing detection (fast first pass)
    # ------------------------------------------------------------------

    def _detect_all_phishing(self, dataset: Dataset) -> list[PhishingEvent]:
        events: list[PhishingEvent] = []
        events.extend(self._scan_sms(dataset))
        events.extend(self._scan_mails(dataset))
        return events

    def _scan_sms(self, dataset: Dataset) -> list[PhishingEvent]:
        events: list[PhishingEvent] = []
        for entry in dataset.sms_messages:
            text = entry.get("sms", "")
            score = self._heuristic_phishing_score(text)
            if score < 0.3:
                continue

            ts = self._extract_sms_timestamp(text)
            target_biotag = self._sms_to_biotag(text, dataset.users)
            if not target_biotag:
                continue

            events.append(PhishingEvent(
                user_biotag=target_biotag,
                timestamp=ts,
                source="sms",
                severity=score,
                message_snippet=text[:200],
            ))
        return events

    def _scan_mails(self, dataset: Dataset) -> list[PhishingEvent]:
        events: list[PhishingEvent] = []
        for entry in dataset.mail_messages:
            raw = entry.get("mail", "")
            plain = strip_html(raw) if "<html" in raw.lower() else raw
            score = self._heuristic_phishing_score(raw)
            if score < 0.3:
                continue

            ts = self._extract_mail_timestamp(raw)
            target_biotag = self._mail_to_biotag(raw, dataset.users)
            if not target_biotag:
                continue

            events.append(PhishingEvent(
                user_biotag=target_biotag,
                timestamp=ts,
                source="email",
                severity=score,
                message_snippet=plain[:200],
            ))
        return events

    # ------------------------------------------------------------------
    # Heuristic scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _heuristic_phishing_score(text: str) -> float:
        score = 0.0
        text_lower = text.lower()

        domains = _DOMAIN_RX.findall(text)
        for domain in domains:
            dl = domain.lower()
            for pat in config.PHISHING_DOMAIN_PATTERNS:
                if re.search(pat, dl):
                    score = max(score, config.PHISHING_CONFIRMED_SCORE)
                    break

        for kw in config.URGENCY_KEYWORDS:
            if kw.lower() in text_lower:
                score = max(score, config.PHISHING_SUSPECTED_SCORE)
                break

        from_phish = re.search(r"From:.*?(paypa1|amaz0n|netfl1x|ub3r|ch4se)", text, re.I)
        if from_phish:
            score = max(score, 0.90)

        return score

    # ------------------------------------------------------------------
    # Claude 4.5 Sonnet deep analysis
    # ------------------------------------------------------------------

    @observe(name="comms_analyst.claude_deep_analysis")
    def _claude_deep_analysis(
        self,
        dataset: Dataset,
        events: list[PhishingEvent],
        biotag_map: dict[str, User],
    ) -> None:
        """Leverage Claude 4.5 Sonnet's 1M context window to analyze
        full conversation threads for social engineering patterns."""
        user_groups: dict[str, list[PhishingEvent]] = {}
        for ev in events:
            user_groups.setdefault(ev.user_biotag, []).append(ev)

        for biotag, evts in user_groups.items():
            user = biotag_map.get(biotag)
            if not user:
                continue

            # Build comprehensive context for Claude
            all_sms_for_user = []
            for entry in dataset.sms_messages:
                text = entry.get("sms", "")
                if user.first_name.lower() in text.lower():
                    all_sms_for_user.append(text[:500])

            all_mails_for_user = []
            for entry in dataset.mail_messages:
                raw = entry.get("mail", "")
                if user.first_name.lower() in raw.lower():
                    plain = strip_html(raw) if "<html" in raw.lower() else raw
                    all_mails_for_user.append(plain[:500])

            snippets = "\n---\n".join(
                f"[{i+1}] ({ev.source}, {ev.timestamp:%Y-%m-%d}) {ev.message_snippet[:200]}"
                for i, ev in enumerate(evts[:20])
            )

            context_msgs = ""
            if all_sms_for_user:
                context_msgs += "\n\nAll SMS involving this user:\n" + "\n---\n".join(all_sms_for_user[:15])
            if all_mails_for_user:
                context_msgs += "\n\nAll emails involving this user:\n" + "\n---\n".join(all_mails_for_user[:15])

            system = (
                "You are a senior cybersecurity analyst specializing in social "
                "engineering detection. Analyze the FULL conversation history "
                "for this user. Look for:\n"
                "1. Typosquat domains (e.g. paypa1.com, amaz0n.net)\n"
                "2. Multi-stage social engineering (trust building -> urgency -> action)\n"
                "3. Urgency patterns (account suspension threats, time limits)\n"
                "4. Credential harvesting attempts\n"
                "5. Impersonation of legitimate entities\n"
                "6. Subtle manipulation over multiple messages\n\n"
                "For each flagged message, classify severity (0-1 where 1 = "
                "certain phishing). Return JSON: a list of objects with keys "
                "'index' (1-based) and 'severity' (float 0-1)."
            )
            user_prompt = (
                f"User: {user.first_name} {user.last_name}, "
                f"{user.job} in {user.city}.\n"
                f"Phishing susceptibility: {user.phishing_susceptibility:.0%}.\n\n"
                f"Flagged messages:\n{snippets}"
                f"{context_msgs}\n\n"
                "Respond ONLY with the JSON array."
            )

            try:
                result = self._call_llm_json(system, user_prompt)
                if isinstance(result, list):
                    for item in result:
                        idx = int(item.get("index", 0)) - 1
                        sev = float(item.get("severity", 0.5))
                        if 0 <= idx < len(evts):
                            evts[idx].severity = sev
            except Exception as exc:
                log.warning("Claude deep analysis failed for %s: %s", biotag, exc)

    # ------------------------------------------------------------------
    # Entity resolution helpers
    # ------------------------------------------------------------------

    def _sms_to_biotag(self, text: str, users: list[User]) -> str | None:
        for u in users:
            if u.first_name.lower() in text.lower() and u.biotag:
                return u.biotag
        phones = re.findall(r"\+\d{10,15}", text)
        for ph in phones:
            for u in users:
                if u.phone == ph and u.biotag:
                    return u.biotag
        return None

    def _mail_to_biotag(self, text: str, users: list[User]) -> str | None:
        text_lower = text.lower()
        for u in users:
            target = f"{u.first_name.lower()}.{u.last_name.lower()}"
            if target in text_lower and u.biotag:
                return u.biotag
            if u.first_name.lower() in text_lower and u.biotag:
                return u.biotag
        return None

    @staticmethod
    def _extract_sms_timestamp(text: str) -> datetime:
        m = _DATE_RX.search(text)
        if m:
            try:
                return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
            except ValueError:
                pass
        return datetime(2087, 6, 1)

    @staticmethod
    def _extract_mail_timestamp(text: str) -> datetime:
        m = _EMAIL_DATE_RX.search(text)
        if m:
            try:
                return datetime.strptime(m.group(1), "%d %b %Y %H:%M:%S")
            except ValueError:
                pass
        return datetime(2087, 6, 1)
