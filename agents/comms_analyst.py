"""Comms Analyst (Tier 3) — Gemini 2.5 Pro.

Scans SMS and email for phishing, propagates risk to transactions.

Strategy: Flag all non-salary transactions from phished users within
the window, but BOOST score when corroborating signals exist (new
recipient, unusual amount).  This catches more fraud while the
scoring system and final LLM review filter false positives.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from datetime import datetime, timedelta
from html.parser import HTMLParser
from statistics import median

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


def _is_salary(tx: Transaction) -> bool:
    if tx.sender_id.startswith("EMP"):
        return True
    desc = (tx.description or "").lower()
    return "salary" in desc


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

        if phishing_events:
            self._llm_refine(dataset, phishing_events, biotag_map)

        user_recipients = self._build_user_recipient_history(dataset)
        user_medians = self._build_user_amount_medians(dataset, biotag_map)

        risks: dict[str, TransactionRisk] = {}
        window = timedelta(days=config.POST_PHISHING_WINDOW_DAYS)

        for tx in transactions:
            user = biotag_map.get(tx.sender_id)
            if not user or not user.biotag:
                continue

            # Only exclude salary — everything else is fair game
            if _is_salary(tx):
                continue

            events = user_events.get(user.biotag, [])
            if not events:
                continue

            # Detect corroborating signals for score boosting
            is_new_recipient = tx.recipient_id not in user_recipients.get(user.biotag, set())
            is_high_amount = self._is_unusually_high(tx, user, user_medians)
            is_night = tx.timestamp.hour in config.NIGHT_HOURS

            for ev in events:
                if ev.timestamp <= tx.timestamp <= ev.timestamp + window:
                    suscept = user.phishing_susceptibility
                    days_after = (tx.timestamp - ev.timestamp).days + 1
                    decay = max(0.3, 1.0 - (days_after / config.POST_PHISHING_WINDOW_DAYS))

                    base = ev.severity * (0.5 + 0.5 * suscept) * decay

                    # Boost for corroborating signals
                    boost = 1.0
                    corr = []
                    if is_new_recipient:
                        boost += 0.25
                        corr.append("new_recipient")
                    if is_high_amount:
                        boost += 0.20
                        corr.append("unusual_amount")
                    if is_night:
                        boost += 0.10
                        corr.append("night_time")

                    # Dampen for clearly routine (rent, subscription)
                    desc_lower = (tx.description or "").lower()
                    if any(kw in desc_lower for kw in ("rent payment", "subscription", "insurance")):
                        boost *= 0.6

                    score = base * boost
                    score = min(score, 0.85)
                    if score < 0.20:
                        continue

                    tr = risks.setdefault(
                        tx.transaction_id,
                        TransactionRisk(transaction_id=tx.transaction_id),
                    )
                    tr.signals.append(RiskSignal(
                        score=score,
                        reason=(
                            f"Post-phishing ({ev.source}, {days_after}d): "
                            f"{ev.message_snippet[:50]}… | "
                            f"corr=[{','.join(corr) or 'none'}] "
                            f"suscept={suscept:.0%}"
                        ),
                        agent=self.name,
                    ))
                    break

        log.info("CommsAnalyst: %d phishing events -> %d transactions flagged",
                 len(phishing_events), len(risks))
        return risks

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_user_recipient_history(dataset: Dataset) -> dict[str, set[str]]:
        history: dict[str, set[str]] = defaultdict(set)
        for tx in dataset.transactions:
            if tx.sender_id and tx.recipient_id:
                history[tx.sender_id].add(tx.recipient_id)
        return dict(history)

    @staticmethod
    def _build_user_amount_medians(
        dataset: Dataset, biotag_map: dict[str, User]
    ) -> dict[tuple[str, str], float]:
        buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
        for tx in dataset.transactions:
            user = biotag_map.get(tx.sender_id)
            if user and user.biotag:
                buckets[(user.biotag, tx.transaction_type)].append(tx.amount)
        return {k: median(v) for k, v in buckets.items() if v}

    @staticmethod
    def _is_unusually_high(
        tx: Transaction,
        user: User,
        medians: dict[tuple[str, str], float],
    ) -> bool:
        if not user.biotag:
            return False
        key = (user.biotag, tx.transaction_type)
        med = medians.get(key)
        if med and med > 0:
            return tx.amount > med * 2.0
        return tx.amount > 2000

    # ------------------------------------------------------------------
    # Phishing detection
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
                user_biotag=target_biotag, timestamp=ts,
                source="sms", severity=score,
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
                user_biotag=target_biotag, timestamp=ts,
                source="email", severity=score,
                message_snippet=plain[:200],
            ))
        return events

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

        urgency_count = sum(1 for kw in config.URGENCY_KEYWORDS if kw.lower() in text_lower)
        if urgency_count >= 2:
            score = max(score, config.PHISHING_SUSPECTED_SCORE)
        elif urgency_count == 1 and score >= 0.5:
            score = max(score, config.PHISHING_SUSPECTED_SCORE)

        from_phish = re.search(r"From:.*?(paypa1|amaz0n|netfl1x|ub3r|ch4se)", text, re.I)
        if from_phish:
            score = max(score, 0.90)

        return score

    # ------------------------------------------------------------------
    # LLM refinement
    # ------------------------------------------------------------------

    @observe(name="comms_analyst.llm_refine")
    def _llm_refine(
        self,
        dataset: Dataset,
        events: list[PhishingEvent],
        biotag_map: dict[str, User],
    ) -> None:
        user_groups: dict[str, list[PhishingEvent]] = {}
        for ev in events:
            user_groups.setdefault(ev.user_biotag, []).append(ev)

        for biotag, evts in user_groups.items():
            user = biotag_map.get(biotag)
            if not user:
                continue

            snippets = "\n---\n".join(
                f"[{i+1}] ({ev.source}, {ev.timestamp:%Y-%m-%d}) {ev.message_snippet[:200]}"
                for i, ev in enumerate(evts[:20])
            )

            system = (
                "You are a cybersecurity analyst. Classify each message: "
                "is it PHISHING or LEGITIMATE? Be strict: legit services "
                "also use urgency language. Only mark phishing if there are "
                "typosquat domains, fake senders, or clear social engineering. "
                "Return JSON: [{\"index\": <1-based>, \"severity\": <float 0-1>}]"
            )
            user_prompt = (
                f"User: {user.first_name} {user.last_name}, {user.job} in {user.city}.\n\n"
                f"Messages:\n{snippets}\n\nRespond ONLY with the JSON array."
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
                log.warning("LLM refinement failed for %s: %s", biotag, exc)

    # ------------------------------------------------------------------
    # Entity resolution
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
