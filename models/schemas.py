"""Domain data-classes shared across agents."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# ---------------------------------------------------------------------------
# Core entities
# ---------------------------------------------------------------------------

@dataclass
class User:
    first_name: str
    last_name: str
    birth_year: int
    salary: float
    job: str
    iban: str
    city: str
    lat: float
    lng: float
    description: str
    biotag: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    phishing_susceptibility: float = 0.35

    def __post_init__(self):
        self.phishing_susceptibility = _extract_susceptibility(self.description)


@dataclass
class Transaction:
    transaction_id: str
    sender_id: str
    recipient_id: str
    transaction_type: str
    amount: float
    location: str
    payment_method: str
    sender_iban: str
    recipient_iban: str
    balance_after: float
    description: str
    timestamp: datetime


@dataclass
class LocationPing:
    biotag: str
    timestamp: datetime
    lat: float
    lng: float
    city: str


# ---------------------------------------------------------------------------
# Agent outputs
# ---------------------------------------------------------------------------

@dataclass
class RiskSignal:
    score: float
    reason: str
    agent: str


@dataclass
class TransactionRisk:
    transaction_id: str
    signals: list[RiskSignal] = field(default_factory=list)

    @property
    def max_score(self) -> float:
        return max((s.score for s in self.signals), default=0.0)

    @property
    def combined_score(self) -> float:
        if not self.signals:
            return 0.0
        top = self.max_score
        n_flagging = sum(1 for s in self.signals if s.score > 0.25)
        boosted = top * (1.0 + 0.1 * max(n_flagging - 1, 0))
        return min(boosted, 1.0)


@dataclass
class PhishingEvent:
    user_biotag: str
    timestamp: datetime
    source: str            # "sms" or "email"
    severity: float        # 0-1
    message_snippet: str


@dataclass
class Dataset:
    name: str
    path: str
    users: list[User]
    transactions: list[Transaction]
    locations: list[LocationPing]
    sms_messages: list[dict]
    mail_messages: list[dict]
    audio_files: list[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_susceptibility(desc: str) -> float:
    """Pull a phishing-susceptibility estimate from the user description."""
    pct = re.search(r"(\d{1,3})\s*%", desc)
    if pct:
        return int(pct.group(1)) / 100.0

    low_trust = [
        "too trusting", "dubious links", "suspicious link",
        "flashy online lure", "online lure", "clicked dubious",
        "risky links", "online mishaps", "too confident",
    ]
    for phrase in low_trust:
        if phrase in desc.lower():
            return 0.55

    moderate_trust = [
        "pragmatically susceptible", "not always prudent",
        "pas toujours prudent", "mäßig vorsichtig",
        "fidarsi dei messaggi", "confiance vis-à-vis",
    ]
    for phrase in moderate_trust:
        if phrase in desc.lower():
            return 0.48

    return 0.35
