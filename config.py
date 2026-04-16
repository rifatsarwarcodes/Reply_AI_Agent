"""Central configuration for the fraud detection pipeline."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = BASE_DIR / "Datasets"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 4096

# ---------------------------------------------------------------------------
# Agent scoring weights  (used in orchestrator fusion)
# ---------------------------------------------------------------------------
AGENT_WEIGHTS = {
    "transaction": 0.35,
    "mobility": 0.25,
    "comms": 0.25,
    "audio": 0.15,
}

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
# Final risk score above which a transaction is flagged as fraud.
# Biased low because false negatives cost much more than false positives.
FRAUD_THRESHOLD = 0.35

# Max fraction of transactions we're willing to flag (safety cap).
MAX_FLAG_RATIO = 0.50

# ---------------------------------------------------------------------------
# Transaction analyst
# ---------------------------------------------------------------------------
IBAN_MISMATCH_SCORE = 0.90
AMOUNT_ZSCORE_HIGH = 3.0
AMOUNT_ZSCORE_MED = 2.0
AMOUNT_HIGH_SCORE = 0.50
AMOUNT_MED_SCORE = 0.30
NIGHT_HOURS = range(0, 6)
NIGHT_SCORE = 0.15
NEW_RECIPIENT_SCORE = 0.10

# ---------------------------------------------------------------------------
# Mobility analyst
# ---------------------------------------------------------------------------
IMPOSSIBLE_SPEED_KMH = 900       # faster than any commercial aircraft
SUSPICIOUS_SPEED_KMH = 500
IMPOSSIBLE_TRAVEL_SCORE = 0.85
SUSPICIOUS_TRAVEL_SCORE = 0.55
FAR_FROM_HOME_KM = 500
FAR_FROM_HOME_SCORE = 0.30
LOCATION_TIME_WINDOW_H = 24      # hours around a tx to search for pings

# ---------------------------------------------------------------------------
# Comms analyst
# ---------------------------------------------------------------------------
PHISHING_DOMAIN_PATTERNS = [
    r"paypa1", r"amaz0n", r"netfl1x", r"ub3r", r"ch4se",
    r"dhl-secure", r"dhl-release", r"coinbase-secure",
    r"socsec-verify", r"ssa-secure", r"ss-aid",
    r"-verify\d{4}", r"-secure\d{4}", r"secure-login",
    r"wiesbadn-verify", r"eurobank-secure", r"wealthx-secure",
    r"invst-secure", r"northfinancc", r"claims-\d{4}-global",
    r"firstnational-alerts",
]
URGENCY_KEYWORDS = [
    "urgent", "immediately", "account lock", "verify now",
    "suspend", "within 24h", "within 48h", "avoid lock",
    "action required", "avoid suspension", "verify identity",
]
PHISHING_CONFIRMED_SCORE = 0.80
PHISHING_SUSPECTED_SCORE = 0.50
POST_PHISHING_WINDOW_DAYS = 14
DEFAULT_SUSCEPTIBILITY = 0.35
