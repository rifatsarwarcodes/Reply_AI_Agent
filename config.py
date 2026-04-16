"""Central configuration for the heterogeneous multi-model fraud detection pipeline.

Model Tiers (waterfall routing):
  Tier 1  Pre-Filter        — Gemini 2.5 Flash Lite  ($)
  Tier 2  Logic Verification — DeepSeek R1            ($$)
  Tier 3  Deep Analysis      — Gemini 2.5 Pro / Gemini 3.1 Pro ($$$)
  Tier 4  Final Decisions    — Gemini 2.5 Pro          ($$$$)
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = BASE_DIR / "Datasets"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ---------------------------------------------------------------------------
# Per-role model assignments — only using models that reliably work
# (Claude/GPT hit 404 privacy errors on this OpenRouter account)
# ---------------------------------------------------------------------------
MODELS = {
    "prefilter": {
        "id": "google/gemini-2.5-flash-lite",
        "temperature": 0.1,
        "max_tokens": 2048,
    },
    "transaction": {
        "id": "deepseek/deepseek-r1-0528",
        "temperature": 0.1,
        "max_tokens": 4096,
    },
    "mobility": {
        "id": "deepseek/deepseek-r1-0528",
        "temperature": 0.1,
        "max_tokens": 4096,
    },
    "memory": {
        "id": "deepseek/deepseek-r1",
        "temperature": 0.2,
        "max_tokens": 4096,
    },
    "comms": {
        "id": "google/gemini-2.5-pro",
        "temperature": 0.15,
        "max_tokens": 4096,
    },
    "audio": {
        "id": "google/gemini-3.1-pro-preview",
        "temperature": 0.15,
        "max_tokens": 4096,
    },
    "orchestrator": {
        "id": "google/gemini-2.5-pro",
        "temperature": 0.2,
        "max_tokens": 4096,
    },
}

DEFAULT_MODEL = "google/gemini-2.5-flash-lite"

# ---------------------------------------------------------------------------
# Agent scoring weights
# ---------------------------------------------------------------------------
AGENT_WEIGHTS = {
    "transaction": 0.35,
    "mobility": 0.25,
    "comms": 0.25,
    "audio": 0.15,
}

# ---------------------------------------------------------------------------
# Thresholds — balanced to catch 15-25% of transactions
# ---------------------------------------------------------------------------
FRAUD_THRESHOLD = 0.38
MAX_FLAG_RATIO = 0.40

FRAUD_THRESHOLD_MIN = 0.32
FRAUD_THRESHOLD_MAX = 0.50

# ---------------------------------------------------------------------------
# Hard exclusion: ONLY salary payments are truly immune
# ---------------------------------------------------------------------------
LEGIT_SENDER_PREFIXES = ("EMP",)
LEGIT_DESCRIPTION_KEYWORDS = ("salary payment", "salary")

# ---------------------------------------------------------------------------
# Transaction analyst
# ---------------------------------------------------------------------------
IBAN_MISMATCH_SCORE = 0.90
AMOUNT_ZSCORE_HIGH = 3.0
AMOUNT_ZSCORE_MED = 2.0
AMOUNT_HIGH_SCORE = 0.50
AMOUNT_MED_SCORE = 0.30
NIGHT_HOURS = range(0, 6)
NIGHT_SCORE = 0.20
NEW_RECIPIENT_SCORE = 0.15

# ---------------------------------------------------------------------------
# Mobility analyst
# ---------------------------------------------------------------------------
IMPOSSIBLE_SPEED_KMH = 900
SUSPICIOUS_SPEED_KMH = 500
IMPOSSIBLE_TRAVEL_SCORE = 0.85
SUSPICIOUS_TRAVEL_SCORE = 0.55
FAR_FROM_HOME_KM = 500
FAR_FROM_HOME_SCORE = 0.30
LOCATION_TIME_WINDOW_H = 24

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

POST_PHISHING_WINDOW_DAYS = 10

# Corroboration: only exclude salary, propagate to all other tx from phished user
PHISHING_REQUIRE_CORROBORATION = False

DEFAULT_SUSCEPTIBILITY = 0.35
