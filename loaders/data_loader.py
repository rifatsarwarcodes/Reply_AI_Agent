"""Load every file type in a dataset directory into domain objects."""

from __future__ import annotations

import csv
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from models.schemas import Dataset, LocationPing, Transaction, User

log = logging.getLogger(__name__)

_TS_FMT = "%Y-%m-%dT%H:%M:%S"


def _parse_ts(raw: str) -> datetime:
    return datetime.strptime(raw[:19], _TS_FMT)


def _safe_float(val: str) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


# ---------------------------------------------------------------------------
# Individual loaders
# ---------------------------------------------------------------------------

def load_users(path: Path) -> list[User]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    users: list[User] = []
    for u in raw:
        res = u.get("residence", {})
        users.append(User(
            first_name=u["first_name"],
            last_name=u["last_name"],
            birth_year=u["birth_year"],
            salary=u["salary"],
            job=u["job"],
            iban=u["iban"],
            city=res.get("city", ""),
            lat=float(res.get("lat", 0)),
            lng=float(res.get("lng", 0)),
            description=u.get("description", ""),
        ))
    return users


def load_transactions(path: Path) -> list[Transaction]:
    txs: list[Transaction] = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            txs.append(Transaction(
                transaction_id=row["transaction_id"],
                sender_id=row.get("sender_id", ""),
                recipient_id=row.get("recipient_id", ""),
                transaction_type=row.get("transaction_type", ""),
                amount=_safe_float(row.get("amount", "0")),
                location=row.get("location", ""),
                payment_method=row.get("payment_method", ""),
                sender_iban=row.get("sender_iban", ""),
                recipient_iban=row.get("recipient_iban", ""),
                balance_after=_safe_float(row.get("balance_after", "0")),
                description=row.get("description", ""),
                timestamp=_parse_ts(row["timestamp"]),
            ))
    return txs


def load_locations(path: Path) -> list[LocationPing]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return [
        LocationPing(
            biotag=p["biotag"],
            timestamp=_parse_ts(p["timestamp"]),
            lat=float(p["lat"]),
            lng=float(p["lng"]),
            city=p.get("city", ""),
        )
        for p in raw
    ]


def load_sms(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_mails(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def list_audio_files(folder: Path) -> list[str]:
    if not folder.exists():
        return []
    return sorted(str(p) for p in folder.glob("*.mp3"))


# ---------------------------------------------------------------------------
# Biotag resolver — maps users.json entries to their biotag strings that
# appear in transactions and locations.
# ---------------------------------------------------------------------------

_BIOTAG_RE = re.compile(r"^[A-ZÀ-ÖØ-Þ]{3,5}-[A-ZÀ-ÖØ-Þ]{3,5}-[A-Z0-9]{2,4}-[A-Z]{2,4}-[01]$", re.IGNORECASE)


def _is_biotag(val: str) -> bool:
    return bool(_BIOTAG_RE.match(val))


def resolve_biotags(users: list[User], transactions: list[Transaction],
                    locations: list[LocationPing]) -> dict[str, User]:
    """Return a mapping  biotag → User  by matching IBANs."""
    iban_to_user = {u.iban: u for u in users}

    biotag_ibans: dict[str, set[str]] = {}
    for tx in transactions:
        if _is_biotag(tx.sender_id):
            biotag_ibans.setdefault(tx.sender_id, set()).add(tx.sender_iban)
        if _is_biotag(tx.recipient_id):
            biotag_ibans.setdefault(tx.recipient_id, set()).add(tx.recipient_iban)

    mapping: dict[str, User] = {}
    for biotag, ibans in biotag_ibans.items():
        for iban in ibans:
            if iban in iban_to_user:
                user = iban_to_user[iban]
                user.biotag = biotag
                mapping[biotag] = user
                break

    loc_biotags = {p.biotag for p in locations}
    for biotag in loc_biotags:
        if biotag in mapping:
            continue
        for user in users:
            ln = user.last_name.upper()[:4]
            fn = user.first_name.upper()[:4]
            if ln[:3] in biotag.upper() and fn[:2] in biotag.upper():
                user.biotag = biotag
                mapping[biotag] = user
                break

    return mapping


# ---------------------------------------------------------------------------
# Phone / email resolution from comms data
# ---------------------------------------------------------------------------

def resolve_phones(users: list[User], sms_messages: list[dict]) -> None:
    """Best-effort association of phone numbers to users via name mentions."""
    phone_names: dict[str, set[str]] = {}
    for entry in sms_messages:
        text = entry.get("sms", "")
        phones = re.findall(r"\+\d{10,15}", text)
        for ph in phones:
            phone_names.setdefault(ph, set())
            for u in users:
                if u.first_name.lower() in text.lower():
                    phone_names[ph].add(u.first_name)

    for ph, names in phone_names.items():
        if len(names) == 1:
            name = next(iter(names))
            for u in users:
                if u.first_name == name and u.phone is None:
                    u.phone = ph


def resolve_emails(users: list[User], mail_messages: list[dict]) -> None:
    for entry in mail_messages:
        text = entry.get("mail", "")
        for u in users:
            pattern = f"{u.first_name.lower()}.{u.last_name.lower()}"
            if pattern in text.lower():
                match = re.search(
                    rf'["\']?{re.escape(u.first_name)}[^"]*?<([^>]+)>',
                    text, re.IGNORECASE,
                )
                if match:
                    u.email = match.group(1)
                elif u.email is None:
                    u.email = f"{u.first_name.lower()}.{u.last_name.lower()}@example.com"


# ---------------------------------------------------------------------------
# Top-level loader
# ---------------------------------------------------------------------------

def load_dataset(dataset_path: str | Path) -> Dataset:
    p = Path(dataset_path)
    name = p.name

    log.info("Loading dataset: %s", name)

    users = load_users(p / "users.json")
    transactions = load_transactions(p / "transactions.csv")
    locations = load_locations(p / "locations.json")
    sms = load_sms(p / "sms.json") if (p / "sms.json").exists() else []
    mails = load_mails(p / "mails.json") if (p / "mails.json").exists() else []
    audio = list_audio_files(p / "audio")

    biotag_map = resolve_biotags(users, transactions, locations)
    resolve_phones(users, sms)
    resolve_emails(users, mails)

    log.info(
        "Loaded %d users, %d txs, %d locations, %d sms, %d mails, %d audio — %d biotags resolved",
        len(users), len(transactions), len(locations),
        len(sms), len(mails), len(audio), len(biotag_map),
    )

    return Dataset(
        name=name, path=str(p),
        users=users, transactions=transactions, locations=locations,
        sms_messages=sms, mail_messages=mails, audio_files=audio,
    )
