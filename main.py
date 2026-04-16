#!/usr/bin/env python3
"""Entry point for the MirrorPay fraud-detection agent pipeline.

Usage
-----
  # Run on a single dataset
  python main.py "Datasets/The Truman Show - train"

  # Run on every dataset found in Datasets/
  python main.py --all
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import ulid
from dotenv import load_dotenv
from langfuse import Langfuse

load_dotenv()

import config  # noqa: E402  (needs env loaded first)
from agents.orchestrator import Orchestrator  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main")


# ---------------------------------------------------------------------------
# Langfuse session management
# ---------------------------------------------------------------------------

def _init_langfuse() -> Langfuse:
    return Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST", "https://challenges.reply.com/langfuse"),
    )


def _session_id() -> str:
    team = os.getenv("TEAM_NAME", "default-team").replace(" ", "-")
    return f"{team}-{ulid.new().str}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _discover_datasets() -> list[Path]:
    found = sorted(config.DATASETS_DIR.glob("*- train"))
    if not found:
        found = sorted(config.DATASETS_DIR.glob("*- eval"))
    if not found:
        found = sorted(
            p for p in config.DATASETS_DIR.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        )
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description="MirrorPay fraud detection pipeline")
    parser.add_argument(
        "dataset",
        nargs="?",
        help="Path to dataset folder (relative to project root)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run on every dataset discovered in Datasets/",
    )
    args = parser.parse_args()

    langfuse = _init_langfuse()
    session = _session_id()
    os.environ["LANGFUSE_SESSION_ID"] = session
    log.info("Session: %s", session)

    orchestrator = Orchestrator()

    if args.all:
        datasets = _discover_datasets()
        if not datasets:
            log.error("No datasets found under %s", config.DATASETS_DIR)
            sys.exit(1)
        log.info("Discovered %d dataset(s)", len(datasets))
    elif args.dataset:
        p = config.BASE_DIR / args.dataset
        if not p.exists():
            p = Path(args.dataset)
        datasets = [p]
    else:
        datasets = _discover_datasets()
        if not datasets:
            parser.print_help()
            sys.exit(1)

    total_flagged = 0
    for ds_path in datasets:
        log.info("━" * 60)
        log.info("Processing: %s", ds_path.name)
        log.info("━" * 60)
        flagged = orchestrator.run(str(ds_path))
        total_flagged += len(flagged)
        log.info("  → %d transactions flagged as fraudulent\n", len(flagged))

    langfuse.flush()
    log.info("Done. Total flagged across all datasets: %d", total_flagged)
    log.info("Session ID: %s", session)
    log.info("Output files in: %s", config.OUTPUT_DIR)


if __name__ == "__main__":
    main()
