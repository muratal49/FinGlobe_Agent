#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Root Agent Orchestrator

Runs the full BoE pipeline end-to-end:

  1A. meeting_scraper.py                -> data/raw/minutes_boe.json
  1B. scrape_boe_speeches.py            -> data/raw/boe_filtered_speeches_conclusion.csv (or tools/data/raw fallback)
  2.  preparing_scraped_docs.py         -> data/raw/minutes_boe_monthly.json,
                                            data/raw/speeches_boe_monthly.json,
                                            data/raw/reference_boe_monthly.json
  3A. roberta_merged_score_evaluate.py  -> data/raw/merged_boe_scores.csv + plots
  3B. openai_merge_justify.py           -> data/raw/justifications_openai.csv

Usage (CLI):
  python3 tools/root_agent.py --start-date 2024-08-01 --end-date 2025-01-01

Notes:
  - GPT model defaults to gpt-4o-mini for 3B.
  - We pass --end-date to all steps for interface consistency; steps that ignore it will do so safely.
"""

from __future__ import annotations
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

# ---------------- Paths ----------------
REPO = Path("/Users/murat/Desktop/Capstone/FinGlobe_Agent").resolve()
TOOLS_DIR = REPO / "tools"
DATA_RAW = REPO / "data" / "raw"
PLOTS_DIR = REPO / "data" / "plots"

PYTHON = "/opt/anaconda3/bin/python3"  # keep consistent with your logs

# Artifacts we verify
MINUTES_JSON = DATA_RAW / "minutes_boe.json"
SPEECHES_CSV_PRIMARY = DATA_RAW / "boe_filtered_speeches_conclusion.csv"
SPEECHES_CSV_FALLBACKS = [
    TOOLS_DIR / "data" / "raw" / "boe_filtered_speeches_conclusion.csv",
    TOOLS_DIR / "boe_filtered_speeches_conclusion.csv",
]
MINUTES_MONTHLY = DATA_RAW / "minutes_boe_monthly.json"
SPEECHES_MONTHLY = DATA_RAW / "speeches_boe_monthly.json"
REFERENCE_MONTHLY = DATA_RAW / "reference_boe_monthly.json"  # optional
SCORES_CSV = DATA_RAW / "merged_boe_scores.csv"
JUSTIFY_CSV = DATA_RAW / "justifications_openai.csv"

# Script paths
MEETING_SCRAPER = TOOLS_DIR / "meeting_scraper.py"
SPEECH_SCRAPER = TOOLS_DIR / "scrape_boe_speeches.py"
PREPARE_STEP = TOOLS_DIR / "preparing_scraped_docs.py"
ROBERTA_STEP = TOOLS_DIR / "roberta_merged_score_evaluate.py"
JUSTIFY_STEP = TOOLS_DIR / "openai_merge_justify.py"


def banner(title: str):
    line = "=" * 55
    print(f"\n{line}\n🚀 {title}\n{line}", flush=True)


def run_cmd(cmd: str, cwd: Path | None = None) -> None:
    """Run a shell command, stream output, raise on non-zero return."""
    print(f"EXECUTING: {cmd}", flush=True)
    proc = subprocess.Popen(
        shlex.split(cmd),
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line.rstrip())
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"Command failed with exit code {ret}: {cmd}")
    print("✅ SUCCESS.", flush=True)


def find_speeches_csv() -> Path | None:
    if SPEECHES_CSV_PRIMARY.exists():
        return SPEECHES_CSV_PRIMARY
    for fb in SPEECHES_CSV_FALLBACKS:
        if fb.exists():
            return fb
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--headless", action="store_true", default=True, help="Run scrapers headless (default True)")
    ap.add_argument("--gpt-model", default="gpt-4o-mini", help="Model for justification step (default: gpt-4o-mini)")
    args = ap.parse_args()

    start_date = args.start_date
    end_date = args.end_date
    headless_flag = "--headless" if args.headless else ""
    gpt_model = args.gpt_model

    # Ensure directories
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------- 1A. Minutes scraper ----------------
    banner("STEP: 1A. SCRAPE Minutes")
    cmd_1a = f'{PYTHON} "{MEETING_SCRAPER}" --start-date {start_date} --end-date {end_date} {headless_flag}'
    run_cmd(cmd_1a, cwd=REPO)

    if not MINUTES_JSON.exists():
        raise FileNotFoundError(
            f"Expected minutes JSON not found:\n  {MINUTES_JSON}\n"
            "Check meeting_scraper.py output path."
        )

    # ---------------- 1B. Speeches scraper ----------------
    banner("STEP: 1B. SCRAPE Speeches")
    cmd_1b = f'{PYTHON} "{SPEECH_SCRAPER}" --start-date {start_date} --end-date {end_date}'
    run_cmd(cmd_1b, cwd=REPO)

    speeches_csv = find_speeches_csv()
    if speeches_csv is None:
        raise FileNotFoundError(
            "Speeches CSV not found in any known locations:\n"
            f"  {SPEECHES_CSV_PRIMARY}\n"
            + "\n".join([f"  {p}" for p in SPEECHES_CSV_FALLBACKS])
        )
    print(f"📄 Using speeches CSV: {speeches_csv}")

    # ---------------- 2. Prepare (monthly aggregation) ----------------
    banner("STEP: 2. PREPARE Monthly Aggregations")
    # end-date is accepted but not needed; we pass it for signature consistency
    cmd_2 = f'{PYTHON} "{PREPARE_STEP}" --start-date {start_date} --end-date {end_date}'
    run_cmd(cmd_2, cwd=REPO)

    # Verify monthly outputs
    if not MINUTES_MONTHLY.exists():
        raise FileNotFoundError(f"Missing minutes monthly JSON: {MINUTES_MONTHLY}")
    if not SPEECHES_MONTHLY.exists():
        raise FileNotFoundError(f"Missing speeches monthly JSON: {SPEECHES_MONTHLY}")
    # reference is optional, but warn if missing
    if not REFERENCE_MONTHLY.exists():
        print(f"⚠️  Reference monthly JSON not found (optional): {REFERENCE_MONTHLY}")

    # ---------------- 3A. Scoring & plots ----------------
    banner("STEP: 3A. SCORE with RoBERTa + Merge + Plots")
    # 3A only needs start-date; we pass it (and end) for logging homogeneity
    cmd_3a = f'{PYTHON} "{ROBERTA_STEP}" --start-date {start_date}'
    run_cmd(cmd_3a, cwd=REPO)

    if not SCORES_CSV.exists():
        raise FileNotFoundError(f"Missing merged scores CSV: {SCORES_CSV}")

    # ---------------- 3B. Justification via GPT ----------------
    banner("STEP: 3B. GPT Justifications (reference → else merged)")
    cmd_3b = (
        f'{PYTHON} "{JUSTIFY_STEP}" '
        f'--start-date {start_date} --end-date {end_date} --model {gpt_model}'
    )
    run_cmd(cmd_3b, cwd=REPO)

    if not JUSTIFY_CSV.exists():
        raise FileNotFoundError(f"Missing justifications CSV: {JUSTIFY_CSV}")

    banner("✅ PIPELINE COMPLETE")
    print("Outputs:")
    print(f"  Minutes map:           {MINUTES_JSON}")
    print(f"  Speeches CSV:          {speeches_csv}")
    print(f"  Minutes monthly:       {MINUTES_MONTHLY}")
    print(f"  Speeches monthly:      {SPEECHES_MONTHLY}")
    print(f"  Reference monthly:     {REFERENCE_MONTHLY} (optional)")
    print(f"  Scores CSV:            {SCORES_CSV}")
    print(f"  Justifications (CSV):  {JUSTIFY_CSV}")
    print(f"  Plots dir:             {PLOTS_DIR}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n❌ ROOT PIPELINE FAILED")
        print(str(e))
        sys.exit(1)
