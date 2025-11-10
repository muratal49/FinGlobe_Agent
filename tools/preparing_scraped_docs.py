#!/usr/bin/env python3
"""
Preparing & Aggregation Pipeline (Step 2)

- Reads:
    * Minutes JSON (now a MAP: {"YYYY-MM-DD": "text", ...} OR legacy LIST of dicts)
    * Speeches CSV (date, title, speaker, text, conclusion_text, url)
    * Reference CSV (headers: "Hawkishness Date", "Hawkishness")

- Outputs (under <REPO>/data/raw/):
    * minutes_boe_monthly.json   -> {"YYYY-MM": "minutes_text (joined if multiple)..."}
    * speeches_boe_monthly.json  -> {"YYYY-MM": "speech_text (joined per month)..."} (ONLY for months that exist in minutes)
    * reference_boe_monthly.json -> {"YYYY-MM": mean(Hawkishness) }

Behavior:
  - For speeches, uses "conclusion_text" if present (else falls back to "text").
  - For minutes, accepts both dict-map and legacy list formats.
  - Reference CSV is aggregated by month using columns "Hawkishness Date" and "Hawkishness".
"""

from __future__ import annotations
import argparse
import json
from datetime import datetime
from pathlib import Path
import sys
from typing import Dict, List

import pandas as pd


# ------------- CONFIG: set your repo root here (ABS path) -----------------
BASE_PATH = Path("/Users/murat/Desktop/Capstone/FinGlobe_Agent").resolve()
DATA_RAW = (BASE_PATH / "data" / "raw")
DATA_RAW.mkdir(parents=True, exist_ok=True)

# Canonical inputs we look for:
MINUTES_JSON_INPUT = DATA_RAW / "minutes_boe.json"                       # map {"YYYY-MM-DD": "text"} (new) OR list (legacy)
SPEECHES_CSV_INPUT = DATA_RAW / "boe_filtered_speeches_conclusion.csv"   # speeches CSV (canonical)
REFERENCE_CSV_INPUT = DATA_RAW / "boe_reference_scores.csv"              # reference CSV, but with new headers

# Fallbacks (in case 1B wrote to tools/data/raw):
FALLBACK_SPEECHES = [
    BASE_PATH / "tools" / "data" / "raw" / "boe_filtered_speeches_conclusion.csv",
    BASE_PATH / "tools" / "boe_filtered_speeches_conclusion.csv",
]

# Canonical outputs:
MINUTES_JSON_MONTHLY_OUT = DATA_RAW / "minutes_boe_monthly.json"
SPEECHES_JSON_MONTHLY_OUT = DATA_RAW / "speeches_boe_monthly.json"
REFERENCE_JSON_MONTHLY_OUT = DATA_RAW / "reference_boe_monthly.json"


def _resolve_input(path: Path, fallbacks: List[Path]) -> Path:
    if path.exists():
        print(f"🔎 Found canonical input: {path}")
        return path
    for fb in fallbacks:
        if fb.exists():
            print(f"⚠️  Canonical missing, using fallback: {fb}")
            return fb
    return path


def _to_month(iso_or_datetime) -> str:
    if isinstance(iso_or_datetime, datetime):
        return iso_or_datetime.strftime("%Y-%m")
    s = str(iso_or_datetime)[:10]
    try:
        d = datetime.fromisoformat(s)
        return d.strftime("%Y-%m")
    except Exception:
        return ""


def _join_nonempty_texts(texts: List[str]) -> str:
    texts = [t for t in (texts or []) if isinstance(t, str) and t.strip()]
    return ("\n\n\n").join(texts).strip()


# ---------------------- Minutes aggregation ----------------------
def aggregate_minutes_monthly(start_iso: str | None) -> Dict[str, str]:
    """
    Accepts minutes as either:
      - MAP: {"YYYY-MM-DD": "text", ...}
      - LIST: [{"minutes_date": "...", "minutes_text": "..."}, ...]

    Returns dict {"YYYY-MM": "joined minutes text for that month"} and writes to file.
    """
    if not MINUTES_JSON_INPUT.exists():
        raise FileNotFoundError(
            f"Minutes JSON not found: {MINUTES_JSON_INPUT}\n"
            "Ensure Step 1A wrote minutes_boe.json (date->text map)."
        )

    print("\n--- Minutes: monthly aggregation ---")
    print(f"Reading: {MINUTES_JSON_INPUT}")
    with MINUTES_JSON_INPUT.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    rows: List[dict] = []
    # New format: dict map
    if isinstance(raw, dict):
        for k, v in raw.items():
            rows.append({"date": k, "text": v})
    # Legacy format: list of dicts
    elif isinstance(raw, list):
        for obj in raw:
            d = obj.get("minutes_date", "")
            t = obj.get("minutes_text", "")
            rows.append({"date": d, "text": t})
    else:
        raise ValueError("Unsupported minutes JSON format.")

    df = pd.DataFrame(rows)
    if df.empty:
        print("⚠️  Minutes file loaded but empty.")
        monthly_map: Dict[str, str] = {}
    else:
        # Optional start filter if supplied
        if start_iso:
            try:
                df = df[pd.to_datetime(df["date"], errors="coerce") >= pd.to_datetime(start_iso)]
            except Exception:
                pass
        df["Month"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m")
        # Join per month (if multiple entries)
        agg = (
            df.groupby("Month")["text"]
              .apply(lambda x: _join_nonempty_texts([str(v) for v in x]))
              .reset_index()
        )
        monthly_map = {row["Month"]: row["text"] for _, row in agg.iterrows() if row["Month"]}

    with MINUTES_JSON_MONTHLY_OUT.open("w", encoding="utf-8") as f:
        json.dump(monthly_map, f, ensure_ascii=False, indent=2)

    print(f"💾 Wrote minutes monthly JSON: {MINUTES_JSON_MONTHLY_OUT} ({len(monthly_map)} months)")
    return monthly_map


# ---------------------- Speeches aggregation ----------------------
def aggregate_speeches_monthly(minutes_months: set[str], start_iso: str | None) -> Dict[str, str]:
    """
    Reads speeches CSV, prefers 'conclusion_text' (else falls back to 'text'),
    then aggregates by Month, but **only for months present in 'minutes_months'**.
    """
    src = _resolve_input(SPEECHES_CSV_INPUT, FALLBACK_SPEECHES)
    print("\n--- Speeches: monthly aggregation ---")
    print(f"Reading: {src}")
    if not src.exists():
        print("⚠️  Speeches CSV not found; returning empty aggregation.")
        speeches_map: Dict[str, str] = {}
        with SPEECHES_JSON_MONTHLY_OUT.open("w", encoding="utf-8") as f:
            json.dump(speeches_map, f, ensure_ascii=False, indent=2)
        return speeches_map

    df = pd.read_csv(src)
    if "date" not in df.columns:
        raise ValueError(f"Speeches CSV missing 'date' column. Columns are: {list(df.columns)}")

    # Choose text column
    text_col = "conclusion_text" if "conclusion_text" in df.columns else ("text" if "text" in df.columns else None)
    if text_col is None:
        raise ValueError("Speeches CSV must have either 'conclusion_text' or 'text' column.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if start_iso:
        try:
            df = df[df["date"] >= pd.to_datetime(start_iso)]
        except Exception:
            pass
    df["Month"] = df["date"].dt.strftime("%Y-%m")

    # Keep only months that exist in minutes
    if minutes_months:
        df = df[df["Month"].isin(minutes_months)]

    agg = (
        df.groupby("Month")[text_col]
          .apply(lambda x: _join_nonempty_texts([str(v) for v in x]))
          .reset_index()
    )
    speeches_map = {row["Month"]: row[text_col] for _, row in agg.iterrows() if row["Month"]}

    with SPEECHES_JSON_MONTHLY_OUT.open("w", encoding="utf-8") as f:
        json.dump(speeches_map, f, ensure_ascii=False, indent=2)

    print(f"💾 Wrote speeches monthly JSON: {SPEECHES_JSON_MONTHLY_OUT} ({len(speeches_map)} months)")
    return speeches_map


# ---------------------- Reference aggregation ----------------------
def aggregate_reference_monthly(start_iso: str | None) -> Dict[str, float]:
    """
    Reads reference CSV with headers: 'Hawkishness Date', 'Hawkishness'
    Aggregates monthly mean hawkishness -> {"YYYY-MM": value}
    """
    if not REFERENCE_CSV_INPUT.exists():
        print(f"\n--- Reference series ---\nℹ️  Skipping: not found -> {REFERENCE_CSV_INPUT}")
        with REFERENCE_JSON_MONTHLY_OUT.open("w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=2)
        return {}

    print("\n--- Reference series: monthly aggregation ---")
    print(f"Reading: {REFERENCE_CSV_INPUT}")

    df = pd.read_csv(REFERENCE_CSV_INPUT)

    # Accept your headers
    date_col = None
    val_col = None
    for cand in ["Hawkishness Date", "date"]:
        if cand in df.columns:
            date_col = cand
            break
    for cand in ["Hawkishness", "value"]:
        if cand in df.columns:
            val_col = cand
            break

    if date_col is None:
        raise ValueError("Reference CSV must have a 'Hawkishness Date' (or 'date') column.")
    if val_col is None:
        raise ValueError("Reference CSV must have a 'Hawkishness' (or 'value') column.")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    if start_iso:
        try:
            df = df[df[date_col] >= pd.to_datetime(start_iso)]
        except Exception:
            pass
    df["Month"] = df[date_col].dt.strftime("%Y-%m")
    out_map = df.groupby("Month")[val_col].mean().round(6).to_dict()

    with REFERENCE_JSON_MONTHLY_OUT.open("w", encoding="utf-8") as f:
        json.dump(out_map, f, ensure_ascii=False, indent=2)

    print(f"💾 Wrote reference monthly JSON: {REFERENCE_JSON_MONTHLY_OUT} ({len(out_map)} months)")
    return out_map


# ------------------------------ Main ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-date", required=True, help="YYYY-MM-DD (inclusive filter)")
    ap.add_argument("--end-date", required=False, help="YYYY-MM-DD (unused in this step)")
    args = ap.parse_args()

    start_iso = args.start_date

    print(f"Starting Data Preparation Pipeline, filtering all data from: {args.start_date}")

    # 1) Minutes (produces monthly map; we need its months for speeches)
    minutes_monthly = aggregate_minutes_monthly(start_iso)
    minutes_months = set(minutes_monthly.keys())

    # 2) Speeches (only for months that exist in minutes)
    aggregate_speeches_monthly(minutes_months, start_iso)

    # 3) Reference (new headers; monthly mean)
    aggregate_reference_monthly(start_iso)

    print("\n✅ Preparation complete.")


if __name__ == "__main__":
    main()
