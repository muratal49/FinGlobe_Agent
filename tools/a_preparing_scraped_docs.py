#!/usr/bin/env python3
"""
Unified preparation script for:

🇬🇧 Bank of England
🇨🇦 Bank of Canada

Creates canonical monthly JSON files AND merged corpora.

FIX APPLIED:
- Safe text extraction during merge to avoid dict + str errors

Author: Murat AL – FinGlobe Agent Pipeline
"""

import json
import re
import pandas as pd
from pathlib import Path

# ======================================================================
# Paths
# ======================================================================

REPO = Path("/Users/murat/Desktop/Capstone/FinGlobe_Agent")
RAW = REPO / "data" / "raw"

# Inputs
MINUTES_JSON_INPUT = RAW / "a_boe_minutes_full.json"
SPEECHES_BOE_JSON_INPUT = RAW / "a_boe_speeches_conclusion.json"
REFERENCE_BOE_CSV_INPUT = RAW / "boe_reference_scores.csv"

MPR_BOC_CSV_INPUT = RAW / "boc_policy_mpr.csv"
SPEECHES_BOC_CSV_INPUT = RAW / "bank_of_canada_speeches.csv"
REFERENCE_BOC_CSV_INPUT = RAW / "boc_reference_scores.csv"

# Outputs
MINUTES_JSON_MONTHLY_OUT = RAW / "minutes_boe_monthly.json"
SPEECHES_JSON_MONTHLY_OUT = RAW / "speeches_boe_monthly.json"
REFERENCE_JSON_MONTHLY_OUT = RAW / "reference_boe_monthly.json"

MINUTES_BOC_JSON_MONTHLY_OUT = RAW / "minutes_boc_monthly.json"
SPEECHES_BOC_JSON_MONTHLY_OUT = RAW / "speeches_boc_monthly.json"
REFERENCE_BOC_JSON_MONTHLY_OUT = RAW / "reference_boc_monthly.json"

MERGED_BOE_OUT = RAW / "merged_boe_monthly.json"
MERGED_BOC_OUT = RAW / "merged_boc_monthly.json"


# ======================================================================
# Helpers
# ======================================================================

def month_key(d: str) -> str:
    return str(d)[:7]


def load_json_mixed(path: Path) -> dict:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        out = {}
        for e in data:
            if isinstance(e, dict) and "date" in e:
                out[e["date"]] = e.get("text") or e.get("full_text") or ""
        return out
    return {}


def _as_text(x):
    """🔥 FIX: safely extract text from dict or string"""
    if isinstance(x, dict):
        return (
            x.get("text")
            or x.get("full_text")
            or x.get("summary")
            or ""
        )
    return x if isinstance(x, str) else ""


# ======================================================================
# BoE Minutes
# ======================================================================

def prepare_minutes_boe():
    if not MINUTES_JSON_INPUT.exists():
        print("⚠ Missing BoE minutes file")
        return {}

    d = load_json_mixed(MINUTES_JSON_INPUT)
    rows = []

    for k, v in d.items():
        rows.append({
            "date": k,
            "month": month_key(k),
            "text": _as_text(v),
        })

    df = pd.DataFrame(rows).sort_values("date")
    df_month = df.groupby("month").tail(1)
    df_month = df_month.set_index("month").sort_index()
    df_month["text"] = df_month["text"].ffill()

    out = df_month["text"].to_dict()
    MINUTES_JSON_MONTHLY_OUT.write_text(json.dumps(out, indent=2))
    print(f"✔ Saved BoE minutes monthly → {MINUTES_JSON_MONTHLY_OUT}")
    return out


# ======================================================================
# BoE Speeches
# ======================================================================

def prepare_speeches_boe():
    if not SPEECHES_BOE_JSON_INPUT.exists():
        print("⚠ Missing BoE speeches JSON")
        return {}

    raw = json.loads(SPEECHES_BOE_JSON_INPUT.read_text())
    rows = []

    for date_key, entry in raw.items():
        rows.append({
            "date": entry.get("date", date_key),
            "text": _as_text(entry),
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["month"] = df["date"].dt.to_period("M").astype(str)

    out = (
        df.groupby("month")["text"]
        .apply(lambda x: "\n\n".join(x))
        .to_dict()
    )

    SPEECHES_JSON_MONTHLY_OUT.write_text(json.dumps(out, indent=2))
    print(f"✔ Saved BoE speeches monthly → {SPEECHES_JSON_MONTHLY_OUT}")
    return out


# ======================================================================
# BoE Reference
# ======================================================================

def prepare_reference_boe():
    if not REFERENCE_BOE_CSV_INPUT.exists():
        print("⚠ Missing BoE reference CSV")
        return {}

    df = pd.read_csv(REFERENCE_BOE_CSV_INPUT)
    df["month"] = df["Hawkishness Date"].astype(str).str[:7]
    df = df[["month", "Hawkishness"]].drop_duplicates("month", keep="last")
    df = df.set_index("month").sort_index()
    df["Hawkishness"] = df["Hawkishness"].ffill()

    out = df["Hawkishness"].to_dict()
    REFERENCE_JSON_MONTHLY_OUT.write_text(json.dumps(out, indent=2))
    print(f"✔ Saved BoE reference monthly → {REFERENCE_JSON_MONTHLY_OUT}")
    return out


# ======================================================================
# BoC MPR
# ======================================================================

def prepare_mpr_boc():
    if not MPR_BOC_CSV_INPUT.exists():
        print("⚠ Missing BoC MPR CSV")
        return {}

    df = pd.read_csv(MPR_BOC_CSV_INPUT)
    df["month"] = df["date"].astype(str).str[:7]
    df = df.sort_values("date").drop_duplicates("month", keep="last")

    out = {row["month"]: _as_text(row.get("summary_text") or row.get("cleaned_text"))
           for _, row in df.iterrows()}

    MINUTES_BOC_JSON_MONTHLY_OUT.write_text(json.dumps(out, indent=2))
    print(f"✔ Saved BoC MPR monthly → {MINUTES_BOC_JSON_MONTHLY_OUT}")
    return out


# ======================================================================
# BoC Speeches
# ======================================================================

def prepare_speeches_boc():
    if not SPEECHES_BOC_CSV_INPUT.exists():
        print("⚠ Missing BoC speeches CSV")
        return {}

    df = pd.read_csv(SPEECHES_BOC_CSV_INPUT)
    df["month"] = df["date"].astype(str).str[:7]
    df["text"] = df["conclusion_text"].fillna("")

    out = (
        df.groupby("month")["text"]
        .apply(lambda x: "\n\n".join(x))
        .to_dict()
    )

    SPEECHES_BOC_JSON_MONTHLY_OUT.write_text(json.dumps(out, indent=2))
    print(f"✔ Saved BoC speeches monthly → {SPEECHES_BOC_JSON_MONTHLY_OUT}")
    return out


# ======================================================================
# BoC Reference
# ======================================================================

def prepare_reference_boc():
    if not REFERENCE_BOC_CSV_INPUT.exists():
        print("⚠ Missing BoC reference CSV")
        return {}

    df = pd.read_csv(REFERENCE_BOC_CSV_INPUT)
    df["month"] = df["Hawkishness Date"].astype(str).str[:7]
    df = df.sort_values("month").drop_duplicates("month", keep="last")
    df = df.set_index("month")
    df["Hawkishness"] = df["Hawkishness"].ffill()

    out = df["Hawkishness"].to_dict()
    REFERENCE_BOC_JSON_MONTHLY_OUT.write_text(json.dumps(out, indent=2))
    print(f"✔ Saved BoC reference monthly → {REFERENCE_BOC_JSON_MONTHLY_OUT}")
    return out


# ======================================================================
# MERGE MONTHLY CORPORA (🔥 FIXED)
# ======================================================================

def merge_monthly_sources():
    print("🔄 Creating merged corpora...")

    boe_min = json.loads(MINUTES_JSON_MONTHLY_OUT.read_text())
    boe_spe = json.loads(SPEECHES_JSON_MONTHLY_OUT.read_text())

    boc_min = json.loads(MINUTES_BOC_JSON_MONTHLY_OUT.read_text())
    boc_spe = json.loads(SPEECHES_BOC_JSON_MONTHLY_OUT.read_text())

    merged_boe = {}
    merged_boc = {}

    for m in sorted(set(boe_min) | set(boe_spe)):
        merged_boe[m] = (
            _as_text(boe_min.get(m)) + "\n\n" + _as_text(boe_spe.get(m))
        ).strip()

    for m in sorted(set(boc_min) | set(boc_spe)):
        merged_boc[m] = (
            _as_text(boc_min.get(m)) + "\n\n" + _as_text(boc_spe.get(m))
        ).strip()

    MERGED_BOE_OUT.write_text(json.dumps(merged_boe, indent=2))
    MERGED_BOC_OUT.write_text(json.dumps(merged_boc, indent=2))

    print("✔ merged_boe_monthly.json created")
    print("✔ merged_boc_monthly.json created")


# ======================================================================
# MAIN
# ======================================================================

def main():
    print("\n=== Preparing Monthly Canonical Files ===\n")

    prepare_minutes_boe()
    prepare_speeches_boe()
    prepare_reference_boe()

    prepare_mpr_boc()
    prepare_speeches_boc()
    prepare_reference_boc()

    merge_monthly_sources()

    print("\n✔ ALL DONE.\n")


if __name__ == "__main__":
    main()
