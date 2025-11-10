#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
openai_merge_justify.py

Goal
----
Generate ~300-word justifications for each month by feeding GPT:
 - the month
 - a single score to justify (reference if available, else merged_model score)
 - panel keywords
 - keyword-bearing sentences from that month’s merged text (minutes + speeches)

Inputs (already produced upstream):
  data/raw/minutes_boe_monthly.json   -> {"YYYY-MM": "minutes text"}
  data/raw/speeches_boe_monthly.json  -> {"YYYY-MM": "speeches text"}
  data/raw/merged_boe_scores.csv      -> with columns: month, reference_score, merged_score, ...
  data/raw/reference_boe_monthly.json -> optional {"YYYY-MM": float}

Output:
  data/raw/justifications_openai.csv  -> columns: date, score, justification

Notes:
 - Requires OPENAI_API_KEY in your environment. If you keep it in .env at repo root,
   we load it automatically with python-dotenv.
 - We DO NOT regenerate any model scores; we only read existing files.
"""

from __future__ import annotations
import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

try:
    # load .env if present
    from dotenv import load_dotenv  # type: ignore
except Exception:
    load_dotenv = None

# ---------- Paths ----------
BASE_PATH = Path("/Users/murat/Desktop/Capstone/FinGlobe_Agent").resolve()
DATA_RAW = BASE_PATH / "data" / "raw"
OUT_CSV = DATA_RAW / "justifications_openai.csv"

MINUTES_MONTHLY = DATA_RAW / "minutes_boe_monthly.json"
SPEECHES_MONTHLY = DATA_RAW / "speeches_boe_monthly.json"
REFERENCE_MONTHLY = DATA_RAW / "reference_boe_monthly.json"  # optional
SCORES_CSV = DATA_RAW / "merged_boe_scores.csv"

# ---------- Panel keywords (Ajinkya lists) ----------
PANEL_A1 = [
    "inflation expectation", "interest rate", "bank rate", "fund rate", "price",
    "economic activity", "inflation", "employment"
]
PANEL_B1 = [
    "unemployment", "growth", "exchange rate", "productivity", "deficit",
    "demand", "job market", "monetary policy"
]
PANEL_A2 = [
    "anchor", "cut", "subdue", "decline", "decrease", "reduce", "low", "drop",
    "fall", "fell", "decelerate", "slow", "pause", "pausing", "stable",
    "nonaccelerating", "downward", "tighten"
]
PANEL_B2 = [
    "ease", "easing", "rise", "rising", "increase", "expand", "improve", "strong",
    "upward", "raise", "high", "rapid"
]
ALL_KEYWORDS = sorted(set([*PANEL_A1, *PANEL_B1, *PANEL_A2, *PANEL_B2]))
KW_PATTERNS = [(kw, re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE)) for kw in ALL_KEYWORDS]
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

# ---------- OpenAI client ----------
def get_openai_client():
    # Load .env if present
    if load_dotenv is not None:
        # Try both repo root and tools/
        load_dotenv(BASE_PATH / ".env")
        load_dotenv()  # fallback to CWD

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment. Put it in your .env or export it.")

    # New-style OpenAI Python SDK (>=1.0)
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError("OpenAI SDK not installed. `pip install openai python-dotenv`") from e

    return OpenAI(api_key=api_key)

# ---------- Helpers ----------
def load_json_map(path: Path) -> Dict[str, str | float]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected a JSON object at {path}")
    return obj

def concat_month_text(minutes_map: Dict[str, str], speeches_map: Dict[str, str]) -> Dict[str, str]:
    months = sorted(set(minutes_map.keys()) | set(speeches_map.keys()))
    merged: Dict[str, str] = {}
    for m in months:
        mt = str(minutes_map.get(m, "") or "")
        st = str(speeches_map.get(m, "") or "")
        text = (mt + ("\n\n" + st if st else "")).strip()
        merged[m] = text
    return merged

def choose_score(row: pd.Series, reference_map: Dict[str, float]) -> Tuple[float | None, str]:
    """
    Priority: reference_score (CSV) -> reference_map -> merged_score.
    Returns (score, source).
    """
    # 1) CSV reference_score
    if "reference_score" in row and pd.notna(row["reference_score"]):
        return float(row["reference_score"]), "reference_score"
    # 2) JSON reference (sometimes CSV may lack it)
    month = str(row.get("month", ""))
    if month and (month in reference_map):
        return float(reference_map[month]), "reference_score(json)"
    # 3) merged_score
    if "merged_score" in row and pd.notna(row["merged_score"]):
        return float(row["merged_score"]), "merged_score"
    return None, "none"

def extract_keyword_sentences(text: str, top_k: int = 20) -> List[str]:
    """
    Keep sentences that contain any panel keyword.
    Rank by total keyword matches (desc) then by length (desc).
    """
    if not text or not text.strip():
        return []
    sentences = [s.strip() for s in SENT_SPLIT.split(text) if len(s.strip()) >= 8]
    scored = []
    for s in sentences:
        hits = 0
        for _, pat in KW_PATTERNS:
            if pat.search(s):
                hits += 1
        if hits > 0:
            scored.append((hits, len(s), s))
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return [s for _, _, s in scored[:top_k]]

def build_prompt(month: str, score: float, top_sentences: List[str]) -> str:
    """
    Compose a compact prompt for GPT. We provide:
      - the month and target score
      - the panel keywords
      - ~20 keyword-bearing sentences as evidence
    Ask for ~300 words; cite evidence naturally.
    """
    bullet_sents = "\n- ".join(top_sentences) if top_sentences else "(no keyword-bearing sentences)"
    prompt = (
        "You are an analyst producing a concise justification for a BoE hawkishness/dovishness score.\n\n"
        f"Month: {month}\n"
        f"Target score to justify: {score:+.4f}\n\n"
        "Panel keywords (use them to anchor evidence):\n"
        f"• A1: {', '.join(PANEL_A1)}\n"
        f"• B1: {', '.join(PANEL_B1)}\n"
        f"• A2: {', '.join(PANEL_A2)}\n"
        f"• B2: {', '.join(PANEL_B2)}\n\n"
        "Evidence sentences (from minutes+speeches, already filtered for keywords):\n"
        f"- {bullet_sents}\n\n"
        "Write ~300 words (3–6 short paragraphs). Be precise, avoid fluff, and justify the sign/magnitude of the score "
        "by referencing concrete evidence above (inflation, bank rate, labour market, demand, etc.). "
        "You may refer to the evidence implicitly (no quotes needed), but your reasoning must be verifiable from it. "
        "Do not invent data; keep it consistent with central-bank language."
    )
    return prompt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-date", type=str, required=False, help="Optional lower bound (YYYY-MM or YYYY-MM-DD)")
    ap.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model to use")
    ap.add_argument("--max-months", type=int, default=9999, help="Limit number of months to justify (for testing)")
    args = ap.parse_args()

    # Load inputs (no rescoring!)
    if not MINUTES_MONTHLY.exists() or not SPEECHES_MONTHLY.exists():
        raise FileNotFoundError(
            "Missing monthly JSON files. Expected:\n"
            f"  {MINUTES_MONTHLY}\n"
            f"  {SPEECHES_MONTHLY}\n"
            "Run Step 2 (preparing_scraped_docs.py) first."
        )
    if not SCORES_CSV.exists():
        raise FileNotFoundError(f"Missing scores CSV: {SCORES_CSV}. Run Step 3A first.")

    minutes_map = load_json_map(MINUTES_MONTHLY)      # {"YYYY-MM": str}
    speeches_map = load_json_map(SPEECHES_MONTHLY)    # {"YYYY-MM": str}
    reference_map = load_json_map(REFERENCE_MONTHLY) if REFERENCE_MONTHLY.exists() else {}

    # Optional clamp by start month
    if args.start_date:
        key = args.start_date[:7]
        minutes_map = {k: v for k, v in minutes_map.items() if k >= key}
        speeches_map = {k: v for k, v in speeches_map.items() if k >= key}
        reference_map = {k: v for k, v in reference_map.items() if k >= key}

    merged_text_map = concat_month_text(minutes_map, speeches_map)

    df_scores = pd.read_csv(SCORES_CSV)
    if "month" not in df_scores.columns:
        raise ValueError(f"'month' column missing in {SCORES_CSV}")
    df_scores["month"] = df_scores["month"].astype(str)

    # Keep only months present in merged_text_map (we justify on the merged text)
    months = sorted(set(merged_text_map.keys()) | set(df_scores["month"].tolist()))
    if args.max_months and len(months) > args.max_months:
        months = months[: args.max_months]

    # OpenAI client
    client = get_openai_client()

    out_rows = []
    for m in months:
        # Choose score to justify
        row = df_scores[df_scores["month"] == m].tail(1)  # last if duplicates
        if row.empty:
            score_used, src = (reference_map.get(m, None), "reference_score(json)" if m in reference_map else "none")
        else:
            score_used, src = choose_score(row.iloc[0], reference_map)

        if score_used is None:
            # Skip months we cannot justify (no score available)
            continue

        text = merged_text_map.get(m, "").strip()
        if not text:
            # If merged text is empty, skip
            continue

        # Extract keyword-bearing sentences to keep prompt small & relevant
        top_sents = extract_keyword_sentences(text, top_k=20)
        prompt = build_prompt(m, score_used, top_sents)

        # Call OpenAI
        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": "You are a precise central-bank policy analyst."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=700,  # ~300 words comfortably
            )
            justification = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            justification = f"(Generation error: {e})"

        out_rows.append({"date": m, "score": score_used, "justification": justification})

    # Save output CSV with EXACT columns requested
    pd.DataFrame(out_rows).sort_values("date").to_csv(OUT_CSV, index=False)
    print(f"💾 Wrote justifications CSV -> {OUT_CSV}")
    print("✅ openai_merge_justify complete.")

if __name__ == "__main__":
    main()
