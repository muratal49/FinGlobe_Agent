#!/usr/bin/env python3
"""
Monthly Justification Generator + Query Engine
----------------------------------------------

• Uses merged monthly texts (minutes + speeches)
• Uses model_score & reference_score CSVs
• Generates long-form justification using gpt-4o
• Also generates structured JSON fields
• Saves per-month justification into /justifications/{BOE|BOC}/{YYYY-MM}.json
• Provides query engine for dashboard: answer_user_query(query)

Author: Murat AL – FinGlobe
"""

import json
import re
from pathlib import Path
import pandas as pd
from datetime import datetime

from dotenv import load_dotenv

import os
from openai import OpenAI

# ============================================================
# CONFIG
# ============================================================

BASE = Path("/Users/murat/Desktop/Capstone/FinGlobe_Agent")
JUSTIFY_DIR = BASE / "justifications"
JUSTIFY_DIR.mkdir(exist_ok=True)

# Load .env from project base

load_dotenv(os.path.join(BASE, ".env"))

client = OpenAI()

# Merged corpora JSONs
MERGED = {
    "BOE": BASE / "data/raw/merged_boe_monthly.json",
    "BOC": BASE / "data/raw/merged_boc_monthly.json",
}

# Model score CSVs (BANK models only)
SCORE_CSV = {
    "BOE": BASE / "output_final/BOE_BANK_BOE_MER.csv",
    "BOC": BASE / "output_final/BOC_BANK_BOC_MER.csv",
}

BANK_NAMES = {
    "BOE": "Bank of England",
    "BOC": "Bank of Canada"
}


# ============================================================
# LOAD MERGED TEXT + MODEL SCORES
# ============================================================

def load_monthly_merged(bank):
    """Returns dict: { '2021-03': 'text...' }"""
    return json.loads(MERGED[bank].read_text())


def load_monthly_scores(bank):
    """
    Returns dict:
    {
        '2021-03': { model_score: 0.12, reference_score: 0.05 }
    }
    """
    df = pd.read_csv(SCORE_CSV[bank])
    out = {}
    for _, row in df.iterrows():
        out[row["month"]] = {
            "model_score": float(row["model_score"]),
            "reference_score": float(row["reference_score"])
        }
    return out


# ============================================================
# GPT-4o JUSTIFICATION GENERATOR
# ============================================================

def make_justification_prompt(bank, month, text, model_score, reference_score):
    bank_full = BANK_NAMES[bank]

    return f"""
You are an expert macroeconomic analyst specializing in central bank communications.

You are analyzing the following monthly communication from the **{bank_full}** for **{month}**.
This text is already combined (Minutes + Speeches):

------------------------
{text[:9000]}
------------------------

The stance scoring model produced:

• model_score = {model_score:.4f}
• reference_score = {reference_score:.4f}

Your task:
1. Give a **long, clear, human-readable explanation** of what the communication signaled.
2. Identify any hawkish/dovish/neutral elements WITHOUT assuming the user knows these terms. 
   Explain using plain macro language like:
   “the communication emphasized inflation risks…”
   “the bank expressed concern about economic weakness…”
3. Use direct evidence from the text (quote short phrases).

Then produce a final structured JSON block with fields:

- "month"
- "bank"
- "stance"  : one of "hawkish", "dovish", "neutral", or "mixed"
- "model_score"
- "reference_score"
- "alignment": "agree" / "diverge"
- "summary": a 2–3 sentence summary
- "evidence": list of short supporting quotes

Return the long justification FIRST, then the JSON block at the end.
"""


def generate_justification(bank, month, text, ms, rs):
    """Call GPT-4o and generate justification."""
    prompt = make_justification_prompt(bank, month, text, ms, rs)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content


# ============================================================
# SAVE & LOAD CACHED JUSTIFICATIONS
# ============================================================

def justification_path(bank, month):
    d = JUSTIFY_DIR / bank
    d.mkdir(exist_ok=True)
    return d / f"{month}.json"


def save_justification(bank, month, content):
    p = justification_path(bank, month)
    p.write_text(content, encoding="utf-8")


def load_justification(bank, month):
    p = justification_path(bank, month)
    return p.read_text() if p.exists() else None


# ============================================================
# RENDER MONTH BY MONTH
# ============================================================

def generate_all_monthly(bank):
    merged = load_monthly_merged(bank)
    scores = load_monthly_scores(bank)

    for month, text in merged.items():
        out_path = justification_path(bank, month)

        if out_path.exists():
            print(f"✔ Cached: {bank} {month}")
            continue

        if month not in scores:
            print(f"⚠ Missing score: {bank} {month}")
            continue

        print(f"🧠 Generating: {bank} {month}")

        model_score = scores[month]["model_score"]
        ref_score = scores[month]["reference_score"]

        content = generate_justification(
            bank, month, text, model_score, ref_score
        )

        save_justification(bank, month, content)

    print(f"\n✨ Completed generation for {BANK_NAMES[bank]}")


# ============================================================
# QUERY ENGINE FOR DASHBOARD
# ============================================================

def parse_query(query):
    """Extract bank + date range from user query."""

    q = query.lower()

    # Bank detection
    if "england" in q or "boe" in q:
        bank = "BOE"
    elif "canada" in q or "boc" in q:
        bank = "BOC"
    else:
        bank = None

    # Date(s)
    months = re.findall(r"\d{4}-\d{2}", q)
    years = re.findall(r"\d{4}", q)

    start = end = None

    if len(months) == 1:
        start = end = months[0]
    elif len(months) >= 2:
        start, end = months[0], months[1]
    else:
        # Year-based queries such as "2021–2022"
        if len(years) == 1:
            start = years[0] + "-01"
            end   = years[0] + "-12"
        elif len(years) >= 2:
            start = years[0] + "-01"
            end   = years[1] + "-12"

    return bank, start, end


def answer_user_query(query):
    """Return all stored justifications for a user-defined range."""

    bank, start, end = parse_query(query)

    if not bank:
        return "Could not detect the bank (BOE or BOC) from your query."

    if not start:
        return "Could not detect a valid date or range from your query."

    # Convert to sortable comparable integers
    def to_int(m): return int(m.replace("-", ""))

    s_int = to_int(start)
    e_int = to_int(end)

    outputs = []

    for p in sorted((JUSTIFY_DIR / bank).glob("*.json")):
        month = p.stem
        m_int = to_int(month)
        if s_int <= m_int <= e_int:
            outputs.append(p.read_text())

    if not outputs:
        return f"No justifications available for {BANK_NAMES[bank]} in the requested range."

    return "\n\n============================\n\n".join(outputs)


# ============================================================
# MAIN
# ============================================================

def main():
    print("Generating BOE...")
    generate_all_monthly("BOE")

    print("\nGenerating BOC...")
    generate_all_monthly("BOC")

    print("\n🎉 All justifications generated & cached.")


if __name__ == "__main__":
    main()
