#!/usr/bin/env python3
import json
import subprocess
import re
from pathlib import Path
from datetime import datetime
import pandas as pd

from query_interpreter_llm import interpret_query_llm
from a_openai_justification import generate_justification


# -----------------------------------------------------
# PATHS
# -----------------------------------------------------
BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw"
JUSTIFICATIONS = BASE / "justifications"
OUTPUT_FINAL = BASE / "output_final"

BOE_MERGED = RAW / "merged_boe_monthly.json"
BOC_MERGED = RAW / "merged_boc_monthly.json"


# -----------------------------------------------------
# Helpers
# -----------------------------------------------------
def month_range(start: str, end: str):
    """Return list of YYYY-MM between two YYYY-MM-DD dates (inclusive)."""
    s = datetime.fromisoformat(start)
    e = datetime.fromisoformat(end)
    out = []
    cur = s
    while cur <= e:
        out.append(cur.strftime("%Y-%m"))
        if cur.month == 12:
            cur = datetime(cur.year + 1, 1, 1)
        else:
            cur = datetime(cur.year, cur.month + 1, 1)
    return out


def run_tool(script: str, *args):
    """Run a tool script via python with extra CLI args."""
    script_path = BASE / "tools" / script
    cmd = ["python", str(script_path), *args]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=False)


def load_merged(bank: str):
    """Load merged monthly JSON (BOE or BOC) with normalized YYYY-MM keys."""
    path = BOE_MERGED if bank == "england" else BOC_MERGED
    if not path.exists():
        print(f"⚠️ merged file not found: {path}")
        return {}
    try:
        data = json.loads(path.read_text())
        return {str(k).strip()[:7]: v for k, v in data.items()}
    except Exception as e:
        print(f"⚠️ Could not parse merged file {path}: {e}")
        return {}


# -----------------------------------------------------
# Justification handling
# -----------------------------------------------------
def load_or_generate_justification(
    bank: str,
    ym: str,
    text: str,
    model_score: float,
    reference_score: float,
):
    """
    Return dict:
      {
        "narrative": <long explanation>,
        "structured": { ... JSON block ... or error/raw ... }
      }

    Saved under justifications/{BOE|BOC}/{YYYY-MM}.json
    """
    bank_code = "BOE" if bank == "england" else "BOC"

    jf_dir = JUSTIFICATIONS / bank_code
    jf_dir.mkdir(parents=True, exist_ok=True)
    jf_path = jf_dir / f"{ym}.json"

    # 1) Try cached JSON
    if jf_path.exists():
        try:
            return json.loads(jf_path.read_text())
        except Exception as e:
            print(f"⚠️ Corrupted justification {jf_path}: {e} – regenerating…")

    # 2) Not enough data → minimal record
    if not text or model_score is None:
        return {
            "narrative": "",
            "structured": {
                "month": ym,
                "bank": bank_code,
                "error": "Insufficient data for justification",
            },
        }

    # 3) Call OpenAI to generate long explanation + JSON
    raw = generate_justification(bank_code, ym, text, model_score, reference_score)

    # Extract JSON block at the end (no backticks assumed for NEW ones)
    match = re.search(r"\{[\s\S]*\}$", raw.strip())
    if match:
        try:
            structured = json.loads(match.group(0))
        except Exception as e:
            structured = {"error": f"JSON parse failed: {e}", "raw": raw}
        narrative = raw[: match.start()].strip()
    else:
        structured = {"error": "No JSON block detected", "raw": raw}
        narrative = raw.strip()

    result = {"narrative": narrative, "structured": structured}
    jf_path.write_text(json.dumps(result, indent=2))
    return result


def extract_summary_and_narrative(js: dict):
    """
    Robustly extract:
      - summary  (from structured.summary OR from JSON inside structured.raw)
      - narrative (top-level 'narrative' field)

    Works with both OLD files (where structured.raw contains ```json``` block)
    and NEW files (where structured is already parsed JSON).
    """
    narrative = js.get("narrative", "") or ""

    structured = js.get("structured", {}) or {}
    summary = (
        structured.get("summary")
        or structured.get("summary_text")
        or structured.get("short_summary")
        or structured.get("stance_summary")
        or structured.get("executive_summary")
        or ""
    )

    # If summary not directly present but there is a raw block, parse JSON inside it
    if not summary and "raw" in structured:
        raw = structured.get("raw") or ""

        # 1) Try fenced ```json ... ```
        m = re.search(r"```json(.*?)```", raw, flags=re.S)
        if m:
            try:
                block = json.loads(m.group(1).strip())
                summary = block.get("summary", "") or summary
            except Exception:
                pass

        # 2) Fallback: any { ... } JSON block
        if not summary:
            m2 = re.search(r"\{.*\}", raw, flags=re.S)
            if m2:
                try:
                    block = json.loads(m2.group(0).strip())
                    summary = block.get("summary", "") or summary
                except Exception:
                    pass

    # Final safety fallback: first sentence of narrative
    if not summary and narrative:
        first_sentence = re.split(r"(?<=[.!?])\s+", narrative.strip())[0]
        summary = first_sentence

    return summary or "", narrative


# -----------------------------------------------------
# MAIN ROOT AGENT
# -----------------------------------------------------
def run_root_agent(query: str):
    print("🧠 Interpreting query:", query)
    parsed = interpret_query_llm(query)

    bank = parsed["bank"]          # "england" / "canada" / "no results..."
    start = parsed["start_date"]   # YYYY-MM-DD
    end = parsed["end_date"]       # YYYY-MM-DD

    if bank == "no results available for that bank":
        return {
            "error": "Unsupported bank. Only Bank of England and Bank of Canada are available."
        }

    bank_code = "BOE" if bank == "england" else "BOC"
    csv_path = OUTPUT_FINAL / f"{bank_code}_BANK_{bank_code}_MER.csv"

    requested = month_range(start, end)

    # Load all available months from existing scoring file (if any)
    if csv_path.exists():
        full_df = pd.read_csv(csv_path)
        all_months = sorted(full_df["month"].astype(str).str[:7].unique())
    else:
        all_months = []

    # ---------------- SMART DATE LOGIC ----------------
    no_dates = start == "2000-01-01" and end == "2025-12-31"
    single_month_query = len(requested) == 1
    single_year_query = (
        len(requested) == 12 and requested[0][:4] == requested[-1][:4]
    )

    if no_dates:
        # No explicit dates → default to last 12 scored months
        needed = all_months[-12:]
    elif single_year_query:
        # A single year (e.g. "2023") → that full year only
        needed = requested
    elif single_month_query:
        # A single month → ±3 months for context (if they exist)
        target = requested[0]
        if target in all_months:
            idx = all_months.index(target)
            lo = max(idx - 3, 0)
            hi = min(idx + 4, len(all_months))
            needed = all_months[lo:hi]
        else:
            needed = requested
    else:
        # Explicit range (e.g. "2022–2024") → just that range
        needed = requested

    print("📅 Months selected:", needed)

    # ---------------- MERGED TEXT / SCRAPING ----------------
    merged = load_merged(bank)
    missing = [m for m in needed if m not in merged]

    need_rescore = False

    if missing:
        print("📌 Missing merged months:", missing)

        # Run scrapers only for this date window
        if bank == "england":
            run_tool("a_boe_speech_scraper.py", "--start-date", start, "--end-date", end)
            run_tool("a_boe_minutes_scraper_full.py", "--start-date", start, "--end-date", end)
        else:
            run_tool("a_boc_speeches.py", "--start-date", start, "--end-date", end)
            run_tool("a_boc_policy_mpr.py", "--start-date", start, "--end-date", end)

        # Rebuild all monthly corpora (including merged_*_monthly.json)
        run_tool("a_preparing_scraped_docs.py")
        merged = load_merged(bank)
        need_rescore = True  # we now have new months that need scores

    # ---------------- SCORING: only when needed ----------------
    if (not csv_path.exists()) or need_rescore:
        # This script recomputes full CSVs for both banks.
        # After this, future runs that don't add new months will be instant.
        run_tool("a_roberta_score_evaluate_NoWCB.py")

    if not csv_path.exists():
        # Still missing → something went wrong in scoring script
        return {
            "error": f"Score CSV not found for {bank_code} after scoring run."
        }

    df = pd.read_csv(csv_path)
    df["month"] = df["month"].astype(str).str[:7]

    # Restrict to needed months
    df_sel = df[df["month"].isin(needed)].sort_values("month")

    if df_sel.empty:
        return {
            "bank": bank,
            "range": f"{start} → {end}",
            "table": [],
            "justifications": {},
            "df_plot": [],
            "warning": "No scored months available for the selected period.",
        }

    # ---------------- Build table + justifications ----------------
    table = []
    justifications = {}

    for _, row in df_sel.iterrows():
        ym = row["month"]
        model_score = float(row["model_score"])
        reference_score = float(row["reference_score"])
        text = merged.get(ym, "")

        js = load_or_generate_justification(
            bank, ym, text, model_score, reference_score
        )
        justifications[ym] = js

        summary, narrative = extract_summary_and_narrative(js)

        table.append(
            {
                "month": ym,
                "model_score": model_score,
                "summary": summary,
                "justification": narrative,
            }
        )

    # Data for plotting (only model_score)
    df_plot = df_sel[["month", "model_score"]].copy()

    return {
        "bank": bank,
        "range": f"{start} → {end}",
        "table": table,
        "justifications": justifications,
        "df_plot": df_plot.to_dict(orient="records"),
    }
