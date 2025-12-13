#!/usr/bin/env python3
"""
Bank of Canada – MPR PDF Scraper (Clean Narrative Version)
---------------------------------------------------------
Features:
• Uses cache-first PDF text loading
• Moderate chart removal (keeps narrative, removes chart blocks)
• Keeps only key macroeconomic narrative:
      - Overview
      - Current Conditions
      - Economic Outlook
      - Inflation Outlook
      - Global Outlook
      - Risks
      - Conclusion
• Merges all into a single cleaned_text column
• Extracts real MPR dates by trying all days in Jan / Apr / Jul / Oct
• Saves CSV to: data/raw/boc_policy_mpr.csv
"""

import argparse
import io
import re
from datetime import date, datetime, timedelta
from pathlib import Path

import pdfplumber
import pandas as pd
import requests


# ---------------------------------------------------------------
#   Paths
# ---------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = REPO_ROOT / "data" / "raw"
CACHE_DIR = REPO_ROOT / "data" / "cache" / "boc_mpr"

DATA_RAW.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = DATA_RAW / "boc_policy_mpr.csv"
MPR_MONTHS = (1, 4, 7, 10)  # Jan, Apr, Jul, Oct


# ---------------------------------------------------------------
#   URL construction
# ---------------------------------------------------------------
def pdf_url_for_date(d: date) -> str:
    """Return PDF URL based on YYYY/MM/mpr-YYYY-MM-DD.pdf."""
    yyyy = d.year
    mm = f"{d.month:02d}"
    dd = f"{d.day:02d}"
    return f"https://www.bankofcanada.ca/wp-content/uploads/{yyyy}/{mm}/mpr-{yyyy}-{mm}-{dd}.pdf"


# ---------------------------------------------------------------
#   PDF loader (with caching)
# ---------------------------------------------------------------
def get_pdf_text(d: date) -> str | None:
    """Load PDF text from cache or download. Returns None if invalid."""
    cache_path = CACHE_DIR / f"mpr_{d.isoformat()}.txt"

    # use cache
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    url = pdf_url_for_date(d)
    resp = requests.get(url, timeout=20)

    if resp.status_code != 200:
        return None

    try:
        with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
            pages_text = [(p.extract_text() or "") for p in pdf.pages]
        full = "\n".join(pages_text)
    except Exception as e:
        print(f"!! PDF parse failed for {d}: {e}")
        return None

    cache_path.write_text(full, encoding="utf-8")
    return full


# ---------------------------------------------------------------
#   Text Cleaning
# ---------------------------------------------------------------

def clean_text(text: str) -> str:
    """Moderate cleaning — remove charts, numeric garbage, footnotes, etc."""

    t = text.replace("\r", " ")

    # Remove chart blocks (moderate)
    t = re.sub(r"(?i)Chart\s+\d+.*?(?=\n\n|$)", "", t, flags=re.DOTALL)

    # Remove FIGURE/TABLE blocks
    t = re.sub(r"(?i)(Figure|Table)\s+\d+.*?(?=\n\n|$)", "", t, flags=re.DOTALL)

    # Remove lines that are mostly numbers, axes, percentages, etc.
    t = re.sub(r"(?m)^[0-9\s\.\%\(\)\-]{4,}$", "", t)

    # Remove "Sources:" and metadata
    t = re.sub(r"(?i)Sources?:.*", "", t)
    t = re.sub(r"(?i)Last observation:.*", "", t)
    t = re.sub(r"ISSN\s*\d+", "", t)

        # --- REMOVE TABLE OF CONTENTS BLOCK ---
    # Detect a "Contents" header and remove everything until the first real section.
    toc_pattern = re.compile(
        r"(?is)contents.*?(?=\n[A-Z][A-Za-z].{3,}\n)",
        re.MULTILINE
    )
    t = toc_pattern.sub("", t)


    # Remove stray Unicode ligatures
    t = t.replace("ﬁ", "fi").replace("ﬂ", "fl")

    # Remove excessive newlines
    t = re.sub(r"\n{2,}", "\n\n", t)
    t = re.sub(r"[ \t]+", " ", t)

    return t.strip()


# ---------------------------------------------------------------
#   Section Extractor
# ---------------------------------------------------------------

SECTION_HEADERS = {
    "Overview": ["Overview"],
    "Current Conditions": ["Current conditions", "Current Conditions"],
    "Economic Outlook": [
        "Economic outlook", "Outlook for the Canadian economy",
        "Economic projection", "Canadian Economic Outlook"
    ],
    "Inflation Outlook": [
        "Inflation outlook", "Prices and costs", "Inflation projection"
    ],
    "Global Outlook": ["Global outlook", "Global economy", "International developments"],
    "Risks": ["Risks", "Risks to the outlook", "Upside and downside risks"],
    "Conclusion": ["Conclusion", "Conclusions"],
}


def extract_sections(text: str) -> str:
    """Extract and merge relevant sections only."""
    cleaned = []
    lower = text.lower()

    for section_name, variants in SECTION_HEADERS.items():
        for h in variants:
            idx = lower.find(h.lower())
            if idx != -1:
                # extract text starting at this header
                block = text[idx:]

                # stop at next ALL-CAPS header
                m = re.search(r"\n[A-Z][A-Z0-9 ,\-\(\)]{3,}\n", block)
                if m:
                    block = block[:m.start()]

                # clean isolated broken lines
                block = re.sub(r"\n{2,}", "\n\n", block)

                cleaned.append(f"{section_name}\n{block.strip()}")
                break  # stop after first match

    return "\n\n".join(cleaned).strip()


# ---------------------------------------------------------------
#   Month helper
# ---------------------------------------------------------------

def month_last_day(year: int, month: int) -> int:
    first = date(year, month, 1)
    next_month = (first.replace(day=28) + timedelta(days=4)).replace(day=1)
    return (next_month - timedelta(days=1)).day


# ---------------------------------------------------------------
#   Main MPR Scraper
# ---------------------------------------------------------------

def scrape_mprs(start: date, end: date) -> list[dict]:
    """
    Scrape BoC MPR PDFs, but only search from day 23 → last day of Jan/Apr/Jul/Oct.
    This prevents 90% of useless requests because MPRs always release late in the month.
    """
    rows = []

    for year in range(start.year, end.year + 1):
        for month in MPR_MONTHS:
            # Compute first/last allowable day for this MPR window
            first_of_month = date(year, month, 1)
            last_of_month = date(year, month, month_last_day(year, month))

            # Skip month entirely if out of range
            if last_of_month < start or first_of_month > end:
                continue

            print(f"\n=== {year}-{month:02d} ===")

            # NEW: Only search day 23 → last day
            search_start_day = max(23, first_of_month.day)
            search_end_day = last_of_month.day

            found = False

            for day in range(search_start_day, search_end_day + 1):
                d = date(year, month, day)
                if d < start or d > end:
                    continue

                url = pdf_url_for_date(d)
                print(f"Checking {d} → {url}")

                raw_text = get_pdf_text(d)
                if not raw_text:
                    print("  ✘ No PDF")
                    continue

                print("  ✔ PDF found → cleaning + extracting sections")

                cleaned_raw = clean_text(raw_text)
                merged_text = extract_sections(cleaned_raw)

                title = f"monetary_policy_{d.strftime('%B').lower()}_{year}"

                rows.append({
                    "date": d.isoformat(),
                    "title": title,
                    "pdf_url": url,
                    "cleaned_text": merged_text,
                })

                found = True
                break

            if not found:
                print("  (No MPR PDF found between day 23 and month-end)")

    return rows



# ---------------------------------------------------------------
#   Main CLI
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    args = parser.parse_args()

    start = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end = datetime.strptime(args.end_date, "%Y-%m-%d").date()

    rows = scrape_mprs(start, end)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n✔ DONE. Saved {len(df)} MPR records → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
