#!/usr/bin/env python3
"""
Scrape Bank of England MPC Meeting Minutes (full text + summary section).

This version extends the meeting_scraper.py tool to:
 - Capture the full meeting minutes text
 - Extract the Summary section separately
 - Save as JSON: { date: { "full_text": ..., "summary_text": ... } }

Source: https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes
"""

import json
import re
import time
import datetime as dt
from pathlib import Path
from selenium.common.exceptions import TimeoutException

from utils.utils import (
    logger,
    getDriver,
    applyFilters,
    collectItems,
    extractContent,
    iso_for_filename,
)

# === PATHS ===
BASE_PATH = Path("/Users/murat/Desktop/Capstone/FinGlobe_Agent")
DATA_RAW = BASE_PATH / "data" / "raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = DATA_RAW / "a_boe_minutes_full.json"
CACHE_DIR = DATA_RAW / "a_boe_cache_minutes_full"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# --- Helper Functions ---

def _is_mps_href(href: str) -> bool:
    if not href:
        return False
    href = href.strip()
    if "/monetary-policy-summary-and-minutes/" not in href:
        return False
    bad = ("/news/", "/events", "/statistics", "/speeches", "/publications", "/prudential-regulation")
    return not any(b in href for b in bad)


def _iso_from_hint_or_href(hint, href):
    if hint:
        try:
            return dt.date.fromisoformat(str(hint)[:10]).isoformat()
        except Exception:
            pass
        m = re.search(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", str(hint))
        if m:
            d, mon, y = int(m.group(1)), m.group(2).lower(), int(m.group(3))
            MONTHS = {
                "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
                "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
            }
            if mon in MONTHS:
                return dt.date(y, MONTHS[mon], d).isoformat()
    return None


def _apply_filters_best_effort(driver, start_iso: str, end_iso: str):
    try:
        applyFilters(driver, start_date=start_iso, end_date=end_iso)
    except Exception:
        try:
            applyFilters(driver, start_iso, end_iso)
        except Exception:
            try:
                applyFilters(driver, start=start_iso, end=end_iso)
            except Exception:
                logger.warning("applyFilters failed; falling back to local filtering.")


def _extract_summary_section(full_text: str) -> str:
    """
    Attempt to extract the 'Summary' section from the full minutes text.
    """
    if not full_text:
        return ""
    text = re.sub(r"\s+", " ", full_text.strip())
    # Find "Summary" or "Summary of the meeting" sections
    pattern = re.compile(r"(?i)\b(summary|summary of the meeting)\b")
    match = pattern.search(text)
    if not match:
        return ""
    section = text[match.start():]

    # Stop at next major heading (e.g. "Minutes of the meeting", "Committee", etc.)
    end_pattern = re.compile(r"(?i)\b(minutes of the meeting|committee members|decided|voted|section)\b")
    end_match = end_pattern.search(section, 100)
    if end_match:
        section = section[:end_match.start()]
    return section.strip()


# --- Main Scraper ---

def run(start=None, end=None, headless=False, write_files=True):
    """
    Scrape MPC minutes in a date range and save both full and summary sections.
    """
    driver = getDriver(headless=headless)
    try:
        start_iso = dt.date.fromisoformat(str(start)).isoformat() if start else None
        end_iso = dt.date.fromisoformat(str(end)).isoformat() if end else None

        try:
            _apply_filters_best_effort(driver, start_iso or "", end_iso or "")
        except Exception:
            pass

        items = collectItems(driver)  # from utils, now with full pagination
        logger.info(f"Collected {len(items)} total MPC items from search results.")

        records = {}
        for tup in items or []:
            if isinstance(tup, tuple) and len(tup) >= 2:
                hint, href = tup[0], tup[1]
            elif isinstance(tup, dict):
                hint = tup.get("date") or tup.get("hint")
                href = tup.get("url") or tup.get("href")
            else:
                hint, href = "", str(tup)

            if not _is_mps_href(href):
                continue

            iso = _iso_from_hint_or_href(hint, href)
            if not iso:
                continue

            # local date range filter
            if start_iso and iso < start_iso:
                continue
            if end_iso and iso > end_iso:
                continue

            # Extract full text
            try:
                key, txt = extractContent(driver, href, hint)
            except TypeError:
                ret = extractContent(href)
                txt = ret.get("text") if isinstance(ret, dict) else (ret or "")

            if not txt or "to be published" in txt.lower():
                continue

            summary = _extract_summary_section(txt)
            records[iso] = {
                "full_text": txt.strip(),
                "summary_text": summary.strip(),
            }

        if write_files:
            with OUTPUT_PATH.open("w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)

            stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            ts_file = CACHE_DIR / f"minutes_boe_full_{stamp}.json"
            with ts_file.open("w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)

            logger.info(f"💾 Saved {len(records)} records -> {OUTPUT_PATH}")

        return records

    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        try:
            driver.save_screenshot(str(CACHE_DIR / "error_screenshot.png"))
            with (CACHE_DIR / "error_dom.html").open("w", encoding="utf-8") as f:
                f.write(driver.page_source)
        except Exception:
            pass
        raise
    finally:
        driver.quit()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Scrape BoE MPC minutes (full text + summary).")
    ap.add_argument("--start-date", dest="start", required=True, help="Start date (YYYY-MM-DD).")
    ap.add_argument("--end-date", dest="end", required=True, help="End date (YYYY-MM-DD).")
    ap.add_argument("--headless", action="store_true", default=False)
    args = ap.parse_args()
    run(start=args.start, end=args.end, headless=args.headless, write_files=True)
