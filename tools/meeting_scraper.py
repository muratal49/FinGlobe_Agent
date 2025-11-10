# tools/meeting_scrapper.py
import json, time, datetime as dt, logging, os, re
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

# === PATHS (UNIFIED) ===
BASE_PATH = Path("/Users/murat/Desktop/Capstone/FinGlobe_Agent")
DATA_RAW = BASE_PATH / "data" / "raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)

OUTPUT_FILENAME = "minutes_boe.json"
OUTPUT_PATH = DATA_RAW / OUTPUT_FILENAME   # <— canonical output
CACHE_DIR = DATA_RAW / "cache_minutes"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# --- local helpers (kept here to strictly gate items & dates regardless of utils) ---
MONTHS = {m.lower(): i for i, m in enumerate(
    ["January","February","March","April","May","June","July","August","September","October","November","December"], 1)
}

def _is_mps_href(href: str) -> bool:
    if not href:
        return False
    href = href.strip()
    if "/monetary-policy-summary-and-minutes/" not in href:
        return False
    # exclude hubs/other sections
    bad = ("/news/", "/events", "/statistics", "/speeches", "/publications", "/prudential-regulation")
    return not any(b in href for b in bad)

def _iso_from_hint_or_href(hint, href):
    # ISO in hint
    if hint:
        try:
            return dt.date.fromisoformat(str(hint)[:10]).isoformat()
        except Exception:
            pass
        # "18 September 2018"
        m = re.search(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", str(hint))
        if m:
            d, mon, y = int(m.group(1)), m.group(2).lower(), int(m.group(3))
            if mon in MONTHS:
                try:
                    return dt.date(y, MONTHS[mon], d).isoformat()
                except Exception:
                    pass
    # fallback: parse year & month from URL
    m = re.search(r"/(20\d{2}|19\d{2})/([a-z]+)-\1", (href or "").lower())
    if m:
        y = int(m.group(1)); mon = m.group(2)
        if mon in MONTHS:
            try:
                return dt.date(y, MONTHS[mon], 1).isoformat()
            except Exception:
                pass
    return None

def _apply_filters_best_effort(driver, start_iso: str, end_iso: str):
    """Try common signatures; never crash. This prevents fallback to 'last 6 months'."""
    try:
        applyFilters(driver, start_date=start_iso, end_date=end_iso); return
    except Exception:
        pass
    try:
        applyFilters(driver, start_iso, end_iso); return
    except Exception:
        pass
    try:
        applyFilters(driver, start=start_iso, end=end_iso); return
    except Exception:
        logger.warning("applyFilters could not set date inputs; proceeding with strict local filtering.")

# -------------------------------------------------------------------------------

def run(start=None, end=None, headless=False, write_files=True):
    driver = getDriver(headless=headless)
    try:
        # --- enforce a concrete window from args (prevents 'last 6 months' default) ---
        s_iso = dt.date.fromisoformat(str(start)).isoformat() if start else None
        e_iso = dt.date.fromisoformat(str(end)).isoformat() if end else None

        # navigate + filters (handles cookies & ticking MPC Minutes)
        try:
            _apply_filters_best_effort(driver, s_iso or "", e_iso or "")
        except Exception:
            # never die on UI filtering; we'll hard-filter locally below
            pass

        # collect all paginated items
        items = collectItems(driver)  # -> list[(hint, href)] or similar

        # strict whitelist + date window
        filtered = []
        for tup in items or []:
            # support (hint, href) or dict/str variants
            if isinstance(tup, tuple) and len(tup) >= 2:
                hint, href = tup[0], tup[1]
            elif isinstance(tup, dict):
                hint, href = tup.get("date") or tup.get("hint"), tup.get("url") or tup.get("href")
            else:
                hint, href = "", str(tup)

            if not _is_mps_href(href):
                continue

            iso = _iso_from_hint_or_href(hint, href)
            if not iso:
                continue
            if s_iso and iso < s_iso:
                continue
            if e_iso and iso > e_iso:
                continue
            filtered.append((iso, hint, href))

        # sort ascending by date (change to reverse=True for newest first)
        filtered.sort(key=lambda x: x[0])

        # visit and extract — build DATE -> TEXT map; keep longest text if dup date
        date_map = {}
        for iso, hint, href in filtered:
            try:
                key, txt = extractContent(driver, href, hint)
            except TypeError:
                # some versions may have different signature; try url-only
                ret = extractContent(href)
                # normalize return to text
                txt = ret.get("text") if isinstance(ret, dict) else (ret or "")
            if not txt or "to be published" in str(txt).lower():
                continue
            prev = date_map.get(iso, "")
            if len(str(txt)) > len(prev):
                date_map[iso] = str(txt)

        if write_files:
            # --- write canonical map to data/raw/minutes_boe.json ---
            with OUTPUT_PATH.open("w", encoding="utf-8") as f:
                json.dump(date_map, f, ensure_ascii=False, indent=2)

            # also keep timestamped & range cache copies (same map format)
            stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            ts_cache = CACHE_DIR / f"minutes_boe_{stamp}.json"
            with ts_cache.open("w", encoding="utf-8") as f:
                json.dump(date_map, f, ensure_ascii=False, indent=2)

            s = dt.date.fromisoformat(s_iso) if s_iso else dt.date(dt.date.today().year, 1, 1)
            e = dt.date.fromisoformat(e_iso) if e_iso else dt.date.today()
            s_tag = iso_for_filename(s)
            e_tag = iso_for_filename(e)
            range_cache = CACHE_DIR / f"minutes_boe_{s_tag}_{e_tag}.json"
            with range_cache.open("w", encoding="utf-8") as f:
                json.dump(date_map, f, ensure_ascii=False, indent=2)

            logger.info(
                "Saved %d dates | map=%s | ts=%s | range=%s",
                len(date_map), str(OUTPUT_PATH), str(ts_cache), str(range_cache)
            )

        return date_map

    except Exception as e:
        logger.exception("Fatal error: %s", e)
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
    # keep the same entrypoint signature as your working script
    # so your pipeline command does not change.
    # Example: python tools/meeting_scrapper.py --start-date 2025-01-01 --end-date 2025-03-01
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-date", dest="start", required=False)
    ap.add_argument("--end-date", dest="end", required=False)
    ap.add_argument("--headless", action="store_true", default=False)
    args = ap.parse_args()
    run(start=args.start, end=args.end, headless=args.headless, write_files=True)
