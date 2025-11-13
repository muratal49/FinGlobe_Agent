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
OUTPUT_PATH = DATA_RAW / OUTPUT_FILENAME   # canonical output
CACHE_DIR = DATA_RAW / "cache_minutes"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# --- local helpers ---
MONTHS = {m.lower(): i for i, m in enumerate(
    ["January","February","March","April","May","June","July","August",
     "September","October","November","December"], 1)
}

def _is_mps_href(href: str) -> bool:
    if not href:
        return False
    href = href.strip()
    if "/monetary-policy-summary-and-minutes/" not in href:
        return False
    # exclude hubs/other sections
    bad = ("/news/", "/events", "/statistics", "/speeches",
           "/publications", "/prudential-regulation")
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

    # fallback: parse year/month from URL
    m = re.search(r"/(20\d{2}|19\d{2})/([a-z]+)-\1", (href or "").lower())
    if m:
        y = int(m.group(1))
        mon = m.group(2)
        if mon in MONTHS:
            try:
                return dt.date(y, MONTHS[mon], 1).isoformat()
            except Exception:
                pass
    return None

def _apply_filters_best_effort(driver, start_iso: str, end_iso: str):
    """Try common signatures safely; never break the scraper."""
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
        logger.warning("applyFilters could not set date inputs; using strict local filtering.")


# -------------------------------------------------------------------------------

def run(start=None, end=None, headless=False, write_files=True):
    """
    Scrape MPC minutes strictly between start and end (ISO format only).
    Examples:
        run("2016-01-01", "2019-12-31")
        run("2015-01-01", "2024-12-31")
    """
    driver = getDriver(headless=headless)

    try:
        # user must pass ISO dates → no fallback, no 6-month window
        s_iso = dt.date.fromisoformat(str(start)).isoformat() if start else None
        e_iso = dt.date.fromisoformat(str(end)).isoformat() if end else None

        # apply filtering in UI (best-effort)
        try:
            _apply_filters_best_effort(driver, s_iso or "", e_iso or "")
        except Exception:
            pass

        # collect all paginated items
        items = collectItems(driver)  # → list[(hint, href)] or dicts
        
        # strict local filtering
        filtered = []
        for tup in items or []:

            # normalize structure
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

            # enforce date window
            if s_iso and iso < s_iso:
                continue
            if e_iso and iso > e_iso:
                continue

            filtered.append((iso, hint, href))

        # sort ascending
        filtered.sort(key=lambda x: x[0])

        # extract text
        date_map = {}
        for iso, hint, href in filtered:
            try:
                key, txt = extractContent(driver, href, hint)
            except TypeError:
                # fallback signature
                ret = extractContent(href)
                txt = ret.get("text") if isinstance(ret, dict) else (ret or "")

            if not txt or "to be published" in str(txt).lower():
                continue

            prev_txt = date_map.get(iso, "")
            if len(str(txt)) > len(prev_txt):
                date_map[iso] = str(txt)

        # write outputs
        if write_files:
            # main file
            with OUTPUT_PATH.open("w", encoding="utf-8") as f:
                json.dump(date_map, f, ensure_ascii=False, indent=2)

            # timestamped cache
            stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            ts_file = CACHE_DIR / f"minutes_boe_{stamp}.json"
            with ts_file.open("w", encoding="utf-8") as f:
                json.dump(date_map, f, ensure_ascii=False, indent=2)

            # range-tagged cache
            s = dt.date.fromisoformat(s_iso) if s_iso else dt.date(dt.date.today().year, 1, 1)
            e = dt.date.fromisoformat(e_iso) if e_iso else dt.date.today()
            s_tag = iso_for_filename(s)
            e_tag = iso_for_filename(e)
            range_file = CACHE_DIR / f"minutes_boe_{s_tag}_{e_tag}.json"
            with range_file.open("w", encoding="utf-8") as f:
                json.dump(date_map, f, ensure_ascii=False, indent=2)

            logger.info(
                "Saved %d minutes | main=%s | ts=%s | range=%s",
                len(date_map), OUTPUT_PATH, ts_file, range_file
            )

        return date_map

    except Exception as e:
        logger.exception("Fatal scraper error: %s", e)
        try:
            driver.save_screenshot(str(CACHE_DIR / "error_screenshot.png"))
            with (CACHE_DIR / "error_dom.html").open("w", encoding="utf-8") as f:
                f.write(driver.page_source)
        except Exception:
            pass
        raise

    finally:
        driver.quit()

def collect_all_pages(driver):
    """Scrape ALL pages of BoE results reliably."""
    results = []
    seen = set()

    while True:
        # scrape current page items
        page_items = collectItems(driver)
        for item in page_items:
            # normalize to tuple (hint, href)
            if isinstance(item, tuple) and len(item) >= 2:
                hint, href = item[0], item[1]
            elif isinstance(item, dict):
                hint = item.get("date") or item.get("hint")
                href = item.get("url") or item.get("href")
            else:
                hint, href = "", str(item)

            if href and href not in seen:
                seen.add(href)
                results.append((hint, href))

        # try to click NEXT
        try:
            next_button = driver.find_element("xpath", "//a[contains(@class,'pagination__item--next')]")
            if "disabled" in next_button.get_attribute("class"):
                break
            next_button.click()
            time.sleep(2)
        except Exception:
            break

    return results

# CLI entry
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-date", dest="start")
    ap.add_argument("--end-date", dest="end")
    ap.add_argument("--headless", action="store_true", default=False)
    args = ap.parse_args()
    run(start=args.start, end=args.end, headless=args.headless, write_files=True)
