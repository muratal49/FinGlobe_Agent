# tools/meeting_scrapper.py

import json, time, datetime as dt, logging, os
from selenium.common.exceptions import TimeoutException

from utils.utils import (
    logger,
    getDriver,
    applyFilters,
    collectItems,       # assumes you've replaced utils.collectItems with the pagination-safe version
    extractContent,
    iso_for_filename,
)

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
        import re
        m = re.search(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", str(hint))
        if m:
            d, mon, y = int(m.group(1)), m.group(2).lower(), int(m.group(3))
            if mon in MONTHS:
                try:
                    return dt.date(y, MONTHS[mon], d).isoformat()
                except Exception:
                    pass
    # fallback: parse year & month from URL
    import re
    m = re.search(r"/(20\d{2}|19\d{2})/([a-z]+)-\1", href or "")
    if m:
        y = int(m.group(1)); mon = m.group(2).lower()
        if mon in MONTHS:
            try:
                return dt.date(y, MONTHS[mon], 1).isoformat()
            except Exception:
                pass
    return None
# -------------------------------------------------------------------------------

def run(start=None, end=None, headless=False, write_files=True):
    driver = getDriver(headless=headless)
    try:
        # navigate + filters (handles cookies & ticking MPC Minutes)
        applyFilters(driver, start=start, end=end)

        # collect all paginated items
        items = collectItems(driver)  # -> list[(hint, href)]

        # strict whitelist + date window
        s_iso = dt.date.fromisoformat(str(start)).isoformat() if start else None
        e_iso = dt.date.fromisoformat(str(end)).isoformat() if end else None

        filtered = []
        for hint, href in items:
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

        # visit and extract
        data = {}
        for _, hint, href in filtered:
            key, txt = extractContent(driver, href, hint)
            if txt and "to be published" not in txt.lower():
                data[key] = txt

        if write_files:
            os.makedirs("cache/scrapped_data", exist_ok=True)

            with open("mpc_minutes_boe.json", "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            ts_cache = f"cache/scrapped_data/mpc_minutes_boe_{stamp}.json"
            with open(ts_cache, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            s = start or dt.date(dt.date.today().year, 1, 1)
            e = end   or dt.date.today()
            s_tag = iso_for_filename(s)
            e_tag = iso_for_filename(e)
            range_cache = f"cache/scrapped_data/mpc_minutes_boe{s_tag}_{e_tag}.json"
            with open(range_cache, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.info(
                "Saved %d entries | working=mpc_minutes_boe.json | ts=%s | range=%s",
                len(data), ts_cache, range_cache
            )

        return data

    except Exception as e:
        logger.exception("Fatal error: %s", e)
        try:
            driver.save_screenshot("error_screenshot.png")
            with open("error_dom.html", "w", encoding="utf-8") as f:
                f.write(driver.page_source)
        except Exception:
            pass
        raise
    finally:
        driver.quit()

if __name__ == "__main__":
    run()

