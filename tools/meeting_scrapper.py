
from selenium import webdriver
import json, time, datetime as dt, logging, os

from mcp_project.utils.utils import \
    logger, \
    getDriver, clickCookieIfPresent, clickMpcMinutes, \
    findDateInputs, applyFilters, expandAll, collectItems, parseDate, extractContent, iso_for_filename

# tools/meeting_scrapper.py  (only new/changed bits)

# tools/meeting_scrapper.py (imports)
from mcp_project.utils.utils import \
    logger, getDriver, clickCookieIfPresent, clickMpcMinutes, \
    findDateInputs, applyFilters, expandAll, collectItems, parseDate, extractContent, \
    formatDate, iso_for_filename   # <-- add iso_for_filename

def run(start=None, end=None, headless=False, write_files=True):
    driver = getDriver(headless=headless)
    try:
        applyFilters(driver, start=start, end=end)
        items = collectItems(driver)
        data = {}
        for hint, href in items:
            key, txt = extractContent(driver, href, hint)
            if txt and "to be published" not in txt.lower():
                data[key] = txt

        if write_files:
            os.makedirs("cache/scrapped_data", exist_ok=True)

            # always write the working file
            with open("mpc_minutes.json", "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            # write timestamped snapshot (existing behavior)
            stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            ts_cache = f"cache/scrapped_data/mpc_minutes_{stamp}.json"
            with open(ts_cache, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            # NEW: write range-named cache if dates provided (or defaults used)
            s = start or dt.date(dt.date.today().year, 1, 1)
            e = end   or dt.date.today()
            s_tag = iso_for_filename(s)
            e_tag = iso_for_filename(e)
            range_cache = f"cache/scrapped_data/mpc_minutes_{s_tag}_{e_tag}.json"
            with open(range_cache, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.info(
                "Saved %d entries | working=mpc_minutes.json | ts=%s | range=%s",
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
