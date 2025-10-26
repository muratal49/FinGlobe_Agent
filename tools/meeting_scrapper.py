from selenium import webdriver
import json, time, datetime as dt, logging, os
from pathlib import Path # ADDED

# Assuming utils.py provides the necessary functions:
from mcp_project.utils.utils import \
    logger, \
    getDriver, clickCookieIfPresent, clickMpcMinutes, \
    findDateInputs, applyFilters, expandAll, collectItems, parseDate, extractContent, \
    formatDate, iso_for_filename

# --- NEW CONFIGURATION ---
# Define the root of your project structure
PROJECT_ROOT = Path("/Users/murat/Desktop/Capstone/FinGlobe_Agent")
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_FILENAME = "minutes_boe.json"
# -------------------------

def run(start=None, end=None, headless=False, write_files=True):
    driver = getDriver(headless=headless)
    try:
        # 1. Apply Filters and Collect Items
        applyFilters(driver, start=start, end=end)
        items = collectItems(driver)
        data = {}
        
        # 2. Extract Content
        for hint, href in items:
            # key = date string (ISO 8601 or similar)
            # txt = raw content
            key, txt = extractContent(driver, href, hint)
            if txt and "to be published" not in txt.lower():
                data[key] = txt

        if write_files:
            # Ensure the output directory exists
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            
            # --- WRITE FINAL ALIGNED FILE ---
            final_output_path = OUTPUT_DIR / OUTPUT_FILENAME
            
            with open(final_output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            # --- Simplified Logging ---
            logger.info(
                "✅ Minutes Scraper completed. Saved %d entries to %s",
                len(data), final_output_path.resolve()
            )

        return data

    except Exception as e:
        logger.exception("Fatal error: %s", e)
        # ... (error handling remains the same)
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
    # Example usage: scrape last 12 months in headless mode
    one_year_ago = dt.date.today() - dt.timedelta(days=365)
    run(start=one_year_ago, headless=True)