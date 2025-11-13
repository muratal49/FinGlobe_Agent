import pandas as pd
import json, time, datetime as dt, logging
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
# --- add near imports ---
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException

# ---------------- Logger Config ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("mpc_scraper.log", mode="w"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

URL = "https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes/monetary-policy-summary-and-minutes"

def getDriver(headless=False):
    opts = webdriver.ChromeOptions()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--window-size=1600,1100")
    opts.add_argument("--disable-gpu")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)
    driver.set_page_load_timeout(90)
    logger.info("Initialized Chrome driver (headless=%s)", headless)
    return driver

# --- add helper: only accept true MPC Summary & Minutes article pages ---
def is_mps_href(href: str) -> bool:
    if not href:
        return False
    href = href.strip()
    if "/monetary-policy-summary-and-minutes/" not in href:
        return False
    # exclude section hubs or non-article paths you don't want
    bad_bits = ["/news/", "/events", "/statistics", "/speeches", "/publications", "/prudential-regulation"]
    return not any(b in href for b in bad_bits)

# --- add helpers for pagination & waiting ---

def _results_container(driver):
    # container that holds the list of results/cards
    for sel in [
        "ul.list-results",               # common BoE search result list
        "ol.list-results",
        "[data-component*='search-results']",
        "main #content",
        "main"
    ]:
        els = driver.find_elements(By.CSS_SELECTOR, sel)
        if els:
            return els[0]
    return driver.find_element(By.TAG_NAME, "body")

def _result_cards(driver):
    # grab result "cards" on page; fall back to anchors
    cards = driver.find_elements(By.CSS_SELECTOR,
        "li.search-result, article.search-result, li.result, article.result, li.list-results__item, article")
    if cards:
        return cards
    return driver.find_elements(By.XPATH, "//a[contains(@href,'/monetary-policy-summary-and-minutes/')]")

def _pagination_nav(driver):
    navs = driver.find_elements(By.CSS_SELECTOR, "nav.container-list-pagination[aria-label='pagination']")
    return navs[0] if navs else None

def _pagination_meta(driver):
    nav = _pagination_nav(driver)
    if not nav:
        return 1, 1
    cur = nav.get_attribute("data-page") or "1"
    total = nav.get_attribute("data-pagecount") or "1"
    try:
        return int(cur), int(total)
    except Exception:
        return 1, 1

def _click_page(driver, target_page, timeout=15):
    nav = _pagination_nav(driver)
    if not nav:
        return
    wait = WebDriverWait(driver, timeout)

    # locate a stable old element to wait for staleness/refresh
    old_container = _results_container(driver)
    old_first = None
    try:
        cards = _result_cards(driver)
        old_first = cards[0] if cards else None
    except Exception:
        pass

    # click page link via data-page-link
    link = None
    try:
        link = nav.find_element(By.CSS_SELECTOR, f"a.list-pagination__link[data-page-link='{target_page}']")
    except Exception:
        # fallback: any anchor whose text == target_page
        anchors = nav.find_elements(By.CSS_SELECTOR, "a.list-pagination__link")
        for a in anchors:
            if (a.text or "").strip() == str(target_page):
                link = a
                break

    if not link:
        return

    driver.execute_script("arguments[0].scrollIntoView({block:'center'});", link)
    try:
        link.click()
    except Exception:
        driver.execute_script("arguments[0].click();", link)

    # wait for either: data-page changes, or current class moves, or list refreshes
    def _page_changed(_):
        new_nav = _pagination_nav(driver)
        if not new_nav:
            return False
        dp = new_nav.get_attribute("data-page") or ""
        if dp == str(target_page):
            return True
        # also accept the "current" link state
        try:
            cur = new_nav.find_element(By.CSS_SELECTOR, "a.list-pagination__link--is-current[aria-current='true']")
            if (cur.text or "").strip() == str(target_page):
                return True
        except Exception:
            pass
        return False

    # prefer staleness of the first card; else poll nav attributes; else small sleep
    try:
        if old_first:
            wait.until(EC.staleness_of(old_first))
        else:
            wait.until(_page_changed)
    except TimeoutException:
        time.sleep(0.8)  # best-effort fallback

    # final guard: ensure some cards exist
    try:
        wait.until(lambda d: len(_result_cards(d)) > 0)
    except TimeoutException:
        pass

def clickCookieIfPresent(driver, wait):
    logger.info("Checking for cookie banner...")
    try:
        # Prefer the “Proceed with necessary cookies only” button
        btn = wait.until(EC.element_to_be_clickable((
            By.XPATH,
            "//button[contains(., 'Proceed with necessary cookies only')]"
        )))
        driver.execute_script("arguments[0].click()", btn)
        logger.info("Clicked: Proceed with necessary cookies only")
        time.sleep(1.0)
        return
    except Exception:
        pass

    # Fallback to the previous “Accept recommended settings” or similar
    selectors = [
        "//button[contains(., 'Accept recommended settings')]",
        "//button[contains(., 'Accept all')]",
        "//button[contains(., 'Accept')]"
    ]
    for xp in selectors:
        try:
            btn = wait.until(EC.element_to_be_clickable((By.XPATH, xp)))
            driver.execute_script("arguments[0].click()", btn)
            logger.info("Fallback clicked: %s", xp)
            time.sleep(1.0)
            return
        except Exception:
            continue

    logger.warning("No cookie banner found or clickable.")


from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def clickMpcMinutes(driver, wait):
    logger.info("Selecting MPC Minutes filter...")

    # Try to bring the topic filter panel into view (best effort)
    try:
        panel = wait.until(EC.presence_of_element_located(
            (By.XPATH, "//legend[contains(., 'Filter by topic')]")
        ))
        driver.execute_script("arguments[0].scrollIntoView({block:'center'})", panel)
        time.sleep(0.5)
    except Exception:
        logger.warning("Filter panel not located by legend; continuing.")

    # 1) Correct, robust XPath (case-insensitive match on label text)
    xp = ("//label["
          "contains(translate(normalize-space(.),"
          " 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),"
          " 'monetary policy committee') and "
          "contains(translate(normalize-space(.),"
          " 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),"
          " 'minutes')]")

    try:
        lbl = wait.until(EC.element_to_be_clickable((By.XPATH, xp)))
        driver.execute_script("arguments[0].scrollIntoView({block:'center'})", lbl)
        driver.execute_script("arguments[0].click()", lbl)
        logger.info("Clicked MPC Minutes checkbox via XPath.")
        time.sleep(1)
        return
    except Exception as e:
        logger.warning("XPath click failed (%s). Falling back to label scan.", e.__class__.__name__)

    # 2) Fallback: scan all labels and click the one whose text matches
    try:
        labels = driver.find_elements(By.TAG_NAME, "label")
        target = None
        for l in labels:
            txt = (l.text or "").strip().lower()
            if "monetary policy committee" in txt and "minutes" in txt:
                target = l
                break
        if target:
            driver.execute_script("arguments[0].scrollIntoView({block:'center'})", target)
            driver.execute_script("arguments[0].click()", target)
            logger.info("Clicked MPC Minutes checkbox via label scan.")
            time.sleep(1)
            return
    except Exception:
        pass

    logger.error("Could not find the 'Monetary Policy Committee (MPC) Minutes' checkbox.")


def findDateInputs(driver):
    candidates = driver.find_elements(By.CSS_SELECTOR, 
        "input[name='from'],input[name='to'],input#from,input#to,input[placeholder*='dd/mm']")
    if len(candidates) >= 2:
        logger.info("Found date input fields via CSS selectors.")
        return candidates[0], candidates[1]
    raise RuntimeError("Date inputs not found on page")



# utils/utils.py  (only new/changed bits)

def formatDate(x):
    if isinstance(x, dt.date):
        return x.strftime("%d/%m/%Y")
    s = str(x).strip()
    # ISO -> DD/MM/YYYY
    try:
        return dt.date.fromisoformat(s).strftime("%d/%m/%Y")
    except Exception:
        pass
    # Already DD/MM/YYYY
    try:
        dt.datetime.strptime(s, "%d/%m/%Y")
        return s
    except Exception:
        raise ValueError(f"Unsupported date format: {x!r}. Use YYYY-MM-DD or DD/MM/YYYY.")


def applyFilters(driver, start=None, end=None):
    wait = WebDriverWait(driver, 25)
    logger.info("Navigating to: %s", URL)
    driver.get(URL)
    clickCookieIfPresent(driver, wait)
    clickMpcMinutes(driver, wait)

    if start is None:
        start = dt.date(dt.date.today().year, 1, 1)
    if end is None:
        end = dt.date.today()

    start_str = formatDate(start)
    end_str = formatDate(end)

    from_box, to_box = findDateInputs(driver)
    driver.execute_script("arguments[0].scrollIntoView({block:'center'})", from_box)
    from_box.clear(); from_box.send_keys(start_str)
    to_box.clear(); to_box.send_keys(end_str); to_box.send_keys("\n")
    logger.info("Applied date filters: from %s to %s", start_str, end_str)
    time.sleep(2)


def expandAll(driver):
    wait = WebDriverWait(driver, 10)
    count = 0
    while True:
        try:
            btn = wait.until(EC.element_to_be_clickable((
                By.XPATH, "//button[contains(., 'Load more') or contains(., 'Show more results')]")))
            driver.execute_script("arguments[0].click()", btn)
            time.sleep(1.5)
            count += 1
        except Exception:
            break
    logger.info("Expanded results %d time(s).", count)



# --- replace your collectItems(driver) with this paginated version ---

def collectItems(driver):
    """
    Collect all MPC Minutes links across *all* paginated result pages.
    Fixes the partial pagination issue that skipped 2016–2019.
    """
    wait = WebDriverWait(driver, 20)
    try:
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "main")))
    except TimeoutException:
        logger.warning("Main container not detected; proceeding anyway.")

    all_items, seen = [], set()
    page, total = _pagination_meta(driver)
    logger.info("Detected %d total page(s) at start.", total)

    # iterate until no next page or max page reached
    while True:
        # --- collect cards on current page ---
        try:
            cards = _result_cards(driver)
        except Exception:
            cards = []
        got = 0

        for c in cards:
            href = None
            date_txt = ""
            try:
                a = c.find_element(By.XPATH, ".//a[contains(@href,'/monetary-policy-summary-and-minutes/')]")
                href = a.get_attribute("href")
            except Exception:
                try:
                    href = c.get_attribute("href")
                except Exception:
                    href = None

            if not href or not is_mps_href(href) or href in seen:
                continue

            try:
                # prefer <time datetime>
                t = c.find_element(By.XPATH, ".//time[@datetime]")
                date_txt = (t.get_attribute("datetime") or "")[:10]
            except Exception:
                try:
                    d = c.find_element(By.XPATH, ".//*[contains(@class,'published-date')]")
                    date_txt = (d.text or "").strip()
                except Exception:
                    pass

            all_items.append((date_txt, href))
            seen.add(href)
            got += 1

        cur, total = _pagination_meta(driver)
        logger.info("Collected %d item(s) on page %d/%d (total %d).",
                    got, cur, total, len(all_items))

        # --- move to next page if exists ---
        try:
            next_btn = driver.find_element(By.CSS_SELECTOR, "a.list-pagination__link[data-page-link='%d']" % (cur + 1))
            if "is-disabled" in next_btn.get_attribute("class"):
                break
            driver.execute_script("arguments[0].scrollIntoView({block:'center'});", next_btn)
            driver.execute_script("arguments[0].click();", next_btn)
            time.sleep(2.5)  # give time for results to refresh
            cur2, _ = _pagination_meta(driver)
            if cur2 <= cur:
                break  # pagination didn't advance
        except Exception:
            break

    logger.info("✅ Collected %d MPC items across %d page(s).", len(all_items), max(1, total))
    uniq = {h: d for d, h in all_items}
    return [(v, k) for k, v in uniq.items()]


import re
def parseDate(hint, html):
    # 1) Card hint like "18 September 2025"
    try:
        return dt.datetime.strptime(hint.strip(), "%d %B %Y").date().isoformat()
    except Exception:
        pass

    soup = BeautifulSoup(html, "html.parser")
    # 2) Published on <div class="published-date">Published on  18 September 2025</div>
    pub = soup.select_one("main#main-content .published-date")
    if pub:
        m = re.search(r"(\d{1,2}\s+[A-Za-z]+\s+\d{4})", pub.get_text(" ", strip=True))
        if m:
            try:
                return dt.datetime.strptime(m.group(1), "%d %B %Y").date().isoformat()
            except Exception:
                pass
    # 3) time[datetime] fallback
    t = soup.select_one("time[datetime]")
    if t and t.get("datetime"):
        try:
            return dt.date.fromisoformat(t["datetime"][:10]).isoformat()
        except Exception:
            pass
    return None



def extractContent(driver, href, hint):
    logger.info("Visiting %s", href)
    driver.get(href)
    WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.TAG_NAME, "main")))
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")

    # anchor at the main container
    main = soup.select_one("main#main-content") or soup.select_one("main#main") or soup.select_one("main")

    # find the *inner* content-block that contains #output
    content_block = None
    for cand in main.select("div.content-block"):
        if cand.select_one("#output"):
            content_block = cand
            break
    if not content_block:
        logger.warning("Inner content-block with #output not found; falling back to main.")
        content_block = main

    # first section.page-section under #output (== Monetary Policy Summary)
    section = (content_block.select_one("#output > section.page-section") or
               content_block.select_one("section.page-section"))

    if not section:
        logger.error("No section.page-section found; falling back to whole content_block.")
        section = content_block

    # collect only headings/paragraphs/lists from that first section
    nodes = section.select("h1, h2, h3, p, li")
    text = "\n".join(n.get_text(" ", strip=True) for n in nodes if n.get_text(strip=True)).strip()

    key = parseDate(hint, html) or dt.date.today().isoformat()
    logger.info("Extracted %d chars from first summary section (date %s)", len(text), key)
    return key, text


# utils/utils.py (add near formatDate or at bottom)

def iso_for_filename(x):
    # returns YYYYMMDD for filenames
    if isinstance(x, dt.date):
        return x.strftime("%Y%m%d")
    s = str(x).strip()
    try:
        return dt.date.fromisoformat(s).strftime("%Y%m%d")
    except Exception:
        # DD/MM/YYYY -> YYYYMMDD
        try:
            return dt.datetime.strptime(s, "%d/%m/%Y").strftime("%Y%m%d")
        except Exception:
            raise ValueError(f"Unsupported date format for filename: {x!r}")

# --- add near other small helpers in app.py ---
def filterByRange(data, s, e):
    from datetime import date
    def toiso(x):
        # accepts YYYY-MM-DD or DD/MM/YYYY; returns ISO 'YYYY-MM-DD'
        try: return date.fromisoformat(x).isoformat()
        except: 
            import datetime as dt
            return dt.datetime.strptime(x, "%d/%m/%Y").date().isoformat()
    s_iso = toiso(s) if s else None
    e_iso = toiso(e) if e else None
    out = {}
    for d, txt in (data.items() if isinstance(data, dict) else []):
        if not isinstance(d, str): continue
        if s_iso and d < s_iso: continue
        if e_iso and d > e_iso: continue
        out[d] = txt
    return out
