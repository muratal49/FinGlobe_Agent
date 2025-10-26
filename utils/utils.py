
import json, time, datetime as dt, logging
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

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



def collectItems(driver):
    expandAll(driver)
    links = driver.find_elements(By.XPATH,
        "//a[contains(@href,'/news/') or contains(@href,'/monetary-policy-summary-and-minutes/')]")
    items = []
    for a in links:
        t = (a.text or "").strip()
        if not t or "to be published" in t.lower(): 
            continue
        href = a.get_attribute("href")
        date_txt = ""
        try:
            date_el = a.find_element(By.XPATH, ".//preceding::*[contains(@class,'published-date')][1]")
            date_txt = date_el.text.strip()
        except Exception:
            pass
        items.append((date_txt, href))
    logger.info("Collected %d MPC items.", len(items))
    uniq = {h: d for d, h in items}
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
