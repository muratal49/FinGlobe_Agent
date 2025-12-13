#!/usr/bin/env python3
"""
Scrape Bank of England speech pages (HTML only), filter by date and topic,
clean the full text, and extract the conclusion/summary section.

Now outputs:
- a_boe_speeches_conclusion.json
- a_boe_cache_speeches/a_boe_speeches_conclusion.json
"""

import re
import json
import time
import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

import requests
from bs4 import BeautifulSoup


# === PATHS (UNIFIED) ===
BASE_PATH = Path("/Users/murat/Desktop/Capstone/FinGlobe_Agent")
DATA_PATH = BASE_PATH / "data"
RAW = DATA_PATH / "raw"
# NEW OUTPUT JSON FILES
OUTPUT_JSON = RAW / "a_boe_speeches_conclusion.json"
CACHE_FOLDER = RAW / "a_boe_cache_speeches"
CACHE_FILE = CACHE_FOLDER / "a_boe_speeches_conclusion.json"

CACHE_FOLDER.mkdir(parents=True, exist_ok=True)


# ---------- Config defaults ----------
SITEMAP_URL = "https://www.bankofengland.co.uk/sitemap/speeches"
DEFAULT_KEYWORDS = ["Monetary Policy Committee", "MPC", "inflation"]

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
}

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_4) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"
)

session = requests.Session()
session.headers.update({
    "User-Agent": UA,
    "Accept-Language": "en-GB,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Connection": "keep-alive",
})


# =============== YOUR ORIGINAL HELPERS (UNCHANGED) ===============
# (I left everything the same – only summary logic updated later)

def is_html_speech_url(url: str) -> bool:
    if "/-/media/" in url:
        return False
    if not url.startswith("https://www.bankofengland.co.uk/speech/"):
        return False
    parts = url.rstrip("/").split("/")
    if len(parts) < 6:
        return False
    year = parts[4]
    month = parts[5].lower()
    if not year.isdigit() or month not in MONTHS:
        return False
    return True


def date_from_url(url: str) -> datetime:
    parts = url.rstrip("/").split("/")
    year = int(parts[4])
    month = MONTHS.get(parts[5].lower())
    return datetime(year, month, 1)


def parse_date_from_page(soup, fallback_url_date):
    # unchanged
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            import json
            data = json.loads(tag.get_text(strip=True))
            items = data if isinstance(data, list) else [data]
            for it in items:
                dp = it.get("datePublished") or it.get("dateModified")
                if dp:
                    return datetime.fromisoformat(dp.replace("Z", "+00:00"))
        except: pass

    candidates = [
        ("meta", {"property": "article:published_time"}, "content"),
        ("meta", {"name": "pubdate"}, "content"),
        ("meta", {"name": "date"}, "content"),
    ]
    for name, attrs, attr_field in candidates:
        tag = soup.find(name, attrs=attrs)
        if tag and tag.get(attr_field):
            try:
                txt = tag[attr_field]
                return datetime.fromisoformat(txt.replace("Z", "+00:00"))
            except: pass

    body_text = soup.get_text(" ", strip=True)
    m = re.search(r"Published on\s+(\d{1,2}\s+[A-Za-z]+\s+\d{4})", body_text, flags=re.I)
    if m:
        return datetime.strptime(m.group(1), "%d %B %Y").replace(tzinfo=timezone.utc)

    return fallback_url_date.replace(tzinfo=timezone.utc)


def extract_title(soup):
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(" ", strip=True)
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    return ""


def extract_speaker(title, soup):
    # unchanged – speaker logic kept
    patts = [
        r"(?:speech|remarks|address|lecture)\s+by\s+(.+)$",
        r"(?:in conversation with|conversation with)\s+(.+)$",
        r"(?:by)\s+(.+)$",
    ]
    for p in patts:
        m = re.search(p, title, flags=re.I)
        if m:
            return m.group(1).strip(" .–-")

    meta_author = soup.find("meta", attrs={"name": "author"})
    if meta_author and meta_author.get("content"):
        return meta_author["content"].strip()

    m2 = re.search(r"Given by\s+(.+)", soup.get_text("\n", strip=True), flags=re.I)
    if m2:
        return m2.group(1).strip(" .–-")

    return ""


def extract_body_text(soup):
    container = soup.find("main") or soup.find("article") or soup.body or soup
    other_speeches_tag = None
    for tag in container.find_all(["h2","h3","p","li"]):
        if tag.get_text(strip=True).lower().startswith("other speeches"):
            other_speeches_tag = tag
            break

    texts = []
    for tag in container.find_all(["h2","h3","p","li"]):
        if other_speeches_tag and tag == other_speeches_tag:
            break
        txt = tag.get_text(" ", strip=True)
        if not txt:
            continue
        if txt.lower() in ("back to top", "skip to main content"):
            continue
        texts.append(txt)

    out = "\n".join(texts)
    if re.search(r"Text to be published|to be published", out, flags=re.I):
        return ""
    out = re.sub(r"\s+\n", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def strip_ack_refs_footnotes(text):
    if not text:
        return text
    section_re = re.compile(r"(?mis)(^|\n)\s*(Acknowledgements?|Acknowledgments?|References?|Footnotes?)\s*(:)?\s*(\n|$).*?\Z")
    text = section_re.sub("", text)
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# =============== UPDATED SUMMARY EXTRACTION ===============

def extract_conclusion_or_summary(body: str) -> str:
    """
    1. Try conclusion markers
    2. Try 'Summary' section
    3. Else last 500 words
    """
    if not body or len(body) < 50:
        return ""

    text = re.sub(r"\s+", " ", body)

    # 1. Conclusion markers (your original)
    conclusion_markers = [
        "in conclusion", "to summarize", "conclusion", "summary",
        "to conclude", "final thoughts", "closing remarks"
    ]
    for marker in conclusion_markers:
        m = re.search(rf"\b{marker}\b", text, flags=re.I)
        if m:
            return text[m.start():].strip()

    # 2. Summary heading
    m = re.search(r"\bExecutive Summary\b|\bSummary\b", text, flags=re.I)
    if m:
        return text[m.start():].strip()

    # 3. Fallback → last 500 words
    words = text.split()
    if len(words) > 500:
        return " ".join(words[-500:])
    return text


# =============== YOUR FETCH LOGIC (UNCHANGED) ===============

def fetch_speech(url, keywords_lower):
    try:
        r = session.get(url, timeout=30)
        r.raise_for_status()
    except Exception as e:
        print(f"⚠️ Fetch failed {url}: {e}")
        return None

    soup = BeautifulSoup(r.text, "lxml")
    base_date = date_from_url(url)
    pub_dt = parse_date_from_page(soup, base_date)

    title = extract_title(soup).strip()
    body = extract_body_text(soup)
    body = strip_ack_refs_footnotes(body)

    if not body:
        return None

    text_lower = (title + " " + body).lower()
    if not any(k in text_lower for k in keywords_lower):
        return None

    speaker = extract_speaker(title, soup)

    # UPDATED SUMMARY EXTRACTION
    conclusion_text = extract_conclusion_or_summary(body)

    return {
        "date": pub_dt.date().isoformat(),
        "title": title,
        "speaker": speaker,
        "text": body,
        "conclusion_text": conclusion_text,
        "url": url,
    }


# =============== SCRAPER LOGIC (UNCHANGED) ===============

def scrape(start_date, end_date, keywords, sleep=0.4):
    print(f"🗺 Fetching sitemap… {SITEMAP_URL}")
    try:
        r = session.get(SITEMAP_URL, timeout=30)
        r.raise_for_status()
    except Exception as e:
        print(f"❌ Failed to fetch sitemap: {e}")
        return

    soup = BeautifulSoup(r.text, "lxml")
    links = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/"):
            href = "https://www.bankofengland.co.uk" + href
        if is_html_speech_url(href):
            url_dt = date_from_url(href)
            if start_date <= url_dt.replace(tzinfo=timezone.utc) <= end_date:
                links.append(href)

    links = sorted(set(links), key=lambda u: date_from_url(u), reverse=True)
    print(f"🔗 Candidates: {len(links)}")

    keywords_lower = [k.lower() for k in keywords]
    for url in links:
        rec = fetch_speech(url, keywords_lower)
        if rec:
            yield rec
        time.sleep(sleep)


# =============== MAIN — NOW SAVES JSON ===============

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--keywords", type=str, default=",".join(DEFAULT_KEYWORDS))
    args = parser.parse_args()

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]

    rows = list(scrape(start_date, end_date, keywords))
    print(f"✅ Matched {len(rows)} speeches")

    # Convert to dict keyed by date for JSON
    json_obj = {r["date"]: r for r in rows}

    # Save main JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=4, ensure_ascii=False)

    # Save cache JSON
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=4, ensure_ascii=False)

    print(f"💾 Saved → {OUTPUT_JSON}")
    print(f"💾 Cache → {CACHE_FILE}")


if __name__ == "__main__":
    main()
