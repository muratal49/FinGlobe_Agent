#!/usr/bin/env python3
"""
Bank of Canada speeches scraper (topic-filtered, HTML-only).

- Skips PDFs (uses on-page HTML content only)
- Outputs full text, summary (first paragraphs), conclusion, and tail snippet

Usage:
    python tools/boc_speeches.py --start 2010-01-01 --end 2020-12-31

Outputs:
    data/raw/bank_of_canada_speeches.csv
"""

import argparse
import datetime
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup


BASE_LIST_URL = "https://www.bankofcanada.ca/press/speeches/"
CONTAINS_TERM = "monetary policy report"

# Focused topics (hawkish/dovish relevance)
TOPIC_IDS: Dict[str, str] = {
    "20416": "Monetary policy",
    "20804": "Monetary policy communications",
    "127": "Monetary policy framework",
    "227": "Monetary policy implementation",
    "138": "Monetary policy transmission",
    "230": "Monetary policy and uncertainty",
    "123": "Interest rates",
    "140": "Inflation and prices",
    "80": "Inflation targets",
    "20849": "Price stability",
    "20847": "Expectations",
}

SKIP_SUBSTRINGS = [
    "/education",
    "/education-",
    "/learn-",
    "/learning",
    "/education-resources",
]

DELAY = 0.7

REPO_ROOT = Path("/Users/murat/Desktop/Capstone/FinGlobe_Agent")
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = DATA_RAW / "bank_of_canada_speeches.csv"


session = requests.Session()
session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/119.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
})


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def parse_date_input(s: str) -> Optional[datetime.date]:
    s = s.strip()
    if not s:
        return None
    try:
        return datetime.datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {s} (use YYYY-MM-DD)")


def get_speech_list(topic_id: str, start_date: Optional[datetime.date], end_date: Optional[datetime.date]) -> List[str]:
    """
    Return all speech URLs for a given topic ID, using mt_page pagination.
    """
    page = 1
    links: List[str] = []

    while True:
        params = {"topic[]": topic_id, "mt_page": page, "mtf_search": CONTAINS_TERM}
        if start_date:
            params["mtf_date_after"] = start_date.isoformat()
        if end_date:
            params["mtf_date_before"] = end_date.isoformat()
        try:
            r = session.get(BASE_LIST_URL, params=params, timeout=30)
        except requests.RequestException as e:
            print(f"    Request error on topic {topic_id} page {page}: {e}")
            break

        if r.status_code != 200:
            print(f"    Status {r.status_code} on topic {topic_id} page {page}, stopping.")
            break

        soup = BeautifulSoup(r.text, "lxml")
        page_links: List[str] = []

        for h3 in soup.select("h3"):
            a = h3.find("a")
            if not a or not a.get("href"):
                continue
            url = a["href"]
            if not url.startswith("http"):
                url = "https://www.bankofcanada.ca" + url
            if any(skip in url for skip in SKIP_SUBSTRINGS):
                continue
            if url not in links:
                page_links.append(url)

        if not page_links:
            break

        links.extend(page_links)
        print(f"    Page {page}: found {len(page_links)} links")
        page += 1
        time.sleep(DELAY)

    return links


def extract_pdf_text(pdf_url: str) -> str:
    """(Unused placeholder to keep compatibility; PDFs are skipped in this script.)"""
    return ""


def parse_publication_date(date_str: str) -> Optional[datetime.date]:
    date_str = (date_str or "").strip()
    if not date_str:
        return None

    # Try ISO first
    try:
        return datetime.date.fromisoformat(date_str)
    except ValueError:
        pass

    # Try "Month DD, YYYY"
    try:
        return datetime.datetime.strptime(date_str, "%B %d, %Y").date()
    except ValueError:
        pass

    # Optional: dateutil fallback if installed
    try:
        from dateutil.parser import parse  # type: ignore
        return parse(date_str).date()
    except Exception:
        return None


def strip_acknowledgements(text: str) -> str:
    if not text:
        return text
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    cutoff = len(paras)
    for i, p in enumerate(reversed(paras)):
        low = p.lower()
        if low.startswith("thank") or "acknowledg" in low:
            cutoff = len(paras) - i - 1
            break
    return "\n".join(paras[:cutoff]).strip()


def extract_conclusion_or_tail(text: str) -> str:
    if not text:
        return ""
    lower = text.lower()
    markers = ["in conclusion", "to conclude", "to summarize", "in summary", "finally", "conclusion"]
    for m in markers:
        idx = lower.find(m)
        if idx != -1:
            return text[idx:].strip()
    paras = [p.strip() for p in text.split("\n") if len(p.strip()) > 20]
    if not paras:
        return ""
    tail = paras[-3:] if len(paras) >= 3 else paras
    words = " ".join(tail).split()
    if len(words) > 500:
        words = words[-500:]
    return " ".join(words)


def scrape_speech(url: str) -> Tuple[str, str, str]:
    r = session.get(url, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")

    title = ""
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        title = h1.get_text(strip=True)
    elif soup.title and soup.title.string:
        title = soup.title.string.strip()

    date_str = ""
    meta = soup.select_one('meta[name="publication_date"]')
    if meta and meta.get("content"):
        date_str = meta["content"].strip()
    else:
        t = soup.find("time")
        if t:
            date_str = t.get_text(strip=True)

    def collect_text(container) -> List[str]:
        if container is None:
            return []
        fragments: List[str] = []
        for tag in container.find_all(["h2", "h3", "h4", "p", "li"]):
            t = tag.get_text(" ", strip=True)
            if t:
                fragments.append(t)
        return fragments

    text_fragments: List[str] = []

    selectors = [
        ".entry-content",
        ".c-article-body__content",
        ".c-article-content",
        ".post-content",
    ]
    for sel in selectors:
        container = soup.select_one(sel)
        if container:
            text_fragments = collect_text(container)
            if text_fragments:
                break

    if not text_fragments:
        main = soup.select_one("div#content") or soup.select_one("main") or soup
        modules = main.select(".cfct-mod-content")
        collected: List[str] = []

        for mod in modules:
            ps = [p.get_text(" ", strip=True) for p in mod.find_all("p")]
            long_ps = [p for p in ps if len(p) > 60]
            if len(long_ps) >= 2:
                collected.extend(long_ps)

        if collected:
            text_fragments = collected

    if not text_fragments:
        article = soup.find("article")
        if article:
            text_fragments = collect_text(article)

    text = "\n".join(text_fragments)
    text = strip_acknowledgements(text)
    conclusion_text = extract_conclusion_or_tail(text)
    return title, date_str, conclusion_text


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-date", type=parse_date_input, help="Start date YYYY-MM-DD (optional)")
    ap.add_argument("--end-date", type=parse_date_input, help="End date YYYY-MM-DD (optional)")
    args = ap.parse_args()

    start_date: Optional[datetime.date] = args.start_date
    end_date: Optional[datetime.date] = args.end_date   

    print("Using date range:")
    print("  Start:", start_date if start_date else "(no lower bound)")
    print("  End:  ", end_date if end_date else "(no upper bound)")
    print()

    rows: List[Dict[str, str]] = []
    url_topics: Dict[str, Dict[str, str]] = {}
    all_urls: List[str] = []

    for topic_id, topic_name in TOPIC_IDS.items():
        print(f"=== Topic {topic_id} – {topic_name} ===")
        speech_urls = get_speech_list(topic_id, start_date, end_date)
        print(f"  Found {len(speech_urls)} URLs\n")

        for url in speech_urls:
            if url in url_topics:
                continue
            url_topics[url] = {"topic_id": topic_id, "topic_name": topic_name}
            all_urls.append(url)

        print(f"  Unique URLs accumulated: {len(all_urls)}\n")

    for idx, url in enumerate(all_urls, 1):
        time.sleep(DELAY)
        try:
            title, date_str, conclusion_text = scrape_speech(url)
        except Exception as e:
            print(f"    ERROR scraping {url}: {e}")
            continue

        pub_date = parse_publication_date(date_str)

        if (start_date or end_date) and pub_date is None:
            continue
        if start_date and pub_date and pub_date < start_date:
            continue
        if end_date and pub_date and pub_date > end_date:
            continue

        meta = url_topics.get(url, {})
        rows.append({
            "topic_id": meta.get("topic_id", ""),
            "topic_name": meta.get("topic_name", ""),
            "title": title,
            "date": date_str,
            "url": url,
            "conclusion_text": conclusion_text,
        })
        print(f"    [{idx}/{len(all_urls)}] OK: {title[:70]}")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"Done. Saved {len(df)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
