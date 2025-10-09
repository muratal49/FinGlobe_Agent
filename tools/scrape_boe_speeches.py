#!/usr/bin/env python3
"""
Scrape Bank of England speech pages (HTML only) from the public "Speeches sitemap",
filter by date window and topic keywords, clean acknowledgements/references/footnotes,
and save to a CSV (comma-safe quoting).

- Source: https://www.bankofengland.co.uk/sitemap/speeches   (static HTML)
- We skip PDFs entirely.
- We only visit URLs like /speech/YYYY/<month>/<slug> and ignore everything else.
- We filter by:
    * start_date (default: first day of month N months back)
    * keywords in title or body (default: MPC / Monetary Policy Committee / inflation)
- Output CSV columns: date,title,speaker,text,url
"""

import re
import csv
import time
import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

import requests
from bs4 import BeautifulSoup

# ---------- Config defaults ----------
SITEMAP_URL = "https://www.bankofengland.co.uk/sitemap/speeches"
DEFAULT_MONTHS_BACK = 6
DEFAULT_KEYWORDS = ["Monetary Policy Committee", "MPC", "inflation"]
OUTPUT_CSV = "data/boe_filtered_speeches.csv"

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


def first_day_of_month_n_months_ago(n: int) -> datetime:
    """Return UTC datetime at 00:00 of the first day of the month N months ago."""
    now = datetime.now(timezone.utc)
    y, m = now.year, now.month
    total_months = (y * 12 + (m - 1)) - n
    y2, m2 = divmod(total_months, 12)
    m2 += 1
    return datetime(y2, m2, 1, tzinfo=timezone.utc)


def is_html_speech_url(url: str) -> bool:
    """True if link looks like an HTML speech page (not a PDF/media)."""
    if "/-/media/" in url:
        return False
    if not url.startswith("https://www.bankofengland.co.uk/speech/"):
        return False
    # Expect /speech/YYYY/month/slug
    parts = url.rstrip("/").split("/")
    if len(parts) < 6:
        return False
    year = parts[4]
    month = parts[5].lower()
    if not year.isdigit() or month not in MONTHS:
        return False
    return True


def date_from_url(url: str) -> datetime:
    """Parse YYYY, month name from the speech URL and return naive date (YYYY-MM-01)."""
    parts = url.rstrip("/").split("/")
    year = int(parts[4])
    month = MONTHS.get(parts[5].lower())
    return datetime(year, month, 1)


def parse_date_from_page(soup: BeautifulSoup, fallback_url_date: datetime) -> datetime:
    """Try to get exact published date; else fall back to YYYY-MM-01 from URL."""
    # 1) JSON-LD datePublished (if present)
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            import json
            data = json.loads(tag.get_text(strip=True))
            items = data if isinstance(data, list) else [data]
            for it in items:
                dp = it.get("datePublished") or it.get("dateModified")
                if dp:
                    return datetime.fromisoformat(dp.replace("Z", "+00:00"))
        except Exception:
            pass

    # 2) Open Graph / meta
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
            except Exception:
                pass

    # 3) Text pattern "Published on 09 May 2025"
    body_text = soup.get_text(" ", strip=True)
    m = re.search(r"Published on\s+(\d{1,2}\s+[A-Za-z]+\s+\d{4})", body_text, flags=re.I)
    if m:
        try:
            return datetime.strptime(m.group(1), "%d %B %Y").replace(tzinfo=timezone.utc)
        except Exception:
            pass

    # 4) Fallback to URL month 1st day
    return fallback_url_date.replace(tzinfo=timezone.utc)


def extract_title(soup: BeautifulSoup) -> str:
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(" ", strip=True)
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    return ""


def extract_speaker(title: str, soup: BeautifulSoup) -> str:
    """Prefer extracting from title patterns like '... - speech by Andrew Bailey'."""
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

    # Possible 'Given by …' line
    m2 = re.search(r"Given by\s+(.+)", soup.get_text("\n", strip=True), flags=re.I)
    if m2:
        return m2.group(1).strip(" .–-")

    return ""


def extract_body_text(soup: BeautifulSoup) -> str:
    """
    Take the main speech text (preferred: <main> or <article>), prune boilerplate,
    skip "Other speeches" sections, and remove known junk.
    """
    container = soup.find("main") or soup.find("article") or soup.body or soup

    # Stop collecting once we hit "Other speeches" section
    other_speeches_tag = None
    for tag in container.find_all(["h2", "h3", "p", "li"]):
        if tag.get_text(strip=True).lower().startswith("other speeches"):
            other_speeches_tag = tag
            break

    texts = []
    for tag in container.find_all(["h2", "h3", "p", "li"]):
        if other_speeches_tag and tag == other_speeches_tag:
            break  # stop at "Other speeches"
        txt = tag.get_text(" ", strip=True)
        if not txt:
            continue
        if txt.lower() in ("back to top", "skip to main content"):
            continue
        texts.append(txt)

    out = "\n".join(texts)

    # Remove announcement-only pages
    if re.search(r"Text to be published|to be published", out, flags=re.I):
        return ""

    # Light cleanup
    out = re.sub(r"\s+\n", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()



def strip_ack_refs_footnotes(text: str) -> str:
    """Remove 'Acknowledgements', 'References', 'Footnotes' sections + inline footnotes like [1]."""
    if not text:
        return text
    section_re = re.compile(
        r"(?mis)(^|\n)\s*(Acknowledgements?|Acknowledgments?|References?|Footnotes?)\s*(:)?\s*(\n|$).*?\Z"
    )
    text = section_re.sub("", text)
    text = re.sub(r"\[\d+\]", "", text)  # remove [1], [12], etc.
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

    # Remove sections starting from a heading line to end of text
    # Headings usually appear as their own lines because we injected newlines between tags.
    section_re = re.compile(
        r"(?mis)(^|\n)\s*(Acknowledgements?|Acknowledgments?|References?|Footnotes?)\s*(:)?\s*(\n|$).*?\Z"
    )
    text = section_re.sub("", text)

    # Remove inline bracketed numeric footnotes [1], [23]
    text = re.sub(r"\[\d+\]", "", text)

    # Light cleanup around extra spaces/newlines left by removals
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fetch_speech(url: str, keywords_lower):
    """Return dict with date,title,speaker,text,url or None if filtered out/invalid."""
    try:
        r = session.get(url, timeout=30)
        r.raise_for_status()
    except Exception as e:
        print(f"⚠️  Fetch failed {url}: {e}")
        return None

    soup = BeautifulSoup(r.text, "lxml")
    base_date = date_from_url(url)
    pub_dt = parse_date_from_page(soup, base_date)

    title = extract_title(soup).strip()
    body = extract_body_text(soup)
    body = strip_ack_refs_footnotes(body)

    # Require non-empty body
    if not body:
        return None

    # Topic keyword filter in title or body
    text_lower = (title + " " + body).lower()
    if not any(k in text_lower for k in keywords_lower):
        return None

    speaker = extract_speaker(title, soup)

    return {
        "date": pub_dt.date().isoformat(),
        "title": title,
        "speaker": speaker,
        "text": body,
        "url": url,
    }


def scrape(start_date: datetime, keywords, limit_per_year=None, sleep=0.4):
    """Yield filtered speech records from the sitemap."""
    print(f"🗺  Fetching sitemap… {SITEMAP_URL}")
    r = session.get(SITEMAP_URL, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")

    # Collect candidate links
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/"):
            href = "https://www.bankofengland.co.uk" + href
        if is_html_speech_url(href):
            # pre-filter by URL date
            url_dt = date_from_url(href)
            if url_dt >= start_date.replace(tzinfo=None):
                links.append(href)

    # Deduplicate & sort newest first (by url date)
    links = sorted(set(links), key=lambda u: date_from_url(u), reverse=True)
    print(f"🔗 Candidates in window: {len(links)}")

    keywords_lower = [k.lower() for k in keywords]

    for i, url in enumerate(links, 1):
        rec = fetch_speech(url, keywords_lower)
        if rec:
            yield rec
        if sleep:
            time.sleep(sleep)


def main():
    parser = argparse.ArgumentParser(description="Scrape BoE speeches (HTML only).")
    parser.add_argument("--months-back", type=int, default=DEFAULT_MONTHS_BACK,
                        help="How many months back (from now) to include, starting at the 1st of that month.")
    parser.add_argument("--start-date", type=str, default=None,
                        help="Optional explicit start date (YYYY-MM-DD). Overrides --months-back.")
    parser.add_argument("--keywords", type=str, default=",".join(DEFAULT_KEYWORDS),
                        help="Comma-separated keywords to keep (title or body).")
    parser.add_argument("--out", type=str, default=OUTPUT_CSV,
                        help="Output CSV path.")
    args = parser.parse_args()

    if args.start_date:
        try:
            start_date = datetime.strptime(args.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            print("Invalid --start-date (use YYYY-MM-DD)", file=sys.stderr)
            sys.exit(2)
    else:
        start_date = first_day_of_month_n_months_ago(args.months_back)

    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]
    print(f"📅 Start date: {start_date.date().isoformat()} | 🔎 Keywords: {keywords}")

    rows = list(scrape(start_date, keywords))
    print(f"✅ Matched speeches: {len(rows)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["date", "title", "speaker", "text", "url"],
            quoting=csv.QUOTE_ALL,           # <-- comma-safe quoting
            escapechar="\\",                 # escape embedded quotes if any
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"💾 Saved CSV -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
