#!/usr/bin/env python3
"""
Publication Scraper: Collects Bank of England Publications using RSS and XML Sitemaps.
Outputs a CSV file containing structured data.
"""

from __future__ import annotations
import os, re, io, gzip, json, time, html, datetime as dt
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Iterable, Tuple
import requests, feedparser
from bs4 import BeautifulSoup
from xml.etree import ElementTree as ET
import argparse
from pathlib import Path
import logging
import pandas as pd # ADDED: For final CSV output

# ---------------- Logger Config ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("publication_scraper.log", mode="w"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# --- Configuration ---
# Set paths relative to the current execution directory (tools/)
TOOL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TOOL_DIR.parent
OUT_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "boe_publications.csv"
START_DEFAULT = (dt.date.today() - dt.timedelta(days=365*5)).isoformat() # 5 years back by default
END_DEFAULT = dt.date.today().isoformat()

BASE = "https://www.bankofengland.co.uk"
HUB  = f"{BASE}/sitemap"
RSS  = f"{BASE}/rss/publications"

UA = "MacroX-BankWatch/1.3 (+https://example.org/contact)"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": UA, "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"})
REQUEST_TIMEOUT = 40
RETRIES = 2
BACKOFF = 1.5
SLEEP   = 0.25

# Regexes needed from the notebook context
TZ_TRAIL_RE = re.compile(r'(Z|[+-]\d{2}:\d{2})$')   
DATE_ONLY_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')

# ---------------- Model ----------------
@dataclass
class FetchedItem:
    source: str
    category: str
    title: str
    url: str
    published: str # ISO 8601 string
    summary: str
    content_text: str
    content_html: str
    topics: List[str]
    meta: Dict[str, str]

# ---------------- Utils (Adapted from Notebook) ----------------
# NOTE: All supporting functions (clean_space, normalize_date, within_range, fetch_text, etc.) 
# from the Jupyter notebook must be defined in this file. (Omitting them here for brevity).

def clean_space(s: str) -> str:
    s = html.unescape(s or "")
    s = re.sub(r"\r+|\t+", " ", s)
    s = re.sub(r"[ \u00A0]+\s*", " ", s)
    s = re.sub(r"\s+\n", "\n", s)
    s = re.sub(r"\n\s+", "\n", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def normalize_date(s: str) -> str:
    if not s: return ""
    s = s.strip()
    if s.endswith('Z'): s = s[:-1] + '+00:00'
    if TZ_TRAIL_RE.search(s): return s
    if DATE_ONLY_RE.match(s): return s + 'T00:00:00+00:00'
    if 'T' in s: return s + '+00:00'
    try:
        _ = dt.datetime.fromisoformat(s)
        return s
    except Exception:
        return s + 'T00:00:00+00:00'

def parse_iso_dt(s: str) -> dt.datetime:
    s = (s or "").strip()
    if not s: raise ValueError("empty ISO string")
    s = normalize_date(s)
    return dt.datetime.fromisoformat(s)

def within_range(iso: str, start: str | None, end: str | None) -> bool:
    try:
        x = parse_iso_dt(iso)
    except Exception:
        return True

    if start:
        try:
            sdt = parse_iso_dt(start)
        except Exception:
            sdt = parse_iso_dt(start + 'T00:00:00+00:00')
        if x < sdt:
            return False

    if end:
        try:
            edt = parse_iso_dt(end)
        except Exception:
            edt = parse_iso_dt(end + 'T23:59:59+00:00')
        if x > edt:
            return False

    return True

def fetch_text(url: str) -> str:
    last = None
    for i in range(RETRIES+1):
        try:
            r = SESSION.get(url, timeout=REQUEST_TIMEOUT)
            last = r.status_code
            if r.status_code == 200 and r.text: return r.text
        except requests.RequestException:
            pass
        time.sleep(BACKOFF*(i+1))
    raise RuntimeError(f"Fetch failed: {url} (status={last})")

def fetch_bytes(url: str) -> bytes:
    last = None
    for i in range(RETRIES+1):
        try:
            r = SESSION.get(url, timeout=REQUEST_TIMEOUT)
            last = r.status_code
            if r.status_code == 200 and r.content: return r.content
        except requests.RequestException:
            pass
        time.sleep(BACKOFF*(i+1))
    raise RuntimeError(f"Fetch bytes failed: {url} (status={last})")

MAIN_SELECTORS = ["main","[role='main']","article",".o-article__content",".article-content",".content"]

def extract_main_html(html_text: str) -> str:
    soup = BeautifulSoup(html_text, "lxml")
    for sel in ["script","style","noscript","header","nav","footer","aside",".o-cookie-banner",".cookie-banner"]:
        for el in soup.select(sel): el.decompose()
    for sel in MAIN_SELECTORS:
        el = soup.select_one(sel)
        if el and len(el.get_text(strip=True)) > 200: return str(el)
    cands = soup.select("main, article, .content, .page-content, .rich-text, .govuk-width-container")
    if cands:
        longest = max(cands, key=lambda e: len(e.get_text()))
        return str(longest)
    return str(soup.body or soup)

def html_to_text(html_fragment: str) -> str:
    soup = BeautifulSoup(html_fragment or "", "lxml")
    for bad in soup(["script","style","noscript"]): bad.decompose()
    out = []
    for el in soup.find_all(["h1","h2","h3","p","li"]):
        t = el.get_text(" ", strip=True)
        if not t: continue
        if el.name == "li": t = f"- {t}"
        out.append(t)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()

def parse_published_from_html(html_text: str) -> str:
    soup = BeautifulSoup(html_text or "", "lxml")
    m = soup.select_one('meta[property="article:published_time"]')
    if m and m.get("content"): return normalize_date(m.get("content"))
    for name in ["pubdate","date","publishdate","dc.date","dc.date.issued"]:
        tag = soup.select_one(f'meta[name="{name}"]')
        if tag and tag.get("content"): return normalize_date(tag.get("content"))
    t = soup.select_one("time[datetime]")
    if t and t.get("datetime"): return normalize_date(t.get("datetime"))
    for tag in soup.find_all("script", type="application/ld+json"):
        try: data = json.loads(tag.get_text(strip=True))
        except Exception: continue
        def find_date(obj):
            if isinstance(obj, dict):
                for k,v in obj.items():
                    if k in ("datePublished","dateCreated","uploadDate"):
                        if isinstance(v,str): return v
                    else:
                        got = find_date(v)
                        if got: return got
            elif isinstance(obj, list):
                for x in obj:
                    got = find_date(x)
                    if got: return got
            return None
        v = find_date(data)
        if v: return normalize_date(v)
    return ""

def guess_title_from_url(url: str) -> str:
    slug = url.rstrip("/").rsplit("/",1)[-1]
    slug = re.sub(r"[-_]+"," ", slug)
    return slug.strip().title()

def guess_title_from_html(html_text: str) -> str:
    soup = BeautifulSoup(html_text, "lxml")
    if soup.title and soup.title.string:
        return clean_space(soup.title.string)
    h1 = soup.find("h1")
    if h1:
        return clean_space(h1.get_text(" ", strip=True))
    return ""

def fetch_article(url: str) -> Dict[str,str]:
    html_text = fetch_text(url)
    main_html  = extract_main_html(html_text)
    title      = guess_title_from_html(html_text) or guess_title_from_url(url)
    content    = html_to_text(main_html)
    page_pub   = parse_published_from_html(html_text)
    time.sleep(SLEEP)
    return {"title": title, "content_html": main_html, "content_text": content, "page_published": page_pub}

def fetch_boe_rss(start: Optional[str], end: Optional[str], fetch_full=True) -> List[FetchedItem]:
    out, seen = [], set()
    fp = feedparser.parse(RSS)
    for e in getattr(fp, "entries", []):
        url = getattr(e, "link", "") or ""
        if not url or url in seen: continue
        seen.add(url)
        title = clean_space(getattr(e, "title", "")) or ""
        published = getattr(e, "published", "") or ""
        if hasattr(e, "published_parsed"):
            published = dt.datetime(*e.published_parsed[:6], tzinfo=dt.timezone.utc).isoformat()
        summary = clean_space(getattr(e, "summary", ""))
        topics = [t.get("term") for t in getattr(e, "tags", []) if t.get("term")]
        content_html, content_text, page_pub = "", summary, ""
        if fetch_full:
            try:
                art = fetch_article(url)
                content_html = art["content_html"]
                content_text = art["content_text"] or summary
                page_pub = art["page_published"]
            except Exception:
                pass
        published_final = page_pub or published
        if not within_range(normalize_date(published_final or ""), start, end): continue
        out.append(FetchedItem("boe","publications", title or guess_title_from_url(url), url,
                               published_final, summary, content_text, content_html, topics,
                               {"via":"rss"}))
    return out

def absolute(href: str) -> str:
    if href.startswith("http"): return href
    if href.startswith("//"):  return "https:" + href
    if href.startswith("/"):   return BASE + href
    return BASE + "/" + href.lstrip("./")

def discover_category_pages_from_hub(hub_url: str) -> List[str]:
    html_text = fetch_text(hub_url)
    soup = BeautifulSoup(html_text, "lxml")
    cats = []
    for a in soup.select("a[href]"):
        href = a["href"]
        if href.startswith("/sitemap/") or (href.startswith(BASE+"/sitemap/")):
            cats.append(absolute(href))
    seen, out = set(), []
    for u in cats:
        if u not in seen:
            seen.add(u); out.append(u)
    return out

def discover_xml_links_from_category_page(cat_url: str) -> List[str]:
    html_text = fetch_text(cat_url)
    soup = BeautifulSoup(html_text, "lxml")
    xmls = []
    for a in soup.select("a[href]"):
        text = (a.get_text() or "").strip().lower()
        href = a["href"]
        if "xml sitemap" in text:
            xmls.append(absolute(href))
    for a in soup.select("a[href$='.xml'], a[href$='.xml.gz']"):
        xmls.append(absolute(a["href"]))

    seen, out = set(), []
    for x in xmls:
        if x.startswith(BASE) and x not in seen:
            seen.add(x); out.append(x)
    return out

NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

def parse_xml_root(url: str) -> ET.Element:
    if url.endswith(".gz"):
        data = fetch_bytes(url)
        with gzip.GzipFile(fileobj=io.BytesIO(data)) as f:
            raw = f.read()
        return ET.fromstring(raw)
    else:
        txt = fetch_text(url)
        return ET.fromstring(txt.encode("utf-8"))

def extract_url_entries(xml_root: ET.Element) -> List[Tuple[str, str]]:
    tag = xml_root.tag.lower()
    out = []
    if tag.endswith("urlset"):
        for uel in xml_root.findall(".//sm:url", NS):
            loc = uel.findtext("sm:loc", default="", namespaces=NS).strip()
            if not loc: continue
            lm  = uel.findtext("sm:lastmod", default="", namespaces=NS).strip()
            out.append((loc, lm))
    return out

KEEP_SEGMENTS = [
    "/publications/", "/publication/", "/news/", "/report", "/reports/",
    "/article", "/articles/", "/speeches", "/minutes", "/monetary-policy",
    "/quarterly-bulletin", "/agents-summary", "/market-notices", "/statistics",
]
def looks_like_publication(url: str) -> bool:
    return any(seg in url for seg in KEEP_SEGMENTS)

def build_items_from_urls(urls: List[str], start: Optional[str], end: Optional[str], fetch_full=True) -> List[FetchedItem]:
    items: List[FetchedItem] = []
    for i, url in enumerate(urls, 1):
        try:
            art = fetch_article(url) if fetch_full else {"title":"", "content_html":"", "content_text":"", "page_published":""}
        except Exception:
            continue
        published = art.get("page_published","")
        if start or end:
            if published and not within_range(normalize_date(published), start, end):
                continue
        title = art.get("title") or guess_title_from_url(url)
        summary = clean_space((art.get("content_text") or "")[:300])
        items.append(FetchedItem("boe","publications", title, url, published, summary,
                                 art.get("content_text",""), art.get("content_html",""), [],
                                 {"via":"sitemap-category-xml"}))
        if i % 50 == 0:
            logger.info(f"  scraped {i}/{len(urls)}…")
    return items

# ---------------- Orchestrator ----------------
def run_boe_publications(start: str, end: str, out_path: Path,
                         use_rss=True, use_sitemap=True,
                         fetch_full_text=True, cap_per_category: Optional[int]=None):
    all_items: List[FetchedItem] = []

    if use_rss:
        logger.info("Fetching via RSS…")
        rss_items = fetch_boe_rss(start, end, fetch_full=fetch_full_text)
        logger.info(f"  RSS items: {len(rss_items)}")
        all_items.extend(rss_items)

    if use_sitemap:
        logger.info("Discovering category pages from hub…")
        cats = discover_category_pages_from_hub(HUB)
        logger.info(f"  Found {len(cats)} category sitemap pages")
        if not cats:
            cats = [
                f"{BASE}/sitemap/reports",
                f"{BASE}/sitemap/statements",
                f"{BASE}/sitemap/statistics",
                f"{BASE}/sitemap/markets",
                f"{BASE}/sitemap/staff-working-paper",
                f"{BASE}/sitemap/annual-report",
            ]

        candidate_urls = []
        seen_url = set()
        for cat in cats:
            try:
                xmls = discover_xml_links_from_category_page(cat)
            except Exception as e:
                logger.warning(f"Failed to process category {cat}: {e}")
                continue
            if not xmls:
                continue
            for sx in xmls:
                try:
                    root = parse_xml_root(sx)
                except Exception as e:
                    logger.warning(f"Failed to parse XML {sx}: {e}")
                    continue
                for loc, lastmod in extract_url_entries(root):
                    if not loc.startswith(BASE): continue
                    if not looks_like_publication(loc): continue
                    if lastmod and not within_range(normalize_date(lastmod), start, end):
                        continue
                    if loc in seen_url: continue
                    seen_url.add(loc)
                    candidate_urls.append(loc)
            if cap_per_category:
                candidate_urls = candidate_urls[:cap_per_category]

        logger.info(f"  Candidate URLs from category XMLs: {len(candidate_urls)}")

        rss_urls = {it.url for it in all_items}
        candidate_urls = [u for u in candidate_urls if u not in rss_urls]
        logger.info(f"  After dedupe vs RSS: {len(candidate_urls)}")

        logger.info("Scraping candidate pages…")
        site_items = build_items_from_urls(candidate_urls, start, end, fetch_full=fetch_full_text)
        logger.info(f"  Sitemap-scraped items: {len(site_items)}")
        all_items.extend(site_items)

    # Final dedupe + date screen
    seen, deduped = set(), []
    for it in all_items:
        if it.url in seen: continue
        seen.add(it.url); deduped.append(it)
    final = [it for it in deduped if within_range(normalize_date(it.published or ""), start, end)]
    logger.info(f"Final items after dedupe/date filter: {len(final)}")

    # Convert to list of dictionaries for Pandas CSV output
    final_data_list = [asdict(it) for it in final]
    df = pd.DataFrame(final_data_list)
    
    # Write to CSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding='utf-8')
    
    logger.info(f"✔ Wrote {len(final)} items to CSV → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Scrape BoE Publications data.")
    parser.add_argument("--start-date", type=str, default=START_DEFAULT, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", type=str, default=END_DEFAULT, help="End date (YYYY-MM-DD).")
    args = parser.parse_args()
    
    # Construct the final output path dynamically
    OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "boe_publications.csv"
    
    run_boe_publications(args.start_date, args.end_date, OUT_PATH,
                         use_rss=True,
                         use_sitemap=True,
                         fetch_full_text=True,
                         cap_per_category=None)


if __name__ == "__main__":
    main()