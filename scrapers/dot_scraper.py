#!/usr/bin/env python3
"""Polite document scraper for state DOT websites (Caltrans, CDOT).

Crawls HTML pages within a DOT's domain, harvests linked PDF documents,
categorises each into the WYDOT 10-bucket ``document_series`` taxonomy,
downloads them to ``data/<dot>/raw/``, and writes a provenance manifest at
``data/<dot>/manifest.jsonl`` (one JSON object per document).

Must run on a LOGIN node -- ARCC compute nodes have no outbound network.
Idempotent: re-running skips URLs / sha256s already in the manifest.

Usage:
    python scrapers/dot_scraper.py --dot cdot     --max-pages 800 --max-pdfs 450
    python scrapers/dot_scraper.py --dot caltrans --max-pages 800 --max-pdfs 450
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

REPO = Path(__file__).resolve().parents[1]
CONTACT = "anonymous@example.com"
USER_AGENT = (
    f"WYDOT-Research-Crawler/1.0 "
    f"(Anonymous Institution academic RAG study; contact {CONTACT})"
)
HTML_DELAY = 1.0          # seconds between HTML page fetches
PDF_DELAY = 1.5           # seconds between PDF downloads
MAX_DEPTH = 4
MAX_PDF_BYTES = 220 * 1024 * 1024
TIMEOUT = 45

# --- per-DOT crawl configuration -------------------------------------------
CONFIG = {
    "cdot": {
        "allowed_domains": {"www.codot.gov", "codot.gov"},
        "seeds": [
            "https://www.codot.gov/",
            "https://www.codot.gov/business",
            "https://www.codot.gov/business/designsupport",
            "https://www.codot.gov/business/designsupport/construction-specifications",
            "https://www.codot.gov/business/designsupport/standard-plans",
            "https://www.codot.gov/business/designsupport/materials",
            "https://www.codot.gov/business/designsupport/bridge",
            "https://www.codot.gov/business/designsupport/cdot-manuals",
            "https://www.codot.gov/library",
            "https://www.codot.gov/programs/planning",
            "https://www.codot.gov/safety",
        ],
    },
    "caltrans": {
        "allowed_domains": {"dot.ca.gov", "www.dot.ca.gov"},
        "seeds": [
            "https://dot.ca.gov/",
            "https://dot.ca.gov/programs/design",
            "https://dot.ca.gov/programs/design/ccs-standard-plans-and-standard-specifications",
            "https://dot.ca.gov/programs/design/manual-highway-design-manual-hdm",
            "https://dot.ca.gov/programs/construction",
            "https://dot.ca.gov/programs/construction/construction-manual",
            "https://dot.ca.gov/programs/engineering-services",
            "https://dot.ca.gov/programs/engineering-services/manuals",
            "https://dot.ca.gov/programs/traffic-operations",
            "https://dot.ca.gov/programs/transportation-planning",
            "https://dot.ca.gov/programs/maintenance",
        ],
    },
}

# --- document taxonomy (same buckets as the WYDOT corpus) ------------------
# Order matters: most specific rules first. Matched against URL path + filename.
CATEGORY_RULES = [
    ("Standard Plans", ["standard plan", "standard-plan", "std plan", "std-plan",
                        "m-and-s", "m&s standard", "m & s standard", "mands",
                        "m-standard", "s-standard", "m standard", "s standard"]),
    ("Standard Specs", ["standard spec", "standard-spec", "std spec", "std-spec",
                        "specification", "special provision"]),
    ("Construction Manual", ["construction manual", "construction-manual",
                             "construction bulletin", "construction handbook"]),
    ("Materials Testing", ["material", "test method", "test-method", "field testing",
                           "mets", "ctm ", "california test", "lab manual"]),
    ("Bridge Program", ["bridge", "structure design", "memo to designer"]),
    ("Design Manual", ["design manual", "design-manual", "design guide", "design-guide",
                       "highway design", "roadway design", "hdm", "pavement design",
                       "drainage design", "geometric design"]),
    ("Traffic & Safety", ["traffic", "mutcd", "safety", "crash", "signal", "signing",
                          "striping", "work zone", "speed"]),
    ("STIP", ["stip", "tip", "improvement program", "programming", "transportation plan",
              "long range", "long-range"]),
    ("Annual Reports", ["annual report", "annual-report", "performance report",
                        "performance-report", "fact book", "factbook", "yearbook"]),
]


def _norm(s: str) -> str:
    return s.lower().replace("_", " ").replace("%20", " ").replace("-", " ")


def categorize(url: str, filename: str, source_page: str = "") -> str:
    """Provisional category from filename + PDF URL, falling back to the source
    page URL. The Qwen content-classifier in Phase 2 refines the 'General' tail."""
    primary = _norm(f"{urlparse(url).path}  {filename}")
    for series, keys in CATEGORY_RULES:
        if any(_norm(k) in primary for k in keys):
            return series
    secondary = _norm(urlparse(source_page).path)
    for series, keys in CATEGORY_RULES:
        if any(_norm(k) in secondary for k in keys):
            return series
    return "General"


def is_pdf_link(href: str) -> bool:
    h = href.lower().split("#")[0]
    return h.endswith(".pdf") or ".pdf?" in h or "/pdf/" in h and h.endswith(".pdf")


# URL-path tokens that make a page worth crawling sooner.
PRIORITY_TOKENS = ("design", "spec", "manual", "construct", "material", "bridge",
                   "traffic", "plan", "publication", "document", "library", "report",
                   "engineering", "standard", "safety", "program")
SKIP_TOKENS = ("news", "career", "espanol", "/es/", "contact", "calendar", "event",
               "press-release", "newsroom", "login", "search", "subscribe", "twitter",
               "facebook", "youtube", "instagram", "linkedin")


def page_priority(url: str) -> int:
    p = urlparse(url).path.lower()
    if any(t in p for t in SKIP_TOKENS):
        return 2
    if any(t in p for t in PRIORITY_TOKENS):
        return 0
    return 1


def safe_filename(url: str) -> str:
    name = urlparse(url).path.split("/")[-1] or "document.pdf"
    name = re.sub(r"%20", " ", name)
    name = re.sub(r"[^A-Za-z0-9 ._-]", "_", name).strip()
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    return name[:180]


def load_manifest(path: Path):
    seen_urls, seen_sha = set(), set()
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rows.append(rec)
            seen_urls.add(rec["url"])
            if rec.get("sha256"):
                seen_sha.add(rec["sha256"])
    return rows, seen_urls, seen_sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dot", required=True, choices=sorted(CONFIG))
    ap.add_argument("--max-pages", type=int, default=800, help="HTML pages to crawl")
    ap.add_argument("--max-pdfs", type=int, default=450, help="PDFs to download")
    args = ap.parse_args()

    cfg = CONFIG[args.dot]
    raw_dir = REPO / "data" / args.dot / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = REPO / "data" / args.dot / "manifest.jsonl"

    rows, seen_urls, seen_sha = load_manifest(manifest_path)
    downloaded = len([r for r in rows if r.get("status") == "downloaded"])
    print(f"[{args.dot}] resume: {len(rows)} manifest rows, {downloaded} downloaded")

    sess = requests.Session()
    sess.headers.update({"User-Agent": USER_AGENT, "Accept": "*/*"})

    # frontier: 3 priority buckets
    frontier = [deque(), deque(), deque()]
    queued = set()
    for s in cfg["seeds"]:
        u = urldefrag(s)[0]
        frontier[page_priority(u)].append((u, 0))
        queued.add(u)

    visited_pages = 0
    pdf_candidates: dict[str, str] = {}   # pdf_url -> source page
    manifest_f = manifest_path.open("a")

    def next_url():
        for b in frontier:
            if b:
                return b.popleft()
        return None

    # --- crawl phase --------------------------------------------------------
    while visited_pages < args.max_pages:
        item = next_url()
        if item is None:
            break
        url, depth = item
        try:
            resp = sess.get(url, timeout=TIMEOUT)
        except Exception as e:
            print(f"  [skip] {url}  ({e})")
            time.sleep(HTML_DELAY)
            continue
        ctype = resp.headers.get("Content-Type", "").lower()
        if resp.status_code != 200 or "html" not in ctype:
            time.sleep(HTML_DELAY)
            continue
        visited_pages += 1
        soup = BeautifulSoup(resp.text, "html.parser")
        new_links = 0
        for a in soup.find_all("a", href=True):
            href = urldefrag(urljoin(url, a["href"].strip()))[0]
            if not href.lower().startswith("http"):
                continue
            host = urlparse(href).netloc.lower()
            if is_pdf_link(href):
                if href not in pdf_candidates:
                    pdf_candidates[href] = url
                continue
            # only crawl HTML inside the DOT's own domain
            if host not in cfg["allowed_domains"]:
                continue
            if depth + 1 > MAX_DEPTH or href in queued:
                continue
            queued.add(href)
            frontier[page_priority(href)].append((href, depth + 1))
            new_links += 1
        if visited_pages % 25 == 0:
            print(f"  [crawl] {visited_pages} pages | "
                  f"{len(pdf_candidates)} pdf links | frontier "
                  f"{sum(len(b) for b in frontier)}")
        time.sleep(HTML_DELAY)

    print(f"[{args.dot}] crawl done: {visited_pages} pages, "
          f"{len(pdf_candidates)} unique PDF links found")

    # --- download phase -----------------------------------------------------
    n_new = 0
    for pdf_url, src_page in pdf_candidates.items():
        if n_new >= args.max_pdfs:
            break
        if pdf_url in seen_urls:
            continue
        try:
            r = sess.get(pdf_url, timeout=TIMEOUT, stream=True)
        except Exception as e:
            print(f"  [pdf skip] {pdf_url}  ({e})")
            time.sleep(PDF_DELAY)
            continue
        if r.status_code != 200:
            time.sleep(PDF_DELAY)
            continue
        ctype = r.headers.get("Content-Type", "").lower()
        body = b""
        too_big = False
        for chunk in r.iter_content(chunk_size=65536):
            body += chunk
            if len(body) > MAX_PDF_BYTES:
                too_big = True
                break
        if too_big or len(body) < 1024:
            time.sleep(PDF_DELAY)
            continue
        if not (body[:5] == b"%PDF-" or "pdf" in ctype):
            time.sleep(PDF_DELAY)
            continue
        sha = hashlib.sha256(body).hexdigest()
        if sha in seen_sha:
            time.sleep(PDF_DELAY)
            continue

        fname = safe_filename(pdf_url)
        dest = raw_dir / fname
        if dest.exists():
            dest = raw_dir / f"{dest.stem}__{sha[:8]}.pdf"
        dest.write_bytes(body)
        series = categorize(pdf_url, fname, src_page)
        rec = {
            "dot": args.dot,
            "url": pdf_url,
            "source_page": src_page,
            "filename": dest.name,
            "local_path": str(dest.relative_to(REPO)),
            "document_series": series,
            "sha256": sha,
            "bytes": len(body),
            "status": "downloaded",
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
        }
        manifest_f.write(json.dumps(rec) + "\n")
        manifest_f.flush()
        seen_urls.add(pdf_url)
        seen_sha.add(sha)
        n_new += 1
        if n_new % 10 == 0:
            print(f"  [pdf] {n_new} downloaded ({series})")
        time.sleep(PDF_DELAY)

    manifest_f.close()
    print(f"[{args.dot}] DONE: {n_new} new PDFs -> {raw_dir}")
    print(f"[{args.dot}] manifest: {manifest_path}")


if __name__ == "__main__":
    sys.exit(main())
