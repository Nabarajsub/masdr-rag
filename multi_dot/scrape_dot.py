#!/usr/bin/env python3
"""Phase 1: scrape a state DOT's public PDF library into data/<dot>/raw/.

Reproduces the manifest format of the existing cdot/caltrans scrapes
(A22 methodology): one JSONL row per PDF with url, source_page, sha256,
provisional document_series (from the seed page), status, retrieved_at.

Seeds live in data/dot_seeds.json:
    {"txdot": [{"url": "https://...", "series": "Standard Specs", "depth": 1}, ...]}

Politeness: 1s delay between requests, 60s timeout, 80MB size cap,
same-host only, honors robots.txt Disallow for our user-agent ('*').

Usage:
    python -m graph_processing.multi_dot.scrape_dot --dot txdot --dry-run
    python -m graph_processing.multi_dot.scrape_dot --dot txdot
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.parse
import urllib.robotparser
from datetime import datetime, timezone
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[2]
SEEDS = REPO / "data" / "dot_seeds.json"
UA = ("Mozilla/5.0 (research crawler; Anonymous Institution; "
      "anonymous@example.com; respectful-rate)")
DELAY_S = 1.0
MAX_BYTES = 80 * 1024 * 1024
HREF_RE = re.compile(r"""href\s*=\s*["']([^"'#]+)""", re.IGNORECASE)

_robots: dict = {}


def allowed(url: str) -> bool:
    host = urllib.parse.urlsplit(url).netloc
    if host not in _robots:
        # Fetch robots.txt with our own UA: urllib.robotparser's read() uses the
        # default Python-urllib agent, which some sites 403 — and a 403 makes it
        # silently disallow everything (bit us on fdot.gov).
        rp = urllib.robotparser.RobotFileParser()
        try:
            r = requests.get(f"https://{host}/robots.txt",
                             headers={"User-Agent": UA}, timeout=30)
            if r.status_code == 200:
                rp.parse(r.text.splitlines())
            else:
                rp = None          # no readable robots -> default allow
        except Exception:
            rp = None
        _robots[host] = rp
    rp = _robots[host]
    return True if rp is None else rp.can_fetch(UA, url)


def fetch(url: str, *, stream=False):
    time.sleep(DELAY_S)
    return requests.get(url, headers={"User-Agent": UA}, timeout=60,
                        stream=stream, allow_redirects=True)


def harvest(seed_url: str, depth: int, seen_pages: set,
            allow_hosts: list[str] | None = None) -> list[tuple[str, str]]:
    """Return [(pdf_url, source_page)] reachable from seed within depth.

    Pages are crawled same-host only; PDF links are accepted from the seed
    host plus any `allow_hosts` (e.g. a DOT's CDN / blob-storage domain)."""
    out, queue = [], [(seed_url, 0)]
    host = urllib.parse.urlsplit(seed_url).netloc
    pdf_hosts = {host, *(allow_hosts or [])}
    while queue:
        page, d = queue.pop(0)
        if page in seen_pages or not allowed(page):
            continue
        seen_pages.add(page)
        try:
            r = fetch(page)
            if r.status_code != 200 or "html" not in r.headers.get("content-type", ""):
                continue
        except Exception as e:
            print(f"    [warn] {page}: {e}"); continue
        # Follow redirects transparently: the final host (e.g. udot.utah.gov ->
        # connect.udot.utah.gov) becomes crawlable/acceptable too.
        final_host = urllib.parse.urlsplit(r.url).netloc
        pdf_hosts.add(final_host)
        base = r.url
        for href in HREF_RE.findall(r.text):
            u = urllib.parse.urljoin(base, href.strip())
            u = urllib.parse.urldefrag(u).url
            u_host = urllib.parse.urlsplit(u).netloc
            if u.lower().split("?")[0].endswith(".pdf"):
                if u_host in pdf_hosts:
                    out.append((u, page))
            elif (u_host in pdf_hosts and d < depth and u not in seen_pages
                  and not u.lower().split("?")[0].endswith(
                      (".css", ".js", ".png", ".jpg", ".svg", ".gif", ".ico",
                       ".xml", ".zip", ".dwg", ".dgn", ".docx", ".xlsx"))):
                queue.append((u, d + 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dot", required=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="harvest links + report counts; download nothing")
    ap.add_argument("--max-pdfs", type=int, default=600)
    args = ap.parse_args()

    seeds = json.loads(SEEDS.read_text())[args.dot]
    raw_dir = REPO / "data" / args.dot / "raw"
    manifest = REPO / "data" / args.dot / "manifest.jsonl"
    raw_dir.mkdir(parents=True, exist_ok=True)

    have_sha, have_url = set(), set()
    if manifest.exists():
        for line in manifest.open():
            r = json.loads(line)
            have_sha.add(r.get("sha256")); have_url.add(r.get("url"))

    seen_pages: set = set()
    todo: list[tuple[str, str, str]] = []          # (pdf_url, source_page, series)
    seen_pdf: set = set()
    for seed in seeds:
        found = harvest(seed["url"], seed.get("depth", 0), seen_pages,
                        allow_hosts=seed.get("allow_hosts"))
        fresh = [u for u, _ in found if u not in seen_pdf]
        print(f"[{args.dot}] seed {seed['url']}\n"
              f"    series={seed['series']} depth={seed.get('depth', 0)} "
              f"pdfs={len(found)} new={len(fresh)}")
        for u, page in found:
            if u not in seen_pdf:
                seen_pdf.add(u)
                todo.append((u, page, seed["series"]))
    print(f"[{args.dot}] total unique pdf urls: {len(todo)}")
    if args.dry_run:
        return

    n_ok = n_skip = n_err = 0
    with manifest.open("a") as mf:
        for url, page, series in todo[: args.max_pdfs]:
            if url in have_url:
                n_skip += 1; continue
            fname = re.sub(r"[^\w.-]", "_",
                           Path(urllib.parse.urlsplit(url).path).name) or "doc.pdf"
            dest = raw_dir / fname
            if dest.exists():
                stem, suf = dest.stem, dest.suffix
                dest = raw_dir / f"{stem}_{hashlib.md5(url.encode()).hexdigest()[:6]}{suf}"
            try:
                r = fetch(url, stream=True)
                if r.status_code != 200:
                    n_err += 1; continue
                buf, size = [], 0
                for chunk in r.iter_content(1 << 16):
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise ValueError("size cap")
                    buf.append(chunk)
                blob = b"".join(buf)
                if not blob.startswith(b"%PDF"):
                    n_err += 1; continue
                sha = hashlib.sha256(blob).hexdigest()
                if sha in have_sha:
                    n_skip += 1; continue
                dest.write_bytes(blob)
                have_sha.add(sha); have_url.add(url)
                mf.write(json.dumps({
                    "dot": args.dot, "url": url, "source_page": page,
                    "filename": dest.name,
                    "local_path": str(dest.relative_to(REPO)),
                    "document_series": series, "sha256": sha,
                    "bytes": len(blob), "status": "downloaded",
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                }) + "\n")
                mf.flush()
                n_ok += 1
                if n_ok % 25 == 0:
                    print(f"[{args.dot}] downloaded {n_ok}...")
            except Exception as e:
                n_err += 1
                print(f"    [err] {url}: {e}")
    print(f"[{args.dot}] done: {n_ok} downloaded, {n_skip} skipped, {n_err} errors")


if __name__ == "__main__":
    main()
