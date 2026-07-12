"""
Assemble an "enterprise-like" composite corpus from 9 public HuggingFace datasets.

We approximate the original EnterpriseRAG-Bench spec (Slack / Gmail / GitHub /
Jira / Confluence / Docs / SO / Helpdesk / Reports) using real public data:

    source_type   HF dataset (default)                         analog
    ------------- ------------------------------------------    ----------
    gmail         snoop2head/enron_aeslc_emails                 Enron email
    slack         daily_dialog                                  multi-turn chat
    github        codeparrot/github-issues  (or lvwerra/...)    issue threads
    jira          codeparrot/github-issues  (filter: bug label) tracker
    confluence    wikipedia (en.20220301)                       wiki pages
    docs          ms_marco                                      doc paragraphs
    stackoverflow pacovaldez/stackoverflow-questions             Q&A
    helpdesk      nvidia/HelpSteer                              support
    reports       JanosAudran/financial-reports-sec             SEC filings

The script tolerates per-source failures (network, schema, license). Each
source contributes at most --limit-per-source rows so a dry-run on
~9k docs is feasible. Bump the limit later for the full ~200k version.

Run from a LOGIN node — compute nodes have no outbound network.

Usage:
    python -m emnlp_evaluation.composite_corpus.assemble --limit-per-source 1000
    -> writes emnlp_evaluation/composite_corpus/data/corpus.parquet
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _safe_text(*parts) -> str:
    return " ".join(str(p).strip() for p in parts if p is not None and str(p).strip())


def _truncate(text: str, max_chars: int = 1200) -> str:
    return text[:max_chars] if len(text) > max_chars else text


def _chunk_text(text: str, max_chars: int = 800) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    sents = re.split(r"(?<=[.!?])\s+", text)
    chunks, cur = [], ""
    for s in sents:
        if len(cur) + len(s) > max_chars and cur:
            chunks.append(cur.strip()); cur = ""
        cur += s + " "
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


# Each loader returns an iterable of dicts:
#   {source_type, doc_id, title, text}
# text MUST be string. doc_id can repeat across sources (we add source as prefix).


def _try(name: str, fn: Callable[[int], Iterable[Dict]], limit: int) -> List[Dict]:
    try:
        print(f"[assemble] {name}: loading...", flush=True)
        rows = list(fn(limit))
        print(f"[assemble] {name}: {len(rows)} docs")
        return rows
    except Exception as e:
        print(f"[assemble] {name}: FAILED ({type(e).__name__}: {e})", flush=True)
        traceback.print_exc()
        return []


# ── per-source loaders ─────────────────────────────────────────────────────

def _load_gmail(limit: int):
    from datasets import load_dataset
    # Enron email corpus via aeslc
    ds = load_dataset("snoop2head/enron_aeslc_emails", split="train", streaming=True)
    for i, ex in enumerate(ds):
        if i >= limit: return
        body = ex.get("email_body") or ex.get("body") or ex.get("text") or ""
        subj = ex.get("subject_line") or ex.get("subject") or ""
        yield {
            "source_type": "gmail",
            "doc_id": f"enron_{i:06d}",
            "title": subj[:120],
            "text": _safe_text(subj, body),
        }


def _load_slack(limit: int):
    """Use OpenAssistant/oasst1 conversation threads as Slack-style chat."""
    from datasets import load_dataset
    ds = load_dataset("OpenAssistant/oasst1", split="train", streaming=True)
    # Group messages into threads by message_tree_id.
    threads: Dict[str, List[Dict]] = {}
    for ex in ds:
        tid = ex.get("message_tree_id") or ex.get("parent_id") or ex.get("message_id")
        threads.setdefault(tid, []).append(ex)
        if len(threads) >= limit * 3:  # over-sample to skip short ones
            break
    yielded = 0
    for tid, msgs in threads.items():
        if yielded >= limit:
            return
        # Sort by created_date if available.
        msgs = sorted(msgs, key=lambda m: m.get("created_date") or "")
        if len(msgs) < 2:
            continue
        text = "\n".join(
            f"U{j+1} ({m.get('role','user')}): {(m.get('text') or '')[:400]}"
            for j, m in enumerate(msgs[:10])
        )
        yield {
            "source_type": "slack",
            "doc_id": f"oasst_{yielded:06d}",
            "title": f"thread {tid[:24]}",
            "text": text,
        }
        yielded += 1


def _load_github(limit: int):
    """GitHub issues via lewtun/github-issues (parquet, no script)."""
    from datasets import load_dataset
    ds = load_dataset("lewtun/github-issues", split="train", streaming=True)
    for i, ex in enumerate(ds):
        if i >= limit: return
        title = ex.get("title") or ""
        body = ex.get("body") or ""
        yield {
            "source_type": "github",
            "doc_id": f"gh_{i:06d}",
            "title": str(title)[:120],
            "text": _safe_text(str(title), str(body)),
        }


def _load_jira(limit: int):
    """Bug-labelled GitHub issues stand in for Jira bug tickets."""
    from datasets import load_dataset
    ds = load_dataset("lewtun/github-issues", split="train", streaming=True)
    bug_re = re.compile(r"\b(bug|crash|error|exception|regress|broken|fail|failure)\b", re.IGNORECASE)
    yielded = 0
    for i, ex in enumerate(ds):
        if yielded >= limit: return
        title = str(ex.get("title") or "")
        body = str(ex.get("body") or "")
        labels = ex.get("labels") or []
        label_names = [l.get("name", "") if isinstance(l, dict) else str(l) for l in labels]
        is_bug = bug_re.search(title + " " + body) or any(
            bug_re.search(ln) for ln in label_names
        )
        if not is_bug:
            continue
        yield {
            "source_type": "jira",
            "doc_id": f"jira_{yielded:06d}",
            "title": f"[BUG] {title[:120]}",
            "text": _safe_text("Issue Type: Bug\nPriority: P2", title, body),
        }
        yielded += 1


def _load_confluence(limit: int):
    """Wikipedia paragraphs stand in for wiki pages."""
    from datasets import load_dataset
    try:
        ds = load_dataset("wikipedia", "20220301.simple", split="train", streaming=True,
                          trust_remote_code=True)
    except Exception:
        ds = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train", streaming=True)
    for i, ex in enumerate(ds):
        if i >= limit: return
        title = ex.get("title", "")
        text = ex.get("text", "")
        yield {
            "source_type": "confluence",
            "doc_id": f"wiki_{i:06d}",
            "title": title[:120],
            "text": _truncate(_safe_text(title, text), 2000),
        }


def _load_docs(limit: int):
    """MS-MARCO passages serve as 'documentation pages'."""
    from datasets import load_dataset
    try:
        ds = load_dataset("ms_marco", "v2.1", split="train", streaming=True, trust_remote_code=True)
        for i, ex in enumerate(ds):
            if i >= limit: return
            passages = ex.get("passages") or {}
            texts = passages.get("passage_text") or []
            for j, t in enumerate(texts[:1]):  # one passage per example
                yield {
                    "source_type": "docs",
                    "doc_id": f"msmarco_{i:06d}_{j}",
                    "title": ex.get("query", "")[:120],
                    "text": t,
                }
    except Exception:
        # Fallback: squad
        ds = load_dataset("squad", split="train", streaming=True)
        for i, ex in enumerate(ds):
            if i >= limit: return
            yield {
                "source_type": "docs",
                "doc_id": f"squad_{i:06d}",
                "title": ex.get("title", "")[:120],
                "text": ex.get("context", ""),
            }


def _load_stackoverflow(limit: int):
    from datasets import load_dataset
    try:
        ds = load_dataset("pacovaldez/stackoverflow-questions", split="train", streaming=True)
    except Exception:
        ds = load_dataset("mteb/stackoverflowdupquestions-reranking", split="train", streaming=True)
    for i, ex in enumerate(ds):
        if i >= limit: return
        title = ex.get("title") or ex.get("query") or ""
        body = ex.get("body") or ex.get("question") or ex.get("text") or ""
        yield {
            "source_type": "stackoverflow",
            "doc_id": f"so_{i:06d}",
            "title": title[:120],
            "text": _safe_text(title, body),
        }


def _load_helpdesk(limit: int):
    from datasets import load_dataset
    try:
        ds = load_dataset("nvidia/HelpSteer", split="train", streaming=True)
    except Exception:
        ds = load_dataset("Bingsu/CustomerService_Conversation", split="train", streaming=True)
    for i, ex in enumerate(ds):
        if i >= limit: return
        prompt = ex.get("prompt") or ex.get("question") or ""
        response = ex.get("response") or ex.get("answer") or ""
        yield {
            "source_type": "helpdesk",
            "doc_id": f"help_{i:06d}",
            "title": prompt[:120],
            "text": _safe_text("Customer:", prompt, "Agent:", response),
        }


def _load_reports(limit: int):
    """SEC 10-K filings via virattt/financial-qa-10K (parquet)."""
    from datasets import load_dataset
    ds = load_dataset("virattt/financial-qa-10K", split="train", streaming=True)
    for i, ex in enumerate(ds):
        if i >= limit: return
        ticker = ex.get("ticker") or "10K"
        filing = ex.get("filing") or "Annual Report"
        # Use the long context as the "report" content; question/answer as title metadata.
        ctx = ex.get("context") or ""
        q = ex.get("question") or ""
        yield {
            "source_type": "reports",
            "doc_id": f"sec_{i:06d}",
            "title": f"{ticker} {filing}: {q[:90]}",
            "text": _truncate(_safe_text(f"Company: {ticker}", f"Filing: {filing}", ctx), 2000),
        }


LOADERS: List[tuple] = [
    ("gmail", _load_gmail),
    ("slack", _load_slack),
    ("github", _load_github),
    ("jira", _load_jira),
    ("confluence", _load_confluence),
    ("docs", _load_docs),
    ("stackoverflow", _load_stackoverflow),
    ("helpdesk", _load_helpdesk),
    ("reports", _load_reports),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-per-source", type=int, default=1000)
    ap.add_argument("--max-chars", type=int, default=800)
    ap.add_argument("--out", default=str(DATA_DIR / "corpus.parquet"))
    ap.add_argument("--meta", default=str(DATA_DIR / "summary.json"))
    args = ap.parse_args()

    t0 = time.time()
    all_rows: List[Dict] = []
    summary: Dict[str, int] = {}

    for name, fn in LOADERS:
        rows = _try(name, fn, args.limit_per_source)
        summary[name] = len(rows)
        # chunk every doc and emit one row per chunk
        for r in rows:
            for j, ch in enumerate(_chunk_text(r["text"], max_chars=args.max_chars)):
                all_rows.append({
                    "id": f"{r['source_type']}::{r['doc_id']}::{j:03d}",
                    "doc_id": r["doc_id"],
                    "source_type": r["source_type"],
                    "title": r.get("title", ""),
                    "text": ch,
                    "idx": j,
                })

    if not all_rows:
        print("[assemble] no rows assembled; nothing written", file=sys.stderr)
        sys.exit(2)

    df = pd.DataFrame(all_rows)
    df.to_parquet(args.out, index=False)
    counts = df.groupby("source_type").size().to_dict()
    with open(args.meta, "w") as f:
        json.dump({
            "docs_per_source": summary,
            "chunks_per_source": counts,
            "total_chunks": int(len(df)),
            "elapsed_s": time.time() - t0,
        }, f, indent=2)

    print(f"[assemble] wrote {args.out} :: {len(df)} chunks total")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
