"""Find a representative monolithic-fails / scoped-wins query for the
failure-case figure in §3."""
import json
from collections import defaultdict, Counter
from pathlib import Path

INFILE = Path(
    "<DATA_ROOT>"
    "emnlp_evaluation/results/"
    "composite_qwen_monolithic-regex_scoped-hybrid_routed-masdr_rag-react"
    ".judged.judged.judged.judged.jsonl"
)
records = [json.loads(l) for l in INFILE.open()]
by_query = defaultdict(dict)
for r in records:
    qid = r.get("query_id")
    sys = r.get("system")
    by_query[qid][sys] = r

cands = []
for qid, sysmap in by_query.items():
    mono = sysmap.get("monolithic")
    scop = sysmap.get("regex_scoped") or sysmap.get("hybrid_routed")
    if not mono or not scop:
        continue
    gold = mono.get("category")
    if not gold:
        continue
    mc = mono.get("correctness")
    sc = scop.get("correctness")
    if mc != 0 or sc != 1:
        continue
    sources_mono = mono.get("chunk_sources") or []
    sources_scop = scop.get("chunk_sources") or []
    frac_gold_mono = sum(1 for s in sources_mono if s == gold) / max(1, len(sources_mono))
    frac_gold_scop = sum(1 for s in sources_scop if s == gold) / max(1, len(sources_scop))
    cands.append((qid, gold, frac_gold_mono, frac_gold_scop, mono, scop))

cands.sort(key=lambda x: (x[2] - x[3]))
print(f"candidates: {len(cands)}")
for qid, gold, fmono, fscop, mono, scop in cands[:8]:
    print()
    print(f"=== {qid}  gold={gold}  mono-on-gold={fmono:.2f}  scop-on-gold={fscop:.2f}")
    print("Q:", mono.get("query"))
    print("ref:", str(mono.get("reference_answer",""))[:200])
    print("MONO ans:", str(mono.get("answer",""))[:200])
    print("  mono sources:", Counter(mono.get("chunk_sources") or []))
    print("  mono titles[:5]:", (mono.get("chunk_titles") or [])[:5])
    print("SCOP ans:", str(scop.get("answer",""))[:200])
    print("  scop sources:", Counter(scop.get("chunk_sources") or []))
    print("  scop titles[:5]:", (scop.get("chunk_titles") or [])[:5])
