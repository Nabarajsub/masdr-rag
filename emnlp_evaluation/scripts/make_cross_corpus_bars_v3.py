"""Grouped bar chart of per-system correctness across the four corpora evaluated
under the SHARED corpus-neutral answer prompt (final_v3). Replaces the v2 figure,
whose bars predated both the router fix and the prompt control."""
import json, glob
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("<DATA_ROOT>")
OUT  = Path("<DATA_ROOT>")

SYSTEMS = [("Monolithic","monolithic"), ("Regex-Scoped","regex_scoped"),
           ("Hybrid-Routed","hybrid_routed"), ("Soft-Scoped","soft_scoped"),
           ("MASDR-RAG","masdr_rag")]
COLORS = {"Monolithic":"#8a8f98","Regex-Scoped":"#4c9be8","Hybrid-Routed":"#2ca02c",
          "Soft-Scoped":"#e8a33d","MASDR-RAG":"#9467bd"}
BENCH = [("WYDOT",       ["wydot_hand_qwen_promptctl_singlecall.judged.jsonl",
                          "wydot_hand_qwen_promptctl_masdr.judged.jsonl"]),
         ("Composite-9", ["composite_qwen_promptctl_singlecall.judged.jsonl",
                          "composite_qwen_promptctl_masdr.judged.jsonl"]),
         ("MultiHop-RAG",["multihop_qwen_promptctl_singlecall.judged.jsonl",
                          "multihop_qwen_promptctl_masdr.judged.jsonl"]),
         ("MMLU-Pro",    ["mmlu_pro_qwen_promptctl_singlecall.judged.jsonl",
                          "mmlu_pro_qwen_promptctl_masdr.judged.jsonl"])]

def corr(files, system):
    seen={}
    for f in files:
        p=ROOT/f
        if not p.exists(): continue
        for line in open(p):
            r=json.loads(line)
            if "error" in r or r.get("system")!=system: continue
            seen[r["query_id"]]=int(r.get("correctness",0))
    return 100*sum(seen.values())/len(seen) if seen else np.nan

data={n:[corr(fs,s) for _,fs in BENCH] for n,s in SYSTEMS}
x=np.arange(len(BENCH)); w=0.16
fig,ax=plt.subplots(figsize=(7.2,2.9))
for i,(name,_) in enumerate(SYSTEMS):
    v=data[name]
    ax.bar(x+(i-2)*w, v, w, label=name, color=COLORS[name], edgecolor="white", linewidth=.6)
ax.set_xticks(x); ax.set_xticklabels([b for b,_ in BENCH], fontsize=9)
ax.set_ylabel("Correctness (%)", fontsize=9); ax.set_ylim(0,100)
ax.grid(axis="y", alpha=.25, linewidth=.6); ax.set_axisbelow(True)
for sp in ("top","right"): ax.spines[sp].set_visible(False)
ax.legend(ncol=5, fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(.5,1.20))
plt.tight_layout()
plt.savefig(OUT/"cross_corpus_bars_v3.pdf", bbox_inches="tight")
print("wrote", OUT/"cross_corpus_bars_v3.pdf")
for n,_ in SYSTEMS: print(f"  {n:<14}", " ".join(f"{v:5.1f}" if v==v else "  n/a" for v in data[n]))
