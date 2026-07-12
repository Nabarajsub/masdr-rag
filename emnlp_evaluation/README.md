# EMNLP Evaluation

Code and SLURM scripts that back the EMNLP submission for **MASDR-RAG**
(domain-scoped, agentic retrieval).

This is *not* a rewrite of `graph_processing/agentic_solution/`. It is a
parallel evaluation harness that:

1. Swaps the proprietary Gemini stack for open-source models (Qwen2.5-7B,
   Llama-3-8B, BGE-M3) so results are reproducible by reviewers.
2. Adds a **second benchmark** — MultiHop-RAG — so the dilution and
   precision/faithfulness claims do not rest on a single in-house corpus.
3. Implements a **ReAct baseline** that uses the same nine domain tools, so
   the speed/quality tradeoff vs iterative reasoning can be measured fairly.
4. Produces per-query JSONL traces (chunks, answer, latency, tokens, routing
   decision) that feed the analysis scripts and the paper tables.

## Layout

```
emnlp_evaluation/
├── configs/
│   └── oss_config.py        # model paths, index names, judge config
├── llm_providers/           # uniform LLM interface
│   ├── base.py
│   ├── gemini_provider.py
│   ├── qwen_provider.py     # Qwen2.5-7B-Instruct via HF transformers
│   └── llama_provider.py    # Meta-Llama-3-8B-Instruct via HF transformers
├── embeddings/
│   ├── base.py
│   ├── gemini_embed.py
│   ├── bge_m3.py            # local BGE-M3 via sentence-transformers
│   └── reindex_neo4j.py     # re-embed all WYDOT chunks with BGE-M3
├── agents/
│   ├── tools_oss.py         # scoped Neo4j tools, embed-agnostic
│   ├── orchestrator_oss.py  # MASDR-RAG with pluggable LLM
│   ├── hybrid_routed.py     # Regex → LLM router → scoped single-agent
│   ├── naive_rag.py         # retrieve→generate, no routing
│   ├── monolithic_rag.py    # global vector, no domain scope
│   └── react_baseline.py    # ReAct loop over the same tool set
├── runners/
│   ├── run_wydot_oss.py     # 200-query WYDOT eval, all systems × all LLMs
│   └── run_multihop_rag.py  # external benchmark runner
├── benchmark/
│   └── multihop_rag/
│       ├── download.py      # pull MultiHop-RAG from HF
│       ├── ingest.py        # ingest corpus into Neo4j (or FAISS)
│       └── source_map.py    # source-type → scoped agent
├── slurm/                   # sbatch scripts targeting mb-l40s / mb-a30
├── analysis/
│   ├── judge.py             # LLM-as-judge (faithfulness, correctness)
│   ├── latency_pareto.py    # P50/P95 latency + token cost + Pareto plot
│   └── dilution_analysis.py # cross-corpus dilution factor
└── results/                 # JSONL traces land here
```

## Running

All runners default to the existing AuraDB used by `agentic_solution`.
SLURM submission from a login node:

```bash
cd graph_processing/emnlp_evaluation
sbatch slurm/reindex_bge_m3.sbatch       # one-time: re-embed Neo4j with BGE-M3
sbatch slurm/run_qwen_wydot.sbatch       # 200 queries × 6 systems with Qwen
sbatch slurm/run_llama_wydot.sbatch      # same, Llama-3-8B
sbatch slurm/run_multihop_ingest.sbatch  # ingest MultiHop-RAG corpus
sbatch slurm/run_multihop_eval.sbatch    # MultiHop-RAG eval
python analysis/latency_pareto.py results/   # tables + figures
```

## Open-source stack

| Component         | Replacement                                       |
|-------------------|---------------------------------------------------|
| Query router LLM  | Qwen2.5-7B-Instruct *or* Llama-3-8B-Instruct      |
| Answer LLM        | Same (configurable per stage)                     |
| Embeddings        | BAAI/bge-m3 (1024d, local)                        |
| LLM judge         | Qwen2.5-7B (or pluggable Groq Llama-70B endpoint) |

Models are loaded from local paths on ARCC:
- `<DATA_ROOT>
- `<DATA_ROOT>
- `<DATA_ROOT>

## Reproducing the paper tables

Every table in the EMNLP draft maps to a JSONL trace in `results/`:

| Paper table             | Source                                                   |
|-------------------------|----------------------------------------------------------|
| Table 2 (WYDOT main)    | `results/wydot_qwen_*.jsonl` + `analysis/latency_pareto` |
| Table 3 (model-agnostic)| `results/wydot_{gemini,qwen,llama}_*.jsonl`              |
| Table 4 (MultiHop-RAG)  | `results/multihop_*.jsonl`                               |
| Table 5 (ReAct vs Hybrid)| `results/*react*.jsonl` + `latency_pareto`              |
| Figure: Pareto curve    | `analysis/latency_pareto.py --plot`                      |
