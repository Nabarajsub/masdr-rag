# MASDR-RAG — Reproducibility Package

Code, datasets, and evaluation artifacts for the anonymous ARR submission
*"When More Documents Hurt RAG: Measuring and Reducing Vector Search Dilution
with Domain-Scoped Retrieval"* (ARR resubmission of Submission 13333).

Every quantitative claim in the paper is traceable to a script in this
repository, and **every results table can be recomputed on a laptop (no GPU)**
from the released judged evaluation logs (`data/results/`).

## Repository layout

```
├── emnlp_evaluation/      evaluation harness
│   ├── agents/            the systems compared (Monolithic, Regex-Scoped,
│   │                      Hybrid-Routed, R2-Routed, MASDR-RAG, SingleCall,
│   │                      ReAct, MA-RAG port, SCOUT-RAG reimpl)
│   ├── runners/           per-corpus eval entry points
│   ├── analysis/          judge, metrics aggregation, statistical tests
│   ├── scripts/           figure generation
│   ├── router/            trained R2 router (BGE-M3 linear probe) + 5-fold CV
│   ├── llm_providers/     Qwen / Llama / Gemini / OpenRouter wrappers
│   ├── embeddings/        BGE-M3 / Gemini embedding wrappers
│   ├── benchmark/         per-corpus download + FAISS index build
│   ├── composite_corpus/  EnterpriseComposite-9 assembly
│   ├── configs/           all configuration via environment variables
│   └── slurm/             SLURM submission scripts used for the paper
├── multi_dot/             §3.4 seven-DOT pipeline: scrape → chunk → embed → dilution δ,
│                          plus scope discovery (discover_scopes.py, §7)
├── scrapers/              polite single-threaded DOT website crawler (§3.4)
├── agentic_solution/      shared scope-filter config (import dependency)
└── data/
    ├── queries/           ★ the datasets ★ — see data/README.md
    │   ├── wydot_test_suite_200.json   200 human-validated WYDOT queries
    │   ├── composite_queries.json      225 Composite-9 queries
    │   ├── wydotv3_queries.json        WYDOT suite with hand / discovered scope labels
    │   └── {multihop_rag,mmlu_pro,nq,financebench}_queries.json
    ├── judge_chunk_dbs/   chunk-id → text databases for the judge (gz)
    ├── results/           judged per-query evaluation logs (gz) — recompute
    │                      any paper table without running a model
    ├── dilution/          per-category dilution δ for all seven DOTs
    ├── human_eval/        blind human faithfulness ratings (§8) + answer key
    └── k_sweep.json       R@k sweep data
```

## Where each result in the revised paper comes from

| Paper result | Script | Data |
|---|---|---|
| Table 6, shared-prompt cross-corpus results and significance (incl. Soft-Scoped) | `analysis/rq2_scoping_test.py` (run by `make tables`) | `data/results/*promptctl*`, `data/results/rq2/` |
| Table 2, controlled corpus growth (§3.3) | `analysis/dilution_scaling_curve.py`, `analysis/rq1_dilution_test.py` | `data/results/dilution_scaling_*.json`, `data/results/rq1/` |
| Table 3, pooled ρ over seven DOTs (§3.3–3.4) | `multi_dot/dilution.py`, `analysis/dilution_analysis.py` | `data/dilution/` |
| Scope discovery (§7, App. A6.2) | `multi_dot/discover_scopes.py`, `router/train_discovered_router.py`, `analysis/rq3_discovery_test.py` | `data/results/rq3/` |
| Human faithfulness study (§8) | `analysis/score_human_eval_paradox.py` (run by `make tables`) | `data/human_eval/` |
| Retrieval budget and confidence fallback (App. A6.1) | `analysis/k_sweep.py`, `analysis/fallback_sweep.py` | `data/k_sweep.json`, `data/results/fallback_sweep.json` |
| Soft scoping and the centroid router (§4) | `runners/run_wydot_oss.py`, `runners/run_generic_corpus.py` (`--systems soft_scoped`) | — |
| Cost figure (Figure 2) | `scripts/make_efficiency_figure_v2.py` | `data/results/` |

## Quickstart — recompute the paper's tables (CPU-only, ~2 minutes)

```bash
pip install -r requirements.txt   # only numpy/pandas/scipy needed for tables
make setup-data                   # decompress judged logs + chunk DBs
make tables                       # aggregate every judged run into the tables
```

`make tables` reads `data/results/*.judged.jsonl` (one JSON per query per
system, with the judge's correctness/faithfulness verdicts and the retrieved
chunk ids) and prints the per-system aggregates behind each table in the
paper. The dilution correlations (§3, §9) recompute from `data/dilution/`:

```bash
python -m emnlp_evaluation.analysis.dilution_analysis     # per-category δ
```

## Full re-run (GPU)

The full sweep (retrieval + generation + judging) needs one 48GB GPU
(L40S-class) and the public HuggingFace checkpoints `Qwen/Qwen2.5-7B-Instruct`,
`meta-llama/Meta-Llama-3-8B-Instruct`, and `BAAI/bge-m3`:

```bash
cp .env.example .env              # fill in what you use (all optional)
make build-indexes                # download corpora + build FAISS indexes
make sweep                        # WYDOT-200 + Composite-9, all systems
make judge                        # LLM-as-judge over the fresh runs
```

Per-corpus runners accept `--systems`, `--limit`, `--llm {qwen,llama,gemini,
openrouter:<slug>}`, and `--backend {faiss,neo4j,splade}`; see each runner's
docstring. SLURM users: `emnlp_evaluation/slurm/*.sbatch` are the exact
scripts used for the paper (account/partition names redacted).

## Paper section → code map

| Paper artifact | Script |
|---|---|
| §3 dilution factor δ, per-category table | `emnlp_evaluation/analysis/dilution_analysis.py`, `multi_dot/dilution.py` |
| §3 Fig. dilution-vs-scale | `emnlp_evaluation/scripts/make_dilution_curve.py` |
| §3/§8 retrieval-confusion heatmap (Fig. 5) | `emnlp_evaluation/scripts/make_confusion_heatmap.py` |
| §4 MASDR-RAG / SingleCall / Hybrid-Routed / R2-Routed | `emnlp_evaluation/agents/{orchestrator_oss,orchestrator_singlecall,hybrid_routed,r2_router}.py` |
| §5–§6 WYDOT tables | `emnlp_evaluation/runners/run_wydot_oss.py` |
| §7 efficiency vs ReAct | `emnlp_evaluation/agents/react_baseline.py` + runner `--systems react` |
| §8 cross-domain tables | `emnlp_evaluation/runners/{run_composite,run_generic_corpus,run_multihop}.py` |
| §8 MA-RAG / SCOUT-RAG baselines | `emnlp_evaluation/agents/{ma_rag,scout_rag}.py` |
| §9 cross-DOT replication | `scrapers/`, `multi_dot/` |
| §10 paradox ablations (rerank / SPLADE / index parity / SingleCall) | `emnlp_evaluation/analysis/aggregate_paradox.py` + runners `--rerank/--backend splade/--backend neo4j` |
| §10 forced-format control, dedup ablation | `emnlp_evaluation/runners/rerun_masdr_forced_format.py` |
| Router fallback sweep | `emnlp_evaluation/analysis/fallback_sweep.py` |
| LLM-as-judge | `emnlp_evaluation/analysis/judge.py` (prompts in file) |
| Statistical tests | `emnlp_evaluation/analysis/permutation_tests.py` |

## Notes

* All corpora are public: state-DOT documents are public government records;
  the benchmark suites (HotpotQA, MultiHop-RAG, NQ-Open, FinanceBench,
  MMLU-Pro) download from their official sources via
  `emnlp_evaluation/benchmark/`.
* No credentials ship with this repository; every external service is
  configured through environment variables (`.env.example`).
* Released under the MIT License (see `LICENSE`).
