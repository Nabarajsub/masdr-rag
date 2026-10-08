#!/bin/bash
#SBATCH --job-name=e17-gapfill
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-h100,mb-l40s,mb-a6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=20:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/e17_gapfill_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/e17_gapfill_%j.err

# E17: close the remaining shared-prompt gaps so every number in the paper is
# produced under one protocol.
#
#   (a) FinanceBench MASDR -- the single-call arms were re-run under the shared
#       prompt but MASDR never was, so the FinanceBench block mixed protocols.
#   (b) Composite-9 masdr_singlecall at n=225 -- tab:singlecall_composite is
#       still old-protocol n=50 and its numbers contradict tab:cross_domain
#       (Hybrid .857 there vs 56.4 in the main table). Its ReAct row also
#       reprints the 94.4 that App. C already discloses as a 10-query artifact.
#   (c) NQ-Open, all five systems -- it had ZERO shared-prompt runs, yet the
#       introduction lists it among six corpora evaluated "under a single
#       corpus-neutral answer prompt". n=500 to match the other corpora.
source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs

# (a) FinanceBench MASDR
OUT_FB=emnlp_evaluation/results/financebench_qwen_promptctl_masdr.jsonl
$PYBIN -m emnlp_evaluation.runners.run_generic_corpus \
    --benchmark financebench --llm qwen --systems masdr_rag \
    --shared-answer-prompt --limit 500 --out $OUT_FB
$PYBIN -m emnlp_evaluation.analysis.judge --in "$OUT_FB" --judge qwen \
    --chunk-text-db emnlp_evaluation/judge_assets/financebench_chunks.json

# (b) Composite-9 single-call MASDR ablation, full query set
OUT_SC=emnlp_evaluation/results/composite_qwen_promptctl_masdr_singlecall.jsonl
$PYBIN -m emnlp_evaluation.runners.run_composite \
    --llm qwen --systems masdr_singlecall \
    --shared-answer-prompt --out $OUT_SC
$PYBIN -m emnlp_evaluation.analysis.judge --in "$OUT_SC" --judge qwen \
    --chunk-text-db emnlp_evaluation/judge_assets/composite_chunks.json

# (c) NQ-Open, all five systems
OUT_NQ=emnlp_evaluation/results/nq_qwen_promptctl_all.jsonl
$PYBIN -m emnlp_evaluation.runners.run_generic_corpus \
    --benchmark nq --llm qwen \
    --systems monolithic,regex_scoped,hybrid_routed,soft_scoped,masdr_rag \
    --backend faiss --shared-answer-prompt --limit 500 --out $OUT_NQ
$PYBIN -m emnlp_evaluation.analysis.judge --in "$OUT_NQ" --judge qwen \
    --chunk-text-db emnlp_evaluation/judge_assets/nq_chunks.json

echo "[e17-gapfill] done"
