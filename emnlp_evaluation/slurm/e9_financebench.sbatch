#!/bin/bash
#SBATCH --job-name=e9-finance
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-a30,mb-h100,mb-a6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=8:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/e9_finance_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/e9_finance_%j.err

# E9: second independent-benchmark test for soft-scoping (reviewer ask — a win
# beyond the author-built Composite-9). FinanceBench has 84 organizational
# scopes (company 10-K/8-K filings), a strong routing signal. Shared prompt.
source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs

OUT=emnlp_evaluation/results/financebench_qwen_promptctl_singlecall.jsonl
$PYBIN -m emnlp_evaluation.runners.run_generic_corpus \
    --benchmark financebench --llm qwen \
    --systems monolithic,regex_scoped,hybrid_routed,soft_scoped \
    --shared-answer-prompt --out $OUT
$PYBIN -m emnlp_evaluation.analysis.judge \
    --in "$OUT" --judge qwen \
    --chunk-text-db emnlp_evaluation/judge_assets/financebench_chunks.json

echo "[e9-finance] done"
