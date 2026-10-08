#!/bin/bash
#SBATCH --job-name=e12-cbscoped
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-a30,mb-h100,mb-a6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=8:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/e12_cbscoped_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/e12_cbscoped_%j.err

# E12: ColBERTv2-scoped on the FULL Composite-9 set (n=225) under the shared prompt.
# NOTE: colbert_scoped routes by the query's GOLD scope (oracle routing), so it is an
# upper bound on scoped retrieval, not a deployable system. Label it as such.
source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs
OUT=emnlp_evaluation/results/composite_qwen_promptctl_colbert_scoped.jsonl
$PYBIN -m emnlp_evaluation.runners.run_colbert_baseline \
    --benchmark composite --llm qwen --systems colbert_scoped_qwen \
    --shared-answer-prompt --out $OUT
$PYBIN -m emnlp_evaluation.analysis.judge --in "$OUT" --judge qwen \
    --chunk-text-db emnlp_evaluation/judge_assets/composite_chunks.json
echo "[e12] done"
