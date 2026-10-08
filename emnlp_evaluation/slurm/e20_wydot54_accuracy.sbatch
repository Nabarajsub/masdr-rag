#!/bin/bash
#SBATCH --job-name=e20-wydot54
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-h100,mb-l40s,mb-a6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=3:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/e20_wydot54_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/e20_wydot54_%j.err

# E20: answer accuracy on the 54-document subset, same 32 gold-labelled queries
# as the full-corpus run. Completes the accuracy row of the dilution curve.
source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs
set -e   # a failed generation must not fall through into the judge
OUT=emnlp_evaluation/results/wydot54_qwen_promptctl_monolithic.jsonl
$PYBIN -m emnlp_evaluation.runners.run_generic_corpus \
    --benchmark wydot54 --llm qwen --systems monolithic \
    --backend faiss --shared-answer-prompt --out $OUT
$PYBIN -m emnlp_evaluation.analysis.judge --in "$OUT" --judge qwen \
    --chunk-text-db emnlp_evaluation/judge_assets/wydotv3_chunks.json
echo "[e20] done"
