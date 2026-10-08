#!/bin/bash
#SBATCH --job-name=e16-crag-soft
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-h100,mb-l40s,mb-a6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=16:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/e16_crag_soft_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/e16_crag_soft_%j.err

# E16: soft-scoping on HotpotQA-distractor. run_crag.py never implemented
# soft_scoped, so the paper's headline mechanism ("safe on every corpus") was
# silently untested on this corpus. Soft scoping routes by embedding centroid
# and never consults the tool catalog or the LLM router, so running it through
# run_generic_corpus (generic tool descriptions) is protocol-identical to the
# run_crag arms for this system.
source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs

OUT=emnlp_evaluation/results/crag_qwen_promptctl_soft.jsonl
$PYBIN -m emnlp_evaluation.runners.run_generic_corpus \
    --benchmark crag --llm qwen --systems soft_scoped \
    --backend faiss --shared-answer-prompt --limit 2000 --out $OUT
$PYBIN -m emnlp_evaluation.analysis.judge --in "$OUT" --judge qwen \
    --chunk-text-db emnlp_evaluation/judge_assets/crag_chunks.json
echo "[e16-crag-soft] done"
