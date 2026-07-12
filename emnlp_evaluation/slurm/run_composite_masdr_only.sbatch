#!/bin/bash
#SBATCH --job-name=composite-masdr
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/composite_masdr_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/composite_masdr_%j.err

# Re-run MASDR-RAG only on EnterpriseComposite-9 (Qwen-2.5-7B). Fills the
# missing MASDR row in Table 6 / Figure 6. Output is written to a fresh
# JSONL so judge auto-chain doesn't collide with already-judged files.

source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs

LLM="${LLM:-qwen}"

$PYBIN -m emnlp_evaluation.runners.run_composite \
    --llm "$LLM" \
    --systems masdr_rag \
    --out emnlp_evaluation/results/composite_${LLM}_masdr_rag.jsonl
