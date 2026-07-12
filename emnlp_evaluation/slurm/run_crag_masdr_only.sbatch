#!/bin/bash
#SBATCH --job-name=crag-masdr
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/crag_masdr_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/crag_masdr_%j.err

# Run MASDR-RAG only on HotpotQA (CRAG adapter). Fills the missing
# MASDR rows in Table 6 / Figure 6 for the multi-hop benchmark.

source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs

LLM="${LLM:-qwen}"

$PYBIN -m emnlp_evaluation.runners.run_crag \
    --llm "$LLM" \
    --systems masdr_rag \
    --out emnlp_evaluation/results/crag_${LLM}_masdr_rag.jsonl
