#!/bin/bash
#SBATCH --job-name=smoke-qwen
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/smoke_qwen_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/smoke_qwen_%j.err

source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs emnlp_evaluation/results

# 3 queries × every system, against the local BGE-M3 index (no Gemini API).
# Requires reindex_bge_m3.sbatch to have run once to populate the index.
$PYBIN -m emnlp_evaluation.runners.run_wydot_oss \
    --llm qwen \
    --embedder bge_m3 \
    --systems monolithic,regex_scoped,hybrid_routed,masdr_rag,react \
    --limit 3 \
    --out emnlp_evaluation/results/smoke_qwen.jsonl
