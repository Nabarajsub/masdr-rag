#!/bin/bash
#SBATCH --job-name=wydot-soft-gemini
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/soft_gemini_%x_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/soft_gemini_%x_%j.err
# Soft-Scoped on the production Gemini-embedding Neo4j store, original answer
# prompt -- same setting as the wydot_{qwen,llama}_gemini cost runs (Figure: efficiency).
source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs
LLM="${LLM:-qwen}"
$PYBIN -m emnlp_evaluation.runners.run_wydot_oss_cached_gemini \
    --llm "$LLM" --embedder gemini --systems soft_scoped --resume
echo "[soft-gemini] done llm=$LLM"
