#!/bin/bash
#SBATCH --job-name=wydot-bge-baselines
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/wydot_bge_baselines_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/wydot_bge_baselines_%j.err

# Needed for the S-2 paradox-cause comparison: same embedder (bge_m3),
# same LLM (qwen), same scoping, only the synthesis schedule differs.

source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs

$PYBIN -m emnlp_evaluation.runners.run_wydot_oss \
    --llm qwen \
    --embedder bge_m3 \
    --systems monolithic,regex_scoped,hybrid_routed,masdr_rag \
    --out emnlp_evaluation/results/wydot_qwen_bge_m3_baselines.jsonl \
    --resume
