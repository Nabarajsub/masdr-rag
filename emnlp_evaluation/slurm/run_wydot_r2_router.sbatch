#!/bin/bash
#SBATCH --job-name=wydot-r2
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/wydot_r2_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/wydot_r2_%j.err

# Run R2-Routed (BGE-M3 + LogReg trained router) end-to-end on WYDOT 200
# alongside the regex-routed Hybrid for direct delta on Faith/Corr.

source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs

$PYBIN -m emnlp_evaluation.runners.run_wydot_oss \
    --llm qwen \
    --embedder bge_m3 \
    --systems r2_routed \
    --out emnlp_evaluation/results/wydot_qwen_bge_m3_r2_routed.jsonl \
    --resume
