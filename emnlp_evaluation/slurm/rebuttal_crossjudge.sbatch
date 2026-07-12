#!/bin/bash
#SBATCH --job-name=reb-xjudge
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/reb_xjudge_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/reb_xjudge_%j.err

source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs

# Cross-family judge (Llama-3-8B) over the canonical WYDOT run (all 4 systems,
# ~800 rows) for judge-agreement analysis vs the Qwen judge (mFpD W8).
$PYBIN -m emnlp_evaluation.analysis.judge \
    --in "emnlp_evaluation/results/wydot_qwen_bge_m3_llamajudge.jsonl" \
    --judge llama \
    --chunk-text-db emnlp_evaluation/judge_assets/wydot_chunks.json
