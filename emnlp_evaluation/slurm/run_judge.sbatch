#!/bin/bash
#SBATCH --job-name=judge-eval
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/judge_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/judge_%j.err

source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs

GLOB="${GLOB:-emnlp_evaluation/results/*.jsonl}"
JUDGE="${JUDGE:-qwen}"

$PYBIN -m emnlp_evaluation.analysis.judge --in "$GLOB" --judge "$JUDGE"
