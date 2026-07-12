#!/bin/bash
#SBATCH --job-name=reb-rejudge
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/reb_rejudge_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/reb_rejudge_%j.err

source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs

$PYBIN -m emnlp_evaluation.analysis.judge \
    --in "emnlp_evaluation/results/wydot_qwen_extbase_rejudge.jsonl" \
    --judge qwen --chunk-text-db emnlp_evaluation/judge_assets/wydot_chunks.json

$PYBIN -m emnlp_evaluation.analysis.judge \
    --in "emnlp_evaluation/results/composite_qwen_extbase_rejudge.jsonl" \
    --judge qwen --chunk-text-db emnlp_evaluation/judge_assets/composite_chunks.json

$PYBIN -m emnlp_evaluation.analysis.judge \
    --in "emnlp_evaluation/results/multihop_qwen_extbase_rejudge.jsonl" \
    --judge qwen --chunk-text-db emnlp_evaluation/judge_assets/multihop_chunks.json
