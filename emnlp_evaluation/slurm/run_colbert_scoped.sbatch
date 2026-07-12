#!/bin/bash
#SBATCH --job-name=colbert-scoped
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/colbert_scoped_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/colbert_scoped_%j.err

source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs

$PYBIN -m emnlp_evaluation.runners.run_colbert_baseline \
    --benchmark composite \
    --systems colbert_scoped,colbert_scoped_qwen \
    --out emnlp_evaluation/results/composite_colbert_scoped_qwen.jsonl
