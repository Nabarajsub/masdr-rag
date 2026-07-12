#!/bin/bash
#SBATCH --job-name=langchain-composite
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/langchain_%A_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/langchain_%A_%j.err

source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs emnlp_evaluation/results

BM=${BM:-composite}
$PYBIN -m emnlp_evaluation.runners.run_langchain_composite --benchmark $BM
