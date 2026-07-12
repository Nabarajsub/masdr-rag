#!/bin/bash
#SBATCH --job-name=bm25-baseline
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/bm25_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/bm25_%j.err

source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs emnlp_evaluation/results

BM=${BM:-composite}
$PYBIN -m emnlp_evaluation.runners.run_bm25_baseline --benchmark $BM
