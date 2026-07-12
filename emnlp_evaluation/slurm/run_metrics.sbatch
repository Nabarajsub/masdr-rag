#!/bin/bash
#SBATCH --job-name=metrics-final
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=teton
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/metrics_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/metrics_%j.err

source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs
$PYBIN -m emnlp_evaluation.analysis.compile_metrics
