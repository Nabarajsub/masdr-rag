#!/bin/bash
#SBATCH --job-name=colbert-baseline
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/colbert_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/colbert_%j.err

source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs emnlp_evaluation/results

# Keep offline mode on; the backend points at a local snapshot path.
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

BM=${BM:-composite}
$PYBIN -m emnlp_evaluation.runners.run_colbert_baseline --benchmark $BM
