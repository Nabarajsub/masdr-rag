#!/bin/bash
#SBATCH --job-name=beir-msmarco
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/beir_msmarco_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/beir_msmarco_%j.err

source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs

# Datasets already cached on login node; keep offline mode set so we use the cache.
# (Compute nodes typically have no outbound network anyway.)
export HF_DATASETS_CACHE=<DATA_ROOT>

$PYBIN -m emnlp_evaluation.benchmark.beir.run_msmarco --limit 1000 --limit-corpus 200000
