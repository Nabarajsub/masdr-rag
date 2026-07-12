#!/bin/bash
#SBATCH --job-name=nq-faiss
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/nq_faiss_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/nq_faiss_%j.err

source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs

export HF_DATASETS_CACHE=<DATA_ROOT>

$PYBIN -m emnlp_evaluation.benchmark.nq.build_faiss
