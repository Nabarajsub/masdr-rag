#!/bin/bash
#SBATCH --job-name=mhrag-faiss
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/mhrag_faiss_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/mhrag_faiss_%j.err

source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs

$PYBIN -m emnlp_evaluation.benchmark.multihop_rag.build_faiss
