#!/bin/bash
#SBATCH --job-name=wydot-faiss
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/wydot_faiss_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/wydot_faiss_%j.err

# S-5: build a unified FAISS index for WYDOT. Read-only MATCH on prod Neo4j.

source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs

$PYBIN -m emnlp_evaluation.composite_corpus.build_wydot_faiss
