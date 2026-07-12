#!/bin/bash
#SBATCH --job-name=bge-m3-reindex
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/reindex_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/reindex_%j.err

source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs

$PYBIN -m emnlp_evaluation.embeddings.reindex_neo4j --batch-size 64
