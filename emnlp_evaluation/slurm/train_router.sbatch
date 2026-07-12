#!/bin/bash
#SBATCH --job-name=train-router
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=emnlp_evaluation/slurm/logs/train_router_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/train_router_%j.err

# G-2: trained domain router baselines.
#   T1 = TF-IDF + logistic regression (CPU)
#   T2 = BGE-M3 linear probe (needs GPU for the embedding pass)
# Reports 5-fold CV accuracy/F1 against the regex baseline + zero-shot LLM.

source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs

$PYBIN -m emnlp_evaluation.router.train_classifier
