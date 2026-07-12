#!/bin/bash
#SBATCH --job-name=composite-embed
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/composite_embed_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/composite_embed_%j.err

source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs

$PYBIN -m emnlp_evaluation.composite_corpus.embed_and_index --batch-size 96
