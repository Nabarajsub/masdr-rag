#!/bin/bash
#SBATCH --job-name=composite-querygen
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/composite_querygen_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/composite_querygen_%j.err

source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs

PER_SOURCE="${PER_SOURCE:-10}"
$PYBIN -m emnlp_evaluation.composite_corpus.query_gen --per-source $PER_SOURCE
