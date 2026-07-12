#!/bin/bash
#SBATCH --job-name=composite-llama
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/composite_llama_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/composite_llama_%j.err

source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs

SYSTEMS="${SYSTEMS:-monolithic,regex_scoped,hybrid_routed,masdr_rag,react}"
LIMIT="${LIMIT:-}"
LIMIT_ARG=${LIMIT:+--limit $LIMIT}

$PYBIN -m emnlp_evaluation.runners.run_composite \
    --llm llama --systems "$SYSTEMS" $LIMIT_ARG
