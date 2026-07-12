#!/bin/bash
#SBATCH --job-name=crag-llama
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/crag_llama_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/crag_llama_%j.err

source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs

SYSTEMS="${SYSTEMS:-monolithic,hybrid_routed,masdr_rag,react}"
LIMIT="${LIMIT:-}"
LIMIT_ARG=${LIMIT:+--limit $LIMIT}

$PYBIN -m emnlp_evaluation.runners.run_crag \
    --llm llama --systems "$SYSTEMS" $LIMIT_ARG
