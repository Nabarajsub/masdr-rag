#!/bin/bash
#SBATCH --job-name=generic-eval
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/generic_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/generic_%j.err

source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs

BENCH="${BENCH:-financebench}"
LIMIT="${LIMIT:-500}"
LLM="${LLM:-qwen}"

$PYBIN -m emnlp_evaluation.runners.run_generic_corpus \
    --benchmark "$BENCH" \
    --llm "$LLM" \
    --systems monolithic,regex_scoped,hybrid_routed,masdr_rag \
    --limit "$LIMIT"
