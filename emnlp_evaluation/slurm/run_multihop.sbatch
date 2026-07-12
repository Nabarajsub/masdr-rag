#!/bin/bash
#SBATCH --job-name=multihop-eval
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/multihop_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/multihop_%j.err

source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs

LLM="${LLM:-qwen}"
LIMIT="${LIMIT:-500}"

$PYBIN -m emnlp_evaluation.runners.run_multihop \
    --llm "$LLM" \
    --systems monolithic,regex_scoped,hybrid_routed,masdr_rag \
    --limit "$LIMIT"
