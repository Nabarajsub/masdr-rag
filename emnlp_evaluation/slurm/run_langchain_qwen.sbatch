#!/bin/bash
#SBATCH --job-name=langchain-qwen
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/langchain_qwen_%A_%a.out
#SBATCH --error=emnlp_evaluation/slurm/logs/langchain_qwen_%A_%a.err

source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs emnlp_evaluation/results

SHARD_IDX=${SLURM_ARRAY_TASK_ID:-0}
N_SHARDS=${SLURM_ARRAY_TASK_COUNT:-${ARRAY_SIZE:-4}}
EMBEDDER="${EMBEDDER:-bge_m3}"

$PYBIN -m emnlp_evaluation.runners.run_langchain_baseline \
    --llm qwen --embedder "$EMBEDDER" \
    --shard $SHARD_IDX --of $N_SHARDS
