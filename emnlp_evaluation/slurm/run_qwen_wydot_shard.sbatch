#!/bin/bash
#SBATCH --job-name=wydot-qwen-shard
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/qwen_shard_%A_%a.out
#SBATCH --error=emnlp_evaluation/slurm/logs/qwen_shard_%A_%a.err

# Submit with e.g.:
#   sbatch --partition=mb-l40s --array=0-3 \
#          emnlp_evaluation/slurm/run_qwen_wydot_shard.sbatch
# That gives 4 parallel shards, each handling ~50 of the 200 queries.

source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs emnlp_evaluation/results

SHARD_IDX=${SLURM_ARRAY_TASK_ID:-0}
N_SHARDS=${SLURM_ARRAY_TASK_COUNT:-${ARRAY_SIZE:-4}}
SYSTEMS="${SYSTEMS:-monolithic,regex_scoped,hybrid_routed,masdr_rag,react}"
EMBEDDER="${EMBEDDER:-bge_m3}"

echo "[shard] idx=$SHARD_IDX of=$N_SHARDS systems=$SYSTEMS"

$PYBIN -m emnlp_evaluation.runners.run_wydot_oss \
    --llm qwen \
    --embedder "$EMBEDDER" \
    --systems "$SYSTEMS" \
    --shard $SHARD_IDX --of $N_SHARDS \
    --resume
