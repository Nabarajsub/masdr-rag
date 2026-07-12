#!/bin/bash
# Fan out the 200-query WYDOT evaluation across whatever GPUs are free.
#
# Strategy: 4 shards of Qwen on mb-l40s, 4 shards of Llama on mb-a30 or mb-h100.
# That's 8 parallel GPU jobs => ~8x speedup over a single full run.
#
# Usage:
#   bash emnlp_evaluation/slurm/submit_fanout.sh
#
# To dry-run (print sbatch commands only):
#   DRY=1 bash emnlp_evaluation/slurm/submit_fanout.sh
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> graph_processing/

submit() {
    local desc="$1"; shift
    if [[ "${DRY:-}" = "1" ]]; then
        echo "[dry] $desc :: sbatch $*"
    else
        echo "[submit] $desc"
        sbatch "$@"
    fi
}

# Qwen × 4 shards on mb-l40s
submit "qwen × 4 shards (mb-l40s)" \
    --partition=mb-l40s \
    --array=0-3 \
    --export=ALL,ARRAY_SIZE=4 \
    emnlp_evaluation/slurm/run_qwen_wydot_shard.sbatch

# Llama × 4 shards — prefer mb-a30 (has an idle node), fall back to mb-h100.
LLAMA_PART="${LLAMA_PART:-mb-a30}"
submit "llama × 4 shards ($LLAMA_PART)" \
    --partition=$LLAMA_PART \
    --array=0-3 \
    --export=ALL,ARRAY_SIZE=4 \
    emnlp_evaluation/slurm/run_llama_wydot_shard.sbatch

echo "[done] fan-out submitted. Tail logs with:"
echo "       tail -F emnlp_evaluation/slurm/logs/*shard_*.out"
echo "       and watch queue: squeue -u \$USER"
