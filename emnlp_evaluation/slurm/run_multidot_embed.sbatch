#!/bin/bash
#SBATCH --job-name=multidot-embed
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/multidot_embed_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/multidot_embed_%j.err

# Phase 2b: Qwen document_series classification + BGE-M3 chunk embedding for
# each DOT corpus. Produces data/<dot>/derived/<dot>.meta.parquet + .npy.
#
# Submit:   sbatch graph_processing/emnlp_evaluation/slurm/run_multidot_embed.sbatch
#   override DOTS:   DOTS="cdot caltrans wydot" sbatch ... run_multidot_embed.sbatch
# WYDOT is embedded with --skip-classify (document_series already authoritative).

source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs

DOTS="${DOTS:-cdot caltrans}"
SKIP_CLASSIFY="${SKIP_CLASSIFY:-}"   # set to "all" for embed-only (no Qwen)
echo "[multidot-embed] DOTS=$DOTS  SKIP_CLASSIFY=$SKIP_CLASSIFY"

for dot in $DOTS; do
    echo "=================  embed: $dot  ================="
    EXTRA=""
    if [ "$dot" = "wydot" ] || [ "$SKIP_CLASSIFY" = "all" ]; then
        EXTRA="--skip-classify"
    fi
    $PYBIN -m graph_processing.multi_dot.classify_embed --dot "$dot" $EXTRA \
        --batch-size "${BATCH:-16}" --max-seq-len "${MAX_SEQ:-512}" \
        --dtype "${DTYPE:-bf16}" \
        || { echo "[multidot-embed] FAILED on $dot"; exit 1; }
done

echo "[multidot-embed] all done: $DOTS"
