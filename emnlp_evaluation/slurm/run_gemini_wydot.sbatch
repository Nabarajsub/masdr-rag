#!/bin/bash
#SBATCH --job-name=wydot-gemini
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=teton
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/gemini_wydot_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/gemini_wydot_%j.err

# Gemini runs are network-bound, not GPU-bound — keep them on a CPU partition.
source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs

SYSTEMS="${SYSTEMS:-monolithic,regex_scoped,hybrid_routed,masdr_rag,react}"
EMBEDDER="${EMBEDDER:-gemini}"
EXTRA="${EXTRA:---resume}"

$PYBIN -m emnlp_evaluation.runners.run_wydot_oss \
    --llm gemini \
    --embedder "$EMBEDDER" \
    --systems "$SYSTEMS" \
    $EXTRA
