#!/bin/bash
#SBATCH --job-name=wydot-qwen
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/qwen_wydot_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/qwen_wydot_%j.err

source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs

SYSTEMS="${SYSTEMS:-monolithic,regex_scoped,hybrid_routed,masdr_rag,react}"
EMBEDDER="${EMBEDDER:-bge_m3}"
EXTRA="${EXTRA:---resume}"

$PYBIN -m emnlp_evaluation.runners.run_wydot_oss \
    --llm qwen \
    --embedder "$EMBEDDER" \
    --systems "$SYSTEMS" \
    $EXTRA
