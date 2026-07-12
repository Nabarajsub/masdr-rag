#!/bin/bash
#SBATCH --job-name=or-wydot
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=teton
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/or_wydot_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/or_wydot_%j.err

# 3rd-backbone WYDOT run via OpenRouter. CPU-only (no local model load),
# but still hits production Neo4j READ for retrieval and BGE-M3 on the
# embedder side (need GPU? embedder is on CPU when BGE_M3_USE_GPU=0).

source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs

MODEL="${OR_MODEL:-anthropic/claude-haiku-4.5}"
LIMIT="${OR_LIMIT:-100}"

$PYBIN -m emnlp_evaluation.runners.run_wydot_oss \
    --llm "openrouter:${MODEL}" \
    --embedder bge_m3 \
    --systems monolithic,regex_scoped,hybrid_routed,masdr_rag \
    --limit "$LIMIT" \
    --resume
