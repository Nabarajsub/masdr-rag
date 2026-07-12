#!/bin/bash
#SBATCH --job-name=masdr-singlecall
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/masdr_singlecall_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/masdr_singlecall_%j.err

# S-2: concat-all-fragments ablation.
# Same routing + scoped retrieval as MASDR-RAG, but every chunk lands in a
# single context for one synthesis call (no multi-round observation loop).
# Run on WYDOT 200 (Qwen stack) + Composite-9. If faithfulness > MASDR-RAG
# AND >= Hybrid-Routed, this is direct evidence that context fragmentation
# is the cause of the precision-faithfulness paradox.

source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs

CORPUS="${CORPUS:-composite}"
LLM="${LLM:-qwen}"

if [[ "$CORPUS" == "composite" ]]; then
    $PYBIN -m emnlp_evaluation.runners.run_composite \
        --llm "$LLM" \
        --systems masdr_singlecall \
        --out emnlp_evaluation/results/composite_${LLM}_masdr_singlecall.jsonl
elif [[ "$CORPUS" == "wydot" ]]; then
    $PYBIN -m emnlp_evaluation.runners.run_wydot_oss \
        --llm "$LLM" \
        --embedder bge_m3 \
        --systems masdr_singlecall \
        --out emnlp_evaluation/results/wydot_${LLM}_bge_m3_masdr_singlecall.jsonl
else
    echo "Unknown CORPUS=$CORPUS"; exit 1
fi
