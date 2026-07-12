#!/bin/bash
#SBATCH --job-name=rerank-all
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/rerank_all_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/rerank_all_%j.err

# Cross-encoder (BGE-reranker-v2-m3) rerank ablation across all corpora.
# Compares monolithic + hybrid_routed (clean single-call paths) WITH and
# WITHOUT reranking. Reranker keeps top-30 bi-encoder candidates and
# returns top-10 by cross-encoder score.

source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs

CORPUS="${CORPUS:-multihop}"
LIMIT="${LIMIT:-500}"

if [[ "$CORPUS" == "wydot" ]]; then
    $PYBIN -m emnlp_evaluation.runners.run_wydot_oss \
        --llm qwen --embedder bge_m3 --rerank \
        --systems monolithic,regex_scoped,hybrid_routed \
        --limit "$LIMIT"
else
    $PYBIN -m emnlp_evaluation.runners.run_generic_corpus \
        --benchmark "$CORPUS" --llm qwen --rerank \
        --systems monolithic,hybrid_routed \
        --limit "$LIMIT"
fi
