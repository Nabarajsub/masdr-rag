#!/bin/bash
#SBATCH --job-name=e14-hp-extbase
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-a30,mb-h100,mb-a6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=16:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/e14_hp_extbase_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/e14_hp_extbase_%j.err

# E14: re-run the HotpotQA-distractor retrieval-swap baselines under the SHARED
# corpus-neutral prompt, so tab:hotpot_emf1 is protocol-matched to the E13 rows
# rather than mixing prompts within one block.
#
# Scope note (mirrors the E11 rationale for Composite-9): the shared-prompt
# control applies to systems that differ only in RETRIEVAL (BM25/ColBERT swaps).
# LangChain is deliberately NOT re-run -- its prompts are internal to the
# framework, so overriding them would break implementation fidelity; it stays in
# the original-protocol block of the table.
source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs

CHUNKDB=emnlp_evaluation/judge_assets/crag_chunks.json

# 1. BM25 (retrieval-only + Qwen generation)
OUT_BM=emnlp_evaluation/results/crag_qwen_promptctl_bm25.jsonl
$PYBIN -m emnlp_evaluation.runners.run_bm25_baseline \
    --benchmark crag --llm qwen --systems bm25_only,bm25_qwen \
    --shared-answer-prompt --out $OUT_BM
$PYBIN -m emnlp_evaluation.analysis.judge --in "$OUT_BM" --judge qwen --chunk-text-db $CHUNKDB

# 2. ColBERTv2 (retrieval-only + Qwen generation)
OUT_CB=emnlp_evaluation/results/crag_qwen_promptctl_colbert.jsonl
$PYBIN -m emnlp_evaluation.runners.run_colbert_baseline \
    --benchmark crag --llm qwen --systems colbert_only,colbert_qwen \
    --shared-answer-prompt --out $OUT_CB
$PYBIN -m emnlp_evaluation.analysis.judge --in "$OUT_CB" --judge qwen --chunk-text-db $CHUNKDB

echo "[e14-extbase] done"
