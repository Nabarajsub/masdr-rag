#!/bin/bash
#SBATCH --job-name=e11-extbase
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-a30,mb-h100,mb-a6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/e11_extbase_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/e11_extbase_%j.err

# E11: re-run the external baselines on Composite-9 under the SHARED corpus-neutral
# prompt, so the appendix baseline table uses the same protocol as the main table.
#
# Scope note (documented in the paper): the shared-prompt control applies to systems
# that differ only in RETRIEVAL (BM25/ColBERT swaps) or that carry a corpus persona
# we authored (ReAct). MA-RAG and SCOUT-RAG prompts are ported verbatim from their
# source papers -- overriding them would break implementation fidelity, so they are
# equalised at the JUDGE instead and are deliberately NOT re-run here.
source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs

CHUNKDB=emnlp_evaluation/judge_assets/composite_chunks.json

# 1. ReAct (persona-neutral system prompt, identical loop protocol)
OUT_RE=emnlp_evaluation/results/composite_qwen_promptctl_react.jsonl
$PYBIN -m emnlp_evaluation.runners.run_composite \
    --llm qwen --systems react \
    --shared-answer-prompt --out $OUT_RE
$PYBIN -m emnlp_evaluation.analysis.judge --in "$OUT_RE" --judge qwen --chunk-text-db $CHUNKDB

# 2. BM25 retrieval swap
OUT_BM=emnlp_evaluation/results/composite_qwen_promptctl_bm25.jsonl
$PYBIN -m emnlp_evaluation.runners.run_bm25_baseline \
    --benchmark composite --llm qwen --systems bm25_qwen \
    --shared-answer-prompt --out $OUT_BM
$PYBIN -m emnlp_evaluation.analysis.judge --in "$OUT_BM" --judge qwen --chunk-text-db $CHUNKDB

# 3. ColBERTv2 retrieval swap (unscoped + scoped)
OUT_CB=emnlp_evaluation/results/composite_qwen_promptctl_colbert.jsonl
$PYBIN -m emnlp_evaluation.runners.run_colbert_baseline \
    --benchmark composite --llm qwen --systems colbert_qwen \
    --shared-answer-prompt --out $OUT_CB
$PYBIN -m emnlp_evaluation.analysis.judge --in "$OUT_CB" --judge qwen --chunk-text-db $CHUNKDB

echo "[e11-extbase] done"
