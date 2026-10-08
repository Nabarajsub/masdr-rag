#!/bin/bash
#SBATCH --job-name=e7-masdr
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-a30,mb-h100,mb-a6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/e7_masdr_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/e7_masdr_%j.err

# E7 MASDR-RAG arm — multi-hop orchestrator (several LLM calls/query, slow),
# under the neutral orchestrator system prompt (set by --shared-answer-prompt).
# Separate job so the single-call table lands first.
source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs

run_and_judge () {
  local bench=$1 chunkdb=$2
  local out=emnlp_evaluation/results/${bench}_qwen_promptctl_masdr.jsonl
  $PYBIN -m emnlp_evaluation.runners.run_generic_corpus \
      --benchmark $bench --llm qwen --systems masdr_rag \
      --shared-answer-prompt --limit 500 --out $out
  $PYBIN -m emnlp_evaluation.analysis.judge \
      --in "$out" --judge qwen --chunk-text-db $chunkdb
}

run_and_judge wydot_hand emnlp_evaluation/judge_assets/wydotv3_chunks.json
run_and_judge composite  emnlp_evaluation/judge_assets/composite_chunks.json
run_and_judge multihop   emnlp_evaluation/judge_assets/multihop_chunks.json
run_and_judge mmlu_pro   emnlp_evaluation/judge_assets/mmlu_pro_chunks.json

echo "[e7-masdr] done"
