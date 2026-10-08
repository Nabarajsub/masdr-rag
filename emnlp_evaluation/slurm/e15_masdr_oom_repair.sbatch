#!/bin/bash
#SBATCH --job-name=e15-masdr-repair
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-h100,mb-l40s,mb-a6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=20:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/e15_masdr_repair_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/e15_masdr_repair_%j.err

# E15: repair the MASDR arms of the shared-prompt cross-corpus table.
#
# The original E7 run used --partition mb-a30,... --mem 80G. On the A30 (23.6GB)
# the orchestrator OOM'd and wrote empty answers, and the judge then scored only
# the survivors:
#     multihop  90/500 lost (18%)  -> reported n=410, caption claims n=500
#     wydot    12/200 lost ( 6%)  -> reported n=188, caption claims n=200
#     mmlu_pro  6/500 lost ( 1%)  -> reported n=494, caption claims n=500
#     composite 0/225 lost        -> clean, NOT re-run here
#
# The dropout is non-random (OOM tracks prompt length, and MASDR has the widest
# context of any system), so the survivor-only means are not interchangeable
# with the n=500/n=200 baselines they are tabulated against. On WYDOT the
# denominator decides the direction of the MASDR-vs-Monolithic comparison
# (.330 vs .315 judged-only; .310 vs .315 over the full set).
#
# Re-run on an H100 (80GB) to NEW filenames so the originals survive for
# comparison; merge/compare happens after the job lands.
source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs

run_and_judge () {
  local bench=$1 chunkdb=$2
  local out=emnlp_evaluation/results/${bench}_qwen_promptctl_masdr_h100.jsonl
  $PYBIN -m emnlp_evaluation.runners.run_generic_corpus \
      --benchmark $bench --llm qwen --systems masdr_rag \
      --shared-answer-prompt --limit 500 --out $out
  $PYBIN -m emnlp_evaluation.analysis.judge \
      --in "$out" --judge qwen --chunk-text-db $chunkdb
}

run_and_judge multihop   emnlp_evaluation/judge_assets/multihop_chunks.json
run_and_judge wydot_hand emnlp_evaluation/judge_assets/wydotv3_chunks.json
run_and_judge mmlu_pro   emnlp_evaluation/judge_assets/mmlu_pro_chunks.json

echo "[e15-masdr-repair] done"
