#!/bin/bash
#SBATCH --job-name=e13-hp-singlecall
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-a30,mb-h100,mb-a6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/e13_hp_singlecall_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/e13_hp_singlecall_%j.err

# E13: HotpotQA-distractor (CRAG harness, n=2000) re-run under the SHARED
# corpus-neutral prompt + per-corpus router, fulfilling the Author-Response
# commitment that HotpotQA receive the same fix as the other corpora.
source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs
OUT=emnlp_evaluation/results/crag_qwen_promptctl_singlecall.jsonl
$PYBIN -m emnlp_evaluation.runners.run_crag \
    --llm qwen --systems "monolithic,regex_scoped,hybrid_routed" --shared-answer-prompt --out $OUT
$PYBIN -m emnlp_evaluation.analysis.judge --in "$OUT" --judge qwen --chunk-text-db emnlp_evaluation/judge_assets/crag_chunks.json
echo "[e13-singlecall] done"
