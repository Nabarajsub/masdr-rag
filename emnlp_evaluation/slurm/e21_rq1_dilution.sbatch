#!/bin/bash
#SBATCH --job-name=e21-rq1
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-h100,mb-l40s,mb-a6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=6:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/e21_rq1_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/e21_rq1_%j.err

# E21 (RQ1 deep test). Part 1: retrieval-only dilution with 20 distractor draws,
# paired stats small(54) vs full(1,083). Part 2: end-to-end answer accuracy on the
# 32 gold-labelled queries, full corpus vs FIVE independent 54-doc draws, all
# generated and judged in this one job with identical code, prompt and judge.
source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs
set -e
R=emnlp_evaluation/results/rq1
mkdir -p $R
D=../data/wydot/derived

$PYBIN -m emnlp_evaluation.analysis.rq1_dilution_test retrieval \
    --prefix $D/wydotv3 --seeds 20 --out $R/rq1_retrieval.json

gen () {   # $1 = prefix, $2 = out
  $PYBIN -m emnlp_evaluation.runners.run_generic_corpus --benchmark wydot54 \
      --prefix "$1" --queries $D/rq1/wydot54_s0.queries.json \
      --llm qwen --systems monolithic --backend faiss --shared-answer-prompt --out "$2"
  $PYBIN -m emnlp_evaluation.analysis.judge --in "$2" --judge qwen \
      --chunk-text-db emnlp_evaluation/judge_assets/wydotv3_chunks.json
}
gen $D/wydotv3 $R/full_monolithic.jsonl
for s in 0 1 2 3 4; do gen $D/rq1/wydot54_s$s $R/small54_s${s}_monolithic.jsonl; done

$PYBIN -m emnlp_evaluation.analysis.rq1_dilution_test e2e \
    --full $R/full_monolithic.judged.jsonl \
    --small $R/small54_s{0,1,2,3,4}_monolithic.judged.jsonl \
    --out $R/rq1_e2e.json
echo "[e21] done"
