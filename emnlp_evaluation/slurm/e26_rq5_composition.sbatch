#!/bin/bash
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-h100,mb-l40s,mb-a6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --job-name=e26-rq5
#SBATCH --output=emnlp_evaluation/slurm/logs/e26_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/e26_%j.err
# E26 (RQ5): direct test of "cross-source synthesis". One synthesis call over exactly 15 chunks
# drawn from the top-1 / top-3 / top-5 centroid scopes; judged for correctness (judge.py) and
# faithfulness with full evidence (judge_v2).
source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs
set -e
J=emnlp_evaluation/judge_assets
V2="$PYBIN -m emnlp_evaluation.analysis.judge_v2"
Q=emnlp_evaluation/results/rq5
run () { local b=$1 db=$2 o=$3
  $PYBIN -m emnlp_evaluation.runners.run_generic_corpus --benchmark $b --llm qwen --shared-answer-prompt       --limit 500 --systems composed_1,composed_3,composed_5 --out $o
  $PYBIN -m emnlp_evaluation.analysis.judge --in "$o" --judge qwen --chunk-text-db $db
  $V2 --in "${o%.jsonl}.judged.jsonl" --chunk-text-db $db; }
run wydot_hand $J/wydotv3_chunks.json   $Q/wydot_composed.jsonl
run composite  $J/composite_chunks.json $Q/composite_composed.jsonl
run multihop   $J/multihop_chunks.json  $Q/multihop_composed.jsonl
echo "[e26] done"
