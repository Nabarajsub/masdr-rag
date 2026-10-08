#!/bin/bash
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-h100,mb-l40s,mb-a6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --job-name=e25-rq3
#SBATCH --output=emnlp_evaluation/slurm/logs/e25_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/e25_%j.err
# E25 (RQ3): hand vs discovered scopes under the SHARED prompt (the paper's number predates it),
# two sampled repeats each for R2-routed, plus soft scoping over discovered scopes.
source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs
set -e
J=emnlp_evaluation/judge_assets
V2="$PYBIN -m emnlp_evaluation.analysis.judge_v2"
Q=emnlp_evaluation/results/rq3
run () { local b=$1 o=$2; shift 2
  $PYBIN -m emnlp_evaluation.runners.run_generic_corpus --benchmark $b --llm qwen --shared-answer-prompt --out $o "$@"
  $PYBIN -m emnlp_evaluation.analysis.judge --in "$o" --judge qwen --chunk-text-db $J/wydotv3_chunks.json; }
for r in 1 2; do
  run wydot_hand $Q/hand_r2_rep$r.jsonl --systems monolithic,r2_routed
  run wydot_disc $Q/disc_r2_rep$r.jsonl --systems r2_routed
done
run wydot_disc $Q/disc_soft.jsonl --systems soft_scoped
echo "[e25] done"
