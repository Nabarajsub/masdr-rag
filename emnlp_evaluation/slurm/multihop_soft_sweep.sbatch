#!/bin/bash
#SBATCH --job-name=mh-soft-sweep
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-a30,mb-h100,mb-a6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=10:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/mh_soft_sweep_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/mh_soft_sweep_%j.err

# MultiHop-targeted soft-scoping sweep (final_v3 Task 4b), first-500 protocol:
#   config A: m=3, global weight 1  (wider scope union)
#   config B: m=2, global weight 2  (global-dominant fusion)
# Baselines for comparison: mono .682, soft(m=2,gw=1) .588.
source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs

run_cfg () {
  local tag=$1 m=$2 gw=$3
  local out=emnlp_evaluation/results/multihop_qwen_soft_${tag}.jsonl
  $PYBIN -m emnlp_evaluation.runners.run_generic_corpus \
      --benchmark multihop --llm qwen --systems soft_scoped \
      --soft-m $m --soft-global-weight $gw --limit 500 --out $out
  $PYBIN -m emnlp_evaluation.analysis.judge \
      --in "$out" --judge qwen \
      --chunk-text-db emnlp_evaluation/judge_assets/multihop_chunks.json
}

run_cfg m3gw1 3 1.0
run_cfg m2gw2 2 2.0

echo "[mh-soft-sweep] done"
