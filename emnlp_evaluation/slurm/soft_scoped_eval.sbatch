#!/bin/bash
#SBATCH --job-name=soft-scoped
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-a30,mb-h100,mb-a6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/soft_scoped_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/soft_scoped_%j.err

# SOFT-SCOPED (final_v3 Task 4): top-2 centroid-routed scope union + global,
# RRF-fused, single synthesis. Run on the three contrast corpora:
#   multihop   (cross-scope queries — where hard scoping loses, mono 68.2)
#   composite  (no routing signal — where hard scoping loses, mono 80.0)
#   wydot_hand (organizational corpus — must keep WYDOT-class behavior)
# Monolithic baselines reused from the corrected *_fixedrouter_* judged runs.
source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs

run_and_judge () {
  local bench=$1 chunkdb=$2
  local out=emnlp_evaluation/results/${bench}_qwen_soft_scoped.jsonl
  $PYBIN -m emnlp_evaluation.runners.run_generic_corpus \
      --benchmark $bench --llm qwen --systems soft_scoped --out $out
  $PYBIN -m emnlp_evaluation.analysis.judge \
      --in "$out" --judge qwen --chunk-text-db $chunkdb
}

run_and_judge multihop    emnlp_evaluation/judge_assets/multihop_chunks.json
run_and_judge composite   emnlp_evaluation/judge_assets/composite_chunks.json
run_and_judge wydot_hand  emnlp_evaluation/judge_assets/wydotv3_chunks.json

echo "[soft-scoped] done"
