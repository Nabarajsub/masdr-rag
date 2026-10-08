#!/bin/bash
#SBATCH --job-name=e23-rq2
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-h100,mb-l40s,mb-a6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/e23_rq2_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/e23_rq2_%j.err

# E23 (RQ2). Generation samples at T=0.2 with no seed, so a single run carries
# sampling noise (7.5% correctness flips on identical contexts, WYDOT E7).
#  (a) Composite-9 monolithic vs soft_scoped: 4 more sampled repeats + 1 greedy run
#  (b) WYDOT hybrid_routed with the case-insensitive router fix
source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs
set -e
R=emnlp_evaluation/results/rq2
J=emnlp_evaluation/judge_assets
run () {  # bench out chunkdb extra-args...
  local bench=$1 out=$2 db=$3; shift 3
  $PYBIN -m emnlp_evaluation.runners.run_generic_corpus --benchmark $bench --llm qwen \
      --shared-answer-prompt --limit 500 --out $out "$@"
  $PYBIN -m emnlp_evaluation.analysis.judge --in "$out" --judge qwen --chunk-text-db $db
}
for r in 1 2 3 4; do
  run composite $R/composite_monosoft_rep$r.jsonl $J/composite_chunks.json --systems monolithic,soft_scoped
done
GEN_TEMPERATURE=0 run composite $R/composite_monosoft_greedy.jsonl $J/composite_chunks.json --systems monolithic,soft_scoped
run wydot_hand $R/wydot_hybrid_routerfix.jsonl $J/wydotv3_chunks.json --systems hybrid_routed
echo "[e23] done"
