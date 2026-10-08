#!/bin/bash
#SBATCH --job-name=e22-rq2
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-h100,mb-l40s,mb-a6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/e22_rq2_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/e22_rq2_%j.err

# E22 (RQ2). Shared prompt, Qwen-2.5-7B synth + judge, BGE-M3, FAISS -- identical
# to E7 so every new arm pairs with the E7 rows query-by-query.
#  (a) ORACLE hard scoping (gold scope given) on Composite-9 / MMLU-Pro / WYDOT
#  (b) soft-scoping sensitivity on Composite-9: m=1, m=3, and m=2 with NO global list
#  (c) cross-family judge (Llama-3-8B) of the E7 Composite-9 single-call answers
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
run composite  $R/composite_oracle.jsonl  $J/composite_chunks.json --systems oracle_scoped --oracle-field gold_source_type
run mmlu_pro   $R/mmlu_oracle.jsonl       $J/mmlu_pro_chunks.json  --systems oracle_scoped --oracle-field category
run wydot_hand $R/wydot_oracle.jsonl      $J/wydotv3_chunks.json   --systems oracle_scoped --oracle-field gold_hand_scope
run composite  $R/composite_soft_m1.jsonl   $J/composite_chunks.json --systems soft_scoped --soft-m 1
run composite  $R/composite_soft_m3.jsonl   $J/composite_chunks.json --systems soft_scoped --soft-m 3
run composite  $R/composite_soft_m2_gw0.jsonl $J/composite_chunks.json --systems soft_scoped --soft-m 2 --soft-global-weight 0
$PYBIN -m emnlp_evaluation.analysis.judge --in $R/composite_singlecall_llamajudge.jsonl --judge llama \
    --chunk-text-db $J/composite_chunks.json
echo "[e22] done"
