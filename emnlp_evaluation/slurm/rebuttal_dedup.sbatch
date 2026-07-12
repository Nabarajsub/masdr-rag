#!/bin/bash
#SBATCH --job-name=reb-dedup
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/reb_dedup_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/reb_dedup_%j.err

source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs

DB=emnlp_evaluation/judge_assets/wydot_chunks.json

$PYBIN -m emnlp_evaluation.runners.rerun_masdr_forced_format \
    --in emnlp_evaluation/results/wydot_qwen_bge_m3_baselines.jsonl \
    --chunk-db $DB --llm qwen --dedup \
    --out emnlp_evaluation/results/wydot_qwen_bge_m3_masdr_ff_dedup.jsonl

for S in 0 1 2 3; do
  $PYBIN -m emnlp_evaluation.runners.rerun_masdr_forced_format \
    --in emnlp_evaluation/results/wydot_qwen_gemini_monolithic-regex_scoped-hybrid_routed-masdr_rag-react_shard${S}of4.jsonl \
    --chunk-db $DB --llm qwen --dedup \
    --out emnlp_evaluation/results/wydot_qwen_gemini_masdr_ff_dedup.jsonl
done

$PYBIN -m emnlp_evaluation.analysis.judge \
    --in "emnlp_evaluation/results/wydot_qwen_*_masdr_ff_dedup.jsonl" \
    --judge qwen --chunk-text-db $DB
