#!/bin/bash
#SBATCH --job-name=reb-e1-generic
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/reb_e1_generic_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/reb_e1_generic_%j.err

# BENCHMARK env var selects multihop or mmlu_pro; LIMIT caps queries.
source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs

BENCHMARK="${BENCHMARK:-multihop}"
LIMIT="${LIMIT:-500}"
OUT=emnlp_evaluation/results/${BENCHMARK}_qwen_fixedrouter_monolithic-regex_scoped-hybrid_routed-masdr_rag.jsonl

$PYBIN -m emnlp_evaluation.runners.run_generic_corpus \
    --benchmark $BENCHMARK --llm qwen \
    --systems monolithic,regex_scoped,hybrid_routed,masdr_rag \
    --limit $LIMIT --out $OUT

$PYBIN -m emnlp_evaluation.analysis.judge \
    --in "$OUT" --judge qwen \
    --chunk-text-db emnlp_evaluation/judge_assets/${BENCHMARK}_chunks.json
