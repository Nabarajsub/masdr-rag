#!/bin/bash
#SBATCH --job-name=reb-e1-composite
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/reb_e1_composite_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/reb_e1_composite_%j.err

source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs

OUT=emnlp_evaluation/results/composite_qwen_fixedrouter_monolithic-regex_scoped-hybrid_routed-masdr_rag.jsonl

$PYBIN -m emnlp_evaluation.runners.run_composite \
    --llm qwen --systems monolithic,regex_scoped,hybrid_routed,masdr_rag \
    --out $OUT

$PYBIN -m emnlp_evaluation.analysis.judge \
    --in "$OUT" --judge qwen \
    --chunk-text-db emnlp_evaluation/judge_assets/composite_chunks.json
