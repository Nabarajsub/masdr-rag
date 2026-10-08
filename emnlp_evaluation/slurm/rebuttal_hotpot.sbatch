#!/bin/bash
#SBATCH --job-name=reb-hotpot
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/reb_hotpot_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/reb_hotpot_%j.err

# SYSTEMS env var picks the shard.
source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs

SYSTEMS="${SYSTEMS:-monolithic,regex_scoped}"
TAG=$(echo $SYSTEMS | tr ',' '-')
OUT=emnlp_evaluation/results/crag_qwen_fixedrouter_${TAG}.jsonl

$PYBIN -m emnlp_evaluation.runners.run_crag \
    --llm qwen --systems "$SYSTEMS" --out $OUT

$PYBIN -m emnlp_evaluation.analysis.judge \
    --in "$OUT" --judge qwen \
    --chunk-text-db emnlp_evaluation/judge_assets/crag_chunks.json
