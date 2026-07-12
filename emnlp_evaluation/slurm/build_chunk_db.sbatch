#!/bin/bash
#SBATCH --job-name=build-chunk-db
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=teton
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=emnlp_evaluation/slurm/logs/build_chunk_db_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/build_chunk_db_%j.err

source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs emnlp_evaluation/judge_assets

# Composite + CRAG: offline from parquet, fast.
$PYBIN -m emnlp_evaluation.analysis.build_chunk_text_db \
    --corpus composite \
    --out emnlp_evaluation/judge_assets/composite_chunks.json

$PYBIN -m emnlp_evaluation.analysis.build_chunk_text_db \
    --corpus crag \
    --out emnlp_evaluation/judge_assets/crag_chunks.json

# WYDOT: read-only MATCH against production Neo4j. No writes.
$PYBIN -m emnlp_evaluation.analysis.build_chunk_text_db \
    --corpus wydot \
    --out emnlp_evaluation/judge_assets/wydot_chunks.json
