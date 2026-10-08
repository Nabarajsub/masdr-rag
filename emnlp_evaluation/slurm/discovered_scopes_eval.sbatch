#!/bin/bash
#SBATCH --job-name=disc-scopes
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/disc_scopes_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/disc_scopes_%j.err

# Hand-vs-discovered scope experiment (final_v3):
#   same store (wydotv3 FAISS, BGE-M3, 217,752 chunks), same 200 queries,
#   only the scope taxonomy differs. Monolithic is taxonomy-independent and
#   runs once, in the hand arm.
source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs

OUT_HAND=emnlp_evaluation/results/wydot_hand_qwen_monolithic-r2_routed.jsonl
OUT_DISC=emnlp_evaluation/results/wydot_disc_qwen_r2_routed.jsonl

$PYBIN -m emnlp_evaluation.runners.run_generic_corpus \
    --benchmark wydot_hand --llm qwen \
    --systems monolithic,r2_routed --out $OUT_HAND

$PYBIN -m emnlp_evaluation.runners.run_generic_corpus \
    --benchmark wydot_disc --llm qwen \
    --systems r2_routed --out $OUT_DISC

$PYBIN -m emnlp_evaluation.analysis.judge \
    --in "$OUT_HAND" --judge qwen \
    --chunk-text-db emnlp_evaluation/judge_assets/wydotv3_chunks.json

$PYBIN -m emnlp_evaluation.analysis.judge \
    --in "$OUT_DISC" --judge qwen \
    --chunk-text-db emnlp_evaluation/judge_assets/wydotv3_chunks.json

echo "[disc-scopes] done"
