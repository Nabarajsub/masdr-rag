#!/bin/bash
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-h100,mb-l40s,mb-a6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --job-name=e24a-rq4
#SBATCH --output=emnlp_evaluation/slurm/logs/e24a_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/e24a_%j.err
# E24a (RQ4): faithfulness re-judged with the full evidence the generator saw + claim-level score.
source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs
set -e
J=emnlp_evaluation/judge_assets
V2="$PYBIN -m emnlp_evaluation.analysis.judge_v2"
Q=emnlp_evaluation/results/rq4
$V2 --in $Q/wydot_single.judged.jsonl --systems monolithic,soft_scoped --chunk-text-db $J/wydotv3_chunks.json
$V2 --in $Q/wydot_masdr.judged.jsonl --chunk-text-db $J/wydotv3_chunks.json
$V2 --in $Q/composite_single.judged.jsonl --systems monolithic --chunk-text-db $J/composite_chunks.json
$V2 --in $Q/composite_masdr.judged.jsonl --chunk-text-db $J/composite_chunks.json
$V2 --in $Q/composite_masdrsinglecall.judged.jsonl --chunk-text-db $J/composite_chunks.json
echo "[e24a] done"
