#!/bin/bash
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-h100,mb-l40s,mb-a6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --job-name=e24b-rq4
#SBATCH --output=emnlp_evaluation/slurm/logs/e24b_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/e24b_%j.err
# E24b (RQ4): same re-judge on MultiHop and MMLU-Pro (monolithic vs MASDR).
source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs
set -e
J=emnlp_evaluation/judge_assets
V2="$PYBIN -m emnlp_evaluation.analysis.judge_v2"
Q=emnlp_evaluation/results/rq4
$V2 --in $Q/multihop_single.judged.jsonl --systems monolithic --chunk-text-db $J/multihop_chunks.json
$V2 --in $Q/multihop_masdr.judged.jsonl --chunk-text-db $J/multihop_chunks.json
$V2 --in $Q/mmlu_single.judged.jsonl --systems monolithic --chunk-text-db $J/mmlu_pro_chunks.json
$V2 --in $Q/mmlu_masdr.judged.jsonl --chunk-text-db $J/mmlu_pro_chunks.json
echo "[e24b] done"
