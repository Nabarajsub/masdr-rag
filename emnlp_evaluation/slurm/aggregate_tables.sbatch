#!/bin/bash
#SBATCH --job-name=aggregate
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=teton
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=emnlp_evaluation/slurm/logs/aggregate_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/aggregate_%j.err

source <DATA_ROOT>

echo "[aggregate] paradox"
$PYBIN -m emnlp_evaluation.analysis.aggregate_paradox || true
echo "[aggregate] rerank+neo4j"
$PYBIN -m emnlp_evaluation.analysis.aggregate_rerank_and_neo4j || true
echo "[aggregate] bootstrap backbones"
$PYBIN -m emnlp_evaluation.analysis.bootstrap_backbones || true
echo "[aggregate] done"
