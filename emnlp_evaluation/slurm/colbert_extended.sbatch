#!/bin/bash
#SBATCH --job-name=colbert-ext
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/colbert_ext_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/colbert_ext_%j.err

source <DATA_ROOT>
BENCH="${BENCH:-multihop}"
LIMIT="${LIMIT:-500}"
$PYBIN -m emnlp_evaluation.runners.run_colbert_baseline \
    --benchmark "$BENCH" \
    --systems colbert_only,colbert_qwen,colbert_scoped,colbert_scoped_qwen \
    --limit "$LIMIT"
