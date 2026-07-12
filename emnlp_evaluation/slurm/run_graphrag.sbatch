#!/bin/bash
#SBATCH --job-name=graphrag
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/graphrag_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/graphrag_%j.err

# GraphRAG-style ablation on WYDOT: k-means communities + Qwen
# summaries + 2-stage retrieval. Compares vs scoped retrieval to test
# whether community-summary retrieval can substitute for metadata-scoping.

source <DATA_ROOT>
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

$PYBIN -m emnlp_evaluation.scripts.graphrag_ablation
