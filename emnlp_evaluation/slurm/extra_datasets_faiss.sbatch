#!/bin/bash
#SBATCH --job-name=extra-faiss
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/extra_faiss_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/extra_faiss_%j.err

source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs
export HF_DATASETS_CACHE=<DATA_ROOT>

$PYBIN -m emnlp_evaluation.benchmark.financebench.build_faiss
$PYBIN -m emnlp_evaluation.benchmark.mmlu_pro.build_faiss
