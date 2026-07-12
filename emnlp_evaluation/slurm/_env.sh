#!/bin/bash
# Shared environment setup for all emnlp_evaluation SLURM jobs.
# Sourced by every sbatch script in this directory.

module load arcc/1.0 2>/dev/null || true
module load gcc/13.2.0 2>/dev/null || true
module load cuda-toolkit/12.4.1 2>/dev/null || true
module load miniconda3 2>/dev/null || true

export PYTHONNOUSERSITE=1
export REPO_DIR=<DATA_ROOT>
export PROJECT_DIR=$REPO_DIR/graph_processing
export PYTHONPATH=$REPO_DIR:$PROJECT_DIR:$PYTHONPATH

# HuggingFace cache + offline mode (compute nodes have no network).
export HF_HOME=<DATA_ROOT>
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Tokenizer + threading
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

# The qwen env that has transformers, sentence-transformers, neo4j.
export PYBIN=<DATA_ROOT>
# Put the env's bin/ on PATH so subprocess calls (ninja, etc.) find it.
export PATH=<DATA_ROOT>

cd $PROJECT_DIR

echo "[env] host=$(hostname) python=$($PYBIN --version 2>&1) cuda=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
