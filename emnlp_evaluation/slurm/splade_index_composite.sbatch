#!/bin/bash
#SBATCH --job-name=splade-comp
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/splade_comp_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/splade_comp_%j.err

source <DATA_ROOT>
$PYBIN <<'PYEOF'
from emnlp_evaluation.composite_corpus.splade_backend import build_corpus_splade
build_corpus_splade("emnlp_evaluation/composite_corpus/data/composite")
PYEOF
