#!/bin/bash
#SBATCH --job-name=wydot-tsne
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/wydot_tsne_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/wydot_tsne_%j.err

source <DATA_ROOT>

$PYBIN -m emnlp_evaluation.scripts.make_tsne_wydot_figure
