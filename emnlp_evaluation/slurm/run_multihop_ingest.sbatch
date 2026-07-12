#!/bin/bash
#SBATCH --job-name=mh-ingest
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/mh_ingest_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/mh_ingest_%j.err

source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs

# Run download.py from a login node BEFORE submitting this job — compute nodes
# usually have no outbound network.
$PYBIN -m emnlp_evaluation.benchmark.multihop_rag.ingest
