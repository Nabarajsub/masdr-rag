#!/bin/bash
#SBATCH --job-name=e19-dilution-curve
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-h100,mb-l40s,mb-a6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=2:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/e19_dilution_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/e19_dilution_%j.err

# E19: causal dilution curve. Fixed 32-query set with fixed gold documents;
# only the distractor pool grows. Retrieval-only, so no LLM calls.
source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs
$PYBIN -m emnlp_evaluation.analysis.dilution_scaling_curve \
    --prefix ../data/wydot/derived/wydotv3 \
    --mode ${MODE:-doc} --retrieval ${RETR:-global} --sizes 16,54,100,200,400,700,1083 --seeds 5 \
    --out emnlp_evaluation/results/dilution_scaling_wydot_${MODE:-doc}_${RETR:-global}.json
echo "[e19] done"
