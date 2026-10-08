#!/bin/bash
#SBATCH --job-name=e8-granularity
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=teton
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/e8_granularity_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/e8_granularity_%j.err

# E8 fully-powered: chunk-level cluster ↔ axis alignment (NMI) for the 5 DOTs
# not yet run (wydot + caltrans already done). Tests whether NMI(unsupervised
# discovery, document_series) tracks whether the document_series axis resolves
# dilution — the retrieval-free "right-granularity" diagnostic.
source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs
export OMP_NUM_THREADS=16

for dot in cdot mndot odot txdot fdot; do
    echo "########## $dot ##########"
    $PYBIN -m graph_processing.multi_dot.adaptive_granularity \
        --corpus $dot --k-sweep 20 500 --steps 8
done
echo "[e8-granularity] done"
