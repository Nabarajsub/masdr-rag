#!/bin/bash
#SBATCH --job-name=splade-eval
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/splade_eval_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/splade_eval_%j.err

# Run monolithic + hybrid_routed on the sparse SPLADE index for one corpus.
source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BENCH="${BENCH:-composite}"
LIMIT="${LIMIT:-200}"

$PYBIN -m emnlp_evaluation.runners.run_generic_corpus \
    --benchmark "$BENCH" --backend splade --llm qwen \
    --systems monolithic,hybrid_routed --limit "$LIMIT"
