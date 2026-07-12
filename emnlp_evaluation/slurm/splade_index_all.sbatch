#!/bin/bash
#SBATCH --job-name=splade-index
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/splade_index_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/splade_index_%j.err

# Encode each of our 5 corpora with SPLADE (opensearch v2 distill).
# Output: <prefix>.splade.npz alongside the existing FAISS .embeddings.npy.

source <DATA_ROOT>

$PYBIN <<'PYEOF'
from emnlp_evaluation.composite_corpus.splade_backend import build_corpus_splade

for prefix in [
    "emnlp_evaluation/composite_corpus/data/composite",
    "emnlp_evaluation/benchmark/multihop_rag/data/multihop",
    "emnlp_evaluation/benchmark/financebench/data/financebench",
    "emnlp_evaluation/benchmark/mmlu_pro/data/mmlu_pro",
]:
    print(f"\n==== {prefix} ====")
    build_corpus_splade(prefix)
PYEOF
