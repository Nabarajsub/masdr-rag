#!/bin/bash
#SBATCH --job-name=generic-neo4j
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/generic_neo4j_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/generic_neo4j_%j.err

# Run any of the 5 datasets via local Neo4j HNSW (vs the FAISS path used by
# run_generic.sbatch). Direct head-to-head for the infra-parity claim.

source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs

BENCH="${BENCH:-multihop}"
LIMIT="${LIMIT:-500}"
LLM="${LLM:-qwen}"

$PYBIN -m emnlp_evaluation.runners.run_generic_corpus \
    --benchmark "$BENCH" \
    --backend neo4j \
    --llm "$LLM" \
    --systems monolithic,regex_scoped,hybrid_routed,masdr_rag \
    --limit "$LIMIT"
