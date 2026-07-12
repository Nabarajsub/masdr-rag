#!/bin/bash
#SBATCH --job-name=ingest-neo4j
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=teton
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=emnlp_evaluation/slurm/logs/ingest_neo4j_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/ingest_neo4j_%j.err

# Push every dataset we ingested to FAISS into the local Neo4j sandbox.
# Each corpus gets its own node label + vector index so they coexist.
# No production Neo4j writes — local sandbox only.

source <DATA_ROOT>

ingest () {
    local corpus="$1"; local prefix="$2"; local scope="$3"
    echo "==== ingesting $corpus from $prefix (scope=$scope) ===="
    $PYBIN -m emnlp_evaluation.composite_corpus.ingest_to_local_neo4j \
        --corpus "$corpus" --prefix "$prefix" --scope-field "$scope"
}

ingest multihop      emnlp_evaluation/benchmark/multihop_rag/data/multihop  source_type
ingest financebench  emnlp_evaluation/benchmark/financebench/data/financebench  source_type
ingest mmlu_pro      emnlp_evaluation/benchmark/mmlu_pro/data/mmlu_pro  source_type
# NQ is 200k chunks → large. Run it last so a failure doesn't block the others.
ingest nq            emnlp_evaluation/benchmark/nq/data/nq  source_type
# Composite-9 too (the existing FAISS one) for full parity
ingest composite     emnlp_evaluation/composite_corpus/data/composite  source_type
