#!/bin/bash
#SBATCH --job-name=comp9-expand
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/comp9_expand_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/comp9_expand_%j.err

# Expand Composite-9 from 5/9 sources, ~50 queries → 9/9 sources, ~225 queries.
# Re-embed all 18k chunks with BGE-M3, regenerate queries (~25 per source).

source <DATA_ROOT>

# Reduce fragmentation when re-embedding long financial-10K records.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "==== embedding 9-source corpus ===="
$PYBIN -m emnlp_evaluation.composite_corpus.embed_and_index --batch-size 16 --max-chars 4000

echo "==== generating queries (25 per source, 9 sources = 225 queries) ===="
$PYBIN -m emnlp_evaluation.composite_corpus.query_gen --per-source 25

# Refresh chunk DB
echo "==== refreshing chunk_text_db ===="
$PYBIN -m emnlp_evaluation.analysis.build_chunk_text_db \
    --corpus composite \
    --out emnlp_evaluation/judge_assets/composite_chunks.json

echo "==== re-ingesting expanded composite into local Neo4j ===="
$PYBIN -m emnlp_evaluation.composite_corpus.ingest_to_local_neo4j \
    --corpus composite \
    --prefix emnlp_evaluation/composite_corpus/data/composite \
    --scope-field source_type
