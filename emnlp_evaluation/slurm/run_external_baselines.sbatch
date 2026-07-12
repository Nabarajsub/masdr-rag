#!/bin/bash
#SBATCH --job-name=ext-baselines
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=<DATA_ROOT>
#SBATCH --error=<DATA_ROOT>

# Run MA-RAG (ported from thangylvp/MA-RAG) and SCOUT-RAG (best-effort
# reimpl of arXiv:2602.08400 Algorithm 1) against one of the four eval
# corpora. The runner is selected with $CORPUS.
#
# Submit examples:
#   CORPUS=composite  sbatch run_external_baselines.sbatch
#   CORPUS=multihop   LIMIT=500  sbatch run_external_baselines.sbatch
#   CORPUS=wydot                 sbatch run_external_baselines.sbatch
#   CORPUS=hotpot     LIMIT=500  sbatch run_external_baselines.sbatch
#
# Optional overrides:
#   LLM=qwen|llama  (default qwen)
#   LIMIT=<int>     (cap queries for time budget)
#   SYSTEMS=ma_rag,scout_rag  (default both)
#   EMBEDDER=bge_m3 (default; applies to multihop/wydot only)

source <DATA_ROOT>
mkdir -p "$PROJECT_DIR/emnlp_evaluation/slurm/logs"

CORPUS="${CORPUS:?CORPUS must be set: composite|multihop|wydot|hotpot}"
LLM="${LLM:-qwen}"
SYSTEMS="${SYSTEMS:-ma_rag,scout_rag}"
EMBEDDER="${EMBEDDER:-bge_m3}"
LIMIT="${LIMIT:-}"
TAG="${TAG:-extbaselines}"   # output suffix; use a distinct TAG for re-runs

echo "[ext-baselines] CORPUS=$CORPUS LLM=$LLM SYSTEMS=$SYSTEMS LIMIT=${LIMIT:-all} TAG=$TAG"

LIMIT_FLAG=""
[ -n "$LIMIT" ] && LIMIT_FLAG="--limit $LIMIT"

case "$CORPUS" in
    composite)
        # run_composite has no --embedder (composite backend is bge_m3).
        OUT="$PROJECT_DIR/emnlp_evaluation/results/composite_${LLM}_${TAG}.jsonl"
        $PYBIN -m emnlp_evaluation.runners.run_composite \
            --systems "$SYSTEMS" --llm "$LLM" --out "$OUT" $LIMIT_FLAG \
            || { echo "[ext-baselines] FAILED"; exit 1; }
        ;;
    multihop)
        # Use run_generic_corpus with FAISS — the Neo4j multihop index does
        # not exist in the production AuraDB, so MultihopSearchBackend
        # returns 0 chunks silently. FAISS is what older successful runs
        # used.
        OUT="$PROJECT_DIR/emnlp_evaluation/results/multihop_${LLM}_${TAG}.jsonl"
        $PYBIN -m emnlp_evaluation.runners.run_generic_corpus \
            --benchmark multihop --systems "$SYSTEMS" --llm "$LLM" \
            --backend faiss --out "$OUT" $LIMIT_FLAG \
            || { echo "[ext-baselines] FAILED"; exit 1; }
        ;;
    wydot)
        OUT="$PROJECT_DIR/emnlp_evaluation/results/wydot_${LLM}_${TAG}.jsonl"
        $PYBIN -m emnlp_evaluation.runners.run_wydot_oss \
            --systems "$SYSTEMS" --llm "$LLM" --embedder "$EMBEDDER" \
            --out "$OUT" $LIMIT_FLAG \
            || { echo "[ext-baselines] FAILED"; exit 1; }
        ;;
    hotpot|crag)
        # run_crag has no --embedder. HotpotQA-distractor is selected by
        # the runner's default prefix; override with PREFIX if needed.
        OUT="$PROJECT_DIR/emnlp_evaluation/results/${CORPUS}_${LLM}_${TAG}.jsonl"
        PREFIX_FLAG=""
        [ -n "${PREFIX:-}" ] && PREFIX_FLAG="--prefix $PREFIX"
        $PYBIN -m emnlp_evaluation.runners.run_crag \
            --systems "$SYSTEMS" --llm "$LLM" --out "$OUT" $LIMIT_FLAG $PREFIX_FLAG \
            || { echo "[ext-baselines] FAILED"; exit 1; }
        ;;
    *)
        echo "Unknown CORPUS=$CORPUS" >&2; exit 2
        ;;
esac

echo "[ext-baselines] wrote $OUT"
