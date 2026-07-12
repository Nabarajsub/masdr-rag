#!/bin/bash
#SBATCH --job-name=rejudge-all
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/rejudge_all_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/rejudge_all_%j.err

# Re-judge with the chunk_text_db so faithfulness can actually be scored.
# Without --chunk-text-db, the judge gets "(chunk texts unavailable)" and
# faithfulness is always 0. Force-truncates existing .judged.jsonl outputs.

source <DATA_ROOT>

# Reduce CUDA fragmentation on long-context judge prompts.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CORPUS="${CORPUS:-composite}"
DB="emnlp_evaluation/judge_assets/${CORPUS}_chunks.json"

if [[ "$CORPUS" == "composite" ]]; then
    GLOB="${GLOB:-emnlp_evaluation/results/composite_qwen_masdr_rag.jsonl emnlp_evaluation/results/composite_qwen_masdr_singlecall.jsonl}"
elif [[ "$CORPUS" == "wydot" ]]; then
    GLOB="${GLOB:-emnlp_evaluation/results/wydot_qwen_bge_m3_masdr_singlecall.jsonl emnlp_evaluation/results/wydot_qwen_bge_m3_baselines.jsonl}"
elif [[ "$CORPUS" == "crag" ]]; then
    GLOB="${GLOB:-emnlp_evaluation/results/crag_qwen_masdr_rag.jsonl}"
elif [[ "$CORPUS" == "multihop" ]]; then
    GLOB="${GLOB:-emnlp_evaluation/results/multihop_qwen_monolithic-regex_scoped-hybrid_routed-masdr_rag.jsonl}"
elif [[ "$CORPUS" == "financebench" ]]; then
    GLOB="${GLOB:-emnlp_evaluation/results/financebench_qwen_monolithic-regex_scoped-hybrid_routed-masdr_rag.jsonl}"
elif [[ "$CORPUS" == "mmlu_pro" ]]; then
    GLOB="${GLOB:-emnlp_evaluation/results/mmlu_pro_qwen_monolithic-regex_scoped-hybrid_routed-masdr_rag.jsonl}"
elif [[ "$CORPUS" == "nq" ]]; then
    GLOB="${GLOB:-emnlp_evaluation/results/nq_qwen_monolithic-regex_scoped-hybrid_routed-masdr_rag.jsonl}"
else
    echo "Unknown CORPUS=$CORPUS"; exit 1
fi

for path in $GLOB; do
    if [[ ! -f "$path" ]]; then
        echo "[skip] $path not found"; continue
    fi
    $PYBIN -m emnlp_evaluation.analysis.judge \
        --in "$path" \
        --judge qwen \
        --chunk-text-db "$DB" \
        --force
done
