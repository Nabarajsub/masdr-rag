#!/bin/bash
#SBATCH --job-name=e5-prompt-ctl
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-a30,mb-h100,mb-a6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/e5_prompt_ctl_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/e5_prompt_ctl_%j.err

# E5 PROMPT CONTROL (final_v3).
#
# Monolithic ships a bare answer prompt (agents/naive_rag.py) while every scoped
# system ships a "You are the WYDOT Knowledge Graph Assistant" prompt with a
# hedging instruction (agents/hybrid_routed.py). That confounds every
# monolithic-vs-scoped comparison this harness has produced, including tables in
# the submitted paper. --shared-answer-prompt forces both onto ONE identical,
# corpus-neutral prompt so the delta isolates the retrieval architecture.
#
# Corpora chosen to cover both directions of the suspected bias:
#   multihop  — scoped systems get a corpus-MISmatched persona (suspected penalty)
#   wydot_hand— scoped systems get a MATCHED persona (suspected advantage)
source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs

OUT_MH=emnlp_evaluation/results/multihop_qwen_promptctl_monolithic-soft_scoped.jsonl
$PYBIN -m emnlp_evaluation.runners.run_generic_corpus \
    --benchmark multihop --llm qwen \
    --systems monolithic,soft_scoped --shared-answer-prompt \
    --limit 500 --out $OUT_MH
$PYBIN -m emnlp_evaluation.analysis.judge \
    --in "$OUT_MH" --judge qwen \
    --chunk-text-db emnlp_evaluation/judge_assets/multihop_chunks.json

OUT_WY=emnlp_evaluation/results/wydot_hand_qwen_promptctl_monolithic-r2_routed-soft_scoped.jsonl
$PYBIN -m emnlp_evaluation.runners.run_generic_corpus \
    --benchmark wydot_hand --llm qwen \
    --systems monolithic,r2_routed,soft_scoped --shared-answer-prompt \
    --out $OUT_WY
$PYBIN -m emnlp_evaluation.analysis.judge \
    --in "$OUT_WY" --judge qwen \
    --chunk-text-db emnlp_evaluation/judge_assets/wydotv3_chunks.json

echo "[e5-prompt-ctl] done"
