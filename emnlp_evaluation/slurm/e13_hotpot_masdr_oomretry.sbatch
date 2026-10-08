#!/bin/bash
#SBATCH --job-name=e13-hp-oomretry
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=8:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/e13_hp_oomretry_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/e13_hp_oomretry_%j.err

# E13-repair: the main MASDR HotpotQA run (job 11544332, A30 23.6GB) hit
# CUDA OOM on 168/2000 queries and wrote empty answers for them. OOM is
# correlated with prompt length (MASDR retrieves up to 149 chunks / 33.7k
# prompt tokens), so the dropout is NON-RANDOM -- scoring the surviving
# 1832 would bias MASDR upward on exactly the hardest queries. Re-run the
# 168 on an H100 (80GB) and merge before reporting anything.
source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs

OUT=emnlp_evaluation/results/crag_qwen_promptctl_masdr_oomretry.jsonl
$PYBIN -m emnlp_evaluation.runners.run_crag \
    --llm qwen --systems "masdr_rag" --shared-answer-prompt \
    --queries emnlp_evaluation/benchmark/crag/data/crag.queries.oom_retry.json \
    --out $OUT
$PYBIN -m emnlp_evaluation.analysis.judge --in "$OUT" --judge qwen \
    --chunk-text-db emnlp_evaluation/judge_assets/crag_chunks.json
echo "[e13-oomretry] done"
