#!/bin/bash
#SBATCH --job-name=wydot-soft-gemini-retry
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/soft_gemini_retry_%x_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/soft_gemini_retry_%x_%j.err
# QIDS is colon-separated (sbatch --export splits on commas).
# Re-run the soft_scoped queries that hit CUDA OOM on 24 GB cards (needs >=48 GB).
source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs
$PYBIN -m emnlp_evaluation.runners.run_wydot_oss_cached_gemini \
    --llm "$LLM" --embedder gemini --systems soft_scoped \
    --query-ids "${QIDS//:/,}" --resume --out "emnlp_evaluation/results/wydot_${LLM}_gemini_soft_scoped_oomretry.jsonl"
echo "[soft-gemini-retry] done llm=$LLM"
