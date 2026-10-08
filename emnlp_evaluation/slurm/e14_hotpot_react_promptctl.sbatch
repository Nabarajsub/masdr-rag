#!/bin/bash
#SBATCH --job-name=e14-hp-react
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=16:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/e14_hp_react_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/e14_hp_react_%j.err

# E14: Custom ReAct on HotpotQA-distractor under the SHARED corpus-neutral
# prompt (persona removed, tool-calling and loop instructions preserved) --
# the same treatment applied to ReAct on Composite-9 in E11.
#
# H100 only, on purpose: ReAct accumulates context across rounds, and the E13
# MASDR run OOM'd on 168/2000 queries at 23.6GB. Run it where 80GB is available
# so the result is not a favourable subset.
source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs

OUT=emnlp_evaluation/results/crag_qwen_promptctl_react.jsonl
$PYBIN -m emnlp_evaluation.runners.run_crag \
    --llm qwen --systems "react" --shared-answer-prompt --out $OUT
$PYBIN -m emnlp_evaluation.analysis.judge --in "$OUT" --judge qwen \
    --chunk-text-db emnlp_evaluation/judge_assets/crag_chunks.json
echo "[e14-react] done"
