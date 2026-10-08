#!/bin/bash
#SBATCH --job-name=e18-llama-matched
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-h100,mb-l40s,mb-a6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=20:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/e18_llama_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/e18_llama_%j.err

# E18: Llama-3-8B cross-corpus replication under the SHARED corpus-neutral prompt.
#
# Why this is required, not optional: the rewritten paradox control §7(3) claims
# "under matched prompts the collapse is present on every backbone we test" and
# cites Llama-3-8B (FinanceBench .207->.000, MMLU-Pro .162->.096). Those figures
# come from tab:llama_cross, which was run under the ORIGINAL prompt protocol --
# no *_llama_promptctl_* results exist. As written the claim is unsupported: the
# Qwen half is matched, the Llama half is not.
#
# Re-running the four corpora of tab:llama_cross fixes both the table and the
# control. ">=48GB partitions only" -- the orchestrator OOM'd on the 23.6GB A30.
source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs

run_and_judge () {
  local bench=$1 chunkdb=$2
  local out=emnlp_evaluation/results/${bench}_llama_promptctl_all.jsonl
  $PYBIN -m emnlp_evaluation.runners.run_generic_corpus \
      --benchmark $bench --llm llama \
      --systems monolithic,regex_scoped,hybrid_routed,soft_scoped,masdr_rag \
      --backend faiss --shared-answer-prompt --limit 500 --out $out
  $PYBIN -m emnlp_evaluation.analysis.judge \
      --in "$out" --judge qwen --chunk-text-db $chunkdb
}

run_and_judge financebench emnlp_evaluation/judge_assets/financebench_chunks.json
run_and_judge mmlu_pro     emnlp_evaluation/judge_assets/mmlu_pro_chunks.json
run_and_judge multihop     emnlp_evaluation/judge_assets/multihop_chunks.json
run_and_judge nq           emnlp_evaluation/judge_assets/nq_chunks.json

echo "[e18-llama-matched] done"
