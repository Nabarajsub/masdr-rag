#!/bin/bash
#SBATCH --job-name=fill-na
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/fill_na_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/fill_na_%j.err

# Fill the two remaining n/a cells in Figure cross_corpus_bars:
# (1) ReAct on MultiHop-RAG
# (2) Regex-Scoped on HotpotQA/CRAG

source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs

WHICH="${WHICH:-multihop_react}"

if [[ "$WHICH" == "multihop_react" ]]; then
    $PYBIN -m emnlp_evaluation.runners.run_multihop \
        --llm qwen --systems react --limit 500
elif [[ "$WHICH" == "crag_regex" ]]; then
    $PYBIN -m emnlp_evaluation.runners.run_crag \
        --llm qwen --systems regex_scoped --limit 500
else
    echo "Unknown WHICH=$WHICH"; exit 1
fi
