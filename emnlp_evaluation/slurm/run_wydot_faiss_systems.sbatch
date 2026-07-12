#!/bin/bash
#SBATCH --job-name=wydot-faiss-sys
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/wydot_faiss_sys_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/wydot_faiss_sys_%j.err

# Run all 4 baseline systems on WYDOT via the unified FAISS index
# (already built at composite_corpus/data/wydot.*). No Neo4j, no Gemini.

source <DATA_ROOT>
mkdir -p emnlp_evaluation/slurm/logs

# Trick: register WYDOT in run_generic_corpus by adding it to BENCH at runtime.
$PYBIN <<'PYEOF'
import sys, os, json, time, argparse
sys.path.insert(0, '<DATA_ROOT>')
sys.path.insert(0, '<DATA_ROOT>')

from pathlib import Path
from emnlp_evaluation.runners import run_generic_corpus as rgc

# Register WYDOT as a generic-corpus benchmark backed by the unified FAISS index.
rgc.BENCH["wydot_faiss"] = {
    "prefix":  Path("<DATA_ROOT>"),
    "queries": Path("<DATA_ROOT>"),
    "scope_field": "source_type",
}
sys.argv = ["run_generic_corpus",
            "--benchmark", "wydot_faiss",
            "--llm", "qwen",
            "--systems", "monolithic,regex_scoped,hybrid_routed,masdr_rag",
            "--limit", "200"]

# test_suite_200.json uses 'id' not 'query_id'; patch the load step.
orig_main = rgc.main
def patched_main():
    # Load suite, then rename 'id' -> 'query_id' before run_generic_corpus uses it.
    suite_path = rgc.BENCH["wydot_faiss"]["queries"]
    suite_raw = json.load(open(suite_path))
    for q in suite_raw:
        if "id" in q and "query_id" not in q:
            q["query_id"] = q["id"]
    tmp_path = Path("/tmp/wydot_suite_normalised.json")
    tmp_path.write_text(json.dumps(suite_raw))
    rgc.BENCH["wydot_faiss"]["queries"] = tmp_path
    orig_main()

patched_main()
PYEOF
