#!/bin/bash
#SBATCH --job-name=ext-smoke
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=mb-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=<DATA_ROOT>
#SBATCH --error=<DATA_ROOT>

# Fast smoke test: 5 Composite-9 queries through MA-RAG and SCOUT-RAG.
# Catches: prompt rendering bugs, JSON parse failures, retriever wiring,
# domain enumeration. Run THIS before submitting the full per-corpus
# jobs.
#
#   sbatch graph_processing/emnlp_evaluation/slurm/smoke_external_baselines.sbatch

source <DATA_ROOT>
mkdir -p "$PROJECT_DIR/emnlp_evaluation/slurm/logs"

OUT="$PROJECT_DIR/emnlp_evaluation/results/_smoke_extbaselines_qwen.jsonl"
rm -f "$OUT"

# run_composite has no --embedder flag (composite backend is bge_m3 internally).
$PYBIN -m emnlp_evaluation.runners.run_composite \
    --systems ma_rag,scout_rag \
    --llm qwen \
    --limit 5 \
    --out "$OUT" \
    || { echo "[smoke] FAILED"; exit 1; }

echo "[smoke] wrote $OUT"
echo "---"
echo "[smoke] non-empty answer counts per system:"
$PYBIN -c "
import json
from collections import Counter
counts, nonempty, samples = Counter(), Counter(), {}
errors = []
with open('$OUT') as f:
    for line in f:
        rec = json.loads(line)
        sysname = rec.get('system','?')
        counts[sysname] += 1
        if rec.get('error'):
            errors.append((sysname, rec['error']))
            continue
        ans = rec.get('answer','') or ''
        if not isinstance(ans, str):
            ans = str(ans)
        if ans.strip():
            nonempty[sysname] += 1
            samples.setdefault(sysname, ans[:200])
for sysname in counts:
    print(f'  {sysname}: {nonempty[sysname]}/{counts[sysname]} non-empty answers')
    if sysname in samples:
        print(f'      sample: {samples[sysname][:160]}')
if errors:
    print(f'  ERRORS ({len(errors)}):')
    for s, e in errors[:10]:
        print(f'    [{s}] {e}')
"
