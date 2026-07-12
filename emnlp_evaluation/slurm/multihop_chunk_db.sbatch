#!/bin/bash
#SBATCH --job-name=mh-chunk-db
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=teton
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=emnlp_evaluation/slurm/logs/mh_chunk_db_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/mh_chunk_db_%j.err

source <DATA_ROOT>

$PYBIN -c "
import json, pandas as pd
meta = pd.read_parquet('emnlp_evaluation/benchmark/multihop_rag/data/multihop.meta.parquet')
db = dict(zip(meta['id'].astype(str), meta['text'].astype(str)))
import json
with open('emnlp_evaluation/judge_assets/multihop_chunks.json','w') as f:
    json.dump(db, f)
print(f'wrote {len(db)} chunks')
"
