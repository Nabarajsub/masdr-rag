#!/bin/bash
#SBATCH --job-name=neo4j-smoke
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=teton
#SBATCH --time=00:05:00
#SBATCH --mem=4G
#SBATCH --output=emnlp_evaluation/slurm/logs/neo4j_smoke_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/neo4j_smoke_%j.err

source <DATA_ROOT>

$PYBIN <<'PYEOF'
from emnlp_evaluation.configs import oss_config as cfg
from neo4j import GraphDatabase
print('Local URI:', cfg.NEO4J_LOCAL_URI)
print('Prod  URI:', cfg.NEO4J_URI[:40], '...')
d = GraphDatabase.driver(cfg.NEO4J_LOCAL_URI,
                          auth=(cfg.NEO4J_LOCAL_USERNAME, cfg.NEO4J_LOCAL_PASSWORD))
with d.session(database=cfg.NEO4J_LOCAL_DATABASE) as s:
    print('hello:', s.run('RETURN 1 AS n').single()['n'])
    s.run('CREATE (n:Smoke {id: 1, ts: timestamp()})')
    cnt = s.run('MATCH (n:Smoke) RETURN count(n) AS c').single()['c']
    print('wrote count =', cnt)
    s.run('MATCH (n:Smoke) DELETE n')
d.close()
print('OK')
PYEOF
