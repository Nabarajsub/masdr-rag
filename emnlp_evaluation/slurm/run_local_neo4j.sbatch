#!/bin/bash
#SBATCH --job-name=local-neo4j
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=teton
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=7-00:00:00
#SBATCH --output=emnlp_evaluation/slurm/logs/local_neo4j_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/local_neo4j_%j.err

# Long-running Neo4j daemon for sandbox writes. Reads of production Neo4j
# happen elsewhere (per user policy). Bolt port + hostname are advertised
# to <DATA_ROOT> so other jobs
# can discover this node.

set -euo pipefail

NEO4J_HOME=<DATA_ROOT>
export JAVA_HOME=<DATA_ROOT>
export PATH="$JAVA_HOME/bin:$PATH"
export NEO4J_HOME

BOLT_PORT=$(cat <DATA_ROOT>
HTTP_PORT=$(cat <DATA_ROOT>
HOST="$(hostname -s)"

ENDPOINT_FILE=<DATA_ROOT>
cat > "$ENDPOINT_FILE" <<EOF
NEO4J_LOCAL_URI=bolt://${HOST}:${BOLT_PORT}
NEO4J_LOCAL_USERNAME=neo4j
NEO4J_LOCAL_PASSWORD=wydotLocal-2026
NEO4J_LOCAL_DATABASE=neo4j
NEO4J_LOCAL_HTTP=http://${HOST}:${HTTP_PORT}
NEO4J_LOCAL_HOST=${HOST}
NEO4J_LOCAL_BOLT_PORT=${BOLT_PORT}
NEO4J_LOCAL_JOB_ID=${SLURM_JOB_ID}
EOF
echo "Wrote endpoint info to $ENDPOINT_FILE"
cat "$ENDPOINT_FILE"

cleanup() {
    echo "[neo4j] received signal, stopping"
    "$NEO4J_HOME/bin/neo4j" stop || true
    rm -f "$ENDPOINT_FILE"
}
trap cleanup TERM INT

# console mode: keep Neo4j in the foreground; signals end the SLURM step cleanly.
exec "$NEO4J_HOME/bin/neo4j" console
