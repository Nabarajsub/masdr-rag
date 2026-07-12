"""
Central configuration for the open-source evaluation harness.

Paths are pinned to the ARCC filesystem. Neo4j / Gemini secrets are read from
the same .env that agentic_solution/config.py uses, so the OSS runners hit
the same AuraDB as the production stack.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


_REPO_ROOT = Path(__file__).resolve().parents[3]
for _ep in (
    Path("/etc/secrets/.env"),
    _REPO_ROOT / "graph_processing" / ".env",
    _REPO_ROOT / ".env",
):
    if _ep.exists():
        load_dotenv(_ep, override=False)


# ── Neo4j (shared with agentic_solution) ──
# Per user policy: production AuraDB is read-only for the OSS evaluation
# harness. Any write (CREATE / MERGE / SET / DELETE / new index) must go to
# the local sandbox Neo4j started by emnlp_evaluation/slurm/run_local_neo4j.sbatch.
# That job writes its bolt URI + credentials to neo4j_endpoint, which the
# loader below reads first; production creds are then layered in for reads.
_LOCAL_ENDPOINT = Path("<DATA_ROOT>")
if _LOCAL_ENDPOINT.exists():
    load_dotenv(_LOCAL_ENDPOINT, override=False)

NEO4J_URI = os.getenv("NEO4J_URI_GEMINI") or os.getenv("NEO4J_URI", "")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME_GEMINI") or os.getenv("NEO4J_USERNAME", "")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD_GEMINI") or os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE_GEMINI") or os.getenv("NEO4J_DATABASE", "")

# Local sandbox (writes only).
NEO4J_LOCAL_URI = os.getenv("NEO4J_LOCAL_URI", "")
NEO4J_LOCAL_USERNAME = os.getenv("NEO4J_LOCAL_USERNAME", "neo4j")
NEO4J_LOCAL_PASSWORD = os.getenv("NEO4J_LOCAL_PASSWORD", "")
NEO4J_LOCAL_DATABASE = os.getenv("NEO4J_LOCAL_DATABASE", "neo4j")

# Vector index names. The Gemini index is what production uses; the BGE-M3 index
# is what reindex_neo4j.py creates.
NEO4J_GEMINI_INDEX = os.getenv("NEO4J_GEMINI_INDEX", "wydot_gemini_index")
NEO4J_BGE_M3_INDEX = os.getenv("NEO4J_BGE_M3_INDEX", "wydot_bge_m3_index")
NEO4J_FULLTEXT_INDEX = os.getenv("NEO4J_FULLTEXT_INDEX", "chunk_fulltext")
NEO4J_BGE_M3_PROPERTY = "embedding_bge_m3"


# ── Local model paths on ARCC ──
ARCC_HOME = Path("<DATA_ROOT>")

QWEN_PATH = os.getenv(
    "QWEN_MODEL_PATH",
    str(ARCC_HOME / "models" / "Qwen2.5-7B-Instruct"),
)
LLAMA_PATH = os.getenv(
    "LLAMA_MODEL_PATH",
    # HF cache snapshot dir; resolved at load time
    str(ARCC_HOME / "cache" / "huggingface" / "hub" / "models--meta-llama--Meta-Llama-3-8B-Instruct"),
)
BGE_M3_PATH = os.getenv(
    "BGE_M3_PATH",
    str(ARCC_HOME / "cache" / "huggingface" / "hub" / "models--BAAI--bge-m3"),
)


# ── HuggingFace cache (avoid hitting the network from compute nodes) ──
HF_CACHE = os.getenv("HF_HOME", str(ARCC_HOME / "cache" / "huggingface"))
os.environ.setdefault("HF_HOME", HF_CACHE)
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


# ── Defaults ──
DEFAULT_VECTOR_K = 30
DEFAULT_SCOPED_LIMIT = 15
DEFAULT_FULLTEXT_LIMIT = 10
MAX_NEW_TOKENS = 1024
ROUTER_MAX_NEW_TOKENS = 256


# ── Available providers (registered in llm_providers/factory.py) ──
SUPPORTED_PROVIDERS = ("gemini", "qwen", "llama")
SUPPORTED_EMBEDDERS = ("gemini", "bge_m3")


# ── Results folder ──
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
