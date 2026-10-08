"""R2 router: BGE-M3 query embedding -> logistic regression classifier.

Loads the artifact produced by emnlp_evaluation/router/train_classifier.py
and exposes route(query) -> agent_name. Drop-in replacement for the regex
+ LLM routing path in hybrid_routed.py / orchestrator_oss.py.

Maps the 9 WYDOT labels back to the agent_name space used by the rest of
the harness (specs_agent / construction_agent / ... / general_agent).
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np

ARTIFACT_PATH = Path("<DATA_ROOT>")

# Map trained-router labels -> agent_name used by SearchBackend.
LABEL_TO_AGENT = {
    "STANDARD_SPECS":      "specs_agent",
    "CONSTRUCTION_MANUAL": "construction_agent",
    "MATERIALS_TESTING":   "materials_agent",
    "DESIGN_MANUAL":       "design_agent",
    "TRAFFIC_CRASHES":     "safety_agent",
    "BRIDGE_PROGRAM":      "bridge_agent",
    "STIP":                "planning_agent",
    "ANNUAL_REPORT":       "admin_agent",
    "HIGHWAY_SAFETY":      "safety_agent",
    "GENERAL":             "general_agent",
    "CROSS_DOMAIN":        "general_agent",
    "VERSION_COMPARISON":  "general_agent",
}


class R2Router:
    _instance: Optional["R2Router"] = None

    def __init__(self, artifact_path: str | Path = ARTIFACT_PATH,
                 label_to_agent: Optional[dict] = None):
        with open(artifact_path, "rb") as f:
            blob = pickle.load(f)
        # Artifact is either {"model": clf, "labels": [...]} or a bare sklearn classifier.
        if isinstance(blob, dict):
            self.clf = blob["model"]
            self.labels = list(blob.get("labels") or self.clf.classes_)
        else:
            self.clf = blob
            self.labels = list(self.clf.classes_)
        # Label -> agent map; defaults to the WYDOT production mapping.
        self.label_to_agent = label_to_agent or LABEL_TO_AGENT
        # The embedder is heavy — share across calls; we lazy-init.
        self._embedder = None

    @classmethod
    def get(cls) -> "R2Router":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _embedder_lazy(self):
        if self._embedder is None:
            from emnlp_evaluation.embeddings import get_embedder
            self._embedder = get_embedder("bge_m3")
        return self._embedder

    def route(self, query: str, *, topk: int = 1) -> List[str]:
        """Return ranked list of agent names (length = topk)."""
        emb = np.asarray(self._embedder_lazy().embed_query(query),
                         dtype=np.float32).reshape(1, -1)
        proba = self.clf.predict_proba(emb)[0]
        order = np.argsort(-proba)
        agents: list[str] = []
        for i in order[:topk]:
            label = self.labels[i]
            agent = self.label_to_agent.get(label, "general_agent")
            if agent not in agents:
                agents.append(agent)
        return agents

    def route_top1(self, query: str) -> str:
        return self.route(query, topk=1)[0]

    def route_with_confidence(self, query: str) -> tuple:
        """Top-1 agent plus the classifier's probability for it."""
        emb = np.asarray(self._embedder_lazy().embed_query(query),
                         dtype=np.float32).reshape(1, -1)
        proba = self.clf.predict_proba(emb)[0]
        i = int(np.argmax(proba))
        return self.label_to_agent.get(self.labels[i], "general_agent"), float(proba[i])
