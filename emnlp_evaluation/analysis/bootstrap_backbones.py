"""Bootstrap 95% CIs for the cross-backbone table (tab:backbones)."""
from __future__ import annotations
import json
from collections import defaultdict
import numpy as np

PATHS = {
    "Qwen-7B":          "emnlp_evaluation/results/wydot_qwen_bge_m3_baselines.judged.jsonl",
    "Claude-H-4.5":     "emnlp_evaluation/results/wydot_or_anthropic-claude-haiku-4.5_bge_m3_monolithic-regex_scoped-hybrid_routed-masdr_rag.judged.jsonl",
    "GPT-5-mini":       "emnlp_evaluation/results/wydot_or_openai-gpt-5-mini_bge_m3_monolithic-regex_scoped-hybrid_routed-masdr_rag.judged.jsonl",
    "DeepSeek":         "emnlp_evaluation/results/wydot_or_deepseek-deepseek-chat_bge_m3_monolithic-regex_scoped-hybrid_routed-masdr_rag.judged.jsonl",
}
SYSTEMS = ["monolithic", "regex_scoped", "hybrid_routed", "masdr_rag"]
N_BOOT = 1000
SEED = 0


def boot_ci(arr, n_boot=N_BOOT, seed=SEED):
    if not arr:
        return None, None
    rng = np.random.default_rng(seed)
    arr = np.asarray(arr, dtype=np.float32)
    means = np.array([rng.choice(arr, size=len(arr), replace=True).mean()
                      for _ in range(n_boot)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    by = {}  # (backbone, system) -> {faith: [...], corr: [...]}
    for backbone, path in PATHS.items():
        d = defaultdict(lambda: {"faith": [], "corr": []})
        seen = set()
        try:
            for line in open(path):
                try: r = json.loads(line)
                except: continue
                if "error" in r: continue
                s = r.get("system", "?")
                k = (r.get("query_id"), s)
                if k in seen: continue
                seen.add(k)
                if r.get("faithfulness") is not None: d[s]["faith"].append(float(r["faithfulness"]))
                if r.get("correctness") is not None: d[s]["corr"].append(float(r["correctness"]))
        except FileNotFoundError:
            print(f"  skip (no file): {path}")
            continue
        for s in SYSTEMS:
            by[(backbone, s)] = d[s]

    print(f"{'backbone':<14} {'system':<14} {'n':>4}  Faith [95% CI]              Corr [95% CI]")
    for backbone in PATHS:
        for s in SYSTEMS:
            stats = by.get((backbone, s), {"faith": [], "corr": []})
            n = len(stats["faith"])
            if n == 0:
                print(f"{backbone:<14} {s:<14} {n:>4}  (no data)")
                continue
            mf = float(np.mean(stats["faith"]))
            mc = float(np.mean(stats["corr"])) if stats["corr"] else None
            flo, fhi = boot_ci(stats["faith"])
            clo, chi = boot_ci(stats["corr"]) if stats["corr"] else (None, None)
            corr_str = f"{mc:.3f} [{clo:.3f}, {chi:.3f}]" if mc is not None else "n/a"
            print(f"{backbone:<14} {s:<14} {n:>4}  {mf:.3f} [{flo:.3f}, {fhi:.3f}]     {corr_str}")


if __name__ == "__main__":
    main()
