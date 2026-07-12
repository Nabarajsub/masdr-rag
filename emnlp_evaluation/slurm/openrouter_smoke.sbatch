#!/bin/bash
#SBATCH --job-name=or-smoke
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=teton
#SBATCH --time=00:05:00
#SBATCH --mem=4G
#SBATCH --output=emnlp_evaluation/slurm/logs/or_smoke_%j.out
#SBATCH --error=emnlp_evaluation/slurm/logs/or_smoke_%j.err

source <DATA_ROOT>

$PYBIN <<'PYEOF'
from emnlp_evaluation.llm_providers import get_provider

for model in ("anthropic/claude-haiku-4.5", "openai/gpt-5-mini", "deepseek/deepseek-chat"):
    print(f"\n=== {model} ===")
    p = get_provider("openrouter", model_name=model)
    r = p.generate(
        [{"role": "user", "content": "What is 2+2? Reply with just the digit."}],
        max_new_tokens=20,
    )
    print(f"  text: {r.text!r}")
    print(f"  tokens: prompt={r.prompt_tokens} completion={r.completion_tokens}")
    print(f"  finish: {r.finish_reason}")
PYEOF
