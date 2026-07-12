PY ?= python

.PHONY: setup-data tables reproduce build-indexes sweep judge clean

# Decompress the released judged logs + judge chunk DBs into the locations
# the analysis scripts expect.
setup-data:
	mkdir -p emnlp_evaluation/results emnlp_evaluation/judge_assets
	for f in data/results/*.jsonl.gz; do \
	  gunzip -kc $$f > emnlp_evaluation/results/$$(basename $$f .gz); done
	cp data/results/fallback_sweep.json emnlp_evaluation/results/
	for f in data/judge_chunk_dbs/*.json.gz; do \
	  gunzip -kc $$f > emnlp_evaluation/judge_assets/$$(basename $$f .gz); done
	cp data/k_sweep.json emnlp_evaluation/judge_assets/

# Recompute every paper table from the judged logs (CPU-only, ~2 min).
tables:
	$(PY) -m emnlp_evaluation.analysis.compile_metrics || true
	$(PY) -m emnlp_evaluation.analysis.rebuttal_numbers
	$(PY) -m emnlp_evaluation.analysis.permutation_tests || true

reproduce: setup-data tables

# ---- full GPU pipeline ------------------------------------------------
build-indexes:
	$(PY) -m emnlp_evaluation.benchmark.multihop_rag.build || true
	$(PY) -m emnlp_evaluation.benchmark.mmlu_pro.build || true
	$(PY) -m emnlp_evaluation.benchmark.nq.build || true
	$(PY) -m emnlp_evaluation.benchmark.financebench.build || true

sweep:
	$(PY) -m emnlp_evaluation.runners.run_composite --llm qwen \
	  --systems monolithic,regex_scoped,hybrid_routed,masdr_rag
	$(PY) -m emnlp_evaluation.runners.run_wydot_oss --llm qwen \
	  --systems monolithic,regex_scoped,hybrid_routed,masdr_rag

judge:
	$(PY) -m emnlp_evaluation.analysis.judge \
	  --in "emnlp_evaluation/results/*.jsonl" --judge qwen \
	  --chunk-text-db emnlp_evaluation/judge_assets/wydot_chunks.json

clean:
	rm -f emnlp_evaluation/results/*.jsonl
