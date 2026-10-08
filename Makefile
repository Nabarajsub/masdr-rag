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
	cp data/results/dilution_scaling_*.json emnlp_evaluation/results/
	for d in rq1 rq2 rq3 rq4 rq5; do \
	  if [ -d data/results/$$d ]; then mkdir -p emnlp_evaluation/results/$$d; \
	    for f in data/results/$$d/*.gz; do [ -e "$$f" ] && gunzip -kc $$f > emnlp_evaluation/results/$$d/$$(basename $$f .gz); done; \
	    cp data/results/$$d/*.json emnlp_evaluation/results/$$d/ 2>/dev/null || true; fi; done
	mkdir -p emnlp_evaluation/human_eval
	cp data/human_eval/*.json emnlp_evaluation/human_eval/

# Recompute every paper table from the judged logs (CPU-only, ~2 min).
tables:
	$(PY) -m emnlp_evaluation.analysis.compile_metrics || true
	$(PY) -m emnlp_evaluation.analysis.rebuttal_numbers
	$(PY) -m emnlp_evaluation.analysis.permutation_tests || true
	@echo "== Table 6 significance (shared prompt, paired tests, Holm)"
	$(PY) -m emnlp_evaluation.analysis.rq2_scoping_test --out emnlp_evaluation/results/rq2/rq2_partA.json
	@echo "== Human faithfulness study (Section 8)"
	$(PY) -m emnlp_evaluation.analysis.score_human_eval_paradox

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
