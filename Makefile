SHELL := /bin/bash
# Unbuffered: these are multi-hour jobs and progress must appear as it happens,
# not in one burst when the pipe closes.
PY    := PYTHONUNBUFFERED=1 .venv/bin/python
PIP   := .venv/bin/pip

# Homebrew's python3.11, NOT pyenv's - the pyenv build on this machine lacks
# liblzma, which breaks datasets -> sentence-transformers. See docs/HARDWARE.md.
BASE_PYTHON := /usr/local/bin/python3.11

ENRON_URL := https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz
TARBALL   := data/raw/enron_mail_20150507.tar.gz
MAILDIR   := data/raw/maildir
MESSAGES  := data/interim/messages.parquet
SAMPLE    := data/interim/sample.parquet

EVALSET   := data/eval/queries.jsonl
PILOT_IDX := data/index/thread_aware__all-MiniLM-L6-v2

# One factor at a time, NOT a full grid.
#
# The plan asks for six dimensions "each varied independently", which is
# one-factor-at-a-time around a baseline - not a 4x3 cross product. The
# distinction is worth 14 hours here:
#
#   full grid     4 chunkings x 3 models            = 12 builds, ~20.5 h
#   independent   4 chunkings + 2 more models       =  6 builds,  ~6.6 h
#
# Dimension 1 varies chunking against a fixed cheap model; dimension 2 then
# varies the model against whichever chunking won. A full grid would also
# report interaction effects the eval set is far too small (70 answerable
# queries) to resolve.
#
# Measured on this machine at the 50k sample, ~58k chunks of ~512 tokens:
#   MiniLM ~28 min/pass, bge-small ~56 min/pass, bge-base ~3.7 h/pass.
# (The 67/34/9.8 docs/s in HARDWARE.md were on 230-token docs; throughput
# scales roughly inversely with chunk length.)
BASELINE_MODEL := sentence-transformers/all-MiniLM-L6-v2
CHUNKERS       := fixed_512 fixed_512_ov64 whole_message thread_aware
EXTRA_MODELS   := BAAI/bge-small-en-v1.5 BAAI/bge-base-en-v1.5
BEST_CHUNKING  ?= thread_aware

# Dimensions 4 and 5 vary one factor against one index, so they need to be told
# which index. Defaults to the pilot; override once dimension 1 has a winner.
PIVOT_CONFIG   ?= thread_aware__all-MiniLM-L6-v2

.PHONY: help venv download extract corpus sample bench-hardware test clean-derived \
        index-pilot index-all candidates verify validate-eval bench \
        bench-rerank bench-transform bench-rerank-budget export-onnx \
        failures cache-stats cache-clear serve ask extract extract-ceiling \
        extract-compare route-eval pgvector eval-generation gmail-auth gmail-status \
        check-privacy

help:
	@grep -E '^[a-z0-9-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

venv: ## create .venv with the pinned Intel-macOS stack
	$(BASE_PYTHON) -m venv .venv
	$(PIP) install -q --upgrade pip
	$(PIP) install -q -r requirements.txt
	@$(PY) -c "import torch,transformers as t; \
		assert t.utils.is_torch_available(), 'torch backend disabled - check pins'; \
		print('ok: torch',torch.__version__,'transformers',t.__version__)"

download: $(TARBALL) ## fetch the CMU Enron corpus (443 MB)
$(TARBALL):
	@mkdir -p data/raw
	curl -L --retry 3 -C - -o $@ $(ENRON_URL)

extract: $(MAILDIR) ## unpack the maildir (~2.6 GB)
$(MAILDIR): $(TARBALL)
	tar -xzf $(TARBALL) -C data/raw
	@touch $@

corpus: $(MESSAGES) ## parse -> dedup -> bulk filter -> Parquet
$(MESSAGES): scripts/build_corpus.py | $(MAILDIR)
	$(PY) scripts/build_corpus.py --maildir $(MAILDIR) --out $@

sample: $(SAMPLE) ## thread reconstruction + stratified ablation subsample
$(SAMPLE): $(MESSAGES) scripts/build_sample.py
	$(PY) scripts/build_sample.py --messages $(MESSAGES) --out $@

index-pilot: $(PILOT_IDX)/config.json ## build the one index needed to start labelling
$(PILOT_IDX)/config.json: $(SAMPLE)
	$(PY) scripts/build_index.py --chunking thread_aware \
		--model sentence-transformers/all-MiniLM-L6-v2

index-d1: $(SAMPLE) ## dimension 1: 4 chunkings x baseline model (~1.9 h)
	@for c in $(CHUNKERS); do \
	  echo "=== D1  $$c / $(BASELINE_MODEL) ==="; \
	  $(PY) scripts/build_index.py --chunking $$c --model $(BASELINE_MODEL) || exit 1; \
	done

index-d2: $(SAMPLE) ## dimension 2: extra models x BEST_CHUNKING (~4.7 h)
	@echo "dimension 2 against BEST_CHUNKING=$(BEST_CHUNKING)"
	@echo "set it from the D1 winner, e.g. make index-d2 BEST_CHUNKING=fixed_512_ov64"
	@for m in $(EXTRA_MODELS); do \
	  echo "=== D2  $(BEST_CHUNKING) / $$m ==="; \
	  $(PY) scripts/build_index.py --chunking $(BEST_CHUNKING) --model $$m || exit 1; \
	done

index-all: index-d1 index-d2 ## both dimensions back to back (~6.6 h, overnight)

candidates: $(SAMPLE) ## draft candidate eval queries (needs GEMINI_API_KEY)
	$(PY) scripts/make_eval_candidates.py --n-per-class 3

verify: $(PILOT_IDX)/config.json ## hand-verify candidates into the eval set
	$(PY) scripts/verify_eval.py --index $(PILOT_IDX)

validate-eval: ## fail if the eval set is malformed
	$(PY) scripts/validate_eval.py

bench: validate-eval ## reproduce every results table from the published eval set
	$(PY) scripts/run_ablation.py

bench-rerank: validate-eval ## dimension 4: rerank arms vs added latency
	$(PY) scripts/run_ablation.py --dimension rerank --config $(PIVOT_CONFIG)

bench-transform: validate-eval ## dimension 5: HyDE / multi-query / decomposition
	$(PY) scripts/run_ablation.py --dimension transform --config $(PIVOT_CONFIG)

failures: ## dimension 6: classify every recall@20 miss in the newest run
	$(PY) scripts/analyze_failures.py --list

bench-hardware: ## re-measure embed/rerank throughput on this machine
	$(PY) scripts/bench_hardware.py

bench-rerank-budget: ## which rerank arm fits 200 ms here (model x k x passage length)
	$(PY) scripts/bench_rerank_budget.py

export-onnx: ## int8 ONNX export of the shipped cross-encoder, with a quality check
	$(PY) scripts/export_onnx_reranker.py

cache-stats: ## size of the LLM response cache
	@$(PY) -c "import sys; sys.path.insert(0,'src'); \
	  from emailrag.llm.cache import ResponseCache as C; c=C(); \
	  print(f'{len(c):,} cached responses, {c.size_bytes()/1e6:.1f} MB in {c.root}')"

cache-clear: ## drop cached LLM responses (they will be re-paid in quota)
	@$(PY) -c "import sys; sys.path.insert(0,'src'); \
	  from emailrag.llm.cache import ResponseCache as C; \
	  print(f'removed {C().clear():,} cached responses')"

test:
	$(PY) -m pytest tests -q

# The LLM cache is NOT dropped here. It is the difference between re-running
# `make bench` today and waiting for tomorrow's free-tier quota; `make
# cache-clear` exists for when you actually mean it.
clean-derived: ## drop everything regenerable; keeps the tarball, eval set and LLM cache
	rm -rf data/interim data/index runs

serve: ## local web UI over the assembled pipeline (127.0.0.1 only)
	$(PY) scripts/serve.py

ask: ## one question, cited answer, in the terminal: make ask Q="..."
	$(PY) scripts/ask.py "$(Q)"

# -- phase 4: extraction and the router ------------------------------------

extract: ## commitments from the eval-scoped threads (local Qwen via Ollama)
	$(PY) scripts/extract_commitments.py --scope eval --provider ollama

extract-ceiling: ## the same messages through Claude Haiku, for the comparison
	$(PY) scripts/extract_commitments.py --scope eval --provider anthropic

extract-compare: ## local arm vs ceiling: agreement and date accuracy
	$(PY) scripts/extract_commitments.py --compare

route-eval: validate-eval ## router accuracy per query class
	$(PY) scripts/route_eval.py --list

pgvector: ## load the pivot index into pgvector and measure HNSW recall loss
	$(PY) scripts/load_pgvector.py --config $(PIVOT_CONFIG) --measure-recall

# -- phase 5: generation eval ----------------------------------------------

eval-generation: validate-eval ## cited answers over the eval set, then judge them
	$(PY) scripts/eval_generation.py

# -- phase 7: your own mail ------------------------------------------------

gmail-auth: ## authorise Gmail (repeat weekly - Testing apps get 7-day tokens)
	$(PY) scripts/gmail_auth.py

gmail-status: ## how long until the Gmail token needs re-authorising
	$(PY) scripts/gmail_auth.py --status

check-privacy: ## refuse to publish anything derived from private mail
	$(PY) scripts/check_privacy.py
