# Pythia - adaptive retrieval routing for cost-efficient QA
#
# Every heavy stage runs through scripts/run_stage.sh, which pins it to 8 cores,
# nices it, and (when available) caps it inside a systemd user scope.
#
# Paths are derived from this Makefile, so the repo is relocatable.  Override the
# interpreter with `make PY=/path/to/python ...` and the data root with
# `PYTHIA_DATA_ROOT=/path make setup`.

SHELL := /bin/bash
REPO  := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
PY    ?= python
RUN   := bash $(REPO)/scripts/run_stage.sh
CFG   := $(REPO)/configs/default.yaml
# Run name and retrieval depth: override for k-ablation or imported runs, e.g.
#   make label router eval analysis report RUN_NAME=k10 K=10
RUN_NAME ?= main
K        ?= 5
# Path to an external Pythia checkout for `make import-run`.
EXTERNAL_SRC ?=

.PHONY: help setup doctor clean-gpu data indices pilot gen label router tiers eval analysis report smoke import-run all tiers-all clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## create the external data root, symlinks, dirs
	@bash $(REPO)/scripts/00_setup.sh

doctor: ## preflight: RAM / disk / GPU / stale processes
	@$(PY) $(REPO)/scripts/doctor.py

clean-gpu: ## kill stale vLLM engine processes holding VRAM
	@bash $(REPO)/scripts/clean_gpu.sh

data: ## phase 0/1a: query splits, gold mapping, corpus subsample
	@$(RUN) data $(PY) $(REPO)/scripts/01_prepare_data.py --config $(CFG)

indices: ## phase 1b: build BM25 + dense(FAISS) indices
	@$(RUN) indices $(PY) $(REPO)/scripts/02_build_indices.py --config $(CFG)

pilot: ## phase 1c: pilot generation run (500 q/dataset, 4 strategies)
	@$(RUN) pilot $(PY) $(REPO)/scripts/04_run_generation.py --config $(CFG) --pilot --k $(K)

gen: ## phase 1c: full generation (label+tune+eval, 4 strategies)
	@$(RUN) gen $(PY) $(REPO)/scripts/04_run_generation.py --config $(CFG) --k $(K)

label: ## phase 2: build labels + oracle statistics
	@$(RUN) label $(PY) $(REPO)/scripts/05_build_labels.py --config $(CFG) --run $(RUN_NAME) --k $(K)

router: ## phase 3: train pooled/leave-one-dataset-out routers
	@$(RUN) router $(PY) $(REPO)/scripts/06_train_router.py --config $(CFG) --run $(RUN_NAME) --k $(K) --variants all

tiers: ## phase 3b: feature-tier ablation (lex / +emb / +probe / +gen)
	@$(RUN) tiers $(PY) $(REPO)/scripts/06_train_router.py --config $(CFG) --run $(RUN_NAME) --k $(K) --tier-ablation --no-primary

eval: ## phase 5: baselines + evaluation on the official dev splits
	@$(RUN) eval $(PY) $(REPO)/scripts/07_run_eval.py --config $(CFG) --run $(RUN_NAME) --k $(K)

analysis: ## phase 6: breakdowns, failure analysis, figures
	@$(RUN) analysis $(PY) $(REPO)/scripts/08_analysis.py --config $(CFG) --run $(RUN_NAME) --k $(K)

report: ## deliverables: docs/REPORT.md + run manifest
	@cd $(REPO) && $(PY) scripts/09_report.py --config $(CFG) --run $(RUN_NAME)

smoke: ## synthetic end-to-end test of stages 05-09 (no GPU / corpus)
	@bash $(REPO)/tests/smoke.sh

import-run: ## import an external checkout's real logs (EXTERNAL_SRC=... RUN_NAME=...)
	@test -n "$(EXTERNAL_SRC)" || (echo "set EXTERNAL_SRC=/path/to/external/Pythia" && exit 1)
	@$(PY) $(REPO)/scripts/import_external_run.py --source $(EXTERNAL_SRC) --root $(REPO) --run $(RUN_NAME) --overwrite

all: setup doctor data indices gen label router eval analysis report ## full pipeline
tiers-all: setup doctor data indices gen label router tiers eval analysis report ## full pipeline incl. tier ablation

clean: ## remove derived artifacts (keeps raw downloads)
	@rm -rf results/logs/* results/tables/* figures/* __pycache__ */__pycache__
	@echo "cleaned"
