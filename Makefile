# Pythia - adaptive retrieval routing for cost-efficient QA
#
# Every stage runs through scripts/run_stage.sh, which pins it to 8 cores,
# nices it, and caps it at 12 GB inside a cgroup (see docs/DEVIATIONS.md).

SHELL := /bin/bash
REPO  := /home/parth/college/NLP/lab/pythia
PY    := /home/parth/venvs/pythia/bin/python
RUN   := scripts/run_stage.sh
CFG   := configs/default.yaml

.PHONY: help setup doctor clean-gpu data indices pilot label router eval analysis report all clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## create the external data root, symlinks, dirs
	@bash scripts/00_setup.sh

doctor: ## preflight: RAM / disk / GPU / stale processes
	@$(PY) scripts/doctor.py

clean-gpu: ## kill stale vLLM engine processes holding VRAM
	@bash scripts/clean_gpu.sh

data: ## phase 0/1a: query splits, gold mapping, corpus subsample
	@$(RUN) data $(PY) scripts/01_prepare_data.py --config $(CFG)

indices: ## phase 1b: build BM25 + dense(FAISS) indices
	@$(RUN) indices $(PY) scripts/02_build_indices.py --config $(CFG)

pilot: ## phase 1c: pilot generation run (500 q/dataset, 4 strategies)
	@$(RUN) pilot $(PY) scripts/04_run_generation.py --config $(CFG) --pilot

label: ## phase 2: build labels + oracle statistics
	@$(RUN) label $(PY) scripts/05_build_labels.py --config $(CFG)

router: ## phase 3: train router variants
	@$(RUN) router $(PY) scripts/06_train_router.py --config $(CFG)

eval: ## phase 5: baselines + evaluation on the official dev splits
	@$(RUN) eval $(PY) scripts/07_run_eval.py --config $(CFG)

analysis: ## phase 6: breakdowns, failure analysis, figures
	@$(RUN) analysis $(PY) scripts/08_analysis.py --config $(CFG)

report: ## deliverables: figures + report scaffolding
	@$(PY) scripts/09_report.py --config $(CFG)

all: setup doctor data indices pilot label router eval analysis ## full pipeline

clean: ## remove derived artifacts (keeps raw downloads)
	@rm -rf results/logs/* results/tables/* figures/* __pycache__ */__pycache__
	@echo "cleaned"
