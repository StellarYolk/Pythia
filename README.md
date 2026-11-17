# Pythia — cost-aware adaptive retrieval routing for open-domain QA

Pythia learns **when retrieval is worth paying for**. For every question a router chooses
one of four strategies — answer from the generator's parameters (`none`), BM25, dense
(bge-base + FAISS), or hybrid RRF — under a calibrated cost knob. The project trains and
compares logistic regression, gradient-boosted trees, an MLP and a fine-tuned
DeBERTa-v3-small router, ablates feature tiers, and evaluates everything on the same
per-query logs so latency, token and retrieval-call accounting is identical across
systems.

> **Status.** The code for phases 0–6 is complete. The repository does **not** yet
> contain official-dev numbers: running the pipeline needs the 21M-passage DPR corpus,
> a GPU with vLLM, and the retrieval indices (see `docs/STATUS.md`). A CPU-only synthetic
> smoke test exercises stages 05–09 end to end (`bash tests/smoke.sh`). Generated
> artefacts are written to `results/` and `figures/`; `docs/REPORT.md` is produced by
> `scripts/09_report.py` and is never hand-edited.

## What is implemented

| Stage | Script | Output |
| --- | --- | --- |
| 0/1a | `scripts/01_prepare_data.py` | query splits (`label`/`tune`/`eval`), gold mapping, corpus subsample |
| 1b | `scripts/02_build_indices.py` | BM25 (`bm25s`) + dense FAISS index |
| 1c | `scripts/04_run_generation.py` | per-query logs for all four strategies |
| 2 | `scripts/05_build_labels.py` | hard/soft labels, oracle stats, label grid |
| 3 | `scripts/06_train_router.py` | pooled / per-dataset / LODO routers, calibration, tier ablation |
| 4–5 | `scripts/07_run_eval.py` | baselines, Pythia systems, bootstrap/McNemar/Wilcoxon, success criterion |
| 6 | `scripts/08_analysis.py` | breakdowns, confusion, error costs, failure taxonomy, figures |
| report | `scripts/09_report.py` | `docs/REPORT.md`, `results/run_manifest_*.json` |

## Quick start

```bash
# 1. environment (Python 3.10+; GPU strongly recommended)
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# 2. paths: heavy data lives outside the repo
export PYTHIA_DATA_ROOT=$HOME/pythia-data
export PYTHIA_VENV=$PWD/.venv
make setup doctor

# 3. full pipeline
make data indices gen label router tiers eval analysis report
```

`make` targets run through `scripts/run_stage.sh`, which pins each stage to 8 cores and
nices it; on a systemd user session it also applies a 12 GB cgroup cap. Override with
`PYTHIA_CORES`, `PYTHIA_MEM_LIMIT`, `PYTHIA_REPO`, `PYTHIA_DATA_ROOT` and
`PYTHIA_VENV`. Paths no longer point at a specific home directory.

### Useful partial runs

```bash
# only one dataset / subset while iterating
python scripts/04_run_generation.py --config configs/default.yaml \
    --datasets nq --subsets label,tune,eval

# one router family, one seed (fast)
python scripts/06_train_router.py --config configs/default.yaml \
    --models logreg --tiers lex --seeds 20241117

# all variants: pooled + per_dataset + leave-one-dataset-out
python scripts/06_train_router.py --config configs/default.yaml --variants all

# feature-tier ablation only (keeps the primary prediction file intact)
make tiers
```

## Smoke test (no GPU, no corpus, no network)

```bash
bash tests/smoke.sh
```

The script copies the code to `/tmp/opencode/pythia-smoke`, generates synthetic queries
and generation logs that match `generation/runner.py`'s schema, then runs stages
05 → 06 → 07 → 08 → 09. It checks that labels, router training, evaluation alignment,
breakdowns, failure sampling, figures and the report all work. It is a plumbing test,
**not** a scientific result.

If the default interpreter is bare but a scientific stack exists elsewhere, point to it:

```bash
PYTHIA_SITE_PACKAGES=/path/to/site-packages bash tests/smoke.sh /tmp/pythia-smoke /path/to/python3.10
```

## Outputs

- `results/logs/<run>/*.parquet` — raw per-query logs (one row per query × strategy)
- `results/tables/*.csv` — evaluation, significance, success criterion, breakdowns,
  confusion, error costs, failure taxonomy, LODO transfer
- `results/*.json` — cost order, oracle summary, eval metadata, run manifest
- `figures/*.png` — Pareto, label distribution, confusion, failures, breakdowns
- `docs/REPORT.md` — generated paper-style report

## Repository layout

```
configs/        default.yaml (single source of truth), smoke.yaml
data/           dataset prep + gold mapping (heavy files are git-ignored / external)
retrieval/      corpus store, BM25, dense FAISS, hybrid RRF
generation/     vLLM engine wrapper, prompts, strategy runner + logging
labeling/       cost-aware labels, soft labels, oracle statistics
router/         feature tiers, model families, calibration, cost-aware decision
eval/           baselines, paired statistics, Pareto utilities
analysis/       breakdown columns, failure taxonomy, figures
scripts/        00–09 stage entry points, resource launcher, preflight
tests/          synthetic log generator + end-to-end smoke test
docs/           STATUS, DEVIATIONS, ENVIRONMENT, DATA_CARD, failures, REPORT
```

## Method in one paragraph

For each query we run all four strategies once and log EM/F1, latency, tokens and
retrieval calls. The label is the **cheapest strategy within ε of the best score**
(tie-break: cost rank, EM, F1, name). A router predicts a distribution over the four
strategies; a temperature is fitted on the held-out `tune` split, and the deployment
decision is `argmax_s log p(s) − λ·cost(s)` with measured latency as the cost. The
pre-registered success criterion (`PLAN.md` §8) is: EM within **1 absolute point** of the
best always-retrieve baseline while cutting retrieval calls by **≥ 30 %**, evaluated with
paired bootstrap CIs, Wilcoxon and McNemar tests over ≥ 3 seeds.

## Reproducibility

- Every stage is seeded (`seed: 20241117` in `configs/default.yaml`); router seeds are
  listed explicitly.
- `results/run_manifest_<run>.json` records the git SHA, Python/GPU/dependency versions,
  the full config, and every table/figure produced.
- Logs are parquet; labels and tables are deterministic functions of the logs, so
  stages 05–09 can be re-run without touching retrieval or generation.
- Numbers in `docs/REPORT.md` come only from files under `results/` — the report script
  refuses to invent missing values and marks absent stages explicitly.

See `docs/STATUS.md` for what has and has not been executed, `docs/DEVIATIONS.md` for
known deviations from `PLAN.md`, and `docs/ENVIRONMENT.md` for dependency notes.
