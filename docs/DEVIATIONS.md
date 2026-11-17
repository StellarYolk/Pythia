# Deviations from PLAN.md

Deviations are recorded here with a reason; the report repeats the relevant ones. The
pre-registered success criterion (`PLAN.md` §8) was **not** changed: on the 0–1 EM scale
it is `EM ≥ best always-retrieve EM − 0.01` **and** a relative retrieval-call reduction of
at least 30 %.

## Engineering deviations (do not affect the scientific claim)

1. **Relocatable paths.** The original tree was hard-wired to `/home/parth/...` (repo,
   venv, data root, dense-model mirror, doctor disk check). The GitHub version now
   derives the repo from the Makefile/script location and uses `PYTHIA_DATA_ROOT`,
   `PYTHIA_VENV`, `PYTHIA_REPO`, `PYTHIA_CORES`, `PYTHIA_MEM_LIMIT`. The dense model is
   referenced by HuggingFace id `BAAI/bge-base-en-v1.5`; a local mirror can be used by
   editing `retrieval.dense_model` or pre-downloading into `HF_HOME`.
2. **`run_stage.sh` fallback.** The systemd user-scope cgroup cap is applied only when a
   user systemd session is available; otherwise the script keeps CPU affinity and
   niceness but not the 12 GB memory cap.
3. **Missing indices degrade gracefully at router-feature time.** If the BM25 index is
   absent, lexical IDF features are zero and a warning is logged instead of rebuilding a
   21M-passage index implicitly. The `probe` tier still fails loudly because it cannot be
   approximated. This affects only development/CI, not a production run after stage 02.
4. **GPUs.** Embedding features fall back to CPU when CUDA is unavailable. Router
   training itself is CPU-capable; generation is not.
5. **The 1.0 → 0.01 success margin was a unit bug, not a criterion change.** The config
   stored `1.0` while per-query EM is in `[0, 1]`; `PLAN.md` says “1.0 absolute point”.

## Scientific / procedural deviations

1. **Trimmed budget.** The original session agreed on 1500 label / 400 tune / 700 eval
   queries per dataset to fit the machine budget. `configs/default.yaml` still records
   the larger 2000/500/1000 targets; `configs/trimmed.yaml` records the agreed smaller
   budget. Use one config for every stage of a run so labels, routers and evaluation see
   the same split.
2. **k ablation is manual.** The primary run uses k=5. k ∈ {3, 10} must be launched as
   separate runs (stage 04 `--k`), then labels/router/eval with `--k`; there is no single
   Makefile target yet.
3. **LODO temperature calibration.** For a leave-one-dataset-out fold, the temperature
   is fitted on the `tune` split of the *training* datasets, never on the held-out
   dataset. This avoids target-domain leakage but means LODO calibration is slightly
   mismatched to the held-out domain; pooled/per-dataset routers calibrate on their own
   tune split.
4. **Primary system for the descriptive analysis.** `scripts/08_analysis.py` picks the
   highest-EM Pythia configuration among the primary variant/tier on the **evaluation**
   split, because the original run's dev-from-train selection table is not available in
   this checkout. This is a descriptive choice for breakdowns; the pre-registered
   comparison in stage 07 still reports every λ/seed/model, and the report states the
   selection rule. For a formal claim, select the configuration on `tune` and pass
   `--system` explicitly to stage 08.
5. **Rule-based failure taxonomy.** Stage 08 assigns categories with deterministic rules
   (under-retrieval, distractor retrieval, multi-hop, over-retrieval, normalisation,
   label noise, hallucination, wrong/ambiguous gold). The plan requires manual
   verification of ≥ 50 failures per dataset; the sample is exported, but the manual
   pass is still pending and the report labels it as such.
6. **No real official-dev numbers here.** This checkout was validated with synthetic
   logs only (no DPR corpus, no vLLM, no GPU). Nothing in `results/` from the original
   machine was imported into this repository.

## Not changed

- The label rule (cheapest strategy within ε, explicit tie-break).
- The decision rule (`argmax log p(s) − λ·cost(s)`).
- The success criterion and the primary metric (EM, ε=0 declared before official-dev).
- The requirement that every reported number come from a file in `results/`.
