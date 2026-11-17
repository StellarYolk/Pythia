# Environment

## Reference environment (original run)

- Linux laptop, 32 hardware threads, ~29 GB RAM, hybrid graphics (NVIDIA GPU used for
  generation and dense encoding; the display runs on the AMD iGPU).
- Python 3.10 virtualenv, vLLM serving `Qwen/Qwen2.5-7B-Instruct-AWQ`.
- Heavy stages are pinned to 8 cores, niced, and (when a systemd user session exists)
  capped at 12 GB via `scripts/run_stage.sh`.

The exact versions from a real run are recorded automatically in
`results/run_manifest_<run>.json` (`env` block: Python, git SHA, torch/CUDA, vLLM,
transformers, FAISS, bm25s, sklearn, datasets, LightGBM, GPU name/memory).

## Validation environment used for this pass

The repository was completed and smoke-tested on a **different** machine without the DPR
dump, GPU generation, or network package installation:

- Python 3.10.18 (uv-managed interpreter) with the scientific stack from a separate
  virtualenv: pandas 2.3.3, pyarrow 25.0.1, scikit-learn 1.7.2, scipy 1.15.3,
  matplotlib 3.10.9, numpy 2.2.6, torch 2.14.1, FAISS, transformers, PyYAML.
- **Missing / not exercised:** vLLM, LightGBM, bm25s, the official datasets, the DPR
  corpus, and the retrieval indices.
- The system `/usr/bin/python3.10` lacks `_ssl`, which breaks `pyarrow.parquet`; use the
  uv interpreter (or any Python with a working `ssl` module) for parquet I/O.

If your interpreter is bare but a compatible site-packages tree exists:

```bash
PYTHONPATH=/path/to/python3.10/site-packages python3.10 scripts/05_build_labels.py ...
```

`tests/smoke.sh` accepts this via `PYTHIA_SITE_PACKAGES`.

## Dependency notes

- `sentence-transformers` is **not** required: `router/features.py` and
  `retrieval/dense.py` call `transformers` directly.
- `lightgbm` is only needed for the GBT router; the stage logs an error and continues
  with the remaining model families if it is missing.
- `bm25s` is only needed to build/load the BM25 index. Router feature extraction with
  `lex` can run without it, but the IDF features are then zero.
- FAISS: install `faiss-cpu` or a matching GPU build; the dense index stores float16
  vectors and a FAISS index file.
- vLLM is imported lazily by `generation/`; stages 05–09 do not need it.
- PyArrow is required for the parquet log/label/prediction format.

## Resource guidance

| Stage | Runtime driver | Peak memory driver |
| --- | --- | --- |
| 01 data | HuggingFace downloads, gold mapping | query pools + title maps |
| 02 indices | dense encoding (GPU) + BM25 build | corpus store + embeddings |
| 04 generation | vLLM on a 7B AWQ model | ~12 GB GPU; CPU batch prep |
| 05 labels | pandas over logs | logs for all strategies |
| 06 routers | embeddings (optional) + model fit | feature matrix (emb 768-d) |
| 07 eval | vectorised numpy + bootstrap | per-query table × systems |
| 08 analysis | pandas + matplotlib | per-query table |

The 10,000-sample paired bootstrap is limited to the top `--sig-top-k` systems (default
25) to avoid an O(systems × N × resamples) blow-up; increase it for a final table if
needed.
