# Pythia — Cost-Aware Adaptive Retrieval Routing for Open-Domain QA

Pythia is a **per-query router** that decides whether a question should be answered from a large language model's own parameters, or whether it is worth paying the cost of retrieving context from Wikipedia. For every question, the router chooses one of four strategies:

- **`none`** — answer from the generator's parameters (no retrieval, cheapest)
- **`bm25`** — keyword search over a Wikipedia corpus (bm25s)
- **`dense`** — semantic search with bge-base embeddings + FAISS
- **`hybrid`** — BM25 and dense fused with Reciprocal Rank Fusion (RRF)

The router is trained as a cost-sensitive text classifier: it learns *when retrieval is worth paying for* rather than always retrieving. A tunable cost knob `λ` trades accuracy against retrieval calls, so the system can match the accuracy of always-retrieving while cutting retrieval usage by a large margin.

---

## 1. The Problem

Standard RAG retrieves context for **every** question. That wastes compute on questions the model already answers correctly ("easy" queries), and can even *hurt* accuracy when the retrieved context is irrelevant ("retrieval hurts" cases). Pythia asks: *for this specific question, is retrieval worth the cost?*

---

## 2. How It Works

### 2.1 Run all four strategies once per query

For each question, Pythia runs `none`, `bm25`, `dense`, and `hybrid` and logs per-query:

- EM and F1 (answer correctness)
- Latency (retrieval + generation)
- Token count (prompt + completion)
- Retrieval calls (0 for `none`, 1 for `bm25`/`dense`, 2 for `hybrid`)
- Whether a gold passage was retrieved

Everything is cached to Parquet keyed by `(model, prompt-hash, strategy, qid)` so re-runs are free and labeling is exactly reproducible.

### 2.2 Build labels from observed results

For each query, let `best = max_s score(s)` across the four strategies. The **label** is the cheapest strategy whose score is within `ε` of `best` (ε = 0 by default). Tie-breaking prefers higher EM, then higher F1, then lower cost, then a fixed name order.

This gives a ground-truth strategy for every query, derived from actually running all four — not from a heuristic.

### 2.3 Train a router classifier

The router is a 4-class text classifier that predicts the optimal strategy from the question text. Four model families are compared:

| Model | Input | Description |
|---|---|---|
| Logistic Regression | Lexical + embedding features | Scikit-learn multinomial LR, `class_weight="balanced"` |
| Gradient Boosted Trees | Lexical + embedding features | LightGBM, 300 estimators, max depth 6 |
| MLP | Lexical + embedding features | 2-layer network, 256 hidden units, ReLU, dropout |
| DeBERTa-v3-small | Raw question text | Fine-tuned transformer classifier, `num_labels=4` |

**Feature tiers** (ablated separately):

| Tier | Features | Cost |
|---|---|---|
| `lex` | Question length, wh-word one-hots, numerals, capitalised entities, stopword ratio, IDF/rare-term ratios | ~free (µs) |
| `emb` | 768-d bge-base query embedding (raw + SVD-64) | 1 encoder pass |
| `probe` | BM25 top-1 score/gap, dense top-1 similarity, overlap | 1 full retrieval call |
| `gen` | No-retrieval log-prob, entropy, margin | 1 no-retrieval generation |

Training uses three seeds (`20241117`, `20241118`, `20241119`) and supports three variants: **pooled** (one router on all data), **per-dataset** (one router per dataset), and **leave-one-dataset-out** (train on two, test on the third).

### 2.4 Calibrate probabilities with temperature scaling

Raw classifier probabilities are often overconfident. Pythia fits a single **temperature** `T` on the held-out `tune` split by minimizing negative log-likelihood:

```
scaled_logp = log(p_raw) / T
p_calibrated = softmax(scaled_logp)
```

The temperature is fixed after tuning and applied to both tune and eval predictions, producing well-calibrated probabilities `p(s)` for each strategy.

### 2.5 Deploy with a cost-aware decision rule

At inference, for each query the router computes calibrated probabilities `p(s)` and measured per-strategy latency `cost(s)`, then selects:

```
argmax_s log p(s) − λ · cost(s)
```

- **`p(s)`**: calibrated probability that strategy `s` is optimal for this query.
- **`cost(s)`**: measured end-to-end latency (ms) for strategy `s` — `none` is cheapest, `hybrid` most expensive.
- **`λ`**: tunable "retrieval price" knob. Higher `λ` penalises retrieval more strongly, yielding fewer retrieval calls at the cost of some accuracy.

Different `λ` values trace an accuracy–cost Pareto curve. The best `λ` is chosen on the `tune` split.

**Worked example:** A query yields calibrated probabilities `[0.28, 0.08, 0.48, 0.16]` for `[none, bm25, dense, hybrid]` with measured latencies `[50, 300, 400, 600]` ms. At `λ = 0.002`:

| Strategy | `log p(s)` | `−λ · cost(s)` | Utility |
|---|---|---|---|
| none | −1.27 | −0.10 | **−1.37** |
| bm25 | −2.53 | −0.60 | −3.13 |
| dense | −0.73 | −0.80 | −1.53 |
| hybrid | −1.83 | −1.20 | −3.03 |

`none` wins (−1.37 is highest). At `λ = 0.0001`, `dense` would win (−0.77 vs −1.28). The same probabilities, different cost trade-off.

---

## 3. NLP Techniques Used

Pythia applies a range of core NLP concepts, making it well-suited as a final NLP course project:

| Component | NLP Technique | Where it appears |
|---|---|---|
| **Text classification** | 4-class classifier predicting optimal strategy from question text | `router/models.py` |
| **Transformer embeddings** | bge-base-en-v1.5 sentence embeddings for dense retrieval and features | `retrieval/dense.py`, `router/features.py` |
| **Transformer fine-tuning** | DeBERTa-v3-small fine-tuned as a sequence classifier (`num_labels=4`) | `router/models.py` (`DebertaRouter`) |
| **Causal language modeling** | Qwen2.5-7B-Instruct-AWQ generates answers under vLLM | `generation/engine.py` |
| **Information retrieval** | Okapi BM25 (`bm25s`), dense vector search (FAISS `IndexFlatIP`), hybrid RRF fusion | `retrieval/` |
| **Text normalization** | SQuAD-style lowercasing, punctuation/article removal for EM/F1 | `common.py` (`normalize_answer`) |
| **Lexical feature engineering** | Wh-word detection, entity counting, stopword ratio, IDF/rare-term features | `router/features.py` (`lexical_features`) |
| **Language model uncertainty** | Mean log-prob, token entropy, margin from generation logits as features | `router/features.py` (`generator_features`) |
| **Prompt engineering** | Single instruction template with optional context block for QA | `generation/prompts.py` |
| **QA evaluation metrics** | Exact Match (EM) and token F1, max over gold aliases | `common.py` (`exact_match`, `token_f1`) |
| **Cross-dataset generalization** | Train/test on NQ-open, TriviaQA, HotpotQA; leave-one-dataset-out transfer | `scripts/06_train_router.py` |

---

## 4. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  INPUT: Questions (NQ-open, TriviaQA, HotpotQA)             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 1c — Generation (4 strategies × every query)         │
│  none  │  bm25  │  dense  │  hybrid                         │
│  vLLM  │  BM25  │  bge+   │  BM25+dense                     │
│  (no   │  search│  FAISS  │  RRF fusion                     │
│  retr) │        │         │                                 │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Per-query logs: EM, F1, latency, tokens, retrieval calls   │
│  (Parquet, cached by model/prompt/strategy/qid)             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 2 — Labeling                                         │
│  Label = cheapest strategy within ε of best score           │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 3 — Router Training                                  │
│  Features (lex/emb/probe/gen) → 4 models → predict_proba    │
│  Temperature calibration on tune split                      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 5 — Evaluation                                       │
│  Decision: argmax log p(s) − λ · cost(s)                    │
│  Baselines: never, always-*, random, oracle, confidence     │
│  Metrics: EM/F1, retrieval call rate, latency, Pareto curve │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 6 — Analysis                                         │
│  Breakdowns, confusion matrix, error costs, figures         │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Four Router Models

All models expose the same interface: `fit(X, y, seed)` and `predict_proba(X)` returning an `(N, 4)` probability array over `[none, bm25, dense, hybrid]`.

### Logistic Regression (`logreg`)
Scikit-learn `LogisticRegression` with `lbfgs` solver, multinomial loss, `C=1.0`, `max_iter=2000`, `class_weight="balanced"`. Trained on lexical + embedding features.

### Gradient Boosted Trees (`gbt`)
LightGBM `LGBMClassifier` with 300 estimators, learning rate 0.06, max depth 6, 31 leaves, `subsample=0.9`, `class_weight="balanced"`. Trained on lexical + embedding features.

### MLP (`mlp`)
2-layer PyTorch network: `Linear(input, 256) → ReLU → Dropout(0.1) → Linear(256, 4)`. Trained with CrossEntropyLoss and AdamW (lr=1e-3) for 30 epochs. Features standardized using training-set statistics.

### DeBERTa-v3-small (`deberta`)
HuggingFace `AutoModelForSequenceClassification` with `num_labels=4`, fine-tuned on raw question text (max length 64) with AdamW (lr=2e-5) for 4 epochs.

---

## 6. Evaluation & Success Criterion

### Baselines
Evaluated on identical queries and generator:
- `never` — no retrieval ever
- `always-bm25` / `always-dense` / `always-hybrid` — fixed strategy
- `random` — matched to Pythia's strategy proportions
- `oracle` — always picks the best strategy in hindsight
- `confidence-threshold` — retrieve only if no-retrieval confidence is low
- `complexity-classifier` — Adaptive-RAG style simple-vs-complex

### Metrics
- EM and F1 per dataset
- Retrieval call rate
- Mean and p95 end-to-end latency
- Total tokens processed
- Accuracy–cost Pareto curves

### Pre-registered success criterion
Pythia (pooled router, best config chosen on `tune`) must reach:
- **EM within 1.0 absolute point** of the best always-retrieve baseline, and
- **≥ 30% fewer retrieval calls** relative to that baseline

Evaluated with paired bootstrap 95% CIs (10,000 samples), Wilcoxon signed-rank on per-query F1, and McNemar tests, over ≥ 3 seeds.

---

## 8. Glossary

| Term | Meaning |
|---|---|
| `none` / `bm25` / `dense` / `hybrid` | The four answer strategies, ordered by increasing cost |
| `p(s)` | Calibrated probability that strategy `s` is optimal for a query |
| `cost(s)` | Measured end-to-end latency (ms) for strategy `s` |
| `λ` (lambda) | Tunable "retrieval price" knob in the decision rule |
| `ε` (epsilon) | Tolerance for label creation: cheapest strategy within ε of best |
| EM / F1 | Exact Match and token F1 (SQuAD-style QA metrics) |
| RRF | Reciprocal Rank Fusion (hybrid strategy) |
| LODO | Leave-one-dataset-out (cross-dataset transfer variant) |
| Oracle | Hypothetical system that always picks the best strategy |
| Retrieval hurts | Cases where retrieved context reduces answer accuracy |
| Temperature scaling | Post-hoc calibration of classifier probabilities |
| Pareto curve | Trade-off curve between accuracy and retrieval cost |

---

## Quick Start

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

`make` targets run through `scripts/run_stage.sh`, which pins each stage to 8 cores and nices it; on a systemd user session it also applies a 12 GB cgroup cap. Override with `PYTHIA_CORES`, `PYTHIA_MEM_LIMIT`, `PYTHIA_REPO`, `PYTHIA_DATA_ROOT` and `PYTHIA_VENV`.

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

## Pipeline Stages

| Stage | Script | Output |
|---|---|---|
| 0/1a | `scripts/01_prepare_data.py` | query splits (`label`/`tune`/`eval`), gold mapping, corpus subsample |
| 1b | `scripts/02_build_indices.py` | BM25 (`bm25s`) + dense FAISS index |
| 1c | `scripts/04_run_generation.py` | per-query logs for all four strategies |
| 2 | `scripts/05_build_labels.py` | hard/soft labels, oracle stats, label grid |
| 3 | `scripts/06_train_router.py` | pooled / per-dataset / LODO routers, calibration, tier ablation |
| 4–5 | `scripts/07_run_eval.py` | baselines, Pythia systems, bootstrap/McNemar/Wilcoxon, success criterion |
| 6 | `scripts/08_analysis.py` | breakdowns, confusion, error costs, failure taxonomy, figures |
| report | `scripts/09_report.py` | `docs/REPORT.md`, `results/run_manifest_*.json` |

## Repository Layout

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

## Outputs

- `results/logs/<run>/*.parquet` — raw per-query logs (one row per query × strategy)
- `results/tables/*.csv` — evaluation, significance, success criterion, breakdowns, confusion, error costs, failure taxonomy, LODO transfer
- `results/*.json` — cost order, oracle summary, eval metadata, run manifest
- `figures/*.png` — Pareto, label distribution, confusion, failures, breakdowns
- `docs/REPORT.md` — generated paper-style report

## Reproducibility

- Every stage is seeded (`seed: 20241117` in `configs/default.yaml`); router seeds are listed explicitly.
- `results/run_manifest_<run>.json` records the git SHA, Python/GPU/dependency versions, the full config, and every table/figure produced.
- Logs are Parquet; labels and tables are deterministic functions of the logs, so stages 05–09 can be re-run without touching retrieval or generation.
- Numbers in `docs/REPORT.md` come only from files under `results/` — the report script refuses to invent missing values and marks absent stages explicitly.

See `docs/STATUS.md` for per-phase notes, `docs/DEVIATIONS.md` for known deviations from `PLAN.md`, and `docs/ENVIRONMENT.md` for dependency notes.