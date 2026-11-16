# Pythia: Adaptive Retrieval Routing for Cost-Efficient Question Answering

**Project plan** — version 1.0 (written before the first pilot run; deviations are logged in
`docs/STATUS.md` and `docs/DEVIATIONS.md` with justifications).

---

## 0. Problem statement

Standard RAG retrieves for every question. That wastes compute on questions a parametric LLM already
answers ("easy" queries) and can *hurt* accuracy when the retrieved context is irrelevant
("retrieval hurts" cases). Pythia is a **per-query router** that picks one of four strategies before
generation:

```
none  <  bm25  <  dense  <  hybrid          (cost order, verified empirically)
```

The router is trained as a classifier on labels derived from actually running all four strategies on
Natural Questions (NQ-open), TriviaQA (unfiltered) and HotpotQA (fullwiki), with the label being the
**cheapest strategy whose score is within ε of the best strategy's score**.

**Goal.** Match the accuracy of the best always-retrieve baseline (within a pre-registered 1.0 EM
point, 95% bootstrap CI) while cutting retrieval calls and end-to-end latency by a large percentage,
and characterise precisely where routing succeeds and fails.

---

## 1. Hardware reality check and the design decisions it forces

The brief assumes "1× 24–80 GB GPU". The actual machine is:

| Resource | Actual | Brief assumption |
|---|---|---|
| GPU | 1× RTX 5060 Laptop, **8 GB VRAM**, sm_120 (Blackwell) | 24–80 GB |
| CPU | 32 cores | — |
| RAM | 29 GB (≈20 GB free) | — |
| Disk | 520 GB free | — |
| Python | 3.12.11 venv (`/tmp/opencode/venv`) | — |

Consequences (all are **flagged deviations**, see §8):

1. **Generator is 4-bit.** A 7–8B instruct model in fp16 needs ≥14 GB. We use
   `Qwen/Qwen2.5-7B-Instruct-AWQ` (4-bit, ~5.2 GB) under vLLM with a short context
   (`max_model_len=2048`) to leave room for the KV cache inside 8 GB. The fallback, if AWQ
   kernels misbehave on sm_120, is `Qwen/Qwen2.5-3B-Instruct` (fp16, ~6.2 GB) — a further deviation
   down the size ladder. The exact model actually used is recorded in `results/run_manifest.json`.
2. **Corpus is a documented gold-preserving subsample** of DPR Wikipedia (21M → ≈2 M passages).
   The brief explicitly permits this: *"If compute-limited, use a documented subsample that still
   contains gold passages for all evaluation queries, and state this limitation."*
3. **BM25 via `bm25s` instead of Pyserini.** No JVM is installed on the machine and a Pyserini
   Lucene index over 21M passages needs ≈30 GB RAM (we have 29 GB total). `bm25s` builds an
   equivalent Okapi BM25 index over our subsample in minutes and ≪8 GB. Same scoring function
   (k1=0.9/0.2, b=0.4 defaults of Lucene; we use bm25s defaults k1=1.5,b=0.75 and record them).
4. **Scale.** We cannot afford 5–10 k queries × 3 datasets × 4 strategies at 7B. Query counts are
   sized from the *measured* pilot throughput (§5), targeting ≈40–50 k total generations.

---

## 2. Repository layout (Phase 0 deliverable)

```
pythia/
├── PLAN.md                    # this file
├── README.md                  # exact commands to reproduce
├── Makefile                   # `make pilot`, `make label`, `make router`, `make eval`, `make report`
├── requirements.txt           # pinned deps
├── common.py                  # seeding, io, timing, logging, normalisation helpers
├── configs/
│   ├── default.yaml           # single source of truth for paths/models/hyper-params
│   ├── pilot.yaml             # 500 q/dataset override
│   ├── full.yaml              # full-run override
│   └── router.yaml            # router model grid
├── data/
│   ├── raw/                   # psgs_w100.tsv.gz (DPR dump), HF cache
│   ├── processed/             # queries_<split>.jsonl, corpus.jsonl, gold map
│   └── indices/               # faiss.index, bm25 index, doc embeddings
├── retrieval/
│   ├── base.py                # Retriever ABC: retrieve(queries, k) -> Passage lists
│   ├── bm25.py                # BM25Retriever (bm25s)
│   ├── dense.py               # DenseRetriever (FAISS + bge-base-en-v1.5)
│   ├── hybrid.py              # RRF(k=60) + weighted-score fusion variant
│   └── build_indices.py       # encode + index scripts
├── generation/
│   ├── prompts.py             # one template, context block optional
│   ├── engine.py              # vLLM wrapper (temperature 0, seed, token accounting)
│   └── runner.py              # runs 4 strategies over a query file, caches to parquet
├── labeling/
│   ├── metrics.py             # EM / token-F1 (SQuAD normalisation, max over aliases)
│   └── build_labels.py        # labels, oracle stats, soft labels, label distributions
├── router/
│   ├── features.py            # lexical / embedding / probe / generator feature tiers
│   ├── models.py              # LR, GBT, MLP, DeBERTa-v3-small
│   ├── train.py               # training, calibration, cost knob, 3 seeds
│   └── predict.py             # inference + strategy assignment
├── eval/
│   ├── baselines.py           # never/always/random/oracle/confidence/complexity
│   ├── run_eval.py            # main evaluation driver
│   └── stats.py               # bootstrap CIs, paired tests, Pareto curves
├── analysis/
│   ├── breakdowns.py          # dataset / wh-word / answer-type / length / popularity / hops
│   ├── failures.py            # failure categorisation (>=50/dataset)
│   └── figures.py             # all plots
├── scripts/                   # 00_setup.sh ... 09_report.sh (one per phase)
├── results/                   # raw per-query logs (parquet/jsonl) + aggregate CSVs
├── figures/                   # PNG/PDF figures
└── docs/                      # STATUS.md (per-phase notes), DEVIATIONS.md, failures.md
```

**Module contract.** Every strategy implements `Retriever.retrieve(queries, k) -> List[List[Passage]]`;
the runner is the only component that knows about prompts, vLLM and metric computation, so adding a
5th strategy is a ~40-line change. All randomness is seeded from `configs/default.yaml`
(`seed: 20241117`).

---

## 3. Data (Phase 0)

| Dataset | HF id | Train (labels) | Dev-from-train (tuning) | Official dev (final eval) |
|---|---|---|---|---|
| NQ-open | `google-research-datasets/nq_open` | `train` | held-out slice of `train` | `validation` (3 610) |
| TriviaQA (rc.nocontext, unfiltered) | `mandarjoshi/trivia_qa` | `train` | held-out slice of `train` | `validation` (11 313) |
| HotpotQA (fullwiki) | `hotpotqa/hotpot_qa` `fullwiki` | `train` | held-out slice of `train` | `validation` (7 405) |

**Strict separation (enforced in code, asserted by `eval/assert_no_leakage.py`):**

```
official train split  ──► sampled label pool  ──► fit router (train part)
                                │
                                └──► held-out dev-from-train slice ──► tune λ / threshold / model selection
official dev split    ──► final evaluation ONLY (router never sees it)
```

Router *features* and *labels* may only be computed from the train split. The official dev splits
are touched exactly once, at Phase 5, by the same code path that produces the always-retrieve
baselines.

**Corpus.** DPR Wikipedia, `psgs_w100.tsv.gz` (21 015 324 passages, 100-word windows, ids 1..21 015 324),
downloaded from `dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz`.

Subsample construction (`data/process_corpus.py`), deterministic given the seed:

1. **Gold-preserving:** every passage whose id appears in the DPR positive annotations for any NQ /
   TriviaQA train+dev query, plus every passage whose *title* matches a HotpotQA supporting-fact
   title (for any train+dev query), is force-included.
2. **Distractors:** `N_distractor` passages sampled uniformly at random from the 21 M dump.
3. Deduplicate by id, keep the original DPR id and title in the metadata so results remain
   traceable to the full dump.

Target `N_distractor = 2 000 000` (≈2.02 M corpus). Consequences are stated in the report: absolute
recall is optimistic w.r.t. the full 21 M corpus (fewer distractors ⇒ easier retrieval), but the
*relative* comparison between BM25 / dense / hybrid and the *routing* question — which is what this
paper studies — is unaffected, because all strategies see the identical corpus.

Gold coverage is asserted: every query used anywhere in the project must have ≥1 gold passage in the
subsampled corpus, else the query is dropped and the drop is logged (`results/corpus_coverage.json`).

---

## 4. Pipeline (Phase 1)

**Strategies** behind one interface, top-k default 5 (ablate k ∈ {3, 5, 10}):

| Strategy | Implementation | Cost |
|---|---|---|
| `none` | no context block | 1 generation, 0 retrieval |
| `bm25` | `bm25s` Okapi BM25 over the subsample | 1 BM25 query + 1 generation |
| `dense` | `BAAI/bge-base-en-v1.5` + FAISS `IndexFlatIP` (exact, fp16) | 1 dense query + 1 generation |
| `hybrid` | RRF(k=60) over bm25+dense; also a weighted-score variant (α sweep) | 1 BM25 + 1 dense + 1 generation |

**Prompt.** One template for all strategies; the *only* difference is whether the context block is
present:

```
System: You are a helpful assistant. Answer the question using only the provided context...
User:   [CONTEXT]  (k passages, each "Title: ...\nText: ...")   <- omitted for `none`
         Question: {q}
Answer:  <short answer>
```

Context is truncated to a fixed budget (`max_context_tokens`, default 1 400) by dropping passages
whole, then truncating the last one; prompt token counts are logged for every call.

**Generator.** vLLM offline engine, `temperature=0`, `top_p=1`, `seed` fixed, `max_tokens=32`,
`max_model_len=2048`. Logprobs are requested for the generated answer so the "no-retrieval
confidence" tier (logprob / entropy) is available **without an extra generation**.

**Per-query per-strategy record** (`results/logs/<run>/<dataset>__<split>__<strategy>.parquet`):

```
qid, dataset, split, strategy, k, question, golds[], pred,
em, f1, hit (gold passage retrieved), recall_gold,
retrieval_latency_ms, generation_latency_ms, total_latency_ms,
retrieved_ids[], retrieved_scores[], prompt_tokens, completion_tokens,
cost_proxy_tokens, n_retrieval_calls
```

**Cost proxy** (used for the cost-sensitive router and the Pareto curves):

```
cost_proxy = n_retrieval_calls * W_bm25 + n_retrieval_calls * W_dense + prompt_tokens + completion_tokens
```

with `W_bm25`, `W_dense` set to 1 (call-count proxy) in the primary analysis and to measured
*relative* latencies in the sensitivity analysis. Latency is additionally reported directly
(mean / p95 end-to-end) so the conclusion does not hinge on the proxy definition.

Everything is cached to parquet keyed by `(model, prompt-hash, strategy, qid)` so re-runs are free and
labeling is exactly reproducible.

---

## 5. Scale & compute budget (sized from the pilot)

The pilot (Phase 1) measures: vLLM startup, prefill/decode throughput on the 8 GB GPU, BM25 query
latency, dense query latency, encoding throughput for the corpus, index build time.

**Plan (to be confirmed by pilot numbers):**

| Set | Source | N/dataset | Generations (×4 strategies) |
|---|---|---|---|
| Pilot | train | 500 | 6 000 |
| Label pool | train | 2 000 | 24 000 |
| Dev-from-train (tuning) | train (disjoint) | 500 | 6 000 |
| Final eval | **official dev** | 1 000 | 12 000 |

Total ≈ 42 000 generations. Estimated GPU time at 7B-AWQ: prefill ≈ 850 prompt tokens for the 3
retrieval strategies and ≈ 60 for `none` ⇒ ≈ 32 M prefill tokens; at a measured ~2 000 tok/s ⇒ ≈ 4.5 h.
BM25 indexing of 2 M docs ≈ 15 min; dense encoding of 2 M passages ≈ 20 min; FAISS flat index build
≈ 5 min; retrieval for ~9 000 unique queries × 3 strategies ≈ 10 min. Everything else (router
training, evaluation, analysis) is CPU minutes.

**Timeline** (single machine, sequential phases, ~1 working day):

| Day | Phase | Deliverable |
|---|---|---|
| 1 (morning) | 0 + 1a | env, data, corpus subsample, indices |
| 1 (afternoon) | 1b + 1c | retrievers, vLLM engine, pilot run + status note |
| 1 (night) | 1c | full generation runs (background, logged) |
| 2 (morning) | 2 + 3 | labels, oracle stats, features, router training (3 seeds) |
| 2 (afternoon) | 4 + 5 | baselines, official-dev evaluation, CIs, ablations |
| 2 (evening) | 6 + deliverables | breakdowns, failure analysis, figures, report |

**Risk register**

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | vLLM/AWQ kernels unsupported on sm_120 | medium | critical | pilot first; fall back to bf16 3B, then to HF `generate` with batched 7B-4bit (slower but works) |
| R2 | 8 GB VRAM OOM at `max_model_len=2048` | medium | high | reduce to 1024–1536, `gpu_memory_utilization=0.92`, `max_num_seqs=16`, `enforce_eager` if needed |
| R3 | vLLM prefill too slow (>4 000 tok/s would be needed for the full run) | medium | high | shrink label pool to 1 500/dataset; reduce k to 3 for the ablation sweep; share prefixes |
| R4 | BM25 index over 2 M docs exceeds RAM | low | high | fall back to 1 M distractors; `bm25s` `retriever.batch_call` with top-k pruning |
| R5 | Gold-passage coverage gaps in the subsample (HotpotQA title→id mapping) | medium | medium | fall back to title-based gold detection (gold = title match) rather than id match; log coverage |
| R6 | Label noise from strict EM (aliases, dates, normalisation) | high | medium | report F1-based labels + ε sweep; treat as a first-class analysis category |
| R7 | Router "overfits" surface cues (dataset → strategy) | medium | medium | leave-one-dataset-out transfer + cross-dataset probe |
| R8 | Session/time budget exhausted mid-run | medium | medium | every phase commits to git and writes results incrementally; `make resume` continues |

---

## 6. Labels (Phase 2)

For a query *q* with per-strategy scores `s(none), s(bm25), s(dense), s(hybrid)` and measured
per-strategy costs `c`:

* `best = max_s s(s)`.
* **Hard label** = the *cheapest* strategy whose score ≥ `best − ε` (cost order `none < bm25 < dense <
  hybrid`, verified against measured latency; if the measured order disagrees, the measured order is
  used and the discrepancy is reported).
* Tie-breaking (explicit): if two strategies tie in score *and* cost, prefer the one with the higher
  EM, then the lexicographically earlier name (`bm25 < dense < hybrid < none` by cost rank; final
  fallback = lowest cost, then highest EM, then fixed name order).
* Primary metric for labelling: **EM** with ε = 0. Ablation: **F1** with ε ∈ {0, 0.05, 0.1}.
* **Soft labels**: per-strategy utility `u(s) = score(s) − λ · cost(s)` (λ swept); a cost-sensitive
  router can be trained by regressing on the argmax-utility strategy, which softens the hard
  ±1 EM cliff.
* Reported: label distribution per dataset, oracle accuracy + oracle cost, and the fraction of
  queries where `none` beats *every* retrieval strategy (**"retrieval hurts"** rate).

## 7. Router (Phase 3)

**Feature tiers** (each ablated; each has a measured cost):

| Tier | Features | Cost |
|---|---|---|
| `lex` | token/char length, wh-word one-hot, has-numeral, capitalised-entity count, IDF rare-term ratio, stopword ratio | ~free (µs) |
| `emb` | 768-d `bge-base-en-v1.5` query embedding (raw + SVD-64) | 1 encoder pass (~4 ms GPU / ~25 ms CPU) |
| `probe` | BM25 top-1 score, BM25 top-1 gap, dense top-1 similarity, dense margin | **1 full retrieval call** — must be charged! |
| `gen` | no-retrieval answer mean logprob, entropy, self-consistency agreement | 1 no-retrieval generation |

The `probe` tier is critical to reason about honestly: paying for a BM25 probe to decide *whether* to
retrieve means the retrieval call is no longer saved. We therefore (a) charge its cost explicitly in
the Pareto analysis and (b) report it separately as an "oracle-lite / pick-the-strategy" setting where
the probe cost is amortised.

**Models**: (a) multinomial logistic regression on `lex` (+SVD of `emb`), (b) gradient-boosted trees
(LightGBM), (c) 2-layer MLP on the raw embedding, (d) fine-tuned `microsoft/deberta-v3-small`
classifier. All are small enough that router latency (target: < 5 ms/query, measured and reported)
is negligible next to retrieval + generation.

**Training**: class-balanced / cost-sensitive weighting; multinomial calibration (temperature
scaling) fitted on the dev-from-train slice; a cost-penalty knob λ (retrieval "price") that traces
the accuracy–cost curve. Hyper-parameters are chosen *only* on dev-from-train.

**Variants**: per-dataset routers, one pooled router, leave-one-dataset-out (train on 2 datasets,
test on the third) to measure transfer.

## 8. Baselines (Phase 4) and evaluation (Phase 5)

Baselines on **identical queries and generator**: `never`, `always-bm25`, `always-dense`,
`always-hybrid`, `random` (matched to Pythia's strategy proportions), `oracle`, `confidence-threshold`
(answer w/o retrieval; retrieve if no-retrieval confidence < τ), and `complexity-classifier`
(Adaptive-RAG style simple-vs-complex, mapped to `none` vs `hybrid`).

Metrics: EM / F1 per dataset, retrieval call rate, mean & p95 end-to-end latency, total tokens
processed, accuracy–cost Pareto curves.

**Pre-registered success criterion** (fixed before running the official dev split):
Pythia (pooled router, best config chosen on dev-from-train) must reach
**EM ≥ (best always-retrieve baseline EM) − 1.0 absolute point** on the pooled official-dev
evaluation, with paired bootstrap 95 % CIs on the EM difference, **while** reducing retrieval calls
by ≥ 30 % relative to that baseline.

Statistical rigour: 10 000-sample bootstrap CIs; paired bootstrap test for the EM difference vs each
baseline; Wilcoxon signed-rank on per-query F1 as a secondary check; ≥ 3 seeds for router training
(mean ± std over seeds).

Ablations: label metric (EM/F1) × ε ∈ {0, 0.05, 0.1}; k ∈ {3, 5, 10}; feature tier subsets
(lex / +emb / +probe / +gen); router model family; cost-weight λ; per-dataset vs pooled vs LODO.

## 9. Analysis (Phase 6)

Breakdowns: dataset, wh-word, answer type (number / date / person / entity / other — rule-based
classifier + regex), query length quartile, answer popularity (token frequency of the answer in the
retrieval corpus as a proxy), single-hop vs multi-hop (HotpotQA `type` field: `comparison` /
`bridge` / `dev`), gold-retrieval success. Confusion matrix predicted × oracle strategy, with the
cost of each error type quantified (**under-retrieving** = oracle needed retrieval but router chose
`none`; **over-retrieving** = router paid for retrieval that didn't help).

Failure taxonomy, ≥ 50 manually inspected failures per dataset, each tagged with:
`router skipped a long-tail fact` / `retrieval returned distractors` / `multi-hop needs iterative
retrieval` / `answer-normalisation artefact` / `label noise from EM strictness` /
`generator hallucination` / `ambiguous or wrong gold`.

## 10. Deliverables checklist

- [ ] `PLAN.md` (this file), `docs/STATUS.md` (per-phase notes), `docs/DEVIATIONS.md`
- [ ] `README.md` with exact commands; `Makefile`; pinned `requirements.txt`; `configs/*.yaml`
- [ ] `results/logs/*.parquet` (raw per-query logs), `results/tables/*.csv` (aggregates),
      `results/run_manifest.json` (all model ids, seeds, versions, git SHA)
- [ ] `figures/`: accuracy–cost Pareto, label distributions, confusion matrices, failure-category bars,
      per-dataset breakdowns
- [ ] `docs/REPORT.md` — 6–8 page paper-style report
- [ ] Local git history, one commit per phase milestone (never pushed; user pushes manually)

---

## 11. Pre-registered commitments (to avoid post-hoc flexibility)

1. The success criterion in §8 (1.0 EM point + ≥ 30 % fewer retrieval calls) is fixed now.
2. The router's decision rule (argmax of calibrated per-strategy utility with a λ knob) is fixed now.
3. ε and the label metric are reported for the full grid; the "primary" configuration (EM, ε=0) is
   declared before any official-dev number is produced.
4. No number enters the report that was not computed from a file in `results/`.
5. Deviations from this plan are appended to `docs/DEVIATIONS.md` *with a reason*, and the report
   states them in a dedicated subsection.
