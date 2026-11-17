# Data card

Pythia uses **public open-domain QA data only**. No user data, no private corpora, no
crowdworkers.

## Query sets

| Dataset | Source | Config / splits used | License (upstream) |
| --- | --- | --- | --- |
| Natural Questions (open) | `google-research-datasets/nq_open` | `train` (label/tune pools), official `validation` (eval) | CC BY-SA 3.0 |
| TriviaQA | `mandarjoshi/trivia_qa`, `unfiltered.nocontext` | `train` (label/tune), official `validation` (eval) | Apache-2.0 (dataset packaging) |
| HotpotQA | `hotpotqa/hotpot_qa`, `fullwiki` | `train` (label/tune), official `validation` (eval) | CC BY-SA 4.0 |

Splits are sampled per dataset:

| Subset | Purpose | Default size / dataset | Trimmed budget |
| --- | --- | --- | --- |
| `label` | router training pool | 2000 usable | 1500 |
| `tune` | temperature / configuration selection | 500 | 400 |
| `eval` | final numbers (official dev) | 1000 | 700 |

`configs/default.yaml` states the default; `configs/trimmed.yaml` the agreed smaller
budget. The data stage over-samples the train splits (`oversample`: NQ 1.6×, TriviaQA
1.35×, HotpotQA 1.0×) and keeps only queries whose question maps to at least one DPR gold
article. From the original run log: 10,282 queries survived filtering (NQ
2000/500/1000; TriviaQA 2000/500/791; HotpotQA 2000/491/1000 for label/tune/eval).

Answers are kept as the upstream alias lists; EM/F1 use the official SQuAD-style
normalisation in `common.py`.

## Retrieval corpus

- DPR Wikipedia passage dump: `psgs_w100.tsv.gz` (~21M passages, 100 words each), the
  standard open-domain retrieval corpus.
- The index used in the experiments is a **subsample**: 1,000,000 random non-gold
  passages plus every gold passage for the kept queries, deduplicated by title/passage.
  Passage ids are DPR ids; titles are Wikipedia titles.
- The corpus store (`retrieval/corpus_store.py`) writes `corpus.jsonl` plus a passage
  blob; BM25 and dense indices live next to it under `data/indices/`.
- **Known bias:** the 1M distractor sample makes retrieval easier than the full 21M
  corpus. The report must state this; absolute retrieval numbers are therefore an
  upper bound relative to full-wiki retrieval.

## Gold annotations

- NQ and TriviaQA: DPR's question→positive-title mapping (only questions with a DPR
  positive are kept); NQ dev has 5,498 question→title pairs in the DPR annotations.
- HotpotQA: the two gold titles in the `fullwiki` record; `meta.type` (`bridge` /
  `comparison`) and `meta.level` (`easy` / `medium` / `hard`) are preserved for
  breakdowns.
- `hit` is 1 when a retrieved title matches a gold title, 0 otherwise, and -1 for the
  `none` strategy (no retrieval). It is a title-level proxy, not answer-level recall.

## Privacy and ethics

- All questions, answers and passages are public Wikipedia/HotpotQA/TriviaQA content.
- No demographic attributes are collected. Breakdowns use only question surface form,
  answer type, length, popularity proxies and HotpotQA's provided type/level.
- No model is trained on evaluation answers; routers see labels derived from the
  `label` split only. The `tune` split is used for calibration/selection and `eval` is
  held out for final numbers.

## Generated artefacts

`tests/make_synthetic.py` creates **synthetic** queries/answers/logs for plumbing tests.
They are written to a temporary directory by `tests/smoke.sh` and must never be reported
as experimental data.
