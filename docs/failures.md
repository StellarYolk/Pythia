# Failure taxonomy and review protocol

Stage 08 writes two files:

- `results/tables/failures_all_<run>.csv` — every failed primary-system query with its
  rule-based category.
- `results/tables/failures_<run>.csv` — a deterministic sample of up to
  `eval.failure_sample_per_dataset` (default 50) failures per dataset for manual review,
  with `sampled=True`.

Columns: `qid, dataset, question, answer, answers, strategy, best_strategy, em,
oracle_em, hit, failure`.

## Categories

| Category | Meaning | Typical fix |
| --- | --- | --- |
| `router_skipped_long_tail_fact` | router chose `none`, but a retrieval strategy would have answered (under-retrieval) | richer lexical/entity features, lower λ, calibration |
| `retrieval_returned_distractors` | retrieval ran, gold title was not retrieved (`hit=0`) | better retriever, larger k, hybrid weighting |
| `multi_hop_needs_iterative_retrieval` | HotpotQA bridge/comparison failure with no gold hit; single-shot retrieval is structurally insufficient | iterative/multi-step retrieval (out of scope here) |
| `retrieval_hurt_over_retrieval` | router paid for retrieval although `none` would have answered and the retrieved context derailed the generator (over-retrieval) | higher λ, context filtering, better calibration |
| `answer_normalisation_artefact` | answer is a near-miss after normalisation (numbers, dates, punctuation) | manual inspection; consider relaxed matching as a secondary metric |
| `label_noise_em_strictness` | strict EM disagreement between near-equivalent answers | report F1, inspect aliases |
| `generator_hallucination` | gold context was retrieved (`hit=1`) but the model still answered incorrectly | prompt/model change |
| `ambiguous_or_wrong_gold` | question or gold is ambiguous/wrong on inspection | exclude or flag |
| `no_failure` | EM = 1 | — |

The rules are a **first pass**: they are deterministic and deliberately conservative.
`answer_normalisation_artefact`, `label_noise_em_strictness` and `ambiguous_or_wrong_gold`
require a human decision and are therefore assigned during review, not automatically.

## Review protocol (pending until a real run)

1. Run stages 05–09 on the official-dev logs.
2. Open `results/tables/failures_<run>.csv`. Review **at least 50 failures per dataset**
   (the sample is already capped at 50/dataset by default).
3. For each row, read the question, gold aliases, the retrieved titles (`hit`) and the
   prediction. Keep the rule category unless there is clear evidence for another one;
   change it in a copy (`docs/failures_reviewed.csv`, not tracked by the report script).
4. Record counts per category and dataset in the report. If a category has fewer than
   five examples, say so rather than over-interpreting it.
5. Add 3–5 representative examples (question, prediction, gold, category, one-line
   diagnosis) to the report's error-analysis section.

## Rules of thumb

- If `oracle_em = 0`, no strategy could answer: this is a retrieval/generation ceiling,
  not a routing error.
- If `strategy == best_strategy` and `em = 0`, the router made the right choice; look at
  `hit` and the prediction.
- A large `retrieval_hurt_over_retrieval` count is evidence that the cost knob is too
  aggressive at the selected λ.
