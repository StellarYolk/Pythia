#!/usr/bin/env python
"""Create a small synthetic dataset + generation logs for a pipeline smoke test.

This does *not* test retrieval or generation quality; it tests that stages 05-09 run
end-to-end, align queries correctly and produce every table/figure.  The logs mirror the
schema written by ``generation/runner.py`` exactly.

Usage (repo root):
    python tests/make_synthetic.py --root . --n-label 160 --n-tune 80 --n-eval 120
Then:
    python scripts/05_build_labels.py --config configs/default.yaml
    python scripts/06_train_router.py --config configs/default.yaml --models logreg --tiers lex
    python scripts/07_run_eval.py --config configs/default.yaml
    python scripts/08_analysis.py --config configs/default.yaml
    python scripts/09_report.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import hash_text  # noqa: E402

DATASETS = ["nq", "triviaqa", "hotpotqa"]
STRATEGIES = ["none", "bm25", "dense", "hybrid"]
COST_MS = {"none": 0.0, "bm25": 8.0, "dense": 18.0, "hybrid": 24.0}
CALLS = {"none": 0, "bm25": 1, "dense": 1, "hybrid": 2}

TEMPLATES = [
    "who wrote the novel {x}",
    "what is the capital of {x}",
    "when was {x} founded",
    "which river flows through {x}",
    "how many people live in {x}",
    "name the director of {x}",
    "what year did {x} happen",
    "where was {x} born",
]
ENTITIES = [
    "Aster", "Borealis", "Cinder", "Dunmore", "Elder", "Fenwick", "Gale", "Harrow",
    "Ionia", "Juniper", "Kestrel", "Lumen", "Morrow", "Nadir", "Orin", "Pell",
]


def make_queries(ds: str, subset: str, n: int, rng: np.random.Generator) -> List[Dict[str, Any]]:
    out = []
    for i in range(n):
        ent = ENTITIES[rng.integers(len(ENTITIES))] + str(rng.integers(1, 999))
        q = TEMPLATES[rng.integers(len(TEMPLATES))].format(x=ent)
        # longer questions are the "hard/long-tail" ones
        if rng.random() < 0.35:
            q = q + " according to the " + " ".join(["archive"] * rng.integers(2, 7))
        ans = str(int(rng.integers(1000, 2100))) if "year" in q or "when" in q else f"Answer {ent}"
        meta: Dict[str, Any] = {}
        if ds == "hotpotqa":
            meta = {"type": str(rng.choice(["bridge", "comparison"])), "level": str(rng.choice(["easy", "medium", "hard"]))}
        qid = f"{ds}-{subset}-{i:05d}"
        out.append(
            {
                "qid": qid,
                "dataset": ds,
                "subset": subset,
                "question": q,
                "answers": [ans, f"{ans} alias"],
                "meta": meta,
                "gold_titles": [f"Gold {ent}"],
                "n_gold_articles": 1,
                "has_gold": True,
            }
        )
    return out


def strategy_quality(q: Dict[str, Any], strategy: str, rng: np.random.Generator) -> float:
    """Latent per-query success probability that creates all four router labels."""
    n_words = len(q["question"].split())
    hard = min(1.0, max(0.0, (n_words - 6) / 20.0))
    low = n_words < 8
    p = {
        "none": 0.80 - 0.45 * hard,
        "bm25": 0.58 + 0.20 * hard,
        "dense": 0.62 + 0.18 * hard,
        "hybrid": 0.68 + 0.12 * hard,
    }[strategy]
    if q["dataset"] == "hotpotqa" and strategy != "none":
        p -= 0.10
    if low and strategy == "none":
        p += 0.08
    # query-specific taste so that no single strategy dominates
    p += rng.normal(0.0, 0.09)
    return float(np.clip(p, 0.03, 0.97))


def make_logs(queries: List[Dict[str, Any]], rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for q in queries:
        for s in STRATEGIES:
            success = rng.random() < strategy_quality(q, s, rng)
            em = 1.0 if success else 0.0
            f1 = float(np.clip(em if success else rng.random() * 0.45, 0, 1))
            hit = -1 if s == "none" else int(rng.random() < (0.35 + 0.45 * em))
            lat_retr = COST_MS[s] * float(rng.uniform(0.8, 1.25))
            lat_gen = float(rng.uniform(260.0, 360.0))
            n_ctx = 0 if s == "none" else int(rng.integers(120, 700))
            rows.append(
                {
                    "qid": q["qid"],
                    "dataset": q["dataset"],
                    "subset": q["subset"],
                    "strategy": s,
                    "k": 5 if s != "none" else 0,
                    "question": q["question"],
                    "answers": json.dumps(q["answers"]),
                    "pred": q["answers"][0] if em else f"wrong {rng.integers(1000)}",
                    "retrieved_ids": json.dumps(list(range(int(rng.integers(0, 50)), int(rng.integers(50, 100)))) if s != "none" else []),
                    "retrieved_titles": json.dumps([f"Gold {q['gold_titles'][0]}"] if hit == 1 else (["Other title"] if s != "none" else [])),
                    "retrieved_scores": json.dumps([float(rng.random()) for _ in range(5 if s != "none" else 0)]),
                    "n_gold_articles": 1,
                    "hit": hit,
                    "ctx_has_answer": int(s != "none" and hit == 1),
                    "n_passages": 5 if s != "none" else 0,
                    "prompt_tokens": 80 + n_ctx,
                    "completion_tokens": int(rng.integers(2, 9)),
                    "ctx_tokens": n_ctx,
                    "retrieval_latency_ms": lat_retr,
                    "generation_latency_ms": lat_gen,
                    "total_latency_ms": lat_retr + lat_gen,
                    "n_retrieval_calls": CALLS[s],
                    "mean_logprob": float(rng.uniform(-1.6, -0.1)),
                    "first_token_entropy": float(rng.uniform(0.0, 3.0)),
                    "mean_entropy": float(rng.uniform(0.0, 2.5)),
                    "first_token_margin": float(rng.uniform(0.0, 1.0)),
                    "prompt_hash": hash_text(q["qid"], s),
                    "model": "synthetic-model",
                    "em": em,
                    "f1": f1,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--run", default="main")
    ap.add_argument("--n-label", type=int, default=240)
    ap.add_argument("--n-tune", type=int, default=90)
    ap.add_argument("--n-eval", type=int, default=150)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    root = Path(args.root)
    qdir = root / "data" / "processed" / "queries"
    ldir = root / "results" / "logs" / args.run
    qdir.mkdir(parents=True, exist_ok=True)
    ldir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    sizes = {"label": args.n_label, "tune": args.n_tune, "eval": args.n_eval}
    for ds in DATASETS:
        for subset, n in sizes.items():
            qs = make_queries(ds, subset, n, rng)
            with open(qdir / f"{ds}__{subset}.jsonl", "w") as fh:
                for q in qs:
                    fh.write(json.dumps(q) + "\n")
            if subset == "eval":
                # eval logs are produced per strategy in stage 04; same schema
                pass
            df = make_logs(qs, rng)
            for s in STRATEGIES:
                df[df["strategy"] == s].to_parquet(ldir / f"{ds}__{subset}__{s}__k5.parquet", index=False)
    print(f"wrote synthetic queries to {qdir} and logs to {ldir}")
    print("next: python scripts/05_build_labels.py --config configs/default.yaml")


if __name__ == "__main__":
    main()
