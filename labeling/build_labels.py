"""Label construction (Phase 2).

For every query we have one score per strategy.  The hard label is the *cheapest*
strategy whose score is within `epsilon` of the best score.

Cost order (cheapest first): none < bm25 < dense < hybrid.  This is verified against the
measured per-strategy latency in `verify_cost_order()`; if the measurement disagrees the
measured order is used and the discrepancy is reported.

Tie-breaking (explicit, in order):
  1. smaller cost rank
  2. higher EM
  3. higher F1
  4. lexicographically smaller strategy name
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from common import get_logger

LOG = get_logger("labeling")

DEFAULT_COST_ORDER = ["none", "bm25", "dense", "hybrid"]


def verify_cost_order(df: pd.DataFrame, cost_order: Sequence[str] = DEFAULT_COST_ORDER) -> Dict[str, Any]:
    """Check the assumed cost order against measured latency."""
    lat = df.groupby("strategy")["total_latency_ms"].mean().to_dict()
    measured = sorted(cost_order, key=lambda s: lat.get(s, float("inf")))
    return {
        "mean_total_latency_ms": {k: round(v, 2) for k, v in lat.items()},
        "assumed_order": list(cost_order),
        "measured_order": measured,
        "agrees": measured == list(cost_order),
    }


def build_labels(
    df: pd.DataFrame,
    metric: str = "em",
    epsilon: float = 0.0,
    cost_order: Sequence[str] = DEFAULT_COST_ORDER,
    tie_break_metric: str = "em",
) -> pd.DataFrame:
    """Assign one label per query.

    `df` must contain one row per (qid, strategy) with columns qid, strategy, em, f1.
    """
    if metric not in ("em", "f1"):
        raise ValueError(metric)
    strategies = [s for s in cost_order if s in set(df["strategy"].unique())]
    rank = {s: i for i, s in enumerate(strategies)}

    rows: List[Dict[str, Any]] = []
    for qid, g in df.groupby("qid", sort=False):
        scores = {r["strategy"]: float(r[metric]) for r in g.itertuples()}
        ems = {r["strategy"]: float(r["em"]) for r in g.itertuples()}
        f1s = {r["strategy"]: float(r["f1"]) for r in g.itertuples()}
        lat = {r["strategy"]: float(r["total_latency_ms"]) for r in g.itertuples()}
        calls = {r["strategy"]: int(r["n_retrieval_calls"]) for r in g.itertuples()}
        tokens = {r["strategy"]: int(r["prompt_tokens"]) + int(r["completion_tokens"]) for r in g.itertuples()}
        hits = {r["strategy"]: int(r["hit"]) for r in g.itertuples()}
        dataset = g["dataset"].iloc[0]
        subset = g["subset"].iloc[0]

        present = [s for s in strategies if s in scores]
        best_score = max(scores[s] for s in present)
        best_strategies = [s for s in present if scores[s] >= best_score - 1e-12]
        # candidates within epsilon of the best
        cand = [s for s in present if scores[s] >= best_score - epsilon - 1e-12]
        cand.sort(key=lambda s: (rank.get(s, 99), -ems.get(s, 0.0), -f1s.get(s, 0.0), s))
        label = cand[0]

        rows.append(
            {
                "qid": qid,
                "dataset": dataset,
                "subset": subset,
                "label": label,
                "best_strategy": min(best_strategies, key=lambda s: (rank.get(s, 99), s)),
                "best_score": best_score,
                "label_score": scores[label],
                "oracle_score": best_score,
                "score_gap": best_score - scores[label],
                "none_score": scores.get("none", float("nan")),
                "label_latency_ms": lat.get(label, float("nan")),
                "label_calls": calls.get(label, 0),
                "label_tokens": tokens.get(label, 0),
                "retrieval_hurts": int(best_strategies == ["none"]),
                "n_tied_best": len(best_strategies),
                **{f"score_{s}": scores.get(s, float("nan")) for s in strategies},
                **{f"hit_{s}": hits.get(s, -1) for s in strategies},
            }
        )
    out = pd.DataFrame(rows)
    LOG.info("labels: metric=%s eps=%.2f  distribution: %s", metric, epsilon, out["label"].value_counts().to_dict())
    return out


def build_soft_labels(
    df: pd.DataFrame,
    metric: str = "em",
    lambdas: Sequence[float] = (0.0, 0.002, 0.005, 0.01, 0.02, 0.05),
    cost_scale: str = "latency_ms",
) -> pd.DataFrame:
    """Per-strategy utility = score - lambda * cost (cost-sensitivity sweep)."""
    strategies = [s for s in DEFAULT_COST_ORDER if s in set(df["strategy"].unique())]
    out_rows = []
    for lam in lambdas:
        per_q: Dict[str, Dict[str, Any]] = {}
        for r in df.itertuples():
            q = per_q.setdefault(r.qid, {"dataset": r.dataset, "subset": r.subset, "u": {}, "s": {}})
            cost = float(r.total_latency_ms) if cost_scale == "latency_ms" else float(r.n_retrieval_calls)
            q["u"][r.strategy] = float(getattr(r, metric)) - lam * cost
            q["s"][r.strategy] = float(getattr(r, metric))
        for qid, q in per_q.items():
            best = max(q["u"], key=lambda s: (q["u"][s], -DEFAULT_COST_ORDER.index(s)))
            out_rows.append(
                {
                    "qid": qid,
                    "dataset": q["dataset"],
                    "subset": q["subset"],
                    "lambda": lam,
                    "soft_label": best,
                    "utility": q["u"][best],
                    **{f"u_{s}": q["u"].get(s, float("nan")) for s in strategies},
                    **{f"s_{s}": q["s"].get(s, float("nan")) for s in strategies},
                }
            )
    return pd.DataFrame(out_rows)


def oracle_summary(labels: pd.DataFrame, df: pd.DataFrame) -> Dict[str, Any]:
    """Oracle accuracy / cost and the 'retrieval hurts' rate."""
    per_ds = {}
    for ds, g in labels.groupby("dataset"):
        per_ds[ds] = {
            "n": int(len(g)),
            "oracle_em": float(g["oracle_score"].mean()),
            "oracle_f1": float(
                df[(df["dataset"] == ds)].groupby("qid")["f1"].max().mean()
            ),
            "mean_label_calls": float(g["label_calls"].mean()),
            "mean_label_latency_ms": float(g["label_latency_ms"].mean()),
            "retrieval_hurts_rate": float(g["retrieval_hurts"].mean()),
            "label_distribution": g["label"].value_counts(normalize=True).round(4).to_dict(),
        }
    return per_ds
