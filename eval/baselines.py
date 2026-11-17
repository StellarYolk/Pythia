"""Baselines (Phase 4).

Every baseline produces, for each evaluation query, an assignment of one of the four
strategies, so that all systems are scored through exactly the same code path (same
per-strategy logs, same latency accounting, same cost proxy).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from common import get_logger
from router.models import STRATEGIES, STRAT2IDX

LOG = get_logger("eval.baselines")


def fixed(strategy: str, qids: Sequence[str]) -> np.ndarray:
    return np.full(len(qids), STRAT2IDX[strategy], dtype=int)


def random_like(qids: Sequence[str], proportions: Dict[str, float], seed: int = 0) -> np.ndarray:
    """Random router matched to the strategy mix of the system it mimics."""
    rng = np.random.default_rng(seed)
    probs = np.array([proportions.get(s, 0.0) for s in STRATEGIES], dtype=float)
    if probs.sum() <= 0:
        probs = np.ones(len(STRATEGIES)) / len(STRATEGIES)
    probs = probs / probs.sum()
    return rng.choice(len(STRATEGIES), size=len(qids), p=probs)


def oracle(logs: pd.DataFrame, metric: str = "em", qids: Optional[Sequence[str]] = None) -> np.ndarray:
    """Upper bound: the strategy with the best score for each query.

    If ``qids`` is supplied, the result is aligned to that order (the pivot index is
    sorted, which is *not* necessarily the evaluation order).
    """
    piv = logs.pivot_table(index="qid", columns="strategy", values=metric)
    piv = piv[[s for s in STRATEGIES if s in piv.columns]]
    if qids is not None:
        piv = piv.reindex(list(qids)).fillna(0.0)
    return piv.values.argmax(axis=1)


def confidence_threshold(
    logs: pd.DataFrame,
    threshold: float,
    fallback: str = "hybrid",
    confidence_col: str = "mean_logprob",
    higher_is_confident: bool = True,
    qids: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Answer without retrieval unless the generator is not confident enough.

    `confidence_col` comes from the no-retrieval run: mean log-prob (higher = more
    confident) or entropy (lower = more confident).  The output is aligned to ``qids``
    when supplied, otherwise to the (sorted) unique query order.
    """
    none = logs[logs["strategy"] == "none"].set_index("qid")
    conf = none[confidence_col]
    if not higher_is_confident:
        conf = -conf
    qids = list(qids) if qids is not None else sorted(logs["qid"].unique())
    conf = conf.reindex(qids)
    use_retrieval = (conf.isna()) | (conf < threshold)
    out = np.array([STRAT2IDX["none"]] * len(qids), dtype=int)
    out[use_retrieval.values] = STRAT2IDX[fallback]
    return out


def complexity_classifier(
    train_features: pd.DataFrame,
    eval_features: pd.DataFrame,
    labels: pd.DataFrame,
    seed: int = 0,
    threshold: float = 0.5,
) -> np.ndarray:
    """Adaptive-RAG style simple/complex classifier mapped onto our strategy set.

    A binary logistic regression decides "simple" (answer from parameters) vs "complex"
    (needs retrieval); we map simple -> `none` and complex -> `hybrid`.  It is trained on
    the router's *label* split and applied to the evaluation split.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    cols = [
        c
        for c in train_features.columns
        if c not in ("qid", "dataset", "subset", "question", "label")
        and c in eval_features.columns
        and pd.api.types.is_numeric_dtype(train_features[c])
    ]
    merged = train_features.merge(labels[["qid", "label"]], on="qid", how="inner")
    y = (merged["label"] != "none").astype(int).values
    X = merged[cols].values.astype(float)
    if len(set(y)) < 2:
        return np.full(len(eval_features), STRAT2IDX["none"], dtype=int)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    clf.fit(Xs, y)
    Xe = eval_features[cols].values.astype(float)
    p = clf.predict_proba(scaler.transform(Xe))[:, 1]
    out = np.where(p >= threshold, STRAT2IDX["hybrid"], STRAT2IDX["none"])
    return out.astype(int)


def router_decision(probs: np.ndarray, cost_ms: Dict[str, float], lam: float) -> np.ndarray:
    """Cost-aware decision rule: argmax_s log p(s) - lam * cost(s)."""
    logp = np.log(np.clip(probs, 1e-12, None))
    costs = np.array([cost_ms.get(s, 0.0) for s in STRATEGIES], dtype=np.float64)
    return (logp - lam * costs[None, :]).argmax(axis=1)


def strategy_proportions(assignments: np.ndarray) -> Dict[str, float]:
    counts = np.bincount(assignments, minlength=len(STRATEGIES)).astype(float)
    counts = counts / max(1.0, counts.sum())
    return {s: float(counts[i]) for i, s in enumerate(STRATEGIES)}
