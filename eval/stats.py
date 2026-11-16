"""Evaluation statistics: bootstrap CIs, paired tests, Pareto curves."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------
# bootstrap
# --------------------------------------------------------------------------------------
def bootstrap_ci(
    values: Sequence[float],
    n_samples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
    stat=np.mean,
) -> Tuple[float, float]:
    """Percentile bootstrap CI of a statistic."""
    v = np.asarray(values, dtype=np.float64)
    if len(v) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_samples, len(v)))
    stats = stat(v[idx], axis=1)
    lo = float(np.quantile(stats, alpha / 2))
    hi = float(np.quantile(stats, 1 - alpha / 2))
    return lo, hi


def bootstrap_diff_ci(
    a: Sequence[float],
    b: Sequence[float],
    n_samples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, float]:
    """Paired bootstrap CI for mean(a) - mean(b) (a and b are aligned per query)."""
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    assert len(x) == len(y), "paired bootstrap needs aligned arrays"
    d = x - y
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_samples, len(d)))
    stats = d[idx].mean(axis=1)
    return {
        "diff": float(d.mean()),
        "lo": float(np.quantile(stats, alpha / 2)),
        "hi": float(np.quantile(stats, 1 - alpha / 2)),
        "p_two_sided": float(2 * min((stats <= 0).mean(), (stats >= 0).mean())),
    }


def wilcoxon(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Two-sided Wilcoxon signed-rank test on paired per-query scores."""
    from scipy.stats import wilcoxon as _w

    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    try:
        stat, p = _w(x, y, zero_method="zsplit", alternative="two-sided")
        return {"statistic": float(stat), "p_value": float(p)}
    except Exception as exc:  # pragma: no cover
        return {"statistic": float("nan"), "p_value": float("nan"), "error": str(exc)}


def mcnemar(a: Sequence[bool], b: Sequence[bool]) -> Dict[str, float]:
    """Exact McNemar test on paired binary outcomes (EM)."""
    from scipy.stats import binomtest

    x = np.asarray(a, dtype=bool)
    y = np.asarray(b, dtype=bool)
    b01 = int(np.sum(~x & y))
    b10 = int(np.sum(x & ~y))
    n = b01 + b10
    if n == 0:
        return {"statistic": 0.0, "p_value": 1.0, "n01": b01, "n10": b10}
    p = binomtest(b10, n, 0.5).pvalue
    return {"statistic": float(b10 - b01), "p_value": float(p), "n01": b01, "n10": b10}


# --------------------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------------------
def aggregate(df: pd.DataFrame, group_cols: Sequence[str], metrics: Sequence[str], n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Group-wise means with bootstrap CIs."""
    rows = []
    for keys, g in df.groupby(list(group_cols), dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, keys))
        for m in metrics:
            if m not in g.columns:
                continue
            vals = g[m].values.astype(float)
            lo, hi = bootstrap_ci(vals, n_samples=n_boot, seed=seed)
            row[m] = float(np.nanmean(vals))
            row[f"{m}_lo"] = lo
            row[f"{m}_hi"] = hi
        row["n"] = int(len(g))
        rows.append(row)
    return pd.DataFrame(rows)


def pareto_frontier(points: Sequence[Tuple[float, float]]) -> List[int]:
    """Indices of the Pareto-optimal points for (cost, accuracy) - minimise cost, maximise accuracy."""
    pts = list(points)
    keep = []
    for i, (c, a) in enumerate(pts):
        dominated = any((c2 <= c and a2 >= a) and (c2, a2) != (c, a) for j, (c2, a2) in enumerate(pts) if j != i)
        if not dominated:
            keep.append(i)
    return keep


def summary_table(df: pd.DataFrame, group_cols: Sequence[str], metrics: Sequence[str]) -> pd.DataFrame:
    """Mean +/- CI table used for the report."""
    out = df.groupby(list(group_cols))[list(metrics)].agg(["mean", "count"])
    out.columns = ["_".join(c) for c in out.columns]
    return out.reset_index()
