"""Figures for the Pythia report (headless matplotlib, deterministic styling)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from eval.stats import pareto_frontier  # noqa: E402

STRATEGY_COLORS = {
    "none": "#4c72b0",
    "bm25": "#dd8452",
    "dense": "#55a868",
    "hybrid": "#c44e52",
}


def _save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_pareto(
    summary: pd.DataFrame,
    out_path: str | Path,
    x: str = "retrieval_call_rate",
    y: str = "em",
    annotate: Sequence[str] = (),
    title: str = "Accuracy vs. retrieval-call rate",
) -> Path:
    """Scatter every system and draw the Pareto frontier (lower cost, higher accuracy)."""
    df = summary.copy()
    pts = list(zip(df[x].astype(float), df[y].astype(float)))
    keep = set(pareto_frontier(pts))
    fig, ax = plt.subplots(figsize=(8, 5.5))
    is_pythia = df["system"].str.startswith("pythia")
    ax.scatter(df.loc[~is_pythia, x], df.loc[~is_pythia, y], s=32, c="#888888", alpha=0.75, label="baselines")
    ax.scatter(df.loc[is_pythia, x], df.loc[is_pythia, y], s=22, c="#1f77b4", alpha=0.65, label="Pythia routers")
    front = df.iloc[sorted(keep)].sort_values(x)
    ax.plot(front[x], front[y], "--", color="#d62728", lw=1.4, alpha=0.8, label="Pareto frontier")
    labels = set(annotate) | set(df.loc[df.index.isin(keep), "system"].head(12))
    for i, row in df.iterrows():
        if row["system"] in labels:
            ax.annotate(row["system"], (row[x], row[y]), fontsize=6.5, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("retrieval-call rate")
    ax.set_ylabel("EM")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    return _save(fig, Path(out_path))


def plot_label_distribution(labels: pd.DataFrame, out_path: str | Path) -> Path:
    """Share of router labels per dataset."""
    tab = labels.groupby(["dataset", "label"]).size().unstack(fill_value=0)
    tab = tab.div(tab.sum(axis=1), axis=0)
    order = [s for s in ("none", "bm25", "dense", "hybrid") if s in tab.columns]
    tab = tab[order]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bottom = np.zeros(len(tab))
    for s in order:
        ax.bar(tab.index, tab[s].values, bottom=bottom, label=s, color=STRATEGY_COLORS.get(s))
        bottom += tab[s].values
    ax.set_ylabel("share of queries")
    ax.set_title("Router label distribution (EM, cheapest within epsilon)")
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    return _save(fig, Path(out_path))


def plot_confusion(cm: pd.DataFrame, out_path: str | Path, title: str = "Predicted vs. oracle strategy") -> Path:
    """Row-normalised confusion matrix."""
    mat = cm.to_numpy(dtype=float)
    row_sums = mat.sum(axis=1, keepdims=True)
    norm = np.divide(mat, row_sums, out=np.zeros_like(mat), where=row_sums > 0)
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(cm.columns)), cm.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(cm.index)), cm.index)
    ax.set_xlabel("oracle strategy")
    ax.set_ylabel("router strategy")
    ax.set_title(title)
    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            ax.text(j, i, f"{norm[i, j]:.2f}\n({int(mat[i, j])})", ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.55 else "#222222")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row share")
    return _save(fig, Path(out_path))


def plot_failure_bars(failures: pd.DataFrame, out_path: str | Path) -> Path:
    """Failure-category counts, split by dataset."""
    tab = failures[failures["failure"] != "no_failure"].groupby(["dataset", "failure"]).size().unstack(fill_value=0)
    if tab.empty:
        tab = pd.DataFrame({"no_failures": [0]})
    fig, ax = plt.subplots(figsize=(9, 4.6))
    tab.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_ylabel("queries")
    ax.set_title("Failure taxonomy")
    ax.legend(fontsize=6.5, ncol=2, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    return _save(fig, Path(out_path))


def plot_breakdown(
    df: pd.DataFrame,
    out_path: str | Path,
    group_col: str,
    metric: str = "em",
    title: Optional[str] = None,
) -> Path:
    """Grouped bars: one panel per dataset, primary system vs oracle."""
    datasets = sorted(df["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.6 * len(datasets), 3.8), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        sub = df[df["dataset"] == ds]
        piv = sub.pivot_table(index=group_col, columns="system", values=metric, aggfunc="mean")
        piv.plot(kind="bar", ax=ax, color={"primary": "#1f77b4", "oracle": "#d62728"}, legend=False)
        ax.set_title(ds)
        ax.set_xlabel("")
        ax.set_ylabel(metric.upper())
        ax.tick_params(axis="x", rotation=35, labelsize=7)
        ax.grid(axis="y", alpha=0.25)
    axes[0][0].legend(fontsize=8)
    fig.suptitle(title or f"{metric.upper()} by {group_col}", y=1.02)
    return _save(fig, Path(out_path))


def plot_ablation(
    df: pd.DataFrame,
    out_path: str | Path,
    x: str,
    y: str = "em",
    hue: str = "model",
    title: str = "Ablation",
) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for key, g in df.groupby(hue):
        g = g.sort_values(x)
        ax.plot(g[x].astype(str), g[y], marker="o", label=str(key))
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    return _save(fig, Path(out_path))
