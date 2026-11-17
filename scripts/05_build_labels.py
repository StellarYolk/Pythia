#!/usr/bin/env python
"""Phase 2 - build the router labels and the oracle statistics.

Usage:
    python scripts/05_build_labels.py --config configs/default.yaml [--run main]
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

from common import get_logger, load_config, write_json  # noqa: E402
from labeling.build_labels import (  # noqa: E402
    DEFAULT_COST_ORDER,
    build_labels,
    build_soft_labels,
    oracle_summary,
    verify_cost_order,
)

LOG = get_logger("build_labels")


def load_logs(log_dir: Path, run: str, subsets: List[str], datasets: List[str], k: int = 5) -> pd.DataFrame:
    frames = []
    for ds in datasets:
        for sub in subsets:
            for strategy in ("none", "bm25", "dense", "hybrid"):
                p = log_dir / run / f"{ds}__{sub}__{strategy}__k{k}.parquet"
                if p.exists():
                    frames.append(pd.read_parquet(p))
    if not frames:
        raise SystemExit("no logs found - run scripts/04_run_generation.py first")
    df = pd.concat(frames, ignore_index=True)
    LOG.info("loaded %d log rows from %d files", len(df), len(frames))
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--run", default="main")
    ap.add_argument("--subsets", default="label,tune")
    ap.add_argument("--datasets", default="nq,triviaqa,hotpotqa")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg["paths"]["root"])
    log_dir = root / cfg["paths"]["logs"]
    res_dir = root / cfg["paths"]["results"]
    res_dir.mkdir(parents=True, exist_ok=True)

    subsets = args.subsets.split(",")
    datasets = args.datasets.split(",")
    df = load_logs(log_dir, args.run, subsets, datasets, k=args.k)

    # only queries for which every strategy was run
    counts = df.groupby("qid")["strategy"].nunique()
    complete = set(counts[counts == 4].index)
    df = df[df["qid"].isin(complete)].copy()
    LOG.info("%d queries with all four strategies", len(complete))

    # ------------------------------------------------------------------ cost order
    cost = verify_cost_order(df, cfg["labels"]["cost_order"])
    write_json(res_dir / "cost_order.json", cost)
    LOG.info("measured latency order: %s (assumed %s, agrees=%s)",
             cost["measured_order"], cost["assumed_order"], cost["agrees"])
    cost_order = cost["measured_order"] if not cost["agrees"] else cfg["labels"]["cost_order"]

    # ------------------------------------------------------------------ hard labels
    primary_metric = cfg["labels"]["primary_metric"]
    epsilon = cfg["labels"]["epsilon"]
    labels = build_labels(df, metric=primary_metric, epsilon=epsilon, cost_order=cost_order)
    labels["run"] = args.run
    labels.to_parquet(res_dir / f"labels_{primary_metric}_eps{epsilon}_{args.run}.parquet", index=False)
    LOG.info("wrote %d labels (%s, eps=%.2f)", len(labels), primary_metric, epsilon)

    # epsilon grid on F1 (ablation)
    grid_rows: List[Dict[str, Any]] = []
    for metric in ("em", "f1"):
        for eps in cfg["labels"]["epsilon_grid"]:
            lab = build_labels(df, metric=metric, epsilon=eps, cost_order=cost_order)
            for ds, g in lab.groupby("dataset"):
                grid_rows.append(
                    {
                        "metric": metric,
                        "epsilon": eps,
                        "dataset": ds,
                        "n": len(g),
                        "dist": json.dumps(g["label"].value_counts(normalize=True).round(4).to_dict()),
                        "label_em": float((g["label_score"] > 0).mean()),
                        "oracle_em": float(g["oracle_score"].mean()),
                        "mean_calls": float(g["label_calls"].mean()),
                        "retrieval_hurts": float(g["retrieval_hurts"].mean()),
                    }
                )
    pd.DataFrame(grid_rows).to_csv(res_dir / "label_grid.csv", index=False)

    # ------------------------------------------------------------------ soft labels
    soft = build_soft_labels(df, metric=primary_metric, lambdas=cfg["router"]["cost_lambda_grid"])
    soft.to_parquet(res_dir / f"soft_labels_{args.run}.parquet", index=False)

    # ------------------------------------------------------------------ oracle stats
    oracle = oracle_summary(labels, df)
    write_json(res_dir / f"oracle_summary_{args.run}.json", oracle)
    LOG.info("oracle summary: %s", json.dumps(oracle, indent=2))

    # per-strategy baseline numbers (also used by the evaluation stage)
    strat_rows = []
    for (ds, sub, strategy), g in df.groupby(["dataset", "subset", "strategy"]):
        strat_rows.append(
            {
                "dataset": ds,
                "subset": sub,
                "strategy": strategy,
                "n": len(g),
                "em": float(g["em"].mean()),
                "f1": float(g["f1"].mean()),
                "hit": float(g["hit"].mean()),
                "ctx_has_answer": float(g["ctx_has_answer"].mean()),
                "mean_latency_ms": float(g["total_latency_ms"].mean()),
                "p95_latency_ms": float(g["total_latency_ms"].quantile(0.95)),
                "mean_retrieval_latency_ms": float(g["retrieval_latency_ms"].mean()),
                "mean_generation_latency_ms": float(g["generation_latency_ms"].mean()),
                "mean_prompt_tokens": float(g["prompt_tokens"].mean()),
                "mean_completion_tokens": float(g["completion_tokens"].mean()),
                "retrieval_calls": float(g["n_retrieval_calls"].mean()),
            }
        )
    strat_df = pd.DataFrame(strat_rows)
    strat_df.to_csv(res_dir / f"strategy_baselines_{args.run}.csv", index=False)
    LOG.info("strategy baselines:\n%s", strat_df.to_string(index=False))


if __name__ == "__main__":
    main()
