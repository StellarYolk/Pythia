#!/usr/bin/env python
"""Phase 5 - baselines + evaluation on the official dev splits.

Every system is evaluated through the same code path: a per-query strategy assignment
plus the per-strategy logs, so latency / token / retrieval-call accounting is identical.

Usage:
    python scripts/07_run_eval.py --config configs/default.yaml --run main
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import get_logger, load_config, read_jsonl, set_seed, write_json  # noqa: E402
from eval.baselines import (  # noqa: E402
    complexity_classifier,
    confidence_threshold,
    fixed,
    oracle,
    random_like,
    router_decision,
    strategy_proportions,
)
from eval.stats import bootstrap_ci, bootstrap_diff_ci, mcnemar, pareto_frontier, wilcoxon  # noqa: E402
from router.models import STRATEGIES, STRAT2IDX  # noqa: E402

LOG = get_logger("run_eval")

DATASETS = ["nq", "triviaqa", "hotpotqa"]
METRIC_COLS = [
    "em",
    "f1",
    "hit",
    "ctx_has_answer",
    "total_latency_ms",
    "retrieval_latency_ms",
    "generation_latency_ms",
    "prompt_tokens",
    "completion_tokens",
    "n_retrieval_calls",
]


def load_eval_logs(root: Path, run: str, k: int = 5) -> pd.DataFrame:
    frames = []
    for ds in DATASETS:
        for strategy in STRATEGIES:
            p = root / "results" / "logs" / run / f"{ds}__eval__{strategy}__k{k}.parquet"
            if p.exists():
                frames.append(pd.read_parquet(p))
    if not frames:
        raise SystemExit("no eval logs found")
    df = pd.concat(frames, ignore_index=True)
    counts = df.groupby("qid")["strategy"].nunique()
    df = df[df["qid"].isin(set(counts[counts == 4].index))].copy()
    LOG.info("eval logs: %d rows, %d queries", len(df), df["qid"].nunique())
    return df


def wide_frame(logs: pd.DataFrame) -> pd.DataFrame:
    """One row per query with the per-strategy metrics side by side (indexed by qid)."""
    keep = [c for c in METRIC_COLS if c in logs.columns]
    wide = logs.pivot_table(index=["qid", "dataset"], columns="strategy", values=keep, aggfunc="first")
    wide.columns = [f"{c}__{s}" for c, s in wide.columns]
    wide = wide.reset_index()
    q_meta = logs.groupby("qid").first().reset_index()[["qid", "question", "answers"]]
    wide = wide.merge(q_meta, on="qid", how="left")
    return wide.set_index("qid", drop=False).sort_index()


def metric_matrices(wide: pd.DataFrame, qids: Sequence[str]) -> Dict[str, np.ndarray]:
    """(N, 4) matrices of per-strategy metrics, aligned to ``qids``."""
    sub = wide.loc[list(qids)]
    out = {}
    for m in METRIC_COLS:
        cols = [f"{m}__{s}" for s in STRATEGIES]
        if all(c in sub.columns for c in cols):
            out[m] = sub[cols].to_numpy(dtype=float)
    return out


def score_system(
    wide: pd.DataFrame,
    matrices: Dict[str, np.ndarray],
    assignment: np.ndarray,
    qids: Sequence[str],
    name: str,
    extra: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Aggregate the metrics of a system given a per-query strategy assignment."""
    assignment = np.asarray(assignment, dtype=int)
    n = len(qids)
    sub = wide.loc[list(qids)]
    rows: Dict[str, Any] = {
        "qid": list(qids),
        "system": np.full(n, name, dtype=object),
        "strategy": np.array([STRATEGIES[i] for i in assignment], dtype=object),
        "dataset": sub["dataset"].to_numpy(),
    }
    for m, mat in matrices.items():
        rows[m] = mat[np.arange(n), assignment]
    df = pd.DataFrame(rows)
    if extra:
        for k, v in extra.items():
            df[k] = v
    return df


def summarize(df: pd.DataFrame, system: str) -> Dict[str, Any]:
    g = df[df["system"] == system]
    out: Dict[str, Any] = {
        "system": system,
        "n": int(len(g)),
        "em": float(g["em"].mean()),
        "f1": float(g["f1"].mean()),
        "hit": float(g["hit"].mean()),
        "ctx_has_answer": float(g["ctx_has_answer"].mean()),
        "retrieval_call_rate": float((g["n_retrieval_calls"] > 0).mean()),
        "mean_calls": float(g["n_retrieval_calls"].mean()),
        "mean_latency_ms": float(g["total_latency_ms"].mean()),
        "p95_latency_ms": float(g["total_latency_ms"].quantile(0.95)),
        "mean_prompt_tokens": float(g["prompt_tokens"].mean()),
        "total_tokens": float((g["prompt_tokens"] + g["completion_tokens"]).sum()),
        "cost_proxy": float((g["n_retrieval_calls"] * 1000 + g["prompt_tokens"] + g["completion_tokens"]).sum()),
        "strategy_mix": json.dumps(strategy_proportions(g["strategy"].map(STRAT2IDX).values)),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--run", default="main")
    ap.add_argument("--sig-top-k", type=int, default=25, help="limit paired tests to the top-k systems by EM")
    ap.add_argument("--k", type=int, default=5, help="retrieval depth of the evaluation logs")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"], init_cuda=False)
    root = Path(cfg["paths"]["root"])
    res = root / "results"
    tabs = res / "tables"
    tabs.mkdir(parents=True, exist_ok=True)

    logs = load_eval_logs(root, args.run, k=args.k)
    wide = wide_frame(logs)
    qids = wide.index.tolist()
    matrices = metric_matrices(wide, qids)
    LOG.info("evaluating %d queries", len(qids))

    cost_path = res / f"router_cost_model_{args.run}.json"
    cost_ms = json.loads(cost_path.read_text()) if cost_path.exists() else {"none": 0.0, "bm25": 1.0, "dense": 2.0, "hybrid": 3.0}

    systems: Dict[str, np.ndarray] = {}
    meta: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------ baselines
    systems["never_retrieve"] = fixed("none", qids)
    for s in ("bm25", "dense", "hybrid"):
        systems[f"always_{s}"] = fixed(s, qids)
    systems["oracle"] = oracle(logs, cfg["labels"]["primary_metric"], qids=qids)

    # confidence threshold (swept)
    for tau in [-1.5, -1.0, -0.7, -0.5, -0.3, -0.1, 0.0, 0.2]:
        systems[f"conf_thresh_{tau}"] = confidence_threshold(
            logs, tau, fallback="hybrid", confidence_col="mean_logprob", qids=qids
        )

    # ------------------------------------------------------------------ pythia
    pred_frames = []
    for pred_path in sorted(res.glob(f"router_predictions_{args.run}*.parquet")):
        pred_frames.append(pd.read_parquet(pred_path))
    if pred_frames:
        preds = pd.concat(pred_frames, ignore_index=True)
        if "variant" not in preds.columns:
            preds["variant"] = "pooled"
        if "heldout" not in preds.columns:
            preds["heldout"] = ""
        ev = preds[preds["subset"] == "eval"]
        for (variant, model_name, tier_tag, seed), sub in ev.groupby(["variant", "model", "tiers", "seed"]):
            # per_dataset / lodo produce one fold per dataset; the union of folds covers
            # every qid exactly once, so the combined system is scored across all three
            # datasets (the per-dataset breakdown shows the individual folds).
            sub = sub.drop_duplicates("qid").set_index("qid")
            if not set(qids) <= set(sub.index):
                continue
            probs = sub.loc[qids, [f"p_{x}" for x in STRATEGIES]].to_numpy(dtype=float)
            for lam in cfg["router"]["cost_lambda_grid"]:
                name = f"pythia_{variant}_{model_name}_{tier_tag}_s{seed}_lam{lam}"
                systems[name] = router_decision(probs, cost_ms, lam)
                meta[name] = {"variant": variant, "model": model_name, "tiers": tier_tag, "seed": int(seed), "lambda": lam}
        LOG.info("router systems: %d", len([s for s in systems if s.startswith("pythia")]))
    else:
        LOG.warning("no router predictions found - run scripts/06_train_router.py first")

    # random baseline matched to the mix of the primary pooled router at lambda=0
    pythia_systems = [s for s in systems if s.startswith("pythia")]
    if pythia_systems:
        primary = f"pythia_{cfg['eval'].get('primary_variant', 'pooled')}"
        preferred_tiers = cfg["eval"].get("primary_tiers", "lex-emb")
        mix_name = next(
            (s for s in pythia_systems if s.startswith(primary) and f"_{preferred_tiers}_" in s and s.endswith("lam0.0")),
            None,
        ) or next((s for s in pythia_systems if s.endswith("lam0.0")), pythia_systems[0])
        tmp = score_system(wide, matrices, systems[mix_name], qids, "tmp")
        props = strategy_proportions(tmp["strategy"].map(STRAT2IDX).values)
        for seed in cfg["router"]["seeds"]:
            systems[f"random_like_pythia_s{seed}"] = random_like(qids, props, seed=seed)
        meta["random_like_props"] = props
        meta["random_like_source"] = mix_name

    # complexity classifier (Adaptive-RAG style): simple -> none, complex -> hybrid
    tiers = cfg["router"]["feature_tiers"]
    feats_eval = res / "router_features" / f"eval__{'-'.join(tiers)}__k{args.k}.parquet"
    feats_label = res / "router_features" / f"label__{'-'.join(tiers)}__k{args.k}.parquet"
    if feats_eval.exists() and feats_label.exists():
        fe = pd.read_parquet(feats_eval)
        fl = pd.read_parquet(feats_label)
        lab = pd.read_parquet(res / f"labels_{cfg['labels']['primary_metric']}_eps{cfg['labels']['epsilon']}_{args.run}.parquet")
        # the classifier returns one assignment per row of `fe`, in `fe` order
        assign = complexity_classifier(fl, fe, lab, seed=cfg["seed"])
        mapping = dict(zip(fe["qid"].tolist(), assign))
        systems["complexity_classifier"] = np.array([mapping.get(q, STRAT2IDX["hybrid"]) for q in qids])

    # ------------------------------------------------------------------ score everything
    started = time.time()
    per_query = []
    summaries = []
    for name, assignment in systems.items():
        df = score_system(wide, matrices, assignment, qids, name)
        per_query.append(df)
        summaries.append(summarize(df, name))
    per_q = pd.concat(per_query, ignore_index=True)
    per_q.to_parquet(res / f"eval_per_query_{args.run}.parquet", index=False)

    summary_df = pd.DataFrame(summaries).sort_values("em", ascending=False)
    summary_df.to_csv(tabs / f"eval_summary_{args.run}.csv", index=False)
    LOG.info("scored %d systems in %.1f s", len(summary_df), time.time() - started)

    # per-dataset breakdown
    per_ds = []
    for (sysname, ds), g in per_q.groupby(["system", "dataset"]):
        per_ds.append(
            {
                "system": sysname,
                "dataset": ds,
                "n": len(g),
                "em": float(g["em"].mean()),
                "f1": float(g["f1"].mean()),
                "hit": float(g["hit"].mean()),
                "ctx_has_answer": float(g["ctx_has_answer"].mean()),
                "retrieval_call_rate": float((g["n_retrieval_calls"] > 0).mean()),
                "mean_latency_ms": float(g["total_latency_ms"].mean()),
                "p95_latency_ms": float(g["total_latency_ms"].quantile(0.95)),
                "total_tokens": float((g["prompt_tokens"] + g["completion_tokens"]).sum()),
            }
        )
    pd.DataFrame(per_ds).to_csv(tabs / f"eval_per_dataset_{args.run}.csv", index=False)

    # ------------------------------------------------------------------ significance
    always = [s for s in systems if s.startswith("always_")]
    best_always = max(always, key=lambda s: summary_df.loc[summary_df["system"] == s, "em"].iloc[0])
    LOG.info("best always-retrieve baseline: %s (EM=%.4f)", best_always, summary_df.loc[summary_df["system"] == best_always, "em"].iloc[0])

    # Paired bootstrap is O(systems * n * n_boot); restrict it to the top-k systems by EM
    # (plus every Pythia system that passes the pre-registered criterion, see below).
    base = per_q[per_q["system"] == best_always].set_index("qid")
    n_boot = cfg["eval"]["bootstrap_samples"]
    seed = cfg["eval"]["bootstrap_seed"]
    top_systems = summary_df["system"].head(max(1, args.sig_top_k)).tolist()
    sig_rows = []
    for sysname in top_systems:
        if sysname == best_always:
            continue
        s = per_q[per_q["system"] == sysname].set_index("qid")
        common = base.index.intersection(s.index)
        d = bootstrap_diff_ci(s.loc[common, "em"].values, base.loc[common, "em"].values, n_samples=n_boot, seed=seed)
        w = wilcoxon(s.loc[common, "f1"].values, base.loc[common, "f1"].values)
        mc = mcnemar((s.loc[common, "em"] > 0).values, (base.loc[common, "em"] > 0).values)
        sig_rows.append(
            {
                "system": sysname,
                "baseline": best_always,
                "em_diff": d["diff"],
                "em_diff_lo": d["lo"],
                "em_diff_hi": d["hi"],
                "p_bootstrap": d["p_two_sided"],
                "p_wilcoxon_f1": w["p_value"],
                "p_mcnemar_em": mc["p_value"],
                "n": len(common),
            }
        )
    pd.DataFrame(sig_rows).to_csv(tabs / f"eval_significance_{args.run}.csv", index=False)

    # ------------------------------------------------------------------ success criterion
    margin = float(cfg["eval"]["success_margin_em"])
    min_cut = float(cfg["eval"]["success_min_call_reduction"])
    succ = []
    for sysname in pythia_systems:
        row = summary_df[summary_df["system"] == sysname].iloc[0].to_dict()
        base_row = summary_df[summary_df["system"] == best_always].iloc[0]
        em_gap = row["em"] - base_row["em"]
        call_reduction = 1.0 - row["retrieval_call_rate"] / max(1e-9, base_row["retrieval_call_rate"])
        lat_reduction = 1.0 - row["mean_latency_ms"] / max(1e-9, base_row["mean_latency_ms"])
        succ.append(
            {
                "system": sysname,
                **{k: meta.get(sysname, {}).get(k) for k in ("variant", "model", "tiers", "seed", "lambda")},
                "em": row["em"],
                "best_always_em": base_row["em"],
                "em_gap": em_gap,
                "within_margin": bool(em_gap >= -margin),
                "retrieval_call_rate": row["retrieval_call_rate"],
                "call_reduction": call_reduction,
                "meets_call_reduction": bool(call_reduction >= min_cut),
                "mean_latency_ms": row["mean_latency_ms"],
                "latency_reduction": lat_reduction,
                "success": bool(em_gap >= -margin and call_reduction >= min_cut),
            }
        )
    succ_df = pd.DataFrame(succ).sort_values("em", ascending=False)
    succ_df.to_csv(tabs / f"success_criterion_{args.run}.csv", index=False)
    LOG.info("success criterion:\n%s", succ_df.head(10).to_string(index=False))

    write_json(
        res / f"eval_meta_{args.run}.json",
        {
            "best_always": best_always,
            "n_queries": len(qids),
            "datasets": DATASETS,
            "router_cost_model": cost_ms,
            "random_like_props": meta.get("random_like_props"),
            "random_like_source": meta.get("random_like_source"),
            "success_margin_em": margin,
            "success_min_call_reduction": min_cut,
            "n_success": int(succ_df["success"].sum()) if len(succ_df) else 0,
            "systems": len(systems),
        },
    )


if __name__ == "__main__":
    main()
