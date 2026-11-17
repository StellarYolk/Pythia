#!/usr/bin/env python
"""Phase 6 - breakdowns, error-cost analysis, failure taxonomy and figures.

Consumes the artefacts written by stages 05-07 and never re-runs retrieval or
generation.  All numbers in the report come from files written here or in stage 07.

Usage:
    python scripts/08_analysis.py --config configs/default.yaml --run main
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.breakdowns import (  # noqa: E402
    FAILURE_CATEGORIES,
    add_breakdown_columns,
    categorise_failure,
)
from analysis import figures as fig  # noqa: E402
from common import get_logger, load_config, read_json, read_jsonl, set_seed, write_json  # noqa: E402
from router.models import STRATEGIES  # noqa: E402

LOG = get_logger("analysis")

DATASETS = ["nq", "triviaqa", "hotpotqa"]


def load_queries_meta(root: Path, subset: str = "eval") -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for ds in DATASETS:
        p = root / "data" / "processed" / "queries" / f"{ds}__{subset}.jsonl"
        if not p.exists():
            continue
        for q in read_jsonl(p):
            meta = q.get("meta") or {}
            rows.append(
                {
                    "qid": q["qid"],
                    "question": q.get("question", ""),
                    "answers": json.dumps(q.get("answers", [])),
                    "hp_type": meta.get("type", ""),
                    "hp_level": meta.get("level", ""),
                }
            )
    return pd.DataFrame(rows)


def load_eval_logs(root: Path, run: str, k: int = 5) -> pd.DataFrame:
    frames = []
    for ds in DATASETS:
        for strategy in STRATEGIES:
            p = root / "results" / "logs" / run / f"{ds}__eval__{strategy}__k{k}.parquet"
            if p.exists():
                frames.append(pd.read_parquet(p))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def pick_primary(summary: pd.DataFrame, succ: Optional[pd.DataFrame], cfg: Dict[str, Any], override: Optional[str]) -> str:
    if override:
        return override
    pythia = summary[summary["system"].str.startswith("pythia_")]
    if pythia.empty:
        return summary.sort_values("em", ascending=False)["system"].iloc[0]
    variant = cfg["eval"].get("primary_variant", "pooled")
    tiers = cfg["eval"].get("primary_tiers", "lex-emb")
    cand = pythia[pythia["system"].str.contains(f"_pythia_{variant}_|^pythia_{variant}_", regex=True)]
    cand = cand[cand["system"].str.contains(f"_{tiers}_", regex=False)]
    if cand.empty:
        cand = pythia
    if succ is not None and len(succ):
        ok = cand.merge(succ[["system", "success"]], on="system", how="left")
        passed = ok[ok["success"] == True]  # noqa: E712
        if len(passed):
            ok = passed
        # highest EM, then fewest retrieval calls
        return ok.sort_values(["em", "retrieval_call_rate"], ascending=[False, True])["system"].iloc[0]
    return cand.sort_values("em", ascending=False)["system"].iloc[0]


def length_quartile(n_words: pd.Series) -> pd.Series:
    try:
        return pd.qcut(n_words, 4, labels=["Q1 (short)", "Q2", "Q3", "Q4 (long)"], duplicates="drop")
    except ValueError:
        return pd.Series(["all"] * len(n_words), index=n_words.index)


def summarise_long(df: pd.DataFrame, breakdown: str, group_col: str) -> pd.DataFrame:
    rows = []
    for (ds, grp, system), g in df.groupby(["dataset", group_col, "system"], dropna=False, observed=True):
        rows.append(
            {
                "breakdown": breakdown,
                "dataset": ds,
                "group": str(grp),
                "system": system,
                "n": len(g),
                "em": float(g["em"].mean()),
                "f1": float(g["f1"].mean()),
                "hit": float(g["hit"].mean()),
                "call_rate": float((g["n_retrieval_calls"] > 0).mean()),
                "mean_latency_ms": float(g["total_latency_ms"].mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--run", default="main")
    ap.add_argument("--system", default=None, help="override the primary system")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"], init_cuda=False)
    root = Path(cfg["paths"]["root"])
    res = root / "results"
    tabs = res / "tables"
    fig_dir = root / cfg["paths"]["figures"]
    tabs.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    pq_path = res / f"eval_per_query_{args.run}.parquet"
    if not pq_path.exists():
        raise SystemExit(f"missing {pq_path} - run scripts/07_run_eval.py first")
    per_q = pd.read_parquet(pq_path)
    summary = pd.read_csv(tabs / f"eval_summary_{args.run}.csv")
    succ_path = tabs / f"success_criterion_{args.run}.csv"
    succ = pd.read_csv(succ_path) if succ_path.exists() else None
    primary_name = pick_primary(summary, succ, cfg, args.system)
    LOG.info("primary system: %s", primary_name)

    # ------------------------------------------------------------------ enrichment
    meta = load_queries_meta(root, "eval")
    if meta.empty:
        LOG.warning("no eval query meta found - breakdowns limited to dataset/strategy")
        meta = pd.DataFrame({"qid": [], "question": [], "answers": [], "hp_type": [], "hp_level": []})
    labels_path = res / f"labels_{cfg['labels']['primary_metric']}_eps{cfg['labels']['epsilon']}_{args.run}.parquet"
    labels = pd.read_parquet(labels_path) if labels_path.exists() else pd.DataFrame()
    label_merge = pd.DataFrame()
    if len(labels):
        label_merge = labels[["qid", "label", "best_strategy", "label_calls", "oracle_score"]].copy()

    base = per_q.copy()
    base = base.merge(meta, on="qid", how="left", suffixes=("", "_q"))
    if len(label_merge):
        base = base.merge(label_merge, on="qid", how="left")
    base["question"] = base["question"].fillna("")

    primary = base[base["system"] == primary_name].copy()
    if primary.empty:
        raise SystemExit(f"system {primary_name} not found in per-query file")
    oracle = base[base["system"] == "oracle"][["qid", "em"]].rename(columns={"em": "oracle_em"})
    primary = primary.merge(oracle, on="qid", how="left")
    primary["oracle_em"] = primary["oracle_em"].fillna(0.0)
    if "best_strategy" not in primary.columns:
        # fall back to argmax EM over the query's strategy rows
        logs = load_eval_logs(root, args.run, args.k)
        if len(logs):
            piv = logs.pivot_table(index="qid", columns="strategy", values="em").reindex(columns=STRATEGIES).fillna(0.0)
            best = pd.Series([STRATEGIES[i] for i in piv.values.argmax(axis=1)], index=piv.index)
            primary["best_strategy"] = primary["qid"].map(best)
    if "best_strategy" not in primary.columns:
        primary["best_strategy"] = "none"
    primary["best_strategy"] = primary["best_strategy"].fillna("none")

    # answer-popularity proxy: document frequency of the rarest answer token in the corpus
    df_counts, n_docs = None, 0
    index_dir = root / cfg["paths"]["indices"]
    corpus_path = root / cfg["paths"]["data_processed"] / "corpus.jsonl"
    if (index_dir / "bm25.pkl").exists() and corpus_path.exists():
        try:
            from retrieval.bm25 import BM25Retriever
            from retrieval.corpus_store import CorpusStore

            bm25 = BM25Retriever(CorpusStore(corpus_path, index_dir=index_dir), index_dir, n_threads=4)
            df_counts, n_docs = bm25.document_frequencies()
            LOG.info("answer-popularity proxy enabled (|V|=%d, n_docs=%d)", len(df_counts), n_docs)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("answer-popularity proxy disabled: %s", exc)

    # analysis columns (answer type, wh-word, length, popularity proxy)
    primary = add_breakdown_columns(primary, df_counts=df_counts, n_docs=n_docs)
    primary["length_quartile"] = length_quartile(primary["n_words"])
    primary["hp_type"] = primary["hp_type"].fillna("")
    primary["hop"] = np.where(
        primary["dataset"].eq("hotpotqa"),
        np.select(
            [primary["hp_type"].isin(["bridge", "comparison"]), primary["hp_type"].eq("")],
            ["multi-hop", "unknown"],
            default="single-hop",
        ),
        "single-hop",
    )

    # ------------------------------------------------------------------ error costs
    pred_none = primary["strategy"].eq("none")
    oracle_none = primary["best_strategy"].eq("none")
    correct = primary["em"] >= 1.0
    under = pred_none & ~oracle_none
    over = ~pred_none & oracle_none
    wrong = ~pred_none & ~oracle_none & primary["strategy"].ne(primary["best_strategy"])
    agree = primary["strategy"].eq(primary["best_strategy"])
    err_rows = []
    for name, mask in [
        ("correct", correct),
        ("under_retrieval", under & ~correct),
        ("over_retrieval", over & ~correct),
        ("wrong_retrieval_choice", wrong & ~correct),
        ("oracle_agrees_but_failed", agree & ~correct),
    ]:
        g = primary[mask]
        err_rows.append(
            {
                "category": name,
                "n": int(len(g)),
                "share": float(len(g) / max(1, len(primary))),
                "lost_em_points": float((1.0 - g["em"]).sum()),
                "mean_calls": float(g["n_retrieval_calls"].mean()) if len(g) else 0.0,
                "mean_latency_ms": float(g["total_latency_ms"].mean()) if len(g) else 0.0,
                "em": float(g["em"].mean()) if len(g) else 0.0,
            }
        )
    err_df = pd.DataFrame(err_rows)
    err_df.to_csv(tabs / f"error_costs_{args.run}.csv", index=False)
    LOG.info("error costs:\n%s", err_df.to_string(index=False))

    # ------------------------------------------------------------------ confusion
    cm = pd.crosstab(primary["strategy"], primary["best_strategy"])
    cm = cm.reindex(index=[s for s in STRATEGIES if s in cm.index], columns=[s for s in STRATEGIES if s in cm.columns], fill_value=0)
    cm.to_csv(tabs / f"confusion_{args.run}.csv")
    if cm.size:
        fig.plot_confusion(cm, fig_dir / f"confusion_{args.run}.png", f"Predicted vs oracle strategy ({primary_name})")

    # ------------------------------------------------------------------ breakdowns
    breakdowns = {
        "dataset": "dataset",
        "wh_word": "wh_word",
        "answer_type": "answer_type",
        "length_quartile": "length_quartile",
        "answer_popularity": "answer_popularity_bin",
        "hop": "hop",
        "gold_hit": "hit",
    }
    long_parts = []
    primary["system"] = "primary"
    oracle_rows = base[base["system"] == "oracle"].merge(meta, on="qid", how="left", suffixes=("", "_q"))
    oracle_rows["question"] = oracle_rows["question"].fillna("")
    oracle_rows = add_breakdown_columns(oracle_rows, df_counts=df_counts, n_docs=n_docs)
    oracle_rows["length_quartile"] = length_quartile(oracle_rows["n_words"])
    oracle_rows["hp_type"] = oracle_rows["hp_type"].fillna("")
    oracle_rows["hop"] = np.where(
        oracle_rows["dataset"].eq("hotpotqa"),
        np.select(
            [oracle_rows["hp_type"].isin(["bridge", "comparison"]), oracle_rows["hp_type"].eq("")],
            ["multi-hop", "unknown"],
            default="single-hop",
        ),
        "single-hop",
    )
    oracle_rows["system"] = "oracle"
    for name, col in breakdowns.items():
        if col not in primary.columns:
            continue
        parts = [summarise_long(primary, name, col)]
        if col in oracle_rows.columns:
            parts.append(summarise_long(oracle_rows, name, col))
        long_parts.append(pd.concat(parts, ignore_index=True))
    if long_parts:
        bd = pd.concat(long_parts, ignore_index=True)
        bd.to_csv(tabs / f"breakdowns_{args.run}.csv", index=False)
        for name in ("dataset", "wh_word", "answer_type"):
            sub = bd[bd["breakdown"] == name]
            if len(sub):
                try:
                    fig.plot_breakdown(sub, fig_dir / f"breakdown_{name}_{args.run}.png", "group", "em", f"EM by {name}")
                except Exception as exc:  # noqa: BLE001 - a figure must not kill the stage
                    LOG.warning("figure for %s failed: %s", name, exc)

    # ------------------------------------------------------------------ failure taxonomy
    failures = primary.copy()
    failures["failure"] = [
        categorise_failure(r, str(r["best_strategy"]), str(r["strategy"]), float(r["oracle_em"]), float(r["em"]))
        for _, r in failures.iterrows()
    ]
    failures["sampled"] = False
    rng = np.random.default_rng(cfg["seed"])
    sample_n = int(cfg["eval"].get("failure_sample_per_dataset", 50))
    sampled_parts = []
    for ds, g in failures[failures["failure"] != "no_failure"].groupby("dataset"):
        take = min(sample_n, len(g))
        if take:
            idx = rng.choice(g.index.to_numpy(), size=take, replace=False)
            sampled_parts.append(g.loc[idx])
    if sampled_parts:
        sampled = pd.concat(sampled_parts, ignore_index=True)
        sampled["sampled"] = True
    else:
        sampled = failures.head(0)
    keep_cols = [
        c
        for c in ("qid", "dataset", "question", "answer", "answers", "strategy",
                  "best_strategy", "em", "oracle_em", "hit", "failure", "sampled")
        if c in failures.columns
    ]
    failures[keep_cols].to_csv(tabs / f"failures_all_{args.run}.csv", index=False)
    sampled[keep_cols].to_csv(tabs / f"failures_{args.run}.csv", index=False)
    fail_counts = failures.groupby(["dataset", "failure"]).size().reset_index(name="n")
    fail_counts.to_csv(tabs / f"failure_counts_{args.run}.csv", index=False)
    if len(failures):
        fig.plot_failure_bars(failures, fig_dir / f"failures_{args.run}.png")

    # ------------------------------------------------------------------ pareto + labels
    fig.plot_pareto(summary, fig_dir / f"pareto_{args.run}.png", annotate=[primary_name, "always_hybrid", "never_retrieve"])
    if len(labels):
        fig.plot_label_distribution(labels, fig_dir / f"label_distribution_{args.run}.png")

    # ------------------------------------------------------------------ LODO transfer
    per_ds_path = tabs / f"eval_per_dataset_{args.run}.csv"
    lodo_rows = []
    if per_ds_path.exists():
        per_ds = pd.read_csv(per_ds_path)
        lodo = per_ds[per_ds["system"].str.startswith("pythia_lodo_")]
        if len(lodo):
            # best lambda per (model, tiers, seed): choose by pooled EM from the summary
            sys_meta = summary[summary["system"].str.startswith("pythia_lodo_")].sort_values("em", ascending=False)
            best = sys_meta.drop_duplicates(subset=["system"]).head(6)
            lodo_rows = lodo[lodo["system"].isin(best["system"])].to_dict("records")
            pd.DataFrame(lodo_rows).to_csv(tabs / f"lodo_transfer_{args.run}.csv", index=False)
    if not lodo_rows:
        LOG.info("no LODO predictions found (train with --variants all to populate this table)")

    # ------------------------------------------------------------------ summary json
    summary_json = {
        "primary_system": primary_name,
        "n": int(len(primary)),
        "em": float(primary["em"].mean()),
        "f1": float(primary["f1"].mean()),
        "oracle_em": float(primary["oracle_em"].mean()),
        "em_gap_to_oracle": float(primary["oracle_em"].mean() - primary["em"].mean()),
        "retrieval_call_rate": float((primary["n_retrieval_calls"] > 0).mean()),
        "mean_calls": float(primary["n_retrieval_calls"].mean()),
        "mean_latency_ms": float(primary["total_latency_ms"].mean()),
        "strategy_mix": primary["strategy"].value_counts(normalize=True).round(4).to_dict(),
        "error_costs": err_df.to_dict("records"),
        "failure_counts": fail_counts.to_dict("records"),
        "n_failure_categories": int(failures["failure"].ne("no_failure").sum()),
        "breakdown_cols": list(breakdowns),
        "seed": cfg["seed"],
    }
    write_json(tabs / f"analysis_summary_{args.run}.json", summary_json)
    LOG.info("analysis summary: %s", json.dumps({k: summary_json[k] for k in ("primary_system", "n", "em", "oracle_em", "retrieval_call_rate")}, indent=2))


if __name__ == "__main__":
    main()
