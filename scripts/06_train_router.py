#!/usr/bin/env python
"""Phase 3 - train the router variants.

Variants
--------
``pooled``      one router trained on all three datasets
``per_dataset`` one router per dataset
``lodo``        leave-one-dataset-out (train on two, test on the third)

For every (variant, model, seed) we fit the model on the *label* split, fit the
temperature on the held-out *tune* split and store the calibrated probabilities for the
tune and eval splits.  The cost knob ``lambda`` is applied at decision time, so the whole
accuracy/cost curve comes from one set of probabilities.

``--tier-ablation`` additionally trains logreg+gbt on every feature tier
(lex / +emb / +probe / +gen) and writes to a separate prediction file.

Usage:
    python scripts/06_train_router.py --config configs/default.yaml --run main
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import get_logger, load_config, read_jsonl, set_seed, write_json  # noqa: E402
from router.features import build_features  # noqa: E402
from router.models import (  # noqa: E402
    STRATEGIES,
    STRAT2IDX,
    apply_temperature,
    build_model,
    fit_temperature,
)

LOG = get_logger("train_router")

DATASETS = ["nq", "triviaqa", "hotpotqa"]


def log_path(root: Path, run: str, ds: str, sub: str, strategy: str, k: int = 5) -> Path:
    return root / "results" / "logs" / run / f"{ds}__{sub}__{strategy}__k{k}.parquet"


def load_queries(root: Path, ds: str, sub: str) -> List[Dict[str, Any]]:
    p = root / "data" / "processed" / "queries" / f"{ds}__{sub}.jsonl"
    return read_jsonl(p) if p.exists() else []


def load_logs(root: Path, run: str, ds: str, sub: str, strategy: str, k: int = 5) -> Optional[pd.DataFrame]:
    p = log_path(root, run, ds, sub, strategy, k)
    return pd.read_parquet(p) if p.exists() else None


def feature_cache_path(root: Path, subset: str, tiers: Tuple[str, ...]) -> Path:
    tag = "-".join(tiers)
    return root / "results" / "router_features" / f"{subset}__{tag}.parquet"


def build_feature_cache(cfg, root: Path, run: str, subset: str, tiers: List[str], k: int = 5) -> pd.DataFrame:
    """Build (or load) the feature matrix for one subset."""
    path = feature_cache_path(root, subset, tuple(tiers))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        LOG.info("loaded cached features %s", path.name)
        return pd.read_parquet(path)

    qs: List[Dict[str, Any]] = []
    for ds in DATASETS:
        qs.extend(load_queries(root, ds, subset))
    questions = [q["question"] for q in qs]
    LOG.info("building features for %d %s queries (tiers=%s)", len(qs), subset, tiers)

    df_counts, n_docs = {}, 0
    none_log = None
    bm25 = dense = None
    corpus = None
    index_dir = root / cfg["paths"]["indices"]

    if "lex" in tiers:
        from retrieval.bm25 import BM25Retriever
        from retrieval.corpus_store import CorpusStore

        corpus = CorpusStore(root / cfg["paths"]["data_processed"] / "corpus.jsonl", index_dir=index_dir)
        bm25 = BM25Retriever(corpus, index_dir, n_threads=8)
        df_counts, n_docs = bm25.document_frequencies()
    if "probe" in tiers:
        import torch

        from retrieval.corpus_store import CorpusStore
        from retrieval.dense import DenseRetriever

        torch.set_num_threads(4)
        if corpus is None:
            corpus = CorpusStore(root / cfg["paths"]["data_processed"] / "corpus.jsonl", index_dir=index_dir)
        dense = DenseRetriever(
            corpus,
            index_dir,
            model_name=cfg["retrieval"]["dense_model"],
            batch_size=cfg["retrieval"]["dense_batch_size"],
            max_len=cfg["retrieval"]["dense_max_len"],
            device="cuda",
        )
    if "gen" in tiers:
        frames = []
        for ds in DATASETS:
            df = load_logs(root, run, ds, subset, "none", k)
            if df is not None:
                frames.append(df[["qid", "mean_logprob", "first_token_entropy", "mean_entropy", "first_token_margin"]])
        none_log = pd.concat(frames, ignore_index=True) if frames else None

    X, cost = build_features(
        questions,
        tiers,
        df_counts=df_counts,
        n_docs=n_docs,
        bm25=bm25,
        dense=dense,
        none_log=none_log,
    )
    X.insert(0, "qid", [q["qid"] for q in qs])
    X.insert(1, "dataset", [q["dataset"] for q in qs])
    X.insert(2, "subset", [q["subset"] for q in qs])
    X.to_parquet(path, index=False)
    write_json(path.with_suffix(".cost.json"), cost)
    LOG.info("wrote %s (feature cost: %s)", path.name, cost)
    return X


def measured_costs(root: Path, run: str) -> Dict[str, float]:
    """Mean end-to-end latency per strategy (ms) - the cost scale of the lambda knob."""
    frames = []
    for ds in DATASETS:
        for sub in ("label", "tune"):
            for s in STRATEGIES:
                df = load_logs(root, run, ds, sub, s)
                if df is not None:
                    frames.append(df[["strategy", "total_latency_ms", "n_retrieval_calls"]])
    cost_ms = {"none": 0.0, "bm25": 1.0, "dense": 2.0, "hybrid": 3.0}
    if frames:
        all_logs = pd.concat(frames, ignore_index=True)
        mean_lat = all_logs.groupby("strategy")["total_latency_ms"].mean().to_dict()
        calls = all_logs.groupby("strategy")["n_retrieval_calls"].mean().to_dict()
        cost_ms = {s: float(mean_lat.get(s, 0.0)) for s in STRATEGIES}
        write_json(
            root / "results" / "cost_model.json",
            {"mean_total_latency_ms": cost_ms, "mean_retrieval_calls": calls},
        )
        LOG.info("measured per-strategy latency (ms): %s", {k: round(v, 1) for k, v in cost_ms.items()})
    return cost_ms


def train_one(
    model_name: str,
    tier_set: List[str],
    feature_cols: List[str],
    label_df: pd.DataFrame,
    tune_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    seed: int,
    cfg: Dict[str, Any],
    res: Path,
) -> Tuple[Dict[str, Any], List[pd.DataFrame]]:
    """Fit one router, calibrate it, and return its predictions."""
    is_deberta = model_name == "deberta"
    Xtr = label_df["question"].tolist() if is_deberta else label_df[feature_cols]
    ytr = label_df["label"].map(STRAT2IDX).values
    model = build_model(model_name, cfg["router"])
    model.fit(Xtr, ytr, seed=seed)

    Xtu = tune_df["question"].tolist() if is_deberta else tune_df[feature_cols]
    ytu = tune_df["label"].map(STRAT2IDX).values
    p_tu = model.predict_proba(Xtu)
    temp = fit_temperature(np.log(np.clip(p_tu, 1e-12, None)), ytu) if len(ytu) else 1.0

    frames = []
    for name, frame in (("tune", tune_df), ("eval", eval_df)):
        if len(frame) == 0:
            continue
        Xe = frame["question"].tolist() if is_deberta else frame[feature_cols]
        p = apply_temperature(model.predict_proba(Xe), temp)
        frames.append(
            pd.DataFrame(
                {
                    "qid": frame["qid"].values,
                    "dataset": frame["dataset"].values,
                    "subset": name,
                    "model": model_name,
                    "seed": seed,
                    "tiers": "-".join(tier_set),
                    "temperature": temp,
                    **{f"p_{s}": p[:, i] for i, s in enumerate(STRATEGIES)},
                }
            )
        )
    model.save(res / "routers" / f"{model_name}__s{seed}__{'-'.join(tier_set)}.pkl")
    acc = float((np.argmax(p_tu, axis=1) == ytu).mean()) if len(ytu) else float("nan")
    summary = {
        "model": model_name,
        "seed": seed,
        "tiers": "-".join(tier_set),
        "n_train": int(len(label_df)),
        "n_tune": int(len(tune_df)),
        "n_eval": int(len(eval_df)),
        "temperature": temp,
        "tune_acc": acc,
    }
    return summary, frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--run", default="main")
    ap.add_argument("--models", default=None)
    ap.add_argument("--tiers", default=None)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--tier-ablation", action="store_true")
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"], init_cuda=False)
    root = Path(cfg["paths"]["root"])
    res = root / "results"
    (res / "routers").mkdir(parents=True, exist_ok=True)
    (res / "router_features").mkdir(parents=True, exist_ok=True)

    models = args.models.split(",") if args.models else cfg["router"]["models"]
    tiers = args.tiers.split(",") if args.tiers else cfg["router"]["feature_tiers"]
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else cfg["router"]["seeds"]

    # ---------------------------------------------------------------- labels
    label_path = res / f"labels_{cfg['labels']['primary_metric']}_eps{cfg['labels']['epsilon']}_{args.run}.parquet"
    if not label_path.exists():
        raise SystemExit(f"missing {label_path} - run scripts/05_build_labels.py first")
    labels = pd.read_parquet(label_path)
    labels = labels[labels["subset"].isin(["label", "tune"])]
    LOG.info("labels: %d rows", len(labels))

    # ---------------------------------------------------------------- cost scale
    cost_ms = measured_costs(root, args.run)
    write_json(res / f"router_cost_model_{args.run}.json", cost_ms)

    # ---------------------------------------------------------------- feature sets
    tier_sets: List[List[str]] = [tiers]
    if args.tier_ablation:
        tier_sets = [["lex"], ["lex", "emb"], ["lex", "emb", "probe"], ["lex", "emb", "probe", "gen"]]

    all_summary = []
    for tier_set in tier_sets:
        is_primary = tier_set == tiers
        feats = {}
        for sub in ("label", "tune", "eval"):
            feats[sub] = build_feature_cache(cfg, root, args.run, sub, tier_set)
        label_df = feats["label"].merge(labels[["qid", "label"]], on="qid", how="inner")
        tune_df = feats["tune"].merge(labels[["qid", "label"]], on="qid", how="inner")
        eval_df = feats["eval"]
        LOG.info("tiers=%s: label=%d tune=%d eval=%d", "-".join(tier_set), len(label_df), len(tune_df), len(eval_df))

        feature_cols = [c for c in label_df.columns if c not in ("qid", "dataset", "subset", "label")]
        use_models = models if is_primary else ["logreg", "gbt"]

        summaries = []
        frames = []
        for model_name in use_models:
            for seed in seeds:
                t0 = time.time()
                try:
                    summary, pf = train_one(model_name, tier_set, feature_cols, label_df, tune_df, eval_df, seed, cfg, res)
                except Exception as exc:  # a broken model family must not kill the stage
                    LOG.error("model %s failed: %s", model_name, exc)
                    continue
                summary["seconds"] = round(time.time() - t0, 1)
                summaries.append(summary)
                frames.extend(pf)
                LOG.info(
                    "%s tiers=%s seed=%d: tune_acc=%.4f temp=%.2f (%.0f s)",
                    model_name,
                    "-".join(tier_set),
                    seed,
                    summary["tune_acc"],
                    summary["temperature"],
                    time.time() - t0,
                )
        suffix = args.out_suffix or ("_tiers" if args.tier_ablation else "")
        pd.DataFrame(summaries).to_csv(res / f"router_training_summary_{args.run}{suffix}.csv", index=False)
        if frames:
            pd.concat(frames, ignore_index=True).to_parquet(res / f"router_predictions_{args.run}{suffix}.parquet", index=False)
        all_summary.extend(summaries)

    LOG.info("saved %d router runs", len(all_summary))


if __name__ == "__main__":
    main()
