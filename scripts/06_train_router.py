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
(lex / +emb / +probe / +gen) for the pooled variant and writes to a separate file.

Usage:
    python scripts/06_train_router.py --config configs/default.yaml --run main
    python scripts/06_train_router.py --config configs/default.yaml --variants all
    python scripts/06_train_router.py --config configs/default.yaml --tier-ablation
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
from router.features import build_features, embedding_features  # noqa: E402
from router.models import (  # noqa: E402
    STRATEGIES,
    STRAT2IDX,
    apply_temperature,
    build_model,
    fit_temperature,
)

LOG = get_logger("train_router")

DATASETS = ["nq", "triviaqa", "hotpotqa"]
META_COLS = ("qid", "dataset", "subset", "question", "label", "variant", "heldout")


def log_path(root: Path, run: str, ds: str, sub: str, strategy: str, k: int = 5) -> Path:
    return root / "results" / "logs" / run / f"{ds}__{sub}__{strategy}__k{k}.parquet"


def load_queries(root: Path, ds: str, sub: str) -> List[Dict[str, Any]]:
    p = root / "data" / "processed" / "queries" / f"{ds}__{sub}.jsonl"
    return read_jsonl(p) if p.exists() else []


def load_logs(root: Path, run: str, ds: str, sub: str, strategy: str, k: int = 5) -> Optional[pd.DataFrame]:
    p = log_path(root, run, ds, sub, strategy, k)
    return pd.read_parquet(p) if p.exists() else None


def feature_cache_path(root: Path, subset: str, tiers: Tuple[str, ...], k: int = 5) -> Path:
    tag = "-".join(tiers)
    return root / "results" / "router_features" / f"{subset}__{tag}__k{k}.parquet"


def _index_ready(index_dir: Path, kind: str) -> bool:
    if kind == "bm25":
        return (index_dir / "bm25.pkl").exists()
    return all((index_dir / f).exists() for f in ("dense.index", "dense_emb.f16.npy", "dense_meta.json"))


def build_feature_cache(cfg, root: Path, run: str, subset: str, tiers: List[str], k: int = 5) -> pd.DataFrame:
    """Build (or load) the feature matrix for one subset.

    Missing heavy indices are a warning, not a crash: the lexical tier degrades to
    purely surface features (all IDF features become 0) so that router code can still be
    exercised on machines without the corpus.  The ``probe`` tier genuinely needs both
    indices and fails loudly instead.
    """
    path = feature_cache_path(root, subset, tuple(tiers), k=k)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        cached = pd.read_parquet(path)
        if "question" in cached.columns or "deberta" not in cfg["router"].get("models", []):
            LOG.info("loaded cached features %s", path.name)
            return cached
        LOG.warning("cached features %s lack the question column - rebuilding", path.name)

    qs: List[Dict[str, Any]] = []
    for ds in DATASETS:
        qs.extend(load_queries(root, ds, subset))
    if not qs:
        raise SystemExit(f"no queries for subset={subset} - run scripts/01_prepare_data.py first")
    questions = [q["question"] for q in qs]
    LOG.info("building features for %d %s queries (tiers=%s)", len(qs), subset, tiers)

    df_counts: Dict[str, int] = {}
    n_docs = 0
    none_log = None
    bm25 = dense = None
    corpus = None
    index_dir = root / cfg["paths"]["indices"]
    corpus_path = root / cfg["paths"]["data_processed"] / "corpus.jsonl"

    needs_bm25 = "lex" in tiers or "probe" in tiers
    if needs_bm25:
        if _index_ready(index_dir, "bm25") and corpus_path.exists():
            try:
                from retrieval.bm25 import BM25Retriever
                from retrieval.corpus_store import CorpusStore

                corpus = CorpusStore(corpus_path, index_dir=index_dir)
                bm25 = BM25Retriever(corpus, index_dir, n_threads=8)
                df_counts, n_docs = bm25.document_frequencies()
                LOG.info("BM25 document frequencies: |V|=%d, n_docs=%d", len(df_counts), n_docs)
            except Exception as exc:  # noqa: BLE001 - degradation is intentional
                LOG.warning("BM25 index unavailable (%s); lexical IDF features fall back to 0", exc)
                bm25 = None
        else:
            LOG.warning("no BM25 index at %s; lexical IDF features fall back to 0", index_dir)

    if "probe" in tiers:
        if not _index_ready(index_dir, "dense") or not corpus_path.exists():
            raise SystemExit("probe tier needs the BM25 + dense indices - run scripts/02_build_indices.py")
        import torch

        from retrieval.corpus_store import CorpusStore
        from retrieval.dense import DenseRetriever

        torch.set_num_threads(4)
        if corpus is None:
            corpus = CorpusStore(corpus_path, index_dir=index_dir)
        if bm25 is None:
            from retrieval.bm25 import BM25Retriever

            bm25 = BM25Retriever(corpus, index_dir, n_threads=8)
            df_counts, n_docs = bm25.document_frequencies()
        dense = DenseRetriever(
            corpus,
            index_dir,
            model_name=cfg["retrieval"]["dense_model"],
            batch_size=cfg["retrieval"]["dense_batch_size"],
            max_len=cfg["retrieval"]["dense_max_len"],
            device=cfg["router"].get("emb_device", "cuda"),
        )

    emb = None
    if "emb" in tiers:
        emb = embedding_features(
            questions,
            model_name=cfg["retrieval"]["dense_model"],
            device=cfg["router"].get("emb_device", "cuda"),
            batch_size=cfg["retrieval"]["dense_batch_size"],
        )

    if "gen" in tiers:
        frames = []
        for ds in DATASETS:
            df = load_logs(root, run, ds, subset, "none", k)
            if df is not None:
                frames.append(df[["qid", "mean_logprob", "first_token_entropy", "mean_entropy", "first_token_margin"]])
        none_log = pd.concat(frames, ignore_index=True) if frames else None
        if none_log is None:
            raise SystemExit("gen tier needs the no-retrieval logs - run scripts/04_run_generation.py")
        # align to the query order explicitly (concat order is not a contract)
        qid_order = [q["qid"] for q in qs]
        none_log = none_log.drop_duplicates("qid").set_index("qid").reindex(qid_order).reset_index()

    X, cost = build_features(
        questions,
        tiers,
        df_counts=df_counts,
        n_docs=n_docs,
        bm25=bm25,
        dense=dense,
        none_log=none_log,
        emb=emb,
    )
    X.insert(0, "qid", [q["qid"] for q in qs])
    X.insert(1, "dataset", [q["dataset"] for q in qs])
    X.insert(2, "subset", [q["subset"] for q in qs])
    X.insert(3, "question", questions)
    X.to_parquet(path, index=False)
    write_json(path.with_suffix(".cost.json"), cost)
    LOG.info("wrote %s (feature cost: %s)", path.name, cost)
    return X


def measured_costs(root: Path, run: str, k: int = 5) -> Dict[str, float]:
    """Mean end-to-end latency per strategy (ms) - the cost scale of the lambda knob."""
    frames = []
    for ds in DATASETS:
        for sub in ("label", "tune"):
            for s in STRATEGIES:
                df = load_logs(root, run, ds, sub, s, k=k)
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


def _feature_cols(frame: pd.DataFrame) -> List[str]:
    return [c for c in frame.columns if c not in META_COLS]


def train_one(
    model_name: str,
    tier_set: List[str],
    label_df: pd.DataFrame,
    tune_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    seed: int,
    cfg: Dict[str, Any],
    res: Path,
    variant: str,
    heldout: str = "",
    save_tag: str = "",
) -> Tuple[Dict[str, Any], List[pd.DataFrame]]:
    """Fit one router, calibrate it, and return its predictions."""
    feature_cols = _feature_cols(label_df)
    is_deberta = model_name == "deberta"
    Xtr = label_df["question"].tolist() if is_deberta else label_df[feature_cols]
    ytr = label_df["label"].map(STRAT2IDX).values
    if len(np.unique(ytr)) < 2:
        raise ValueError("training labels contain a single class")
    model = build_model(model_name, cfg["router"])
    model.fit(Xtr, ytr, seed=seed)

    temp = 1.0
    acc = float("nan")
    if len(tune_df):
        Xtu = tune_df["question"].tolist() if is_deberta else tune_df[feature_cols]
        ytu = tune_df["label"].map(STRAT2IDX).values
        p_tu = model.predict_proba(Xtu)
        # fit_temperature expects probabilities (it takes the log internally)
        temp = fit_temperature(p_tu, ytu)
        acc = float((np.argmax(apply_temperature(p_tu, temp), axis=1) == ytu).mean())

    frames = []
    for name, frame in (("tune", tune_df), ("eval", eval_df)):
        if frame is None or len(frame) == 0:
            continue
        Xe = frame["question"].tolist() if is_deberta else frame[feature_cols]
        p = apply_temperature(model.predict_proba(Xe), temp)
        out = pd.DataFrame(
            {
                "qid": frame["qid"].values,
                "dataset": frame["dataset"].values,
                "subset": name,
                "model": model_name,
                "seed": seed,
                "tiers": "-".join(tier_set),
                "variant": variant,
                "heldout": heldout,
                "temperature": temp,
                **{f"p_{s}": p[:, i] for i, s in enumerate(STRATEGIES)},
            }
        )
        frames.append(out)
    tag = save_tag or "-".join(x for x in (variant, heldout) if x)
    model.save(res / "routers" / f"{model_name}__s{seed}__{'-'.join(tier_set)}__{tag}.pkl")
    summary = {
        "variant": variant,
        "heldout": heldout,
        "model": model_name,
        "seed": seed,
        "tiers": "-".join(tier_set),
        "n_train": int(len(label_df)),
        "n_tune": int(len(tune_df)),
        "n_eval": int(len(eval_df)) if eval_df is not None else 0,
        "temperature": temp,
        "tune_acc": acc,
    }
    return summary, frames


def run_variant(
    variant: str,
    model_name: str,
    tier_set: List[str],
    label_all: pd.DataFrame,
    tune_all: pd.DataFrame,
    eval_all: pd.DataFrame,
    seed: int,
    cfg: Dict[str, Any],
    res: Path,
) -> Tuple[List[Dict[str, Any]], List[pd.DataFrame]]:
    """Train one (variant, model, seed) combination."""
    summaries: List[Dict[str, Any]] = []
    frames: List[pd.DataFrame] = []
    if variant == "pooled":
        s, f = train_one(model_name, tier_set, label_all, tune_all, eval_all, seed, cfg, res, "pooled")
        return [s], f
    if variant == "per_dataset":
        for ds in sorted(set(label_all["dataset"])):
            lab = label_all[label_all["dataset"] == ds]
            tun = tune_all[tune_all["dataset"] == ds]
            ev = eval_all[eval_all["dataset"] == ds]
            if not len(lab):
                continue
            s, f = train_one(model_name, tier_set, lab, tun, ev, seed, cfg, res, "per_dataset", heldout=ds)
            summaries.append(s)
            frames.extend(f)
        return summaries, frames
    if variant == "lodo":
        for ds in sorted(set(eval_all["dataset"])):
            lab = label_all[label_all["dataset"] != ds]
            tun = tune_all[tune_all["dataset"] != ds]
            ev = eval_all[eval_all["dataset"] == ds]
            if not len(eval_all) or not len(ev):
                continue
            s, f = train_one(model_name, tier_set, lab, tun, ev, seed, cfg, res, "lodo", heldout=ds)
            summaries.append(s)
            frames.extend(f)
        return summaries, frames
    raise ValueError(variant)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--run", default="main")
    ap.add_argument("--models", default=None)
    ap.add_argument("--tiers", default=None)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--variants", default="pooled", help="pooled | per_dataset | lodo | all")
    ap.add_argument("--k", type=int, default=5, help="retrieval depth of the logs used for features/costs")
    ap.add_argument("--tier-ablation", action="store_true")
    ap.add_argument("--no-primary", action="store_true", help="with --tier-ablation, do not overwrite the primary prediction file")
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
    if args.variants == "all":
        variants = ["pooled", "per_dataset", "lodo"]
    else:
        variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    # ---------------------------------------------------------------- labels
    label_path = res / f"labels_{cfg['labels']['primary_metric']}_eps{cfg['labels']['epsilon']}_{args.run}.parquet"
    if not label_path.exists():
        raise SystemExit(f"missing {label_path} - run scripts/05_build_labels.py first")
    labels = pd.read_parquet(label_path)
    labels = labels[labels["subset"].isin(["label", "tune"])]
    LOG.info("labels: %d rows", len(labels))

    # ---------------------------------------------------------------- cost scale
    cost_ms = measured_costs(root, args.run, k=args.k)
    write_json(res / f"router_cost_model_{args.run}.json", cost_ms)

    # ---------------------------------------------------------------- feature sets
    def _run_sweep(tier_sets: List[List[str]], model_list: List[str], variant_list: List[str], suffix: str) -> None:
        all_summary: List[Dict[str, Any]] = []
        all_frames: List[pd.DataFrame] = []
        for tier_set in tier_sets:
            try:
                feats = {sub: build_feature_cache(cfg, root, args.run, sub, tier_set, k=args.k) for sub in ("label", "tune", "eval")}
            except (Exception, SystemExit) as exc:  # e.g. probe tier without indices
                LOG.error("tiers=%s unavailable, skipping: %s", "-".join(tier_set), exc)
                continue
            label_df = feats["label"].merge(labels[["qid", "label"]], on="qid", how="inner")
            tune_df = feats["tune"].merge(labels[["qid", "label"]], on="qid", how="inner")
            eval_df = feats["eval"]
            LOG.info("tiers=%s: label=%d tune=%d eval=%d", "-".join(tier_set), len(label_df), len(tune_df), len(eval_df))
            for variant in variant_list:
                for model_name in model_list:
                    for seed in seeds:
                        t0 = time.time()
                        try:
                            summaries, frames = run_variant(
                                variant, model_name, tier_set, label_df, tune_df, eval_df, seed, cfg, res
                            )
                        except Exception as exc:  # a broken model family must not kill the stage
                            LOG.error("%s %s failed: %s", variant, model_name, exc)
                            continue
                        for s in summaries:
                            s["seconds"] = round(time.time() - t0, 1)
                        all_summary.extend(summaries)
                        all_frames.extend(frames)
                        LOG.info(
                            "%s %s tiers=%s seed=%d: folds=%d tune_acc=%.4f temp=%.2f (%.0f s)",
                            variant,
                            model_name,
                            "-".join(tier_set),
                            seed,
                            len(summaries),
                            float(np.nanmean([s["tune_acc"] for s in summaries])),
                            float(np.nanmean([s["temperature"] for s in summaries])),
                            time.time() - t0,
                        )
        if all_summary:
            pd.DataFrame(all_summary).to_csv(res / f"router_training_summary_{args.run}{suffix}.csv", index=False)
        if all_frames:
            pd.concat(all_frames, ignore_index=True).to_parquet(
                res / f"router_predictions_{args.run}{suffix}.parquet", index=False
            )
        LOG.info("saved %d router runs (%s)", len(all_summary), suffix or "primary")

    if args.tier_ablation:
        tier_sets = [["lex"], ["lex", "emb"], ["lex", "emb", "probe"], ["lex", "emb", "probe", "gen"]]
        _run_sweep(tier_sets, ["logreg", "gbt"], ["pooled"], args.out_suffix or "_tiers")

    if not args.no_primary:
        _run_sweep([tiers], models, variants, args.out_suffix)


if __name__ == "__main__":
    main()
