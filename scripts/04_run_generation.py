#!/usr/bin/env python
"""Phase 1c - run the four strategies over the query sets and log everything.

Usage:
    python scripts/04_run_generation.py --config configs/default.yaml [--pilot]
                                         [--datasets nq,triviaqa,hotpotqa]
                                         [--subsets label,tune,eval] [--k 5]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from common import env_info, get_logger, load_config, read_jsonl, set_seed, write_json  # noqa: E402
from generation.runner import STRATEGIES, StrategyRunner, save_log  # noqa: E402

LOG = get_logger("run_generation")


def load_queries(proc: Path, dataset: str, subset: str) -> List[Dict[str, Any]]:
    path = proc / "queries" / f"{dataset}__{subset}.jsonl"
    if not path.exists():
        return []
    return read_jsonl(path)


def build_retrievers(cfg, root: Path, need_bm25: bool, need_dense: bool):
    from retrieval.bm25 import BM25Retriever
    from retrieval.corpus_store import CorpusStore
    from retrieval.dense import DenseRetriever
    from retrieval.hybrid import HybridRetriever

    index_dir = root / cfg["paths"]["indices"]
    corpus = CorpusStore(root / cfg["paths"]["data_processed"] / "corpus.jsonl", index_dir=index_dir)
    LOG.info("corpus: %d passages", len(corpus))
    out = {}
    bm25 = dense = None
    if need_bm25:
        bm25 = BM25Retriever(
            corpus,
            index_dir,
            k1=cfg["retrieval"]["bm25"]["k1"],
            b=cfg["retrieval"]["bm25"]["b"],
            n_threads=8,
        )
        out["bm25"] = bm25
    if need_dense:
        import torch

        torch.set_num_threads(4)
        dense = DenseRetriever(
            corpus,
            index_dir,
            model_name=cfg["retrieval"]["dense_model"],
            batch_size=cfg["retrieval"]["dense_batch_size"],
            max_len=cfg["retrieval"]["dense_max_len"],
            device="cuda",
        )
        out["dense"] = dense
    if need_bm25 and need_dense:
        out["hybrid"] = HybridRetriever(
            bm25,
            dense,
            rrf_k=cfg["retrieval"]["hybrid"]["rrf_k"],
            fusion=cfg["retrieval"]["hybrid"]["fusion"],
            weighted_alpha=cfg["retrieval"]["hybrid"]["weighted_alpha"],
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--datasets", default="nq,triviaqa,hotpotqa")
    ap.add_argument("--subsets", default="label,tune,eval")
    ap.add_argument("--strategies", default="none,bm25,dense,hybrid")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="cap queries per file (pilot)")
    ap.add_argument("--limits", default=None, help="per-subset caps, e.g. label=1500,tune=400,eval=700")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"], init_cuda=False)
    root = Path(cfg["paths"]["root"])
    proc = root / cfg["paths"]["data_processed"]
    log_dir = root / cfg["paths"]["logs"]
    log_dir.mkdir(parents=True, exist_ok=True)
    run_name = "pilot" if args.pilot else "main"
    k = args.k or cfg["retrieval"]["k"]

    datasets = [d for d in args.datasets.split(",") if d]
    subsets = [s for s in args.subsets.split(",") if s]
    strategies = [s for s in args.strategies.split(",") if s]

    limits: Dict[str, int] = {}
    if args.limits:
        for part in args.limits.split(","):
            k_, v_ = part.split("=")
            limits[k_] = int(v_)
    if args.limit:
        limits = {s: args.limit for s in subsets}

    jobs = []
    for ds in datasets:
        for sub in subsets:
            qs = load_queries(proc, ds, sub)
            if not qs:
                continue
            cap = limits.get(sub)
            if cap:
                qs = qs[:cap]
            jobs.append((ds, sub, qs))
    if not jobs:
        LOG.error("no query files found - run scripts/01_prepare_data.py first")
        return
    total = sum(len(q) for _, _, q in jobs)
    LOG.info("%d jobs, %d queries, strategies=%s, k=%d", len(jobs), total, strategies, k)

    need_bm25 = any(s in ("bm25", "hybrid") for s in strategies)
    need_dense = any(s in ("dense", "hybrid") for s in strategies)
    retrievers = build_retrievers(cfg, root, need_bm25, need_dense)

    from generation.engine import VLLMEngine

    t0 = time.time()
    engine = VLLMEngine(cfg)
    LOG.info("engine ready in %.1f s (model=%s)", time.time() - t0, engine.model_name)

    runner = StrategyRunner(cfg, engine, retrievers)
    manifest = {
        "run": run_name,
        "k": k,
        "strategies": strategies,
        "datasets": datasets,
        "subsets": subsets,
        "generator_model": engine.model_name,
        "retrieval": cfg["retrieval"],
        "generation": cfg["generation"],
        "prompt": cfg["prompt"],
        "seed": cfg["seed"],
        "env": env_info(),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(root / cfg["paths"]["results"] / f"run_manifest_{run_name}.json", manifest)

    t_start = time.time()
    for ds, sub, qs in jobs:
        for strategy in strategies:
            out_path = log_dir / run_name / f"{ds}__{sub}__{strategy}__k{k}.parquet"
            if out_path.exists():
                try:
                    df_old = pd.read_parquet(out_path, columns=["qid"])
                    if set(df_old["qid"]) >= {q["qid"] for q in qs}:
                        LOG.info("skip %s (cached)", out_path.name)
                        continue
                except Exception:
                    pass
            t1 = time.time()
            df = runner.run_strategy(qs, strategy, k=k, dataset=ds, subset=sub)
            save_log(df, out_path)
            LOG.info("wrote %s (%d rows) in %.1f s", out_path.name, len(df), time.time() - t1)

    manifest["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    manifest["total_minutes"] = round((time.time() - t_start) / 60, 2)
    write_json(root / cfg["paths"]["results"] / f"run_manifest_{run_name}.json", manifest)
    LOG.info("done in %.1f min", manifest["total_minutes"])


if __name__ == "__main__":
    main()
