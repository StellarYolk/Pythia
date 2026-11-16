#!/usr/bin/env python
"""Phase 1b - build the BM25 and dense (FAISS) indices over the corpus.

Usage:
    python scripts/02_build_indices.py --config configs/default.yaml [--rebuild]
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import get_logger, load_config, set_seed, write_json  # noqa: E402

LOG = get_logger("build_indices")


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--bm25-only", action="store_true")
    ap.add_argument("--dense-only", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"], init_cuda=False)
    root = Path(cfg["paths"]["root"])
    corpus_path = root / cfg["paths"]["data_processed"] / "corpus.jsonl"
    index_dir = root / cfg["paths"]["indices"]
    index_dir.mkdir(parents=True, exist_ok=True)

    from retrieval.corpus_store import CorpusStore

    t0 = time.time()
    corpus = CorpusStore(corpus_path, index_dir=index_dir, build=args.rebuild)
    LOG.info("corpus store ready: %d passages (%.1f s)", len(corpus), time.time() - t0)

    stats = {}
    if not args.dense_only:
        from retrieval.bm25 import BM25Retriever

        t0 = time.time()
        bm25 = BM25Retriever(
            corpus,
            index_dir,
            k1=cfg["retrieval"]["bm25"]["k1"],
            b=cfg["retrieval"]["bm25"]["b"],
            build=args.rebuild,
            n_threads=8,
        )
        LOG.info("bm25 ready in %.1f min (peak RSS %.2f GB)", (time.time() - t0) / 60, peak_rss_gb())
        stats["bm25"] = bm25.stats()

    if not args.bm25_only:
        import torch

        from retrieval.dense import DenseRetriever

        torch.set_num_threads(4)
        t0 = time.time()
        dense = DenseRetriever(
            corpus,
            index_dir,
            model_name=cfg["retrieval"]["dense_model"],
            batch_size=cfg["retrieval"]["dense_batch_size"],
            max_len=cfg["retrieval"]["dense_max_len"],
            device="cuda",
            build=args.rebuild,
        )
        LOG.info("dense ready in %.1f min (peak RSS %.2f GB)", (time.time() - t0) / 60, peak_rss_gb())
        stats["dense"] = dense.stats()

        # sanity check: retrieve for a couple of queries
        probe = ["who wrote the book origin of species", "when did the eiffel tower open"]
        t0 = time.time()
        res = dense.retrieve(probe, k=3)
        LOG.info("dense probe in %.2f s", time.time() - t0)
        for q, r in zip(probe, res):
            LOG.info("  %-45s -> %s", q[:45], [p["title"] for p in r.passages])

    stats["peak_rss_gb"] = round(peak_rss_gb(), 2)
    write_json(root / cfg["paths"]["results"] / "index_stats.json", stats)
    LOG.info("index stats: %s", json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
