#!/usr/bin/env python
"""Extract gold *article* titles for the OFFICIAL dev splits.

The Tevatron DPR dumps only ship positive passages for the *train* splits (the ``*-test``
files contain questions and answers but no passages), so the official dev splits get
their gold articles from the datasets' own annotations:

* **NQ-open dev** (3 610 questions)   -> ``google-research-datasets/natural_questions``
  (config ``dev``): the document that contains the long answer is the gold article.
* **TriviaQA dev** (11 313 questions) -> ``mandarjoshi/trivia_qa`` config ``unfiltered``:
  the entity pages of the question plus the top Wikipedia search results.

Both are read **over HTTP with parquet column pruning**, so only the small columns we
need are transferred (the full dev split is 1.3 GB / 1.7 GB, the pruned read is ~30 MB).

Output: ``data/processed/gold_dev_titles.json``
    {"<dataset>": {"<loose-normalised question>": [title, ...]}}
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import get_logger, write_json  # noqa: E402
from data.gold import loose_normalize  # noqa: E402

LOG = get_logger("dev_gold")

HF = "https://huggingface.co/datasets"


def _http_parquet(repo: str, path: str):
    import fsspec
    import pyarrow.parquet as pq

    fs = fsspec.filesystem("http")
    url = f"{HF}/{repo}/resolve/main/{path}"
    fh = fs.open(url, block_size=1 << 20)
    return pq.ParquetFile(fh), fh


def _struct_col(table, name: str):
    """Return a list of dicts for a (possibly nested) struct column."""
    if name not in table.schema.names:
        for n in table.schema.names:
            if n.split(".")[-1] == name.split(".")[-1]:
                name = n
                break
        else:
            return None
    vals = table.column(name).to_pylist()
    return [v if isinstance(v, dict) else {} for v in vals]


def extract_nq(out: Dict[str, List[str]], shards: int = 7) -> None:
    for i in range(shards):
        path = f"dev/validation-{i:05d}-of-{shards:05d}.parquet"
        t0 = time.time()
        pf, fh = _http_parquet("google-research-datasets/natural_questions", path)
        tbl = pf.read(columns=["document.title", "question.text", "annotations.long_answer"])
        docs = _struct_col(tbl, "document")
        questions = _struct_col(tbl, "question")
        annotations = _struct_col(tbl, "annotations")
        n_new = 0
        for d, q, a in zip(docs, questions, annotations):
            long_answers = (a or {}).get("long_answer") or []
            if not any((la or {}).get("start_token", -1) >= 0 for la in long_answers):
                continue
            title = d.get("title")
            if not title:
                continue
            key = loose_normalize(q.get("text", ""))
            lst = out.setdefault(key, [])
            if title not in lst:
                lst.append(title)
                n_new += 1
        fh.close()
        LOG.info("  NQ shard %d/%d: %d questions with gold (+%d titles, %.0f s)", i + 1, shards, len(out), n_new, time.time() - t0)
    LOG.info("NQ dev gold articles: %d questions", len(out))


def extract_triviaqa(out: Dict[str, List[str]], shards: int = 7) -> None:
    for i in range(shards):
        path = f"unfiltered/validation-{i:05d}-of-{shards:05d}.parquet"
        t0 = time.time()
        pf, fh = _http_parquet("mandarjoshi/trivia_qa", path)
        tbl = pf.read(columns=["question.text", "entity_pages.title", "search_results.title"])
        questions = _struct_col(tbl, "question")
        entity_pages = _struct_col(tbl, "entity_pages")
        search_results = _struct_col(tbl, "search_results")
        n_new = 0
        for q, ep, sr in zip(questions, entity_pages, search_results):
            titles: List[str] = []
            for t in (ep or {}).get("title", []) or []:
                if t and t not in titles:
                    titles.append(t)
            for t in ((sr or {}).get("title", []) or [])[:3]:
                if t and t not in titles:
                    titles.append(t)
            if not titles:
                continue
            key = loose_normalize(q.get("text", ""))
            lst = out.setdefault(key, titles)
            n_new += len(lst)
        fh.close()
        LOG.info("  TriviaQA shard %d/%d: %d questions with gold (%.0f s)", i + 1, shards, len(out), time.time() - t0)
    LOG.info("TriviaQA dev gold articles: %d questions", len(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--only", default=None, choices=[None, "nq", "triviaqa"])
    args = ap.parse_args()

    from common import load_config

    cfg = load_config(args.config)
    root = Path(cfg["paths"]["root"])
    out_path = root / cfg["paths"]["data_processed"] / "gold_dev_titles.json"

    result: Dict[str, Dict[str, List[str]]] = {}
    for ds, fn in (("nq", extract_nq), ("triviaqa", extract_triviaqa)):
        if args.only and ds != args.only:
            continue
        result[ds] = {}
        fn(result[ds])
    write_json(out_path, result)
    LOG.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
