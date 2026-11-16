"""Gold-passage mapping for the three datasets.

**Gold definition (uniform across datasets): the gold *article* (Wikipedia title).**
A query is answered correctly only if the retriever surfaces a passage that belongs to
one of its gold articles.  Passage-level DPR positives are expanded to article level so
that a neighbouring 100-word window of the same article counts as a hit (the DPR dump
splits articles into windows, so an exact-window gold would unfairly punish BM25 and
dense retrieval alike).

Sources of the gold titles
--------------------------
* **NQ-open**      : DPR positives from ``Tevatron/wikipedia-nq``
                      (``nq-train.jsonl.gz`` for the train split, ``nq-test.jsonl.gz``
                      for the official NQ dev split).
* **TriviaQA**     : DPR positives from ``Tevatron/wikipedia-trivia``
                      (``trivia-train.jsonl.gz`` / ``trivia-test.jsonl.gz``) for the train
                      and official validation splits respectively.
* **HotpotQA**     : ``supporting_facts`` titles of the fullwiki split.

The Tevatron dumps store the DPR passage ``docid`` *and* its ``title``; we use the titles
directly, so no extra pass over the 21M-passage dump is needed.
"""
from __future__ import annotations

import gzip
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from common import get_logger, normalize_answer

LOG = get_logger("data.gold")

# repo -> (filename for the TRAIN split, filename for the OFFICIAL DEV split)
TEVATRON_FILES = {
    "nq": {
        "train": ("Tevatron/wikipedia-nq", "nq-train.jsonl.gz"),
        "dev": ("Tevatron/wikipedia-nq", "nq-test.jsonl.gz"),
    },
    "triviaqa": {
        "train": ("Tevatron/wikipedia-trivia", "trivia-train.jsonl.gz"),
        "dev": ("Tevatron/wikipedia-trivia", "trivia-test.jsonl.gz"),
    },
}

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def loose_normalize(s: str) -> str:
    """Very aggressive normalisation used only for *matching* query strings.

    ``nq_open`` and the DPR dumps tokenise contractions differently
    ("aint" vs "ai nt"), so we strip every non-alphanumeric character; this makes the
    two representations identical.
    """
    s = normalize_answer(s)
    return _NON_ALNUM.sub("", s)


def _download(repo: str, filename: str, dest_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    dest_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=repo,
        filename=filename,
        repo_type="dataset",
        local_dir=str(dest_dir),
    )
    return Path(path)


def _iter_jsonl_gz(path: Path) -> Iterable[Dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def map_positives(
    dataset: str,
    split: str,
    queries: List[Dict[str, Any]],
    cache_dir: Path,
    refresh: bool = False,
) -> Dict[str, List[int]]:
    """Return ``{qid: [dpr_pid, ...]}`` for the given queries using DPR positives."""
    if dataset not in TEVATRON_FILES:
        return {}
    repo, fname = TEVATRON_FILES[dataset][split]
    # the cache is per (dataset, source file) - it stores *every* query found in the
    # file, so later subsets (label / tune / eval) reuse it without re-parsing 1.3 GB.
    cache = cache_dir / f"{dataset}_{Path(fname).stem}_positives.json"
    raw: Dict[str, Any] = {}
    if cache.exists() and not refresh:
        with open(cache) as fh:
            raw = json.load(fh)
        LOG.info("loaded cached positives for %s/%s (%d queries)", dataset, split, len(raw))

    wanted = {q["qid"]: loose_normalize(q["question"]) for q in queries}
    missing = {qid for qid, nq in wanted.items() if nq not in raw}
    if not missing:
        LOG.info("all %d %s/%s queries already cached", len(wanted), dataset, split)
        return {qid: list(raw[nq]) for qid, nq in wanted.items() if nq in raw}

    path = _download(repo, fname, cache_dir)
    want_norm = {nq for nq in wanted.values()}
    LOG.info("scanning %s for %d unmatched %s queries", path.name, len(missing), dataset)
    n_new = 0
    for i, row in enumerate(_iter_jsonl_gz(path)):
        nq = loose_normalize(row.get("query", ""))
        if nq in want_norm and nq not in raw:
            pids = []
            for p in row.get("positive_passages", []) or []:
                try:
                    pids.append(int(p["docid"]))
                except Exception:
                    continue
            raw[nq] = pids
            n_new += 1
            if n_new == len(missing):
                break
        if i % 200000 == 0:
            LOG.info("  ...scanned %d rows, matched %d/%d", i, n_new, len(missing))
    with open(cache, "w") as fh:
        json.dump(raw, fh)
    out = {qid: list(raw[nq]) for qid, nq in wanted.items() if nq in raw}
    LOG.info("matched %d/%d %s/%s queries with DPR positives", len(out), len(wanted), dataset, split)
    return out


def map_positive_titles(
    dataset: str,
    split: str,
    queries: List[Dict[str, Any]],
    cache_dir: Path,
    refresh: bool = False,
) -> Dict[str, List[str]]:
    """Return ``{qid: [wikipedia_title, ...]}`` (article-level gold)."""
    if dataset not in TEVATRON_FILES:
        return {}
    repo, fname = TEVATRON_FILES[dataset][split]
    cache = cache_dir / f"{dataset}_{Path(fname).stem}_titles.json"
    raw: Dict[str, List[str]] = {}
    if cache.exists() and not refresh:
        with open(cache) as fh:
            raw = json.load(fh)

    wanted = {q["qid"]: loose_normalize(q["question"]) for q in queries}
    missing = {nq for nq in wanted.values() if nq not in raw}
    if missing:
        path = _download(repo, fname, cache_dir)
        LOG.info("scanning %s for gold titles (%d unmatched)", path.name, len(missing))
        for i, row in enumerate(_iter_jsonl_gz(path)):
            nq = loose_normalize(row.get("query", ""))
            if nq in missing and nq not in raw:
                titles = []
                for p in row.get("positive_passages", []) or []:
                    t = (p.get("title") or "").strip()
                    if t and t not in titles:
                        titles.append(t)
                raw[nq] = titles
            if i % 200000 == 0 and i:
                LOG.info("  ...scanned %d rows", i)
        with open(cache, "w") as fh:
            json.dump(raw, fh)

    out: Dict[str, List[str]] = {}
    for qid, nq in wanted.items():
        if nq in raw:
            out[qid] = list(raw[nq])
    LOG.info("gold titles for %d/%d %s/%s queries", len(out), len(wanted), dataset, split)
    return out


def hotpot_gold_titles(queries: List[Dict[str, Any]]) -> Set[str]:
    titles: Set[str] = set()
    for q in queries:
        for t in q.get("meta", {}).get("gold_titles", []) or []:
            if t:
                titles.add(t.strip())
    return titles
