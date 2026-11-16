"""Build the retrieval corpus (gold-preserving subsample of DPR Wikipedia).

We cannot index the full 21M-passage dump inside the memory budget of this machine
(8 GB GPU, 29 GB shared with the desktop).  Following PLAN.md §3 we build a
**gold-preserving subsample**:

  corpus = (all DPR passages whose *title* is a gold article for any of our queries)
         + (N_distractor passages sampled uniformly at random from the 21M dump)

The build **streams**: passages are written straight to disk and only tiny metadata
(pid list) is kept in RAM, so peak memory stays well under 1 GB.
"""
from __future__ import annotations

import gzip
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from common import get_logger

LOG = get_logger("data.corpus")

DPR_TOTAL = 21_015_324


def _iter_dpr(path: Path) -> Iterable[Tuple[int, str, str]]:
    """Yield (pid, text, title) from the DPR psgs_w100.tsv.gz dump."""
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            try:
                pid = int(parts[0])
            except ValueError:
                continue
            yield pid, parts[1], parts[2]


def build_corpus(
    dpr_path: Path,
    out_path: Path,
    gold_titles: Set[str],
    n_distractors: int,
    seed: int,
    max_passage_words: int = 100,
) -> Dict[str, Any]:
    """Stream the DPR dump once and write the subsampled corpus.

    `gold_titles` are the *article* titles that must be preserved (lowercased).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    rng = np.random.default_rng(seed)
    n_sample = min(n_distractors + 50_000, DPR_TOTAL)
    sample_ids = set(int(x) for x in rng.choice(DPR_TOTAL, size=n_sample, replace=False) + 1)
    LOG.info("sampling %d candidate distractor ids from %d DPR passages", n_sample, DPR_TOTAL)

    title_norm = {t.strip().lower() for t in gold_titles}
    LOG.info("preserving %d gold article titles", len(title_norm))

    n_gold = 0
    n_dist = 0
    n_lines = 0
    gold_pids: List[int] = []
    with open(out_path, "w", encoding="utf-8") as out:
        for pid, text, title in _iter_dpr(dpr_path):
            n_lines += 1
            is_gold = title.strip().lower() in title_norm
            if is_gold:
                n_gold += 1
                gold_pids.append(pid)
            elif pid in sample_ids:
                n_dist += 1
            else:
                continue
            if max_passage_words:
                words = text.split()
                if len(words) > max_passage_words:
                    text = " ".join(words[:max_passage_words])
            out.write(json.dumps({"pid": pid, "title": title, "text": text, "kind": "gold" if is_gold else "distractor"}, ensure_ascii=False) + "\n")
            if n_lines % 4_000_000 == 0:
                LOG.info(
                    "  streamed %d lines (%.1f min) gold=%d distractors=%d",
                    n_lines,
                    (time.time() - t0) / 60,
                    n_gold,
                    n_dist,
                )

    meta = {
        "dpr_total": DPR_TOTAL,
        "dpr_file": str(dpr_path),
        "corpus_size": n_gold + n_dist,
        "n_gold": n_gold,
        "n_distractor": n_dist,
        "requested_distractors": n_distractors,
        "n_gold_titles": len(title_norm),
        "seed": seed,
        "build_minutes": round((time.time() - t0) / 60, 2),
    }
    LOG.info(
        "corpus written: %s (%d passages: %d gold-article + %d distractor) in %.1f min",
        out_path,
        meta["corpus_size"],
        n_gold,
        n_dist,
        meta["build_minutes"],
    )
    with open(out_path.with_suffix(".meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def corpus_doc_frequency(corpus_path: Path, top_n: int = 200_000) -> Dict[str, int]:
    """Document frequency of the most common tokens (for the IDF / rarity feature).

    Only the `top_n` most frequent tokens are kept - that is all the IDF feature needs
    (rare terms are approximated by the "not in this list" test).
    """
    from collections import Counter

    df: Counter = Counter()
    n_docs = 0
    n_tokens = 0
    with open(corpus_path) as fh:
        for line in fh:
            r = json.loads(line)
            toks = set(r["text"].lower().split())
            n_docs += 1
            n_tokens += len(toks)
            df.update(toks)
    keep = {w: c for w, c in df.most_common(top_n)}
    out = {"n_docs": n_docs, "n_unique_tokens": len(df), "n_tokens": n_tokens, "top_df": keep}
    return out
