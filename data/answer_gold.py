"""Resolve gold Wikipedia articles from answer strings.

The DPR annotations do not cover the official dev splits, and TriviaQA's HF
annotations carry no Wikipedia titles at all.  For those queries we fall back to
**answer -> article title matching** against the DPR dump: an answer such as
"David Seville" identifies the article "David Seville"; the gold articles of the query
are then every DPR passage with that title.

The match is done in one streaming pass over the 21M-passage dump with a hash-set lookup
on the normalised title, so it costs ~2 minutes and no extra memory.
"""
from __future__ import annotations

import gzip
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from common import get_logger, normalize_answer

LOG = get_logger("data.answer_gold")

_PAREN = re.compile(r"\s*\([^)]*\)\s*$")
_LEADING_THE = re.compile(r"^the\s+")


def candidate_title_keys(answer: str) -> List[str]:
    """Normalised title candidates for an answer string (most specific first)."""
    out: List[str] = []

    def add(x: str) -> None:
        x = normalize_answer(x)
        if x and x not in out:
            out.append(x)

    add(answer)
    stripped = _PAREN.sub("", answer).strip()
    add(stripped)
    add(_LEADING_THE.sub("", stripped).strip())
    # "X, Y" -> "X" (e.g. "Sinclair Lewis, Harry Sinclair Lewis")
    if "," in stripped:
        add(stripped.split(",")[0].strip())
    return out


def build_answer_title_index(
    dpr_path: Path,
    answers: Iterable[str],
    out_path: Optional[Path] = None,
) -> Dict[str, List[str]]:
    """Map normalised answer -> [matching DPR article titles]."""
    keys: Dict[str, List[str]] = {}
    for a in answers:
        for k in candidate_title_keys(a):
            keys.setdefault(k, [])
    LOG.info("resolving %d answer strings against the DPR dump", len(keys))
    key_set = set(keys)
    found: Dict[str, List[str]] = {}
    t0 = time.time()
    n = 0
    with gzip.open(dpr_path, "rt", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            n += 1
            title = parts[2]
            key = normalize_answer(title)
            if key in key_set:
                lst = found.setdefault(key, [])
                if title not in lst:
                    lst.append(title)
            if n % 5_000_000 == 0:
                LOG.info("  streamed %d lines (%.1f min), %d titles matched", n, (time.time() - t0) / 60, len(found))
    LOG.info("matched %d/%d answer strings in %.1f min", len(found), len(key_set), (time.time() - t0) / 60)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(found, fh)
    return found


def titles_for_answers(answers: List[str], index: Dict[str, List[str]]) -> List[str]:
    titles: List[str] = []
    for a in answers:
        for k in candidate_title_keys(a):
            for t in index.get(k, []):
                if t not in titles:
                    titles.append(t)
            if titles:
                break
    return titles
