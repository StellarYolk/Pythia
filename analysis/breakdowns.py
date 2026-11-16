"""Analysis helpers: answer types, popularity proxies, failure taxonomy."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from common import normalize_answer

_MONTHS = "january february march april may june july august september october november december".split()
_YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
_NUM_RE = re.compile(r"^[\d\s,.\-/]+$")
_DATE_RE = re.compile(r"\b(19|20)\d{2}\b|\b(" + "|".join(_MONTHS) + r")\b", re.I)


def answer_type(answers: Sequence[str]) -> str:
    """Rule-based answer type: boolean / number / date / person / entity / other."""
    if not answers:
        return "other"
    a = answers[0].strip()
    low = a.lower()
    if low in ("yes", "no"):
        return "boolean"
    if _NUM_RE.match(a):
        return "number"
    if _DATE_RE.search(a):
        return "date"
    words = a.split()
    caps = [w for w in words if w[:1].isupper()]
    if len(caps) >= 2 and len(words) <= 4:
        return "person"
    if caps:
        return "entity"
    return "other"


def wh_word(question: str) -> str:
    toks = re.findall(r"[a-z]+", question.lower())
    for t in toks:
        if t in ("who", "what", "when", "where", "which", "why", "how", "whose", "whom", "name"):
            return t
    return "other"


def hop_type(row: pd.Series) -> str:
    """Single-hop vs multi-hop (HotpotQA provides the type; others are single-hop)."""
    if row.get("dataset") == "hotpotqa":
        return "multi-hop" if row.get("hp_type") in ("bridge", "comparison") else "single-hop"
    return "single-hop"


def add_breakdown_columns(df: pd.DataFrame, df_counts: Optional[Dict[str, int]] = None, n_docs: int = 0) -> pd.DataFrame:
    """Attach the analysis columns used by the breakdowns."""
    df = df.copy()
    answers = [json_answers(a) for a in df["answers"]]
    df["answer_type"] = [answer_type(a) for a in answers]
    df["answer"] = [a[0] if a else "" for a in answers]
    df["wh_word"] = [wh_word(q) for q in df["question"]]
    df["n_words"] = df["question"].str.split().str.len()
    df["log_len"] = np.log1p(df["n_words"])
    if df_counts:
        pops = []
        for a in answers:
            if not a:
                pops.append(np.nan)
                continue
            toks = [t for t in normalize_answer(a[0]).split() if len(t) > 3]
            if not toks:
                pops.append(np.nan)
                continue
            # popularity proxy = document frequency of the rarest content token of the answer
            dfs = [df_counts.get(t, 0) for t in toks]
            pops.append(float(min(dfs)))
        df["answer_popularity"] = pops
        df["answer_popularity_bin"] = pd.cut(
            df["answer_popularity"],
            bins=[-1, 0, 5, 50, 500, np.inf],
            labels=["0", "1-5", "6-50", "51-500", ">500"],
        )
    return df


def json_answers(s: Any) -> List[str]:
    import json

    if isinstance(s, list):
        return [str(x) for x in s]
    try:
        v = json.loads(s)
        return [str(x) for x in v] if isinstance(v, list) else [str(v)]
    except Exception:
        return [str(s)]


# --------------------------------------------------------------------------------------
# failure taxonomy
# --------------------------------------------------------------------------------------
FAILURE_CATEGORIES = [
    "router_skipped_long_tail_fact",
    "retrieval_returned_distractors",
    "multi_hop_needs_iterative_retrieval",
    "answer_normalisation_artefact",
    "label_noise_em_strictness",
    "generator_hallucination",
    "ambiguous_or_wrong_gold",
    "no_failure",
]


def categorise_failure(row: pd.Series, oracle_strategy: str, pred_strategy: str, oracle_em: float, pred_em: float) -> str:
    """Rule-based first pass of the failure taxonomy (manually verified in docs/failures.md)."""
    if pred_em >= 1.0:
        return "no_failure"
    if oracle_em < 1.0 and pred_em < 1.0:
        # the oracle strategy also failed -> not a routing error
        if row.get("hit", 0) == 0:
            return "retrieval_returned_distractors"
        return "generator_hallucination"
    if oracle_em >= 1.0 and pred_em < 1.0:
        if oracle_strategy == "none" or pred_strategy == "none":
            # the router skipped a retrieval that was needed
            if row.get("n_words", 0) >= 12:
                return "router_skipped_long_tail_fact"
            return "router_skipped_long_tail_fact"
        if row.get("hit", 0) == 0:
            return "retrieval_returned_distractors"
        return "generator_hallucination"
    if oracle_em < 1.0 and pred_em >= 1.0:
        return "label_noise_em_strictness"
    return "no_failure"
