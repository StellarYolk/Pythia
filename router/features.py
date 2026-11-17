"""Router feature extraction.

Four tiers, each ablated separately (PLAN.md §7):

``lex``    lexical features of the question - essentially free
``emb``    dense embedding of the question (bge-base) - one cheap encoder pass
``probe``  retrieval-derived features (BM25 top-1 score / gap, dense top-1 similarity) -
           these cost a *full retrieval call*, which has to be charged to the router
``gen``    generator-derived features from the no-retrieval answer (log-prob, entropy,
           margin) - these cost one extra generation

The cost of every tier is measured so the analysis can show that expensive tiers erode
the savings.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from common import get_logger

LOG = get_logger("router.features")

WH_WORDS = ["who", "what", "when", "where", "which", "why", "how", "whose", "whom", "name"]
_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_CAPS_RE = re.compile(r"\b[A-Z][a-z]+")
_NUM_RE = re.compile(r"\b\d[\d,\.]*\b")

STOPWORDS = set(
    """a an the of in on at to for from by with and or but if then than that this these those is are was were
    be been being am do does did doing have has had having will would shall should can could may might must
    what which who whom whose when where why how it its it's as into about over under again further once here
    there all any both each few more most other some such no nor not only own same so too very s t don now""".split()
)


def lexical_features(questions: Sequence[str], df: Optional[Dict[str, int]] = None, n_docs: int = 0) -> pd.DataFrame:
    """Cheap surface features of the question."""
    rows = []
    for q in questions:
        q = q.strip()
        words = _WORD_RE.findall(q.lower())
        nw = max(1, len(words))
        toks = [w for w in words if w not in STOPWORDS]
        feats: Dict[str, Any] = {
            "n_chars": len(q),
            "n_words": len(words),
            "log_n_words": math.log1p(len(words)),
            "mean_word_len": sum(len(w) for w in words) / nw,
            "max_word_len": max((len(w) for w in words), default=0),
            "n_digits": sum(ch.isdigit() for ch in q),
            "has_number": int(bool(_NUM_RE.search(q))),
            "n_caps_entities": len(_CAPS_RE.findall(q[1:] if q else q)),
            "n_caps_all": sum(ch.isupper() for ch in q),
            "stopword_ratio": 1 - len(toks) / nw,
            "rare_ratio": 0.0,
            "mean_idf": 0.0,
            "max_idf": 0.0,
            "n_rare_terms": 0,
        }
        for w in WH_WORDS:
            feats[f"wh_{w}"] = int(w in words)
        feats["wh_other"] = int(not any(w in words for w in WH_WORDS))
        feats["starts_with_wh"] = int(bool(words) and words[0] in WH_WORDS)
        if df:
            idfs = []
            for w in toks:
                d = df.get(w, 0)
                idf = math.log(1 + (n_docs - d + 0.5) / (d + 0.5))
                idfs.append(idf)
            if idfs:
                feats["mean_idf"] = float(np.mean(idfs))
                feats["max_idf"] = float(np.max(idfs))
                feats["rare_ratio"] = float(np.mean([1.0 if df.get(w, 0) <= 10 else 0.0 for w in toks]))
                feats["n_rare_terms"] = int(sum(1 for w in toks if df.get(w, 0) <= 10))
        rows.append(feats)
    df_out = pd.DataFrame(rows)
    df_out.index = range(len(df_out))
    return df_out


def embedding_features(questions: Sequence[str], model_name: str = "BAAI/bge-base-en-v1.5", device: str = "cuda", batch_size: int = 256) -> np.ndarray:
    """Dense query embeddings (L2-normalised)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    from retrieval.dense import QUERY_PREFIX

    if device == "cuda" and not torch.cuda.is_available():
        LOG.warning("CUDA unavailable for the embedding tier - falling back to CPU")
        device = "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(questions), batch_size):
            chunk = [QUERY_PREFIX + q for q in questions[i : i + batch_size]]
            enc = tok(chunk, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
            hidden = model(**enc).last_hidden_state[:, 0]
            hidden = torch.nn.functional.normalize(hidden, dim=-1)
            out.append(hidden.float().cpu().numpy())
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.concatenate(out, axis=0).astype(np.float32)


def probe_features(questions: Sequence[str], bm25, dense, k: int = 5) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Retrieval-derived features.  NOTE: this costs one BM25 *and* one dense query."""
    import time

    t0 = time.perf_counter()
    bm_res = bm25.retrieve(questions, k=k)
    t_bm25 = (time.perf_counter() - t0) * 1000.0 / max(1, len(questions))
    t0 = time.perf_counter()
    dn_res = dense.retrieve(questions, k=k)
    t_dense = (time.perf_counter() - t0) * 1000.0 / max(1, len(questions))

    rows = []
    for b, d in zip(bm_res, dn_res):
        bs = [p["score"] for p in b.passages]
        ds = [p["score"] for p in d.passages]
        bm_top = bs[0] if bs else 0.0
        bm_gap = (bs[0] - bs[1]) if len(bs) > 1 else 0.0
        bm_std = float(np.std(bs)) if bs else 0.0
        dn_top = ds[0] if ds else 0.0
        dn_gap = (ds[0] - ds[1]) if len(ds) > 1 else 0.0
        dn_std = float(np.std(ds)) if ds else 0.0
        overlap = len({p["pid"] for p in b.passages} & {p["pid"] for p in d.passages}) / max(1, k)
        rows.append(
            {
                "probe_bm25_top1": bm_top,
                "probe_bm25_gap": bm_gap,
                "probe_bm25_std": bm_std,
                "probe_dense_top1": dn_top,
                "probe_dense_gap": dn_gap,
                "probe_dense_std": dn_std,
                "probe_overlap": overlap,
            }
        )
    cost = {
        "probe_latency_ms": t_bm25 + t_dense,
        "probe_bm25_ms": t_bm25,
        "probe_dense_ms": t_dense,
    }
    return pd.DataFrame(rows), cost


def generator_features(none_log: pd.DataFrame) -> pd.DataFrame:
    """Features derived from the no-retrieval generation (log-prob / entropy / margin)."""
    df = none_log[["qid", "mean_logprob", "first_token_entropy", "mean_entropy", "first_token_margin"]].copy()
    df = df.rename(
        columns={
            "mean_logprob": "gen_mean_logprob",
            "first_token_entropy": "gen_first_entropy",
            "mean_entropy": "gen_mean_entropy",
            "first_token_margin": "gen_first_margin",
        }
    )
    df = df.fillna({"gen_mean_logprob": -10.0, "gen_first_entropy": 10.0, "gen_mean_entropy": 10.0, "gen_first_margin": 0.0})
    return df.reset_index(drop=True)


def build_features(
    questions: Sequence[str],
    tiers: Sequence[str],
    df_counts: Optional[Dict[str, int]] = None,
    n_docs: int = 0,
    bm25=None,
    dense=None,
    none_log: Optional[pd.DataFrame] = None,
    emb: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Assemble the feature matrix for the requested tiers."""
    frames = []
    cost: Dict[str, float] = {}
    if "lex" in tiers:
        frames.append(lexical_features(questions, df_counts, n_docs))
    if "emb" in tiers:
        if emb is None:
            emb = embedding_features(questions)
        frames.append(pd.DataFrame(emb, columns=[f"emb_{i}" for i in range(emb.shape[1])]))
    if "probe" in tiers:
        if bm25 is None or dense is None:
            raise ValueError("probe tier needs bm25 and dense retrievers")
        pf, pcost = probe_features(questions, bm25, dense)
        frames.append(pf)
        cost.update(pcost)
    if "gen" in tiers:
        if none_log is None:
            raise ValueError("gen tier needs the no-retrieval log")
        frames.append(generator_features(none_log))
    X = pd.concat(frames, axis=1)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X, cost
