"""Dataset loading + deterministic sampling for NQ-open, TriviaQA and HotpotQA.

Split discipline (see PLAN.md §3):
  * ``train`` split  -> router label pool (fit) and held-out dev-from-train slice (tune)
  * ``validation``   -> official dev split, used ONLY for final evaluation
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from common import get_logger, normalize_answer

LOG = get_logger("data.datasets")

DATASETS = ["nq", "triviaqa", "hotpotqa"]

# --------------------------------------------------------------------------------------
# raw loading
# --------------------------------------------------------------------------------------


def _load_hf_split(hf_id: str, cfg: Optional[str], split: str) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(hf_id, cfg, split=split)
    return [dict(r) for r in ds]


def load_nq(cfg) -> Dict[str, List[Dict[str, Any]]]:
    d = cfg["datasets"]["nq"]
    out = {}
    for key, split in (("train", d["train_split"]), ("dev", d["dev_split"])):
        rows = _load_hf_split(d["hf_id"], d["hf_config"], split)
        recs = []
        for i, r in enumerate(rows):
            answers = [str(a) for a in r["answer"]]
            recs.append(
                {
                    "qid": f"nq-{split}-{i}",
                    "dataset": "nq",
                    "split": split,
                    "question": r["question"].strip(),
                    "answers": answers,
                    "meta": {},
                }
            )
        out[key] = recs
    return out


def load_triviaqa(cfg) -> Dict[str, List[Dict[str, Any]]]:
    d = cfg["datasets"]["triviaqa"]
    out = {}
    for key, split in (("train", d["train_split"]), ("dev", d["dev_split"])):
        rows = _load_hf_split(d["hf_id"], d["hf_config"], split)
        recs = []
        for i, r in enumerate(rows):
            ans = r["answer"]
            answers = [ans["value"]] + list(ans.get("aliases", []) or [])
            answers = [a for a in dict.fromkeys(answers) if a]
            recs.append(
                {
                    "qid": f"tqa-{split}-{i}",
                    "dataset": "triviaqa",
                    "split": split,
                    "question": r["question"].strip(),
                    "answers": answers,
                    "meta": {"question_id": r.get("question_id", ""), "question_source": r.get("question_source", "")},
                }
            )
        out[key] = recs
    return out


def load_hotpotqa(cfg) -> Dict[str, List[Dict[str, Any]]]:
    d = cfg["datasets"]["hotpotqa"]
    out = {}
    for key, split in (("train", d["train_split"]), ("dev", d["dev_split"])):
        rows = _load_hf_split(d["hf_id"], d["hf_config"], split)
        recs = []
        for i, r in enumerate(rows):
            sf = r.get("supporting_facts") or {}
            titles = list(sf.get("title", []) or [])
            answers = [str(r["answer"])] if r["answer"] else []
            recs.append(
                {
                    "qid": f"hqa-{split}-{i}",
                    "dataset": "hotpotqa",
                    "split": split,
                    "question": r["question"].strip(),
                    "answers": answers,
                    "meta": {
                        "type": r.get("type", ""),
                        "level": r.get("level", ""),
                        "gold_titles": titles,
                        "hp_id": r.get("id", ""),
                    },
                }
            )
        out[key] = recs
    return out


LOADERS = {"nq": load_nq, "triviaqa": load_triviaqa, "hotpotqa": load_hotpotqa}


def load_all(cfg) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    out = {}
    for name, fn in LOADERS.items():
        LOG.info("loading %s", name)
        out[name] = fn(cfg)
        LOG.info("  %s: train=%d dev=%d", name, len(out[name]["train"]), len(out[name]["dev"]))
    return out


# --------------------------------------------------------------------------------------
# deterministic sampling
# --------------------------------------------------------------------------------------
def stable_hash(text: str) -> int:
    """Deterministic across processes (unlike built-in hash())."""
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def sample_indices(n_total: int, n_want: int, seed: int, tag: str) -> np.ndarray:
    """Deterministic sample of `n_want` indices out of `n_total` (without replacement)."""
    n_want = min(n_want, n_total)
    rng = np.random.default_rng(seed + (stable_hash(tag) % 100000))
    return np.sort(rng.choice(n_total, size=n_want, replace=False))


def build_query_splits(cfg, data: Dict[str, Dict[str, List[Dict[str, Any]]]], oversample: Dict[str, float] | None = None):
    """Split each dataset into: label pool (train), dev-from-train (tune), eval (official dev).

    `oversample` compensates for datasets whose train split is only partially covered by
    the DPR gold annotations (NQ-open train: 58 219 of 87 146 questions).  The extra
    candidates are dropped again once the gold articles are known.
    """
    sizes = cfg["sizes"]
    seed = cfg["seed"]
    oversample = oversample or {}
    out: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for name, splits in data.items():
        train = splits["train"]
        dev = splits["dev"]
        n_lab = sizes["label_pool"] if name != "pilot" else sizes["pilot"]
        n_tune = sizes["dev_from_train"] if name != "pilot" else max(1, sizes["pilot"] // 5)
        n_eval = sizes["eval_dev"] if name != "pilot" else sizes["pilot"]
        ov = oversample.get(name, 1.0)

        idx_lab = sample_indices(len(train), int(round(n_lab * ov)), seed, f"{name}-label")
        idx_tune = sample_indices(len(train), int(round(n_tune * ov)), seed + 7, f"{name}-tune")
        mask = np.ones(len(train), dtype=bool)
        mask[idx_lab] = False
        idx_tune = idx_tune[mask[idx_tune]]  # enforce disjointness
        idx_eval = sample_indices(len(dev), n_eval, seed + 13, f"{name}-eval")

        def mark(rows, idx, tag):
            out_rows = []
            for j in idx:
                r = dict(rows[j])
                r["qid"] = f"{name}-{tag}-{int(j)}"
                r["subset"] = tag
                out_rows.append(r)
            return out_rows

        out[name] = {
            "label": mark(train, idx_lab, "label"),
            "tune": mark(train, idx_tune, "tune"),
            "eval": mark(dev, idx_eval, "eval"),
        }
        LOG.info(
            "%s -> candidates label=%d tune=%d eval=%d (oversample=%.2f, disjoint label/tune: %s)",
            name,
            len(out[name]["label"]),
            len(out[name]["tune"]),
            len(out[name]["eval"]),
            ov,
            len(set(idx_lab) & set(idx_tune)) == 0,
        )
    return out
