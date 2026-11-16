"""Shared utilities for Pythia: config, seeding, IO, timing, normalisation.

Everything in the project is deterministic given the seed in configs/default.yaml.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import string
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "default.yaml"


# --------------------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------------------
def load_config(path: str | Path | None = None, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load a YAML config, resolve a single `extends:` chain, apply dotted overrides."""
    import yaml

    path = Path(path) if path else CONFIG_PATH
    with open(path) as fh:
        cfg = yaml.safe_load(fh)

    parent = cfg.pop("extends", None)
    if parent:
        parent_path = (path.parent / parent).resolve() if not str(parent).startswith("/") else Path(parent)
        base = load_config(parent_path)
        cfg = deep_update(base, cfg)

    if overrides:
        cfg = deep_update(cfg, overrides)
    return cfg


def deep_update(base: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in other.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def set_by_path(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = cfg
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


# --------------------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------------------
def set_seed(seed: int, init_cuda: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if init_cuda and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:  # torch may not be installed in the data-only stage
        pass


# --------------------------------------------------------------------------------------
# logging
# --------------------------------------------------------------------------------------
_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str, logfile: Optional[str | Path] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fh = logging.StreamHandler(sys.stdout)
    fh.setFormatter(logging.Formatter(_LOG_FORMAT))
    logger.addHandler(fh)
    if logfile:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        lf = logging.FileHandler(logfile)
        lf.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(lf)
    logger.propagate = False
    return logger


# --------------------------------------------------------------------------------------
# io
# --------------------------------------------------------------------------------------
def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=str)


def read_json(path: str | Path) -> Any:
    with open(path) as fh:
        return json.load(fh)


def hash_text(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------------------
# timing
# --------------------------------------------------------------------------------------
class Timer:
    """Context manager measuring wall-clock milliseconds."""

    def __init__(self) -> None:
        self.ms = 0.0

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.ms = (time.perf_counter() - self._t0) * 1000.0


# --------------------------------------------------------------------------------------
# SQuAD-style normalisation (official open-domain QA normalisation)
# --------------------------------------------------------------------------------------
_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(s: str) -> str:
    """Lowercase, strip punctuation/articles/extra whitespace (DPR/SQuAD official)."""
    s = s.lower()
    s = s.translate(_PUNCT_TABLE)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def answer_tokens(s: str) -> List[str]:
    return normalize_answer(s).split()


def exact_match(pred: str, golds: Sequence[str]) -> float:
    np_ = normalize_answer(pred)
    return float(any(np_ == normalize_answer(g) for g in golds))


def token_f1(pred: str, golds: Sequence[str]) -> float:
    """Max over gold aliases of SQuAD token F1."""
    pt = answer_tokens(pred)
    best = 0.0
    for g in golds:
        gt = answer_tokens(g)
        if not pt or not gt:
            best = max(best, float(pt == gt))
            continue
        common = Counter(pt) & Counter(gt)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = num_same / len(pt)
        recall = num_same / len(gt)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def best_score(pred: str, golds: Sequence[str]) -> Dict[str, float]:
    return {"em": exact_match(pred, golds), "f1": token_f1(pred, golds)}


# --------------------------------------------------------------------------------------
# misc
# --------------------------------------------------------------------------------------
def git_sha(root: Path = ROOT) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def env_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "git_sha": git_sha(),
        "platform": os.uname().release,
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
            info["capability"] = torch.cuda.get_device_capability(0)
    except Exception:
        pass
    for mod in ("vllm", "transformers", "faiss", "bm25s", "sklearn", "datasets", "lightgbm", "xgboost"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            info[mod] = None
    return info


@dataclass
class Passage:
    pid: int
    title: str
    text: str
    score: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {"pid": self.pid, "title": self.title, "text": self.text, "score": self.score}
