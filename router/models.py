"""Router models: logistic regression, gradient-boosted trees, MLP, DeBERTa-v3-small.

All models expose the same interface so the training / calibration / evaluation code is
model-agnostic:

    m.fit(X, y, seed)        # X: pandas.DataFrame or np.ndarray, y: strategy index
    m.predict_proba(X)       # (N, 4) probabilities over [none, bm25, dense, hybrid]
    m.save(path) / load(path)
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from common import get_logger

LOG = get_logger("router.models")

STRATEGIES = ["none", "bm25", "dense", "hybrid"]
STRAT2IDX = {s: i for i, s in enumerate(STRATEGIES)}
# measured per-query cost (ms) used by the cost knob; filled in from the logs at runtime
DEFAULT_COST_MS = {"none": 0.0, "bm25": 1.0, "dense": 2.0, "hybrid": 3.0}


class BaseRouter:
    name = "base"

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.classes_: List[int] = list(range(4))
        self.feature_names_: List[str] = []

    def fit(self, X, y, seed: int = 0):  # pragma: no cover - interface
        raise NotImplementedError

    def predict_proba(self, X) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({"cls": type(self).__name__, "model": self, "kwargs": self.kwargs}, fh)

    @staticmethod
    def load(path: str | Path) -> "BaseRouter":
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        return blob["model"]


class LogRegRouter(BaseRouter):
    name = "logreg"

    def fit(self, X, y, seed: int = 0):
        from sklearn.linear_model import LogisticRegression

        self.feature_names_ = list(X.columns) if hasattr(X, "columns") else []
        Xa = X.values if hasattr(X, "values") else np.asarray(X)
        ya = np.asarray(y)
        self.model_ = LogisticRegression(
            C=self.kwargs.get("C", 1.0),
            max_iter=self.kwargs.get("max_iter", 2000),
            class_weight="balanced",
            multi_class="multinomial",
            random_state=seed,
            n_jobs=4,
        )
        self.model_.fit(Xa, ya)
        self.classes_ = list(self.model_.classes_)
        return self

    def predict_proba(self, X) -> np.ndarray:
        Xa = X.values if hasattr(X, "values") else np.asarray(X)
        p = self.model_.predict_proba(Xa)
        out = np.zeros((p.shape[0], 4))
        for j, c in enumerate(self.model_.classes_):
            out[:, int(c)] = p[:, j]
        return out


class GBTRouter(BaseRouter):
    name = "gbt"

    def fit(self, X, y, seed: int = 0):
        import lightgbm as lgb

        self.feature_names_ = list(X.columns) if hasattr(X, "columns") else []
        Xa = X.values if hasattr(X, "values") else np.asarray(X)
        ya = np.asarray(y)
        self.model_ = lgb.LGBMClassifier(
            n_estimators=self.kwargs.get("n_estimators", 300),
            learning_rate=self.kwargs.get("learning_rate", 0.06),
            max_depth=self.kwargs.get("max_depth", 6),
            num_leaves=self.kwargs.get("num_leaves", 31),
            subsample=self.kwargs.get("subsample", 0.9),
            subsample_freq=1,
            colsample_bytree=0.9,
            class_weight="balanced",
            random_state=seed,
            n_jobs=4,
            verbose=-1,
        )
        self.model_.fit(Xa, ya)
        self.classes_ = list(self.model_.classes_)
        return self

    def predict_proba(self, X) -> np.ndarray:
        Xa = X.values if hasattr(X, "values") else np.asarray(X)
        p = self.model_.predict_proba(Xa)
        out = np.zeros((p.shape[0], 4))
        for j, c in enumerate(self.model_.classes_):
            out[:, int(c)] = p[:, j]
        return out


class MLPRouter(BaseRouter):
    """Small MLP on the question embedding (+ lexical features), trained on the GPU."""

    name = "mlp"

    def fit(self, X, y, seed: int = 0):
        import torch
        import torch.nn as nn

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.feature_names_ = list(X.columns) if hasattr(X, "columns") else []
        Xa = X.values.astype(np.float32) if hasattr(X, "values") else np.asarray(X, dtype=np.float32)
        ya = np.asarray(y, dtype=np.int64)
        self.mu_ = Xa.mean(axis=0)
        self.sd_ = Xa.std(axis=0) + 1e-6
        Xn = (Xa - self.mu_) / self.sd_

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net_ = nn.Sequential(
            nn.Linear(Xn.shape[1], self.kwargs.get("hidden", 256)),
            nn.ReLU(),
            nn.Dropout(self.kwargs.get("dropout", 0.1)),
            nn.Linear(self.kwargs.get("hidden", 256), 4),
        ).to(device)
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.kwargs.get("lr", 1e-3), weight_decay=1e-4)
        lossf = nn.CrossEntropyLoss()
        Xt = torch.tensor(Xn, device=device)
        yt = torch.tensor(ya, device=device)
        n = len(yt)
        bs = min(self.kwargs.get("batch_size", 64), n)
        epochs = self.kwargs.get("epochs", 30)
        for ep in range(epochs):
            perm = torch.randperm(n, device=device)
            self.net_.train()
            tot = 0.0
            for i in range(0, n, bs):
                idx = perm[i : i + bs]
                opt.zero_grad()
                logits = self.net_(Xt[idx])
                loss = lossf(logits, yt[idx])
                loss.backward()
                opt.step()
                tot += float(loss) * len(idx)
            if ep % 10 == 0:
                LOG.info("  mlp epoch %d loss=%.4f", ep, tot / max(1, n))
        self.net_.eval()
        self.device_ = device
        return self

    def predict_proba(self, X) -> np.ndarray:
        import torch

        Xa = X.values.astype(np.float32) if hasattr(X, "values") else np.asarray(X, dtype=np.float32)
        Xn = (Xa - self.mu_) / self.sd_
        with torch.no_grad():
            logits = self.net_(torch.tensor(Xn, device=self.device_))
            p = torch.softmax(logits, dim=-1).cpu().numpy()
        return p


class DebertaRouter(BaseRouter):
    """Fine-tuned deberta-v3-small classifier over the raw question text."""

    name = "deberta"

    def __init__(self, model_name: str = "microsoft/deberta-v3-small", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.model_name = model_name

    def fit(self, X, y, seed: int = 0):
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.manual_seed(seed)
        np.random.seed(seed)
        texts = list(X)
        ya = np.asarray(y, dtype=np.int64)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=4).to(device)
        max_len = self.kwargs.get("max_len", 64)

        class DS(Dataset):
            def __init__(self, texts, labels):
                self.texts, self.labels = texts, labels

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, i):
                enc = tok(self.texts[i], truncation=True, max_length=max_len, padding="max_length", return_tensors="pt")
                return {k: v.squeeze(0) for k, v in enc.items()}, int(self.labels[i])

        loader = DataLoader(DS(texts, ya), batch_size=self.kwargs.get("batch_size", 32), shuffle=True)
        opt = torch.optim.AdamW(model.parameters(), lr=self.kwargs.get("lr", 2e-5), weight_decay=0.01)
        epochs = self.kwargs.get("epochs", 4)
        model.train()
        for ep in range(epochs):
            tot, cnt = 0.0, 0
            for batch, labels in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = labels.to(device)
                opt.zero_grad()
                out = model(**batch)
                loss = out.loss
                loss.backward()
                opt.step()
                tot += float(loss) * len(labels)
                cnt += len(labels)
            LOG.info("  deberta epoch %d loss=%.4f", ep, tot / max(1, cnt))
        model.eval()
        self.model_ = model
        self.tok_ = tok
        self.device_ = device
        self.max_len_ = max_len
        return self

    def predict_proba(self, X) -> np.ndarray:
        import torch

        texts = list(X)
        out = []
        bs = 64
        for i in range(0, len(texts), bs):
            chunk = texts[i : i + bs]
            enc = self.tok_(chunk, truncation=True, max_length=self.max_len_, padding=True, return_tensors="pt").to(self.device_)
            with torch.no_grad():
                logits = self.model_(**enc).logits
                out.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(out, axis=0)


def build_model(name: str, cfg: Dict[str, Any]) -> BaseRouter:
    if name == "logreg":
        return LogRegRouter(**cfg.get("logreg", {}))
    if name == "gbt":
        return GBTRouter(**cfg.get("gbt", {}))
    if name == "mlp":
        return MLPRouter(**cfg.get("mlp", {}))
    if name == "deberta":
        return DebertaRouter(**cfg.get("deberta", {}).get("model", "microsoft/deberta-v3-small"), **cfg.get("deberta", {}))
    raise ValueError(f"unknown router model {name}")


# --------------------------------------------------------------------------------------
# calibration + cost knob
# --------------------------------------------------------------------------------------
def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Fit a single temperature on log-probabilities (NLL minimisation)."""
    from scipy.optimize import minimize_scalar

    logp = np.log(np.clip(logits, 1e-12, None))

    def nll(t):
        scaled = logp / max(t, 1e-3)
        scaled = scaled - scaled.max(axis=1, keepdims=True)
        lse = np.log(np.exp(scaled).sum(axis=1))
        return float(np.mean(lse - scaled[np.arange(len(y)), y]))

    res = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    return float(res.x)


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    logp = np.log(np.clip(probs, 1e-12, None)) / max(temperature, 1e-3)
    logp = logp - logp.max(axis=1, keepdims=True)
    p = np.exp(logp)
    return p / p.sum(axis=1, keepdims=True)


def decide(probs: np.ndarray, cost_ms: Dict[str, float], lam: float) -> np.ndarray:
    """Cost-aware decision: argmax_s log p(s) - lam * cost(s) (cost in ms)."""
    logp = np.log(np.clip(probs, 1e-12, None))
    costs = np.array([cost_ms.get(s, 0.0) for s in STRATEGIES], dtype=np.float64)
    util = logp - lam * costs[None, :]
    return util.argmax(axis=1)
