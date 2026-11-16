"""Hybrid retrieval: reciprocal rank fusion (k=60) of BM25 and dense, plus a weighted-score variant."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from common import get_logger
from retrieval.base import RetrievalResult, Retriever
from retrieval.bm25 import BM25Retriever
from retrieval.dense import DenseRetriever

LOG = get_logger("retrieval.hybrid")


class HybridRetriever(Retriever):
    """Fuses the BM25 and the dense rankings.

    ``fusion='rrf'``      : reciprocal rank fusion, score = sum_c 1/(rrf_k + rank_c)
    ``fusion='weighted'`` : normalised score fusion, score = alpha * dense + (1-alpha) * bm25
    """

    name = "hybrid"

    def __init__(
        self,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        rrf_k: int = 60,
        fusion: str = "rrf",
        weighted_alpha: float = 0.5,
    ) -> None:
        self.bm25 = bm25
        self.dense = dense
        self.rrf_k = rrf_k
        self.fusion = fusion
        self.weighted_alpha = weighted_alpha

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _rrf(rank_lists: List[List[int]], rrf_k: int) -> Dict[int, float]:
        scores: Dict[int, float] = {}
        for ranks in rank_lists:
            for r, doc in enumerate(ranks):
                scores[doc] = scores.get(doc, 0.0) + 1.0 / (rrf_k + r + 1)
        return scores

    @staticmethod
    def _minmax(xs: List[float]) -> List[float]:
        if not xs:
            return xs
        lo, hi = min(xs), max(xs)
        if hi - lo < 1e-9:
            return [0.0 for _ in xs]
        return [(x - lo) / (hi - lo) for x in xs]

    # ------------------------------------------------------------------ search
    def retrieve(self, queries: Sequence[str], k: int = 5) -> List[RetrievalResult]:
        t0 = time.perf_counter()
        bm_res = self.bm25.retrieve(queries, k=max(k * 4, 20))
        dn_res = self.dense.retrieve(queries, k=max(k * 4, 20))
        dt = (time.perf_counter() - t0) * 1000.0 / max(1, len(queries))

        results = []
        for i in range(len(queries)):
            bm_passages = bm_res[i].passages
            dn_passages = dn_res[i].passages
            bm_by_pid = {p["pid"]: p for p in bm_passages}
            dn_by_pid = {p["pid"]: p for p in dn_passages}

            if self.fusion == "weighted":
                bm_scores = self._minmax([p["score"] for p in bm_passages])
                dn_scores = self._minmax([p["score"] for p in dn_passages])
                a = self.weighted_alpha
                fused: Dict[int, float] = {}
                for j, p in enumerate(bm_passages):
                    fused[p["pid"]] = fused.get(p["pid"], 0.0) + (1 - a) * bm_scores[j]
                for j, p in enumerate(dn_passages):
                    fused[p["pid"]] = fused.get(p["pid"], 0.0) + a * dn_scores[j]
            else:
                bm_rank = [p["pid"] for p in bm_passages]
                dn_rank = [p["pid"] for p in dn_passages]
                fused = self._rrf([bm_rank, dn_rank], self.rrf_k)

            ordered = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
            passages = []
            for pid, score in ordered:
                src = bm_by_pid.get(pid) or dn_by_pid.get(pid)
                passages.append(
                    {
                        "pid": pid,
                        "title": src["title"],
                        "text": src["text"],
                        "score": float(score),
                        "in_bm25": pid in bm_by_pid,
                        "in_dense": pid in dn_by_pid,
                    }
                )
            results.append(
                RetrievalResult(
                    passages=passages,
                    latency_ms=dt,
                    n_calls=2,  # one BM25 + one dense call
                    extra={"ranker": f"hybrid-{self.fusion}"},
                )
            )
        return results

    def stats(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "fusion": self.fusion,
            "rrf_k": self.rrf_k,
            "weighted_alpha": self.weighted_alpha,
            "n_docs": len(self.bm25.corpus),
        }
