"""BM25 retrieval.

The brief suggests Pyserini.  This machine has no JVM and only 29 GB of RAM shared with
the desktop (a Pyserini Lucene index over 21M passages needs ~30 GB), so we use `bm25s`
(Okapi BM25, the Lucene scoring function) over the gold-preserving corpus subsample -
see PLAN.md §1 (deviation 3).

Memory discipline: `bm25s.tokenize` materialises Python lists of ints, which for ~1M
documents would cost >3 GB of RSS.  We therefore tokenise ourselves in chunks and keep
every document's token ids in a numpy int32 array (~0.5 GB for the whole corpus) and hand
the (ids, vocab) pair to bm25s for the index / scoring.  The tokenisation is identical to
bm25s' own (lowercase, `\\w\\w+` pattern, English stopwords, Porter2 stemming).
"""
from __future__ import annotations

import json
import pickle
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from common import get_logger
from retrieval.base import RetrievalResult, Retriever
from retrieval.corpus_store import CorpusStore

LOG = get_logger("retrieval.bm25")

TOKEN_PATTERN = re.compile(r"\b\w\w+\b")


class BM25Tokenizer:
    """Lowercase -> word tokens -> English stopwords -> Porter2 stemming."""

    def __init__(self, stemmer: bool = True) -> None:
        from bm25s.tokenization import STOPWORDS_EN

        self.stopwords = set(STOPWORDS_EN)
        self.stemmer = None
        if stemmer:
            try:
                import Stemmer

                self.stemmer = Stemmer.Stemmer("english").stemWord
            except Exception as exc:  # pragma: no cover
                LOG.warning("PyStemmer unavailable (%s) - BM25 without stemming", exc)
        self._stem_cache: Dict[str, str] = {}

    def tokenize(self, text: str) -> List[str]:
        out = []
        for tok in TOKEN_PATTERN.findall(text.lower()):
            if tok in self.stopwords:
                continue
            if self.stemmer is not None:
                st = self._stem_cache.get(tok)
                if st is None:
                    st = self.stemmer(tok)
                    self._stem_cache[tok] = st
                tok = st
            out.append(tok)
        return out

    def tokenize_many(self, texts: Sequence[str]) -> List[List[str]]:
        return [self.tokenize(t) for t in texts]


class BM25Retriever(Retriever):
    name = "bm25"

    def __init__(
        self,
        corpus: CorpusStore,
        index_dir: str | Path,
        k1: float = 1.5,
        b: float = 0.75,
        build: bool = False,
        n_threads: int = 8,
        chunk_size: int = 20_000,
    ) -> None:
        self.corpus = corpus
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.k1 = k1
        self.b = b
        self.n_threads = n_threads
        self.chunk_size = chunk_size
        self.tokenizer = BM25Tokenizer(stemmer=True)
        self._retriever = None
        self.vocab: Dict[str, int] = {}
        self._load_or_build(build)

    # ------------------------------------------------------------------ index
    def _load_or_build(self, build: bool) -> None:
        idx_path = self.index_dir / "bm25.pkl"
        if idx_path.exists() and not build:
            LOG.info("loading BM25 index from %s", idx_path)
            with open(idx_path, "rb") as fh:
                blob = pickle.load(fh)
            self._retriever = blob["retriever"]
            self.vocab = blob["vocab"]
            self.k1 = blob.get("k1", self.k1)
            self.b = blob.get("b", self.b)
            LOG.info("BM25 index loaded: %d docs, |V|=%d", len(self.corpus), len(self.vocab))
            return
        self._build(idx_path)

    def _build(self, idx_path: Path) -> None:
        import bm25s

        LOG.info("building BM25 index over %d passages (k1=%.2f b=%.2f)", len(self.corpus), self.k1, self.b)
        t0 = time.time()
        vocab: Dict[str, int] = {}
        doc_ids: List[np.ndarray] = []
        n_tokens = 0
        chunk: List[str] = []
        n_docs = len(self.corpus)

        def flush(buf: List[str]) -> None:
            nonlocal n_tokens
            toks = self.tokenizer.tokenize_many(buf)
            for t in toks:
                n_tokens += len(t)
                ids = np.fromiter(
                    (vocab.setdefault(w, len(vocab)) for w in t),
                    dtype=np.int32,
                    count=len(t),
                )
                doc_ids.append(ids)

        for i in range(n_docs):
            chunk.append(self.corpus.passage_text(i))
            if len(chunk) >= self.chunk_size:
                flush(chunk)
                chunk = []
                if len(doc_ids) % 200_000 == 0:
                    LOG.info("  tokenised %d/%d docs (%.1f min)", len(doc_ids), n_docs, (time.time() - t0) / 60)
        if chunk:
            flush(chunk)

        LOG.info("tokenised %d docs, %d tokens, |V|=%d in %.1f min", len(doc_ids), n_tokens, len(vocab), (time.time() - t0) / 60)
        t1 = time.time()
        retriever = bm25s.BM25(k1=self.k1, b=self.b, method="lucene")
        retriever.index((doc_ids, vocab), show_progress=False)
        LOG.info("BM25 sparse index built in %.1f min", (time.time() - t1) / 60)
        with open(idx_path, "wb") as fh:
            pickle.dump({"retriever": retriever, "vocab": vocab, "k1": self.k1, "b": self.b}, fh, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.index_dir / "bm25_meta.json", "w") as fh:
            json.dump(
                {
                    "n_docs": n_docs,
                    "k1": self.k1,
                    "b": self.b,
                    "backend": "bm25s (custom memory-bounded tokenisation)",
                    "vocab_size": len(vocab),
                    "n_tokens": n_tokens,
                },
                fh,
                indent=2,
            )
        self._retriever = retriever
        self.vocab = vocab

    # ------------------------------------------------------------------ search
    def document_frequencies(self) -> Tuple[Dict[str, int], int]:
        """(token -> document frequency, n_docs) taken from the sparse index."""
        scores = self._retriever.scores
        indptr = np.asarray(scores["indptr"])
        dfs = np.diff(indptr)
        vocab = self._retriever.vocab_dict
        out: Dict[str, int] = {}
        for token, col in vocab.items():
            if 0 <= col < len(dfs):
                out[token] = int(dfs[col])
        return out, len(self.corpus)

    def retrieve(self, queries: Sequence[str], k: int = 5) -> List[RetrievalResult]:
        t0 = time.perf_counter()
        q_tokens = self.tokenizer.tokenize_many(list(queries))
        ids, scores = self._retriever.retrieve(q_tokens, k=k, show_progress=False, n_threads=self.n_threads)
        dt = (time.perf_counter() - t0) * 1000.0 / max(1, len(queries))
        results = []
        for i in range(len(queries)):
            passages = []
            for rank in range(ids.shape[1]):
                j = int(ids[i, rank])
                if j < 0:
                    continue
                passages.append(
                    {
                        "pid": self.corpus.pid(j),
                        "title": self.corpus.title(j),
                        "text": self.corpus.text(j),
                        "score": float(scores[i, rank]),
                    }
                )
            results.append(RetrievalResult(passages=passages, latency_ms=dt, n_calls=1, extra={"ranker": "bm25"}))
        return results

    def stats(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "n_docs": len(self.corpus),
            "k1": self.k1,
            "b": self.b,
            "backend": "bm25s",
            "vocab_size": len(self.vocab),
        }
