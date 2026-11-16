"""Dense retrieval with a FAISS index over bge-base-en-v1.5 embeddings."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from common import get_logger
from retrieval.base import RetrievalResult, Retriever
from retrieval.corpus_store import CorpusStore

LOG = get_logger("retrieval.dense")

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class DenseRetriever(Retriever):
    name = "dense"

    def __init__(
        self,
        corpus: CorpusStore,
        index_dir: str | Path,
        model_name: str = "BAAI/bge-base-en-v1.5",
        batch_size: int = 256,
        max_len: int = 256,
        device: str = "cuda",
        build: bool = False,
        use_fp16_index: bool = True,
    ) -> None:
        self.corpus = corpus
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_len = max_len
        self.device = device
        self.use_fp16_index = use_fp16_index
        self._index = None
        self._model = None
        self._tokenizer = None
        self._load_or_build(build)

    # ------------------------------------------------------------------ index
    def _load_or_build(self, build: bool) -> None:
        import faiss

        emb_path = self.index_dir / "dense_emb.f16.npy"
        faiss_path = self.index_dir / "dense.index"
        meta_path = self.index_dir / "dense_meta.json"

        if emb_path.exists() and faiss_path.exists() and meta_path.exists() and not build:
            LOG.info("loading dense index from %s", faiss_path)
            self._index = faiss.read_index(str(faiss_path))
            LOG.info("dense index loaded: %d vectors", self._index.ntotal)
            return

        LOG.info("building dense index over %d passages", len(self.corpus))
        emb = np.load(emb_path, mmap_mode="r") if emb_path.exists() else None
        if emb is None:
            emb = self._encode_corpus()
            np.save(emb_path, emb)
        n_docs, dim = emb.shape
        if self.use_fp16_index:
            # exact search with fp16 storage -> half the RAM of IndexFlatIP
            index = faiss.IndexScalarQuantizer(dim, faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_INNER_PRODUCT)
        else:
            index = faiss.IndexFlatIP(dim)
        index.train(np.ascontiguousarray(np.asarray(emb[: min(50_000, n_docs)], dtype=np.float32)))
        # add in chunks so we never materialise a full float32 copy of the corpus
        step = 100_000
        for i in range(0, n_docs, step):
            chunk = np.ascontiguousarray(np.asarray(emb[i : i + step], dtype=np.float32))
            index.add(chunk)
            del chunk
        faiss.write_index(index, str(faiss_path))
        self._index = index
        with open(meta_path, "w") as fh:
            json.dump(
                {
                    "model": self.model_name,
                    "n_docs": int(index.ntotal),
                    "dim": dim,
                    "index_type": type(index).__name__,
                    "fp16_storage": bool(self.use_fp16_index),
                    "query_prefix": QUERY_PREFIX,
                },
                fh,
                indent=2,
            )
        LOG.info("dense index built: %d vectors (%s)", index.ntotal, type(index).__name__)

    # ------------------------------------------------------------------ encoding
    def _get_model(self):
        if self._model is None:
            import torch
            from transformers import AutoModel, AutoTokenizer

            LOG.info("loading encoder %s on %s (fp16)", self.model_name, self.device)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            # fp16 weights are ~3.4x faster than fp32/TF32 on this GPU and are the
            # standard way this encoder is used; TF32 is enabled for any fp32 op.
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self._model = AutoModel.from_pretrained(self.model_name, torch_dtype=dtype).to(self.device).eval()
        return self._model, self._tokenizer

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        import torch

        model, tok = self._get_model()
        if is_query:
            texts = [QUERY_PREFIX + t for t in texts]
        out = []
        dtype = next(model.parameters()).dtype
        with torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                chunk = list(texts[i : i + self.batch_size])
                enc = tok(chunk, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt").to(self.device)
                enc = {k: (v.to(dtype) if v.dtype.is_floating_point else v) for k, v in enc.items()}
                hidden = model(**enc).last_hidden_state[:, 0]  # BGE uses CLS pooling
                hidden = torch.nn.functional.normalize(hidden.float(), dim=-1)
                out.append(hidden.float().cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)

    def _encode_corpus(self) -> np.ndarray:
        LOG.info("encoding %d passages with %s", len(self.corpus), self.model_name)
        t0 = time.time()
        chunks = []
        step = 20_000
        n = len(self.corpus)
        for i in range(0, n, step):
            chunk = [self.corpus.passage_text(j) for j in range(i, min(i + step, n))]
            embs = self.encode(chunk, is_query=False)
            chunks.append(embs.astype(np.float16))
            done = min(i + step, n)
            rate = done / max(1e-6, time.time() - t0)
            LOG.info(
                "  encoded %d/%d (%.0f p/s, eta %.1f min)",
                done,
                n,
                rate,
                (n - done) / max(1e-6, rate) / 60,
            )
        return np.concatenate(chunks, axis=0)

    # ------------------------------------------------------------------ search
    def retrieve(self, queries: Sequence[str], k: int = 5) -> List[RetrievalResult]:
        t_enc = time.perf_counter()
        q = self.encode(queries, is_query=True)
        enc_ms = (time.perf_counter() - t_enc) * 1000.0 / max(1, len(queries))
        t0 = time.perf_counter()
        scores, ids = self._index.search(q, k)
        search_ms = (time.perf_counter() - t0) * 1000.0 / max(1, len(queries))
        dt = enc_ms + search_ms
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
            results.append(RetrievalResult(passages=passages, latency_ms=dt, n_calls=1, extra={"ranker": "dense"}))
        return results

    def stats(self) -> Dict[str, Any]:
        n = self._index.ntotal if self._index is not None else 0
        return {
            "name": self.name,
            "model": self.model_name,
            "n_docs": n,
            "index": type(self._index).__name__ if self._index is not None else "none",
        }
