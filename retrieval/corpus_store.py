"""Memory-mapped store for the retrieval corpus.

Keeping ~1M passages as Python objects costs several GB of RSS, which on this machine
risks starving the AMD iGPU that drives the display.  Instead we store the corpus in two
flat binary files and memory-map them:

    corpus_ids.npy    int64  (n,)          passage ids
    corpus_blob.bin   bytes                concatenated "title\\ntext" payloads
    corpus_offsets.npy int64 (n+1,)        offsets into the blob

Only the byte ranges actually touched are faulted into the page cache, so a search over
the corpus costs ~0 resident memory.
"""
from __future__ import annotations

import json
import mmap
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from common import get_logger

LOG = get_logger("retrieval.corpus")


class CorpusStore:
    def __init__(self, corpus_path: str | Path, index_dir: Optional[str | Path] = None, build: bool = False) -> None:
        self.corpus_path = Path(corpus_path)
        self.index_dir = Path(index_dir) if index_dir is not None else self.corpus_path.parent
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.ids: np.ndarray = np.zeros(0, dtype=np.int64)
        self.offsets: np.ndarray = np.zeros(1, dtype=np.int64)
        self._blob = None
        self._mm = None
        self._load_or_build(build)

    # ------------------------------------------------------------------
    def _paths(self):
        return (
            self.index_dir / "corpus_ids.npy",
            self.index_dir / "corpus_offsets.npy",
            self.index_dir / "corpus_blob.bin",
        )

    def _load_or_build(self, build: bool) -> None:
        ids_p, off_p, blob_p = self._paths()
        if ids_p.exists() and off_p.exists() and blob_p.exists() and not build:
            LOG.info("memory-mapping corpus from %s", blob_p)
            self.ids = np.load(ids_p, mmap_mode="r")
            self.offsets = np.load(off_p, mmap_mode="r")
            self._open_blob(blob_p)
            LOG.info("corpus store ready: %d passages, %.2f GB blob", len(self.ids), self.blob_size / 1e9)
            return
        self._build(blob_p)

    def _open_blob(self, blob_p: Path) -> None:
        self._blob = open(blob_p, "rb")
        self._mm = mmap.mmap(self._blob.fileno(), 0, access=mmap.ACCESS_READ)
        self.blob_size = self._mm.size()

    def _build(self, blob_p: Path) -> None:
        LOG.info("building memory-mapped corpus store from %s", self.corpus_path)
        t0 = time.time()
        ids: List[int] = []
        offsets: List[int] = [0]
        pos = 0
        with open(blob_p, "wb") as blob:
            with open(self.corpus_path) as fh:
                for line in fh:
                    r = json.loads(line)
                    payload = (r["title"] + "\n" + r["text"]).encode("utf-8")
                    blob.write(payload)
                    pos += len(payload)
                    ids.append(r["pid"])
                    offsets.append(pos)
            blob.flush()
        np.save(self._paths()[0], np.asarray(ids, dtype=np.int64))
        np.save(self._paths()[1], np.asarray(offsets, dtype=np.int64))
        LOG.info("corpus store built in %.1f s (%d passages, %.2f GB)", time.time() - t0, len(ids), pos / 1e9)
        self.ids = np.load(self._paths()[0], mmap_mode="r")
        self.offsets = np.load(self._paths()[1], mmap_mode="r")
        self._open_blob(blob_p)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return int(self.ids.shape[0])

    def title(self, i: int) -> str:
        start = int(self.offsets[i])
        end = int(self.offsets[i + 1])
        raw = self._mm[start:end].decode("utf-8", errors="ignore")
        return raw.split("\n", 1)[0]

    def text(self, i: int) -> str:
        start = int(self.offsets[i])
        end = int(self.offsets[i + 1])
        raw = self._mm[start:end].decode("utf-8", errors="ignore")
        return raw.split("\n", 1)[1] if "\n" in raw else ""

    def passage_text(self, i: int) -> str:
        start = int(self.offsets[i])
        end = int(self.offsets[i + 1])
        return self._mm[start:end].decode("utf-8", errors="ignore")

    def pid(self, i: int) -> int:
        return int(self.ids[i])

    def passages(self, indices: List[int]) -> List[Tuple[int, str, str]]:
        return [(self.pid(i), self.title(i), self.text(i)) for i in indices]

    def titles(self) -> List[str]:
        """All titles (used for gold-title -> passage lookup)."""
        return [self.title(i) for i in range(len(self))]
