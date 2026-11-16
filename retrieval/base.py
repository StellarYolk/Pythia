"""Retriever interface shared by all strategies."""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from common import Passage


@dataclass
class RetrievalResult:
    passages: List[Passage]
    latency_ms: float = 0.0
    n_calls: int = 1
    extra: Dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.extra is None:
            self.extra = {}


class Retriever(abc.ABC):
    """Every retriever maps a list of question strings to a list of passage lists."""

    name: str = "retriever"

    @abc.abstractmethod
    def retrieve(self, queries: Sequence[str], k: int = 5) -> List[RetrievalResult]:
        ...

    def stats(self) -> Dict[str, Any]:
        return {"name": self.name}
