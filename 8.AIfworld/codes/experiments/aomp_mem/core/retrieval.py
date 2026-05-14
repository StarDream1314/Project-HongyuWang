"""Cheap retrieval primitives for A-OMP-Mem."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

from experiments.aomp_mem.core.memory import MemoryEntry


class HashingEmbedder:
    """Small deterministic embedder for tests and CPU-only development."""

    def __init__(self, *, dim: int = 64) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = int(dim)

    def encode(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=float)
        for token in text.lower().split():
            vector[hash(token) % self.dim] += 1.0
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector

    def encode_many(self, texts: Iterable[str]) -> np.ndarray:
        return np.vstack([self.encode(text) for text in texts])


@dataclass(frozen=True)
class RetrievalResult:
    entry: MemoryEntry
    score: float


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def retrieve_top_k(
    *,
    query: str,
    entries: Sequence[MemoryEntry],
    embedder: HashingEmbedder,
    k: int = 3,
) -> List[RetrievalResult]:
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if not entries:
        return []

    query_embedding = embedder.encode(query)
    scored = [
        RetrievalResult(entry=entry, score=cosine_similarity(query_embedding, entry.embedding))
        for entry in entries
    ]
    scored.sort(key=lambda item: item.score, reverse=True)
    return scored[:k]
