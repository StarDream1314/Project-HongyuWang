"""Pluggable embedder abstraction for plan-3.5.

Provides a ``runtime_checkable`` ``EmbedderProtocol`` that lets us swap the
legacy ``HashingEmbedder`` (locked under AC-2 inside ``retrieval.py``) for a
real semantic embedder without touching any locked file.  The protocol is
satisfied structurally by the legacy class — duck typing keeps the SHA256
invariant intact.

Design rules respected here:
- Output vectors are unit-normalized (L2 norm == 1) so the existing
  ``_looks_unit_normalized`` branch in ``evaluation/memory_quality.py``
  continues to apply the correct coverage radius.
- ``encode`` accepts ``str``, ``encode_many`` accepts ``Iterable[str]`` and
  returns a 2-D array; both shapes match the legacy embedder.
- The optional ``SentenceTransformerEmbedder`` lazily imports
  ``sentence_transformers`` so unit tests that do not need a real model
  can run without the heavy dependency installed.
"""

from __future__ import annotations

import os
from typing import Iterable, Optional, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class EmbedderProtocol(Protocol):
    """Structural type compatible with the legacy ``HashingEmbedder``.

    Implementations must:
    - expose an ``int`` ``dim`` attribute,
    - return a 1-D ``np.ndarray`` of length ``dim`` from ``encode(text)``,
    - return a 2-D ``np.ndarray`` of shape ``(len(texts), dim)`` from
      ``encode_many(texts)``,
    - produce unit-normalized vectors (``||v||_2 == 1``) so coverage radius
      defaults remain valid.
    """

    dim: int

    def encode(self, text: str) -> np.ndarray:  # pragma: no cover - protocol
        ...

    def encode_many(self, texts: Iterable[str]) -> np.ndarray:  # pragma: no cover
        ...


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization with zero-norm guard."""
    if matrix.ndim == 1:
        norm = float(np.linalg.norm(matrix))
        if norm == 0.0:
            return matrix
        return matrix / norm
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe = np.where(norms > 0, norms, 1.0)
    return matrix / safe


class SentenceTransformerEmbedder:
    """Real semantic embedder backed by ``sentence-transformers``.

    Defaults to ``all-MiniLM-L6-v2`` (dim=384, ~90MB local model).  Vectors
    are unit-normalized to match ``HashingEmbedder`` semantics so downstream
    coverage / cosine logic stays unchanged.
    """

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self,
        model_name: Optional[str] = None,
        *,
        cache_folder: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        # Lazy import keeps ``import experiments.aomp_mem.core.embedding`` cheap
        # and makes the rest of the package usable when sentence-transformers
        # is not installed (e.g. CI lanes that only exercise legacy code).
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised only when dep missing
            raise ImportError(
                "SentenceTransformerEmbedder requires `sentence-transformers`; "
                "install via `pip install sentence-transformers>=2.5.0`."
            ) from exc

        self.model_name = str(model_name or self.DEFAULT_MODEL)
        resolved_cache = cache_folder or os.environ.get("HF_HOME")
        self._model = SentenceTransformer(
            self.model_name,
            cache_folder=resolved_cache,
            device=device,
        )
        get_dim = getattr(self._model, "get_embedding_dimension", self._model.get_sentence_embedding_dimension)
        self.dim = int(get_dim())
        self._encode_cache: dict[str, np.ndarray] = {}

    def encode(self, text: str) -> np.ndarray:
        key = str(text)
        cached = self._encode_cache.get(key)
        if cached is not None:
            return cached.copy()
        vec = self._model.encode(
            key,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        arr = np.asarray(vec, dtype=float).reshape(-1)
        # ``normalize_embeddings=True`` already normalizes, but we re-apply to
        # defend against subtle device/precision drift across backends.
        normalized = _l2_normalize(arr)
        self._encode_cache[key] = normalized.copy()
        return normalized

    def encode_many(self, texts: Iterable[str]) -> np.ndarray:
        text_list = [str(t) for t in texts]
        if not text_list:
            return np.zeros((0, self.dim), dtype=float)
        matrix = self._model.encode(
            text_list,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        arr = np.asarray(matrix, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return _l2_normalize(arr)


__all__ = [
    "EmbedderProtocol",
    "SentenceTransformerEmbedder",
]
