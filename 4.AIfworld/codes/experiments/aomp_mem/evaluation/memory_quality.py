"""Memory-quality metrics used by MQC and evidence-chain analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from experiments.aomp_mem.core.memory import MemoryEntry
from experiments.aomp_mem.core.retrieval import cosine_similarity


@dataclass(frozen=True)
class MemoryQualityMetrics:
    coverage: float
    precision: float
    redundancy: float
    freshness: float
    no_retrieval_window_flag: bool

    def feature_vector(self) -> np.ndarray:
        return np.asarray(
            [
                1.0 - self.coverage,
                self.redundancy,
                1.0 - self.freshness,
            ],
            dtype=float,
        )


def compute_memory_quality(
    entries: Sequence[MemoryEntry],
    retrieved: Sequence[MemoryEntry],
    anchors: np.ndarray,
    task_step: int,
    lambda_fresh: float = 0.20,
    coverage_radius: float | None = None,
    redundancy_threshold: float = 0.85,
) -> MemoryQualityMetrics:
    """Return Coverage / Precision / Redundancy / Freshness for one memory state."""
    del task_step  # Reserved for future windowed metrics.

    anchor_matrix = np.asarray(anchors, dtype=float)
    if anchor_matrix.ndim != 2:
        raise ValueError("anchors must have shape (n_anchors, dim)")

    expected_dim = anchor_matrix.shape[1] if anchor_matrix.size else None
    entry_embeddings = _embedding_matrix(entries, expected_dim=expected_dim)
    effective_coverage_radius = _resolve_coverage_radius(
        anchor_matrix,
        entry_embeddings,
        coverage_radius=coverage_radius,
    )
    coverage = _compute_coverage(
        entry_embeddings,
        anchor_matrix,
        coverage_radius=effective_coverage_radius,
    )
    precision, no_retrieval_window_flag = _compute_precision(retrieved)
    redundancy = _compute_redundancy(entry_embeddings, threshold=redundancy_threshold)
    freshness = _compute_freshness(entries, lambda_fresh=lambda_fresh)
    return MemoryQualityMetrics(
        coverage=coverage,
        precision=precision,
        redundancy=redundancy,
        freshness=freshness,
        no_retrieval_window_flag=no_retrieval_window_flag,
    )


def _embedding_matrix(entries: Sequence[MemoryEntry], expected_dim: int | None) -> np.ndarray:
    if not entries:
        if expected_dim is None:
            return np.zeros((0, 0), dtype=float)
        return np.zeros((0, expected_dim), dtype=float)
    matrix = np.vstack([np.asarray(entry.embedding, dtype=float) for entry in entries])
    if expected_dim is not None and matrix.shape[1] != expected_dim:
        raise ValueError(f"embedding dim mismatch: expected {expected_dim}, got {matrix.shape[1]}")
    return matrix


def _compute_coverage(entry_embeddings: np.ndarray, anchors: np.ndarray, *, coverage_radius: float) -> float:
    if anchors.size == 0 or entry_embeddings.size == 0:
        return 0.0
    distances = np.linalg.norm(anchors[:, None, :] - entry_embeddings[None, :, :], axis=2)
    covered = np.min(distances, axis=1) <= coverage_radius
    return float(np.mean(covered.astype(float)))


def _resolve_coverage_radius(
    anchors: np.ndarray,
    entry_embeddings: np.ndarray,
    *,
    coverage_radius: float | None,
) -> float:
    if coverage_radius is not None:
        return float(coverage_radius)
    if anchors.ndim != 2 or anchors.size == 0:
        return 0.0

    dim = int(anchors.shape[1])
    inferred = np.sqrt(float(dim)) / 4.0
    if _looks_unit_normalized(anchors) and (entry_embeddings.size == 0 or _looks_unit_normalized(entry_embeddings)):
        # HashingEmbedder emits unit-norm vectors, so unconstrained sqrt(d)/4 can exceed the
        # realizable L2 geometry and collapse coverage to a constant.
        inferred = min(inferred, np.sqrt(2.0) - 0.1)
    return float(inferred)


def _looks_unit_normalized(matrix: np.ndarray, *, atol: float = 1e-3) -> bool:
    if matrix.ndim != 2 or matrix.size == 0:
        return False
    norms = np.linalg.norm(matrix, axis=1)
    return bool(np.all(np.isfinite(norms)) and np.allclose(norms, 1.0, atol=atol))


def _compute_precision(retrieved: Sequence[MemoryEntry]) -> tuple[float, bool]:
    if not retrieved:
        return float("nan"), True
    return float(np.mean([_entry_precision_signal(entry) for entry in retrieved])), False


def _entry_precision_signal(entry: MemoryEntry) -> float:
    explicit = entry.metadata.get("precision_signal")
    if explicit is not None:
        return float(np.clip(float(explicit), 0.0, 1.0))
    hit_count = entry.metadata.get("hit_count")
    if hit_count is not None:
        count = max(0, int(hit_count or 0))
        return float(count / (count + 1.0)) if count > 0 else float(entry.quality_score)
    return float(entry.quality_score)


def _compute_redundancy(entry_embeddings: np.ndarray, *, threshold: float) -> float:
    count = entry_embeddings.shape[0]
    if count < 2:
        return 0.0
    redundant_pairs = 0
    total_pairs = 0
    for left in range(count):
        for right in range(left + 1, count):
            total_pairs += 1
            if cosine_similarity(entry_embeddings[left], entry_embeddings[right]) >= threshold:
                redundant_pairs += 1
    return float(redundant_pairs / total_pairs) if total_pairs else 0.0


def _compute_freshness(entries: Sequence[MemoryEntry], *, lambda_fresh: float) -> float:
    if not entries:
        return 0.0
    max_timestamp = max(int(entry.timestamp) for entry in entries)
    decayed = [np.exp(-lambda_fresh * max(0, max_timestamp - int(entry.timestamp))) for entry in entries]
    return float(np.mean(decayed))
