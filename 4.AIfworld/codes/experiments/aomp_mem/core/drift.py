"""Drift detection helpers for A-OMP-Mem."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def _window_mean(values: Sequence[bool], window: int) -> float:
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values[-window:], dtype=float)))


def detect_performance_drift(
    *,
    success_history: Sequence[bool],
    warmup: int,
    window: int,
    drift_threshold: float,
) -> tuple[bool, float, Optional[float]]:
    if warmup <= 0:
        raise ValueError(f"warmup must be positive, got {warmup}")
    if len(success_history) < warmup:
        baseline = float(np.mean(np.asarray(success_history, dtype=float))) if success_history else 0.0
        return False, baseline, None

    baseline = float(np.mean(np.asarray(success_history[:warmup], dtype=float)))
    current = _window_mean(success_history, window)
    detected = current < baseline - drift_threshold
    return detected, baseline, current


def detect_similarity_warning(
    *,
    query_embedding: np.ndarray,
    history_embeddings: Sequence[np.ndarray],
    similarity_threshold: float,
) -> bool:
    if not history_embeddings:
        return False
    center = np.mean(np.vstack(history_embeddings), axis=0)
    center_norm = np.linalg.norm(center)
    query_norm = np.linalg.norm(query_embedding)
    if center_norm == 0 or query_norm == 0:
        return False
    cosine_distance = 1.0 - float(np.dot(query_embedding, center) / (query_norm * center_norm))
    return cosine_distance > similarity_threshold
