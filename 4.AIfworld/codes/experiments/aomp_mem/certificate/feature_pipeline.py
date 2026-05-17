"""Feature extraction pipeline for the OLE reward model (plan-4 v0.2).

Defines the 12-dim feature vector :math:`\\phi(x_t, m_t)` used by
``LinearRewardModel`` (``reward_model.py``).  Features fall into 5 groups:

==========================  ===  ============================================
Group                        d   Source
==========================  ===  ============================================
Memory-state                 4   ``MemoryQualityMetrics`` (Coverage / Precision /
                                 Redundancy / Freshness)
Embedding-state              3   query embedding + retrieved entries + anchors
Memory-size                  2   memory store size + mean entry age
Burst-history                2   scheduler-tracked last burst step + window count
Bias                         1   constant 1.0
==========================  ===  ============================================

Z-score normalization is applied via ``FeatureStats`` fitted on training data
(see ``fit_stats`` / ``apply_stats``).  This keeps :math:`\\|\\phi\\|_2` bounded
so the projected SGLD step in ``sgld.py`` stays inside the parameter ball
:math:`\\Theta = \\{\\theta : \\|\\theta\\|_2 \\leq B\\}` for a reasonable B.

Special-case handling:
  - ``precision_t`` can be NaN when no retrievals occurred — replaced with 0.5
    (Beta(1,1) prior expectation).
  - empty ``memory_entries`` → memory-size and embedding-distance features
    default to 0.
  - no recorded bursts → burst-history features default to (0, 0).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from experiments.aomp_mem.core.memory import MemoryEntry
from experiments.aomp_mem.evaluation.memory_quality import MemoryQualityMetrics

FEATURE_NAMES: tuple[str, ...] = (
    # Memory-state (4)
    "coverage_t",
    "precision_t",
    "redundancy_t",
    "freshness_t",
    # Embedding-state (3)
    "retrieval_sim_mean",
    "retrieval_sim_max",
    "query_to_anchor_min_l2",
    # Memory-size (2)
    "log_memory_size",
    "mean_entry_age",
    # Burst-history (2)
    "time_since_last_burst",
    "burst_count_in_window",
    # Bias (1)
    "bias",
)
FEATURE_DIM: int = len(FEATURE_NAMES)
PRECISION_DEFAULT_WHEN_MISSING: float = 0.5  # Beta(1,1) prior expectation


@dataclass
class BurstHistoryState:
    """Light-weight burst tracker injected by the scheduler.

    ``last_burst_step`` is the trajectory step index of the most recent burst,
    or ``None`` if no burst has been triggered yet.  ``burst_count_in_window``
    is the count of bursts within the last ``window_size`` steps (the scheduler
    is responsible for updating it).
    """

    last_burst_step: Optional[int] = None
    burst_count_in_window: int = 0
    window_size: int = 50


@dataclass(frozen=True)
class FeatureStats:
    """Mean/std for z-score normalization (fitted on training feature matrix)."""

    mean: np.ndarray  # shape (FEATURE_DIM,)
    std: np.ndarray   # shape (FEATURE_DIM,)

    def __post_init__(self) -> None:
        if self.mean.shape != (FEATURE_DIM,):
            raise ValueError(f"mean shape mismatch: expected ({FEATURE_DIM},), got {self.mean.shape}")
        if self.std.shape != (FEATURE_DIM,):
            raise ValueError(f"std shape mismatch: expected ({FEATURE_DIM},), got {self.std.shape}")


def compute_features(
    metrics: MemoryQualityMetrics,
    *,
    query_embedding: Optional[np.ndarray],
    memory_entries: Sequence[MemoryEntry],
    retrieved: Sequence[MemoryEntry],
    anchors: np.ndarray,
    burst_state: BurstHistoryState,
    task_step: int,
) -> np.ndarray:
    """Build the raw (unnormalized) 12-dim feature vector :math:`\\phi(x_t, m_t)`.

    Caller normalizes via ``apply_stats`` before passing to the reward model.
    The bias dimension is included in the raw vector so normalization can
    later treat it specially (its mean/std are forced to (0.0, 1.0) so it
    passes through unchanged).
    """
    coverage = float(metrics.coverage)
    precision = (
        PRECISION_DEFAULT_WHEN_MISSING
        if metrics.no_retrieval_window_flag or not np.isfinite(metrics.precision)
        else float(metrics.precision)
    )
    redundancy = float(metrics.redundancy)
    freshness = float(metrics.freshness)

    sim_mean, sim_max = _retrieval_similarity_stats(query_embedding, retrieved)
    query_to_anchor = _query_to_anchor_min_l2(query_embedding, anchors)

    n_memory = len(memory_entries)
    log_memory_size = float(np.log1p(n_memory))
    mean_entry_age = _mean_entry_age(memory_entries, task_step=task_step)

    time_since_last_burst = (
        float(task_step - burst_state.last_burst_step)
        if burst_state.last_burst_step is not None
        else 0.0
    )
    burst_count_in_window = float(burst_state.burst_count_in_window)

    raw = np.asarray(
        [
            coverage,
            precision,
            redundancy,
            freshness,
            sim_mean,
            sim_max,
            query_to_anchor,
            log_memory_size,
            mean_entry_age,
            time_since_last_burst,
            burst_count_in_window,
            1.0,  # bias
        ],
        dtype=float,
    )
    if raw.shape != (FEATURE_DIM,):
        raise RuntimeError(f"feature vector shape mismatch: {raw.shape}")
    return raw


def fit_stats(feature_matrix: np.ndarray, *, eps: float = 1e-6) -> FeatureStats:
    """Estimate mean/std from a (n_samples, FEATURE_DIM) training matrix.

    The bias dimension's mean/std are forced to (0.0, 1.0) so applying the
    stats leaves bias unchanged.  ``eps`` floors std to avoid divide-by-zero
    on degenerate features.
    """
    if feature_matrix.ndim != 2 or feature_matrix.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"feature_matrix must have shape (n, {FEATURE_DIM}); got {feature_matrix.shape}"
        )
    mean = feature_matrix.mean(axis=0)
    std = feature_matrix.std(axis=0, ddof=0)
    std = np.maximum(std, eps)
    # Pass-through for the bias dimension.
    mean[-1] = 0.0
    std[-1] = 1.0
    return FeatureStats(mean=mean, std=std)


def apply_stats(features: np.ndarray, stats: FeatureStats) -> np.ndarray:
    """Apply z-score normalization (returns a new array; safe to call repeatedly)."""
    arr = np.asarray(features, dtype=float)
    if arr.ndim == 1:
        if arr.shape[0] != FEATURE_DIM:
            raise ValueError(f"expected ({FEATURE_DIM},), got {arr.shape}")
        return (arr - stats.mean) / stats.std
    if arr.ndim == 2:
        if arr.shape[1] != FEATURE_DIM:
            raise ValueError(f"expected (n, {FEATURE_DIM}), got {arr.shape}")
        return (arr - stats.mean[None, :]) / stats.std[None, :]
    raise ValueError(f"features must be 1-D or 2-D; got ndim={arr.ndim}")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _retrieval_similarity_stats(
    query_embedding: Optional[np.ndarray],
    retrieved: Sequence[MemoryEntry],
) -> tuple[float, float]:
    """Return (mean, max) cosine similarity between query and retrieved entries."""
    if query_embedding is None or not retrieved:
        return 0.0, 0.0
    query = np.asarray(query_embedding, dtype=float).reshape(-1)
    q_norm = float(np.linalg.norm(query))
    if q_norm == 0.0:
        return 0.0, 0.0
    sims = []
    for entry in retrieved:
        emb = np.asarray(entry.embedding, dtype=float).reshape(-1)
        if emb.shape != query.shape:
            continue
        e_norm = float(np.linalg.norm(emb))
        if e_norm == 0.0:
            continue
        sims.append(float(np.dot(query, emb) / (q_norm * e_norm)))
    if not sims:
        return 0.0, 0.0
    arr = np.asarray(sims, dtype=float)
    return float(arr.mean()), float(arr.max())


def _query_to_anchor_min_l2(
    query_embedding: Optional[np.ndarray],
    anchors: np.ndarray,
) -> float:
    """Minimum L2 distance from query to any anchor."""
    if query_embedding is None or anchors.size == 0:
        return 0.0
    query = np.asarray(query_embedding, dtype=float).reshape(-1)
    anchor_matrix = np.asarray(anchors, dtype=float)
    if anchor_matrix.ndim != 2 or anchor_matrix.shape[1] != query.shape[0]:
        return 0.0
    distances = np.linalg.norm(anchor_matrix - query[None, :], axis=1)
    return float(np.min(distances))


def _mean_entry_age(entries: Sequence[MemoryEntry], *, task_step: int) -> float:
    """Mean age of memory entries (task_step - entry.timestamp)."""
    if not entries:
        return 0.0
    ages = [max(0, int(task_step) - int(entry.timestamp)) for entry in entries]
    return float(np.mean(ages))


__all__ = [
    "BurstHistoryState",
    "FEATURE_DIM",
    "FEATURE_NAMES",
    "FeatureStats",
    "PRECISION_DEFAULT_WHEN_MISSING",
    "apply_stats",
    "compute_features",
    "fit_stats",
]
