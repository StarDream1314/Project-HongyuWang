"""Scalar metrics used in the paper's experiment tables.

The experiments write per-iteration trajectories (CSV) and then aggregate
multi-seed summaries for the LaTeX tables.

This module contains small, deterministic helpers that are shared across the
real-data drift experiments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .simplex import moving_average


@dataclass(frozen=True)
class RecoveryConfig:
    """Definition of the time-to-recovery metric.

    Attributes:
        drift_iter: iteration at which the proxy judge begins drifting.
        post_window: window (in iterations) after drift within which we locate
            the post-drift minimum.
        pre_window: number of iterations before drift used to define the
            pre-drift reference reward.
        smooth_window: moving-average window used to compute the metric (to
            reduce sensitivity to iteration-level noise).
        tol: additive tolerance for the recovery threshold.
    """

    drift_iter: int
    post_window: int
    pre_window: int = 50
    smooth_window: int = 25
    tol: float = 0.0


def time_to_recovery_after_drift(
    reward: np.ndarray,
    *,
    cfg: RecoveryConfig,
) -> int:
    """Compute time-to-recovery after drift.

    We use the following operational definition (aligned with the paper's
    drift-stress tests):

      1) Smooth the reward trajectory with a centered moving average.
      2) Define a pre-drift reference level r_pre as the mean reward in the
         last `pre_window` iterations strictly before drift.
      3) Find the post-drift minimum within the first `post_window` iterations
         after drift; call its time index t_min.
      4) Define the recovery time as the first iteration t >= t_min such that
         the smoothed reward is >= r_pre - tol. Return (t - drift_iter).

    If the trajectory never recovers by the end of the run, we return the
    censored value (T - drift_iter + 1) where T is the trajectory length.

    Args:
        reward: 1D array of per-iteration rewards of length T.
        cfg: RecoveryConfig controlling the drift time, windows, and smoothing.

    Returns:
        Integer recovery time in iterations after drift (0 means immediate
        recovery; larger values indicate slower recovery; the maximum value
        indicates "did not recover within horizon").
    """
    r = np.asarray(reward, dtype=float)
    if r.ndim != 1:
        raise ValueError("reward must be a 1D array")
    T = int(r.shape[0])
    drift = int(cfg.drift_iter)
    if drift < 1 or drift > T:
        raise ValueError(f"drift_iter must be in [1, T], got {drift} with T={T}")

    rs = moving_average(r, int(cfg.smooth_window))

    # Iterations are 1-indexed in the CSVs; here we work in 0-indexed arrays.
    # pre range: t in [drift-pre_window, drift-1] in 1-indexed => indices
    # [drift-pre_window-1, drift-2] in 0-indexed.
    pre_start = max(1, drift - int(cfg.pre_window))
    pre_end = drift - 1
    if pre_end < pre_start:
        # If drift is extremely early, fall back to using all pre-drift points.
        pre_start = 1
        pre_end = max(1, drift - 1)
    pre_slice = slice(pre_start - 1, pre_end)
    r_pre = float(np.mean(rs[pre_slice]))
    thresh = r_pre - float(cfg.tol)

    # Post-drift window for locating the minimum.
    post_start = drift
    post_end = min(T, drift + int(cfg.post_window) - 1)
    post_slice = slice(post_start - 1, post_end)
    post_rs = rs[post_slice]
    if post_rs.size == 0:
        return int(T - drift + 1)
    t_min_offset = int(np.argmin(post_rs))
    # Convert back to 1-indexed iteration.
    t_min = post_start + t_min_offset

    # First time after the minimum that we re-attain the pre-drift level.
    after_min = rs[t_min - 1 :]
    hit = np.where(after_min >= thresh)[0]
    if hit.size == 0:
        return int(T - drift + 1)
    t_rec = t_min + int(hit[0])
    return int(t_rec - drift)
