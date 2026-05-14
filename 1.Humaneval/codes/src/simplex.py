"""Simplex/KL-geometry utilities.

This module intentionally keeps dependencies minimal and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


def kl_prox(z: np.ndarray, eta: float, g: np.ndarray, eps: float = 1e-300) -> np.ndarray:
    """KL prox on the simplex: prox_z(eta g) = argmin_{y in Delta} <g, y> + (1/eta) KL(y||z).

    Closed form: y ∝ z ⊙ exp(-eta g).

    Args:
        z: point in simplex (positive, sums to 1)
        eta: step size (>0)
        g: gradient/operator vector
        eps: clamp to avoid log/underflow issues

    Returns:
        y: point in simplex
    """
    z = np.asarray(z, dtype=float)
    g = np.asarray(g, dtype=float)
    if eta <= 0:
        raise ValueError(f"eta must be positive, got {eta}")
    if np.any(z <= 0):
        raise ValueError("kl_prox requires strictly positive simplex point z.")
    if not np.isfinite(z).all() or not np.isfinite(g).all():
        raise ValueError("Non-finite input to kl_prox.")

    log_new = np.log(z) - eta * g
    log_new -= np.max(log_new)  # stabilize
    y = np.exp(log_new)
    y = np.clip(y, eps, None)
    y = y / np.sum(y)
    return y


def mirror_grad_kl(x: np.ndarray, eps: float = 1e-300) -> np.ndarray:
    """Gradient of negative entropy mirror map ψ(x)=Σ x log x on the simplex.

    ∇ψ(x)=log(x)+1 (elementwise).
    """
    x = np.asarray(x, dtype=float)
    x = np.clip(x, eps, None)
    return np.log(x) + 1.0


def F_from_reward(x: np.ndarray, r: np.ndarray, tau: float, x_ref: np.ndarray) -> np.ndarray:
    """Operator for the KL-regularized linear objective: f(x) = -<r,x> + tau KL(x||x_ref)."""
    x = np.asarray(x, dtype=float)
    r = np.asarray(r, dtype=float)
    x_ref = np.asarray(x_ref, dtype=float)
    return -r + tau * (np.log(x / x_ref) + 1.0)


def gapS_simplex(x: np.ndarray, F: np.ndarray) -> float:
    """Stampacchia gap on the simplex for a given operator value F(x).

    Gap_S(x) = max_{y in Δ} <F(x), x - y> = <F(x),x> - min_i F_i(x).
    """
    x = np.asarray(x, dtype=float)
    F = np.asarray(F, dtype=float)
    return float(np.dot(F, x) - np.min(F))


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average with edge padding."""
    x = np.asarray(x, dtype=float)
    if window <= 1:
        return x.copy()
    pad = window // 2
    xpad = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(xpad, kernel, mode="valid")
