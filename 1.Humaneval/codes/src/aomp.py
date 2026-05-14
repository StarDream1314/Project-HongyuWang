"""Audited Optimistic Mirror-Prox (A-OMP) in KL geometry on the simplex.

This implementation is intentionally minimal and matches the algorithmic interface used in the paper:
- hint is the previous lookahead direction g_{t-1}
- lookahead operator estimate g_t is provided by a user-supplied callable
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np

from .simplex import kl_prox


def aomp_step_simplex(
    z: np.ndarray,
    g_prev: np.ndarray,
    eta: float,
    F_est: Callable[[np.ndarray], np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One A-OMP step on the simplex.

    Args:
        z: current iterate in simplex (strictly positive)
        g_prev: previous lookahead direction (hint)
        eta: step size
        F_est: callable that returns an operator estimate at a point (lookahead)

    Returns:
        z_next: next iterate
        w: hint-point (prox step using g_prev)
        g: new lookahead direction
    """
    w = kl_prox(z, eta, g_prev)
    g = np.asarray(F_est(w), dtype=float)
    z_next = kl_prox(z, eta, g)
    return z_next, w, g
