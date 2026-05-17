"""CUSUM-FDR recalibration gate for MQC."""

from __future__ import annotations

import math
from typing import List

import numpy as np
from scipy.stats import norm

try:
    from statsmodels.stats.multitest import multipletests as _multipletests
except ImportError:  # pragma: no cover
    _multipletests = None


class CUSUMFDRGate:
    def __init__(self, q: float = 0.10, k_cusum: float = 0.5, window_size: int = 20) -> None:
        self.q = float(q)
        self.k_cusum = float(k_cusum)
        self.window_size = int(window_size)
        self._cusum_stat = 0.0
        self._p_history: List[float] = []
        self._adjusted_p_history: List[float] = []

    @property
    def cusum_stat(self) -> float:
        return self._cusum_stat

    @property
    def p_history(self) -> List[float]:
        return list(self._p_history)

    @property
    def adjusted_p_history(self) -> List[float]:
        return list(self._adjusted_p_history)

    def update(self, residual: float, sigma: float) -> bool:
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")

        standardized = float(residual / sigma)
        self._cusum_stat = max(0.0, self._cusum_stat + standardized - (self.k_cusum / 2.0))
        threshold = -math.log(max(self.q, 1e-12)) * 2.0
        raw_p = float(norm.sf(self._cusum_stat)) if self._cusum_stat > threshold else 1.0
        self._p_history.append(raw_p)
        self._p_history = self._p_history[-self.window_size :]
        self._adjusted_p_history = _bh_adjust(self._p_history)
        return any(p_value < self.q for p_value in self._adjusted_p_history)


def _bh_adjust(p_values: List[float]) -> List[float]:
    if not p_values:
        return []
    if _multipletests is not None:
        return [float(value) for value in _multipletests(p_values, alpha=0.10, method="fdr_bh")[1]]

    p_array = np.asarray(p_values, dtype=float)
    order = np.argsort(p_array)
    ranked = p_array[order]
    n = ranked.size
    adjusted = np.empty(n, dtype=float)
    running = 1.0
    for index in range(n - 1, -1, -1):
        rank = index + 1
        running = min(running, ranked[index] * n / rank)
        adjusted[index] = running
    restored = np.empty(n, dtype=float)
    restored[order] = adjusted
    return [float(value) for value in restored]
