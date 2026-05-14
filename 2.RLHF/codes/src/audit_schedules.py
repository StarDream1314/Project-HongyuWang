"""Audit scheduling policies for the two-channel (monitoring + calibration) interface.

This module contains reusable audit schedulers that decide how many *calibration*
audits to spend each outer-loop iteration.

The experimental code in this package keeps *monitoring* audits fixed per round
(`n_mon`) and varies only the *calibration* audits (`n_cal`).

We implement a small, self-contained Exp3-based controller that treats each
candidate calibration audit rate as an "arm" and uses a certificate-driven loss
signal to trade off:

  - proxy drift/stationarity (via a monitoring-vs-lookahead discrepancy), and
  - audit cost.

The controller is designed to match the audit-scheduling theorem in the paper:
an adversarial bandit oracle inequality of the form

  min_a { 2*eta*xi^2(a) + rho*c(a) } + O(sqrt(T*K log K)).

In the experiments we do not require the full theorem assumptions; we simply
use the monitoring discrepancy as a bounded bandit loss proxy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


def exp3_default_gamma(K: int, T: int) -> float:
    """Default exploration parameter for Exp3.

    We use the classical choice (Auer et al., 2002):
        gamma = min(1, sqrt(K log K / ((e-1) T))).
    """
    if K <= 1:
        return 0.0
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    return float(min(1.0, np.sqrt((K * np.log(K)) / ((np.e - 1.0) * T))))


@dataclass
class Exp3AuditConfig:
    """Configuration for the Exp3 audit scheduler.

    Args:
        arms: List of possible calibration audit counts per iteration.
        rho: Relative weight on audit cost vs. discrepancy (rho=0 ignores cost).
        gamma: Exp3 exploration parameter. If None, use exp3_default_gamma.
        xi_bound: Upper bound used to clip the discrepancy proxy before
            normalization to [0,1]. Must be > 0.
    """

    arms: Sequence[int]
    rho: float = 0.25
    gamma: float | None = None
    xi_bound: float = 100.0


class Exp3AuditScheduler:
    """Exp3 controller over a finite set of calibration-audit rates."""

    def __init__(self, *, T: int, cfg: Exp3AuditConfig):
        if T <= 0:
            raise ValueError(f"T must be positive, got {T}")
        self.T = int(T)
        self.arms: List[int] = [int(a) for a in cfg.arms]
        if len(self.arms) == 0:
            raise ValueError("Exp3AuditScheduler requires at least one arm")
        if any(a < 0 for a in self.arms):
            raise ValueError(f"Calibration audit counts must be nonnegative, got {self.arms}")
        self.K = len(self.arms)
        self.rho = float(cfg.rho)
        if self.rho < 0:
            raise ValueError(f"rho must be nonnegative, got {self.rho}")
        self.gamma = float(cfg.gamma) if cfg.gamma is not None else exp3_default_gamma(self.K, self.T)
        if not (0.0 <= self.gamma <= 1.0):
            raise ValueError(f"gamma must be in [0,1], got {self.gamma}")
        self.xi_bound = float(cfg.xi_bound)
        if self.xi_bound <= 0:
            raise ValueError(f"xi_bound must be > 0, got {self.xi_bound}")

        # Exp3 weights.
        self._w = np.ones(self.K, dtype=float)

        # Precompute maximum calibration cost for normalization.
        self._max_cost = float(max(self.arms)) if max(self.arms) > 0 else 1.0

    def probs(self) -> np.ndarray:
        """Current arm sampling distribution p_t."""
        w = self._w
        wsum = float(np.sum(w))
        if wsum <= 0 or not np.isfinite(wsum):
            # Reset in pathological cases.
            w = np.ones(self.K, dtype=float)
            self._w = w
            wsum = float(self.K)
        p = (1.0 - self.gamma) * (w / wsum) + (self.gamma / self.K)
        # Numerical safety.
        p = np.clip(p, 1e-12, 1.0)
        p = p / float(np.sum(p))
        return p

    def sample(self, rng: np.random.Generator) -> Tuple[int, int, float]:
        """Sample an arm index, returning (arm_idx, n_cal, p_arm)."""
        p = self.probs()
        arm_idx = int(rng.choice(self.K, p=p))
        return arm_idx, self.arms[arm_idx], float(p[arm_idx])

    def loss_from_proxy(self, *, xi_sq: float, n_cal: int) -> float:
        """Construct a bounded loss in [0,1] from (xi_sq, n_cal).

        Args:
            xi_sq: nonnegative discrepancy proxy (e.g., ||g - F_mon||_*^2).
            n_cal: chosen calibration audits.
        """
        xi_sq = float(max(0.0, xi_sq))
        # Clip discrepancy proxy and normalize.
        xi_scaled = min(xi_sq, self.xi_bound) / self.xi_bound
        # Normalize cost to [0,1].
        cost_scaled = float(max(0.0, n_cal)) / self._max_cost
        # Convex combination determined by rho.
        if self.rho == 0.0:
            loss = xi_scaled
        else:
            loss = (xi_scaled + self.rho * cost_scaled) / (1.0 + self.rho)
        return float(np.clip(loss, 0.0, 1.0))

    def update(self, *, arm_idx: int, p_arm: float, loss: float) -> None:
        """Exp3 weight update from a single bandit loss observation.

        Args:
            arm_idx: chosen arm.
            p_arm: probability with which it was chosen.
            loss: observed loss in [0,1].
        """
        arm_idx = int(arm_idx)
        if arm_idx < 0 or arm_idx >= self.K:
            raise ValueError(f"arm_idx out of range: {arm_idx}")
        p_arm = float(p_arm)
        if p_arm <= 0 or not np.isfinite(p_arm):
            raise ValueError(f"p_arm must be positive finite, got {p_arm}")
        loss = float(loss)
        if not (0.0 <= loss <= 1.0) or not np.isfinite(loss):
            raise ValueError(f"loss must be in [0,1] and finite, got {loss}")

        # Importance-weighted loss estimate.
        loss_hat = loss / p_arm
        # Exp3 multiplicative-weights update.
        eta_hat = self.gamma / float(self.K) if self.K > 0 else 0.0
        self._w[arm_idx] *= float(np.exp(-eta_hat * loss_hat))

        # Numerical guard: renormalize if weights become extreme.
        if not np.isfinite(self._w).all() or float(np.max(self._w)) > 1e100:
            self._w = self._w / float(np.max(self._w) + 1e-12)
