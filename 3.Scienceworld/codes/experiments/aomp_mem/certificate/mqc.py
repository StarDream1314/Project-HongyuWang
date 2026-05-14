"""MQC certificate calibration and inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import numpy as np
from scipy.optimize import nnls

from experiments.aomp_mem.evaluation.memory_quality import MemoryQualityMetrics


def compute_R_eps(n: int, delta: float = 0.05) -> float:
    """Finite-sample tail term used by MQC."""
    if n <= 0:
        return 0.0
    return float(np.sqrt(np.log(1.0 / delta) / (2.0 * n)))


@dataclass(frozen=True)
class MQCResult:
    gap_upper: float
    mode: str
    alpha: float
    beta: float
    gamma: float
    R_eps: float
    triggered: bool
    metrics: MemoryQualityMetrics


class MQCCertificate:
    """Calibrates and evaluates the Memory-Quality-driven Certificate."""

    def __init__(
        self,
        tau: float,
        mode: str = "auto",
        *,
        reward_upper_bound: float = 1.0,
        delta: float = 0.05,
        bootstrap_samples: int = 200,
        stability_threshold: float = 2.0,
    ) -> None:
        self.tau = float(tau)
        self.mode = mode
        self.reward_upper_bound = float(reward_upper_bound)
        self.delta = float(delta)
        self.bootstrap_samples = int(bootstrap_samples)
        self.stability_threshold = float(stability_threshold)

        self.alpha = self.reward_upper_bound
        self.beta = self.reward_upper_bound
        self.gamma = self.reward_upper_bound
        self.r_eps = 0.0
        self.calibrated_mode = "theorem_constants"
        self.point_estimate = np.full(3, self.reward_upper_bound, dtype=float)
        self.ci = np.tile(self.point_estimate[:, None], (1, 2))
        self.stability_ratio = np.zeros(3, dtype=float)
        self.holdout_r2_per_seed: Dict[str, float] = {}
        self.mean_holdout_r2 = 0.0
        self.recalibrate_flag = False

    def calibrate(
        self,
        regret_obs: np.ndarray,
        quality_obs: np.ndarray,
        *,
        seed_ids: Sequence[object] | None = None,
    ) -> None:
        target = np.asarray(regret_obs, dtype=float).reshape(-1)
        features = np.asarray(quality_obs, dtype=float)
        if features.ndim != 2 or features.shape[0] != target.shape[0]:
            raise ValueError("quality_obs must have shape (n_samples, n_features) aligned with regret_obs")
        if features.shape[1] < 3:
            raise ValueError("quality_obs must expose at least 3 MQC features")
        features = features[:, :3]
        if target.size < 3:
            raise ValueError("at least 3 calibration samples are required")

        if seed_ids is None:
            seed_ids = [0] * target.size
        if len(seed_ids) != target.size:
            raise ValueError("seed_ids must align with regret_obs")

        self.point_estimate = _fit_nnls(features, target)
        self.ci = _bootstrap_ci(features, target, bootstrap_samples=self.bootstrap_samples)
        ci_width = self.ci[:, 1] - self.ci[:, 0]
        denom = np.maximum(np.abs(self.point_estimate), 1e-8)
        self.stability_ratio = ci_width / denom
        self.holdout_r2_per_seed = _leave_one_seed_out_r2(features, target, seed_ids)
        self.mean_holdout_r2 = float(np.mean(list(self.holdout_r2_per_seed.values())))
        self.r_eps = compute_R_eps(target.size, delta=self.delta)
        self.calibrated_mode = self._select_mode()
        active = self._active_coefficients()
        self.alpha, self.beta, self.gamma = map(float, active)
        self.recalibrate_flag = False

    def compute(self, metrics: MemoryQualityMetrics, budget_remaining: float) -> MQCResult:
        del budget_remaining  # Reserved for future budget-coupled thresholds.
        active = self._active_coefficients()
        gap_upper = float(np.dot(active, metrics.feature_vector()) + self.r_eps)
        return MQCResult(
            gap_upper=gap_upper,
            mode=self.calibrated_mode,
            alpha=float(active[0]),
            beta=float(active[1]),
            gamma=float(active[2]),
            R_eps=self.r_eps,
            triggered=gap_upper > self.tau,
            metrics=metrics,
        )

    def _select_mode(self) -> str:
        if self.mode != "auto":
            return self.mode
        if np.linalg.norm(self.point_estimate, ord=1) <= 1e-12:
            return "theorem_constants"
        if self.mean_holdout_r2 >= 0.60 and np.all(self.stability_ratio <= self.stability_threshold):
            return "empirical"
        if self.mean_holdout_r2 >= 0.60:
            return "semi_empirical"
        return "theorem_constants"

    def _active_coefficients(self) -> np.ndarray:
        if self.calibrated_mode == "empirical":
            return self.point_estimate
        return np.full(3, self.reward_upper_bound, dtype=float)


def _fit_nnls(features: np.ndarray, target: np.ndarray) -> np.ndarray:
    coefficients, _ = nnls(features, target)
    return coefficients.astype(float, copy=False)


def _bootstrap_ci(features: np.ndarray, target: np.ndarray, *, bootstrap_samples: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    samples = []
    n_samples = target.shape[0]
    for _ in range(max(bootstrap_samples, 20)):
        indices = rng.integers(0, n_samples, size=n_samples)
        samples.append(_fit_nnls(features[indices], target[indices]))
    stacked = np.vstack(samples)
    lower = np.percentile(stacked, 2.5, axis=0)
    upper = np.percentile(stacked, 97.5, axis=0)
    return np.vstack([lower, upper]).T


def _leave_one_seed_out_r2(features: np.ndarray, target: np.ndarray, seed_ids: Sequence[object]) -> Dict[str, float]:
    seed_array = np.asarray(list(seed_ids), dtype=object)
    unique_ids = list(dict.fromkeys(seed_array.tolist()))
    scores: Dict[str, float] = {}
    for seed_id in unique_ids:
        holdout_mask = seed_array == seed_id
        train_mask = ~holdout_mask
        if not np.any(holdout_mask):
            continue
        if not np.any(train_mask):
            prediction = np.repeat(np.mean(target[holdout_mask]), np.count_nonzero(holdout_mask))
        else:
            coefficients = _fit_nnls(features[train_mask], target[train_mask])
            prediction = features[holdout_mask] @ coefficients
        scores[str(seed_id)] = _r2_score(target[holdout_mask], prediction)
    return scores


def _r2_score(target: np.ndarray, prediction: Iterable[float]) -> float:
    observed = np.asarray(target, dtype=float)
    predicted = np.asarray(list(prediction), dtype=float)
    if observed.size == 0:
        return 0.0
    ss_res = float(np.sum((observed - predicted) ** 2))
    ss_tot = float(np.sum((observed - np.mean(observed)) ** 2))
    if ss_tot <= 1e-12:
        return 1.0 if ss_res <= 1e-12 else 0.0
    return float(1.0 - (ss_res / ss_tot))
