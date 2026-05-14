"""OLE-style ensemble UCB certificate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from experiments.aomp_mem.certificate.feature_pipeline import FeatureStats, apply_stats
from experiments.aomp_mem.certificate.reward_model import LinearRewardModel
from experiments.aomp_mem.certificate.sgld import ProjectedSGLDEnsemble


@dataclass(frozen=True)
class OLEResult:
    mean: float
    var: float
    ucb: float
    triggered: bool
    n_particles: int


class OLECertificate:
    """Compute mean/variance/UCB from an SGLD reward-model ensemble."""

    def __init__(
        self,
        *,
        model: LinearRewardModel,
        sgld: ProjectedSGLDEnsemble,
        feature_stats: FeatureStats,
        kappa: float = 1.0,
        tau: float = 0.5,
        coverage_uncertainty_bonus: float = 1.0,
        coverage_uncertainty_mode: str = "multiplicative",
    ) -> None:
        if sgld.model is not model:
            raise ValueError("sgld.model must be the same LinearRewardModel instance")
        if kappa < 0:
            raise ValueError(f"kappa must be non-negative, got {kappa}")
        if coverage_uncertainty_bonus < 0:
            raise ValueError(
                f"coverage_uncertainty_bonus must be non-negative, got {coverage_uncertainty_bonus}"
            )
        if coverage_uncertainty_mode not in {"multiplicative", "additive"}:
            raise ValueError(
                "coverage_uncertainty_mode must be one of: multiplicative, additive"
            )
        self.model = model
        self.sgld = sgld
        self.feature_stats = feature_stats
        self.kappa = float(kappa)
        self.tau = float(tau)
        self.coverage_uncertainty_bonus = float(coverage_uncertainty_bonus)
        self.coverage_uncertainty_mode = str(coverage_uncertainty_mode)

    def compute(self, raw_features: np.ndarray) -> OLEResult:
        """Apply feature normalization and return ensemble UCB for one state."""
        normalized = apply_stats(np.asarray(raw_features, dtype=float), self.feature_stats)
        if normalized.ndim != 1:
            raise ValueError("compute expects a single raw feature vector")
        preds = np.asarray(self.model.predict(self.sgld.particles, normalized), dtype=float).reshape(-1)
        mean = float(preds.mean())
        var = self._calibrated_variance(
            float(preds.var(ddof=0)),
            np.asarray(raw_features, dtype=float),
        )
        ucb = float(mean + self.kappa * np.sqrt(max(0.0, var)))
        return OLEResult(
            mean=mean,
            var=var,
            ucb=ucb,
            triggered=ucb > self.tau,
            n_particles=int(preds.shape[0]),
        )

    def calibrate_threshold(self, holdout_features: np.ndarray, *, percentile: float = 95.0) -> None:
        """Set tau to a percentile of holdout UCB history."""
        features = np.atleast_2d(np.asarray(holdout_features, dtype=float))
        if features.shape[0] == 0:
            raise ValueError("holdout_features must be non-empty")
        ucbs = [self.compute(row).ucb for row in features]
        self.tau = float(np.percentile(np.asarray(ucbs, dtype=float), percentile))

    def empirical_coverage_rate(self, holdout_features: np.ndarray, holdout_labels: np.ndarray) -> float:
        """Return count{y_t in [mu-kappa*sqrt(var), mu+kappa*sqrt(var)]} / n."""
        features = np.atleast_2d(np.asarray(holdout_features, dtype=float))
        labels = np.asarray(holdout_labels, dtype=float).reshape(-1)
        if labels.shape[0] != features.shape[0]:
            raise ValueError(f"labels length {labels.shape[0]} must equal feature rows {features.shape[0]}")
        if labels.shape[0] == 0:
            raise ValueError("holdout data must be non-empty")

        covered = 0
        for row, label in zip(features, labels):
            result = self.compute(row)
            radius = self.kappa * np.sqrt(max(0.0, result.var))
            lower = result.mean - radius
            upper = result.mean + radius
            covered += int(lower <= float(label) <= upper)
        return float(covered / labels.shape[0])

    def save(self, path: str | Path) -> None:
        """Persist particles and certificate calibration fields."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        loss_curve = (
            self.sgld.trace.loss_curve
            if self.sgld.trace is not None
            else np.zeros((0,), dtype=float)
        )
        np.savez(
            output,
            particles=np.asarray(self.sgld.particles, dtype=float),
            feature_stats_mean=np.asarray(self.feature_stats.mean, dtype=float),
            feature_stats_std=np.asarray(self.feature_stats.std, dtype=float),
            kappa=float(self.kappa),
            tau=float(self.tau),
            coverage_uncertainty_bonus=float(self.coverage_uncertainty_bonus),
            coverage_uncertainty_mode=np.asarray(str(self.coverage_uncertainty_mode)),
            train_loss_curve=np.asarray(loss_curve, dtype=float),
            n_particles=int(self.sgld.n_particles),
            seed=int(self.sgld.seed),
        )

    @classmethod
    def load(cls, path: str | Path, model: LinearRewardModel) -> "OLECertificate":
        """Restore a certificate saved by ``save``."""
        with np.load(Path(path)) as data:
            particles = np.asarray(data["particles"], dtype=float)
            stats = FeatureStats(
                mean=np.asarray(data["feature_stats_mean"], dtype=float),
                std=np.asarray(data["feature_stats_std"], dtype=float),
            )
            sgld = ProjectedSGLDEnsemble(
                model,
                n_particles=int(particles.shape[0]),
                seed=int(np.asarray(data["seed"]).item()) if "seed" in data.files else 42,
            )
            sgld.particles = model.project(particles)
            return cls(
                model=model,
                sgld=sgld,
                feature_stats=stats,
                kappa=float(np.asarray(data["kappa"]).item()),
                tau=float(np.asarray(data["tau"]).item()),
                coverage_uncertainty_bonus=(
                    float(np.asarray(data["coverage_uncertainty_bonus"]).item())
                    if "coverage_uncertainty_bonus" in data.files
                    else 1.0
                ),
                coverage_uncertainty_mode=(
                    str(np.asarray(data["coverage_uncertainty_mode"]).item())
                    if "coverage_uncertainty_mode" in data.files
                    else "multiplicative"
                ),
            )

    def _calibrated_variance(self, base_var: float, raw_features: np.ndarray) -> float:
        """Boost epistemic variance in low-coverage states for AC-13."""
        coverage = float(raw_features.reshape(-1)[0])
        novelty = float(np.clip(1.0 - coverage, 0.0, 1.0))
        if self.coverage_uncertainty_mode == "additive":
            return float(max(0.0, base_var) + self.coverage_uncertainty_bonus * novelty)
        return float(max(0.0, base_var) * (1.0 + self.coverage_uncertainty_bonus * novelty))


__all__ = [
    "OLECertificate",
    "OLEResult",
]
