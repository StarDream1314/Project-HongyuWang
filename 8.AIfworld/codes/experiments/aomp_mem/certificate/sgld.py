"""Projected SGLD ensemble for the Full OLE certificate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from experiments.aomp_mem.certificate.reward_model import LinearRewardModel


@dataclass(frozen=True)
class SGLDTrainingTrace:
    """Diagnostics recorded during SGLD training."""

    loss_curve: np.ndarray
    dispersion_curve: np.ndarray


class ProjectedSGLDEnsemble:
    """Projected SGLD sampler over linear reward-model parameters.

    The implementation keeps all particles inside the model's L2 ball via
    ``LinearRewardModel.project`` after every Langevin step.  The prior is a
    zero-mean isotropic Gaussian with scale ``sigma_0``.
    """

    def __init__(
        self,
        model: LinearRewardModel,
        *,
        n_particles: int = 20,
        eta_0: float = 0.1,
        beta: float = 1.0,
        sigma_0: float = 1.0,
        seed: int = 42,
    ) -> None:
        if n_particles <= 0:
            raise ValueError(f"n_particles must be positive, got {n_particles}")
        if eta_0 <= 0:
            raise ValueError(f"eta_0 must be positive, got {eta_0}")
        if beta < 0:
            raise ValueError(f"beta must be non-negative, got {beta}")
        if sigma_0 <= 0:
            raise ValueError(f"sigma_0 must be positive, got {sigma_0}")

        self.model = model
        self.n_particles = int(n_particles)
        self.eta_0 = float(eta_0)
        self.beta = float(beta)
        self.sigma_0 = float(sigma_0)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.particles = self.model.initialize(self.n_particles, seed=self.seed)
        self.trace: Optional[SGLDTrainingTrace] = None

    def learning_rate(self, t: int) -> float:
        """Return eta_t = eta_0 / t with a safe lower bound on t."""
        return float(self.eta_0 / max(1, int(t)))

    def step(
        self,
        particles: np.ndarray,
        t: int,
        batch_features: np.ndarray,
        batch_labels: np.ndarray,
    ) -> np.ndarray:
        """Run one projected SGLD update for every particle."""
        current = np.asarray(particles, dtype=float)
        if current.shape != (self.n_particles, self.model.dim):
            raise ValueError(
                f"particles must have shape ({self.n_particles}, {self.model.dim}); got {current.shape}"
            )
        features = np.atleast_2d(np.asarray(batch_features, dtype=float))
        labels = np.asarray(batch_labels, dtype=float).reshape(-1)
        if features.ndim != 2 or features.shape[1] != self.model.dim:
            raise ValueError(f"batch_features must have shape (n, {self.model.dim}); got {features.shape}")
        if labels.shape[0] != features.shape[0]:
            raise ValueError(f"batch_labels length {labels.shape[0]} must equal batch size {features.shape[0]}")

        eta_t = self.learning_rate(t)
        noise_scale = float(np.sqrt(2.0 * self.beta * eta_t)) if self.beta > 0.0 else 0.0
        updated = np.empty_like(current)
        prior_scale = self.beta / (self.sigma_0 ** 2)

        for index, theta in enumerate(current):
            data_grad = self.model.gradient(theta, features, labels)
            prior_grad = prior_scale * theta
            noise = self.rng.standard_normal(self.model.dim) if noise_scale > 0.0 else np.zeros(self.model.dim)
            updated[index] = theta - eta_t * (data_grad + prior_grad) + noise_scale * noise

        return self.model.project(updated)

    def train(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        n_iters: int = 1000,
        batch_size: int = 32,
    ) -> SGLDTrainingTrace:
        """Train particles on an offline feature/label set."""
        feature_matrix = np.atleast_2d(np.asarray(features, dtype=float))
        label_vector = np.asarray(labels, dtype=float).reshape(-1)
        if feature_matrix.ndim != 2 or feature_matrix.shape[1] != self.model.dim:
            raise ValueError(f"features must have shape (n, {self.model.dim}); got {feature_matrix.shape}")
        if label_vector.shape[0] != feature_matrix.shape[0]:
            raise ValueError(f"labels length {label_vector.shape[0]} must equal feature rows {feature_matrix.shape[0]}")
        if n_iters <= 0:
            raise ValueError(f"n_iters must be positive, got {n_iters}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        n_samples = feature_matrix.shape[0]
        actual_batch_size = min(int(batch_size), n_samples)
        loss_curve = np.zeros(int(n_iters), dtype=float)
        dispersion_curve = np.zeros(int(n_iters), dtype=float)

        for iteration in range(1, int(n_iters) + 1):
            indices = self.rng.choice(n_samples, size=actual_batch_size, replace=n_samples < actual_batch_size)
            self.particles = self.step(
                self.particles,
                iteration,
                feature_matrix[indices],
                label_vector[indices],
            )
            loss_curve[iteration - 1] = self.mean_loss(feature_matrix, label_vector)
            dispersion_curve[iteration - 1] = self.particle_dispersion()

        self.trace = SGLDTrainingTrace(loss_curve=loss_curve, dispersion_curve=dispersion_curve)
        return self.trace

    def mean_loss(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Mean logistic loss averaged over particles."""
        losses = [self.model.loss(theta, features, labels) for theta in self.particles]
        return float(np.mean(losses))

    def particle_dispersion(self) -> float:
        """Average particle distance from the ensemble mean."""
        center = self.particles.mean(axis=0)
        distances = np.linalg.norm(self.particles - center[None, :], axis=1)
        return float(np.mean(distances))


__all__ = [
    "ProjectedSGLDEnsemble",
    "SGLDTrainingTrace",
]
