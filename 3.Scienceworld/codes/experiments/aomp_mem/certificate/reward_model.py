"""Linear reward model for the OLE certificate (plan-4 v0.2).

The model predicts a bounded reward

    r_theta(phi) = sigmoid(theta^T phi),  theta in {||theta||_2 <= B}

Labels may be hard success labels in {0, 1} or soft progress/reward labels in
[0, 1].  Supporting soft labels is important for ScienceWorld runs where the
binary success signal can saturate while progress still carries information.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LOSS_NORMALIZER: float = float(np.log(1.0 + np.exp(1.0)))


def _stable_sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable elementwise logistic sigmoid."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[neg])
    out[neg] = exp_z / (1.0 + exp_z)
    return out


def _stable_log1p_exp(z: np.ndarray) -> np.ndarray:
    """Numerically stable elementwise ``log(1 + exp(z))``."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z > 0
    neg = ~pos
    out[pos] = z[pos] + np.log1p(np.exp(-z[pos]))
    out[neg] = np.log1p(np.exp(z[neg]))
    return out


@dataclass(frozen=True)
class LinearRewardModelConfig:
    """Hyperparameters fixed at construction time."""

    dim: int
    radius: float = 10.0
    normalize_loss: bool = True


class LinearRewardModel:
    """Stateless logistic reward model with bounded parameters."""

    def __init__(self, dim: int, *, radius: float = 10.0, normalize_loss: bool = True) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if radius <= 0:
            raise ValueError(f"radius must be positive, got {radius}")
        self.config = LinearRewardModelConfig(
            dim=int(dim),
            radius=float(radius),
            normalize_loss=bool(normalize_loss),
        )

    @property
    def dim(self) -> int:
        return self.config.dim

    @property
    def radius(self) -> float:
        return self.config.radius

    def predict(self, theta: np.ndarray, features: np.ndarray) -> np.ndarray:
        """Predict ``sigmoid(theta^T features)`` with particle/sample batching."""
        theta_arr = np.asarray(theta, dtype=float)
        features_arr = np.asarray(features, dtype=float)
        self._validate_dim(theta_arr, name="theta")
        self._validate_dim(features_arr, name="features")
        if theta_arr.ndim == 1 and features_arr.ndim == 1:
            return _stable_sigmoid(np.asarray(theta_arr @ features_arr))
        if theta_arr.ndim == 1 and features_arr.ndim == 2:
            return _stable_sigmoid(features_arr @ theta_arr)
        if theta_arr.ndim == 2 and features_arr.ndim == 1:
            return _stable_sigmoid(theta_arr @ features_arr)
        if theta_arr.ndim == 2 and features_arr.ndim == 2:
            return _stable_sigmoid(theta_arr @ features_arr.T)
        raise ValueError("predict supports 1-D or 2-D theta/features only")

    def loss(self, theta: np.ndarray, features: np.ndarray, labels: np.ndarray) -> float:
        """Mean logistic cross-entropy for labels in ``[0, 1]``."""
        theta_arr, features_arr, labels_arr = self._prepare_batch(theta, features, labels)
        margins = features_arr @ theta_arr
        probs = _stable_sigmoid(margins)
        per_sample = -(
            labels_arr * np.log(probs + 1e-12)
            + (1.0 - labels_arr) * np.log(1.0 - probs + 1e-12)
        )
        if self.config.normalize_loss:
            per_sample = per_sample / LOSS_NORMALIZER
        return float(per_sample.mean())

    def gradient(self, theta: np.ndarray, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Gradient of the mean logistic cross-entropy w.r.t. ``theta``."""
        theta_arr, features_arr, labels_arr = self._prepare_batch(theta, features, labels)
        probs = _stable_sigmoid(features_arr @ theta_arr)
        grad = ((probs - labels_arr)[:, None] * features_arr).mean(axis=0)
        if self.config.normalize_loss:
            grad = grad / LOSS_NORMALIZER
        return grad.astype(float, copy=False)

    def project(self, theta: np.ndarray) -> np.ndarray:
        """Euclidean projection onto the L2 ball ``||theta||_2 <= radius``."""
        arr = np.asarray(theta, dtype=float)
        self._validate_dim(arr, name="theta")
        radius = self.config.radius
        if arr.ndim == 1:
            norm = float(np.linalg.norm(arr))
            return arr * (radius / norm) if norm > radius else arr.astype(float, copy=False)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        scale = np.where(norms > radius, radius / np.maximum(norms, 1e-12), 1.0)
        return arr * scale

    def initialize(self, n_particles: int, *, seed: int = 0) -> np.ndarray:
        """Sample Gaussian initial particles and project them into the L2 ball."""
        rng = np.random.default_rng(int(seed))
        particles = rng.standard_normal(size=(int(n_particles), self.config.dim))
        return self.project(particles)

    def _validate_dim(self, arr: np.ndarray, *, name: str) -> None:
        if arr.ndim < 1 or arr.ndim > 2:
            raise ValueError(f"{name} must be 1-D or 2-D; got ndim={arr.ndim}")
        if arr.shape[-1] != self.config.dim:
            raise ValueError(
                f"{name} last dimension must equal {self.config.dim}; got {arr.shape[-1]}"
            )

    def _prepare_batch(
        self,
        theta: np.ndarray,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        theta_arr = np.asarray(theta, dtype=float).reshape(-1)
        features_arr = np.atleast_2d(np.asarray(features, dtype=float))
        labels_arr = np.asarray(labels, dtype=float).reshape(-1)
        if theta_arr.shape[0] != self.config.dim:
            raise ValueError(f"theta must have shape ({self.config.dim},); got {theta_arr.shape}")
        if features_arr.shape[1] != self.config.dim:
            raise ValueError(
                f"features must have shape (B, {self.config.dim}); got {features_arr.shape}"
            )
        if labels_arr.shape[0] != features_arr.shape[0]:
            raise ValueError(
                f"labels length {labels_arr.shape[0]} must equal batch size {features_arr.shape[0]}"
            )
        if np.any(labels_arr < 0.0) or np.any(labels_arr > 1.0):
            raise ValueError("labels must lie in [0, 1]")
        return theta_arr, features_arr, labels_arr


__all__ = [
    "LOSS_NORMALIZER",
    "LinearRewardModel",
    "LinearRewardModelConfig",
    "_stable_log1p_exp",
]
