"""Plotting + aggregation utilities for the reproducibility package.

We keep this module small and deterministic:
  * a single Matplotlib rcParams style to ensure consistent paper aesthetics
  * simple t-based confidence intervals for multi-seed plots / tables
  * an optional bootstrap CI for a lightweight robustness check

The experiments use 5 seeds by default; for n=5 we report mean ± 95% t-CI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import logging

import numpy as np
from scipy.stats import t as student_t

# Reduce noisy, non-actionable logs from optional font-subsetting backends
# (observed in some containerized environments).
logging.getLogger("fontTools").setLevel(logging.ERROR)


def savefig_atomic(path: Path, *, fig=None, **kwargs) -> None:
    """Save a Matplotlib figure atomically.

    Motivation: if an environment is interrupted while writing a PDF/PNG, the
    target path can be left as a zero-byte file, which later causes LaTeX
    compilation failures. We therefore write to a temporary file and then
    replace the target.
    """

    # Lazy import to keep this utility lightweight for non-plotting code paths.
    import matplotlib.pyplot as plt  # type: ignore

    fig = fig if fig is not None else plt.gcf()
    path = Path(path)
    # Keep the original extension so Matplotlib infers the correct writer.
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    fig.savefig(tmp, **kwargs)
    tmp.replace(path)


def set_matplotlib_style() -> Dict[str, object]:
    """Return an ICML-friendly Matplotlib rcParams dict.

    Design goals:
      * readable in two-column PDFs
      * robust across Matplotlib versions
      * vector-friendly (embedded fonts)
      * unobtrusive grid + clean spines

    We intentionally avoid hard-coding a custom color palette so figures remain
    legible in both color and grayscale.
    """
    return {
        # Rendering
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,

        # Typography
        "font.size": 10,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,

        # Lines
        "lines.linewidth": 2.0,
        "lines.markersize": 5,

        # Axes aesthetics
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,

        # Legends
        "legend.frameon": False,

        # Embed TrueType fonts in vector outputs
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }


@dataclass(frozen=True)
class MeanCI:
    mean: np.ndarray
    lo: np.ndarray
    hi: np.ndarray


def t_confidence_interval(samples: np.ndarray, *, alpha: float = 0.05) -> MeanCI:
    """Compute mean ± t-CI along axis=0.

    Args:
        samples: array of shape (n, ...) with n seeds/repetitions.
        alpha: significance level (0.05 => 95% CI).

    Returns:
        MeanCI(mean, lo, hi) with same trailing shape as samples[0].
    """
    x = np.asarray(samples, dtype=float)
    if x.ndim == 0:
        raise ValueError("samples must have at least 1 dimension")
    n = int(x.shape[0])
    mean = np.mean(x, axis=0)
    if n <= 1:
        return MeanCI(mean=mean, lo=mean.copy(), hi=mean.copy())
    # ddof=1 for unbiased sample std
    sd = np.std(x, axis=0, ddof=1)
    se = sd / np.sqrt(float(n))
    tcrit = float(student_t.ppf(1.0 - alpha / 2.0, df=n - 1))
    half = tcrit * se
    return MeanCI(mean=mean, lo=mean - half, hi=mean + half)


def align_and_stack(series_by_seed: Iterable[np.ndarray]) -> np.ndarray:
    """Stack 1D series arrays after validating they share length."""
    xs = [np.asarray(s, dtype=float) for s in series_by_seed]
    if len(xs) == 0:
        raise ValueError("need at least one series")
    T = int(xs[0].shape[0])
    if any(int(x.shape[0]) != T for x in xs):
        raise ValueError("all series must have same length")
    return np.stack(xs, axis=0)


def mean_ci_scalar(values: Iterable[float], *, alpha: float = 0.05) -> Tuple[float, float]:
    """Mean and (two-sided) t-CI half-width for scalar values."""
    v = np.asarray(list(values), dtype=float)
    if v.size == 0:
        raise ValueError("values must be non-empty")
    if v.size == 1:
        return float(v[0]), 0.0
    m = float(np.mean(v))
    sd = float(np.std(v, ddof=1))
    se = sd / np.sqrt(float(v.size))
    tcrit = float(student_t.ppf(1.0 - alpha / 2.0, df=int(v.size) - 1))
    return m, float(tcrit * se)


def bootstrap_ci_scalar(
    values: Iterable[float],
    *,
    alpha: float = 0.05,
    n_boot: int = 2000,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Nonparametric bootstrap CI for the mean (percentile interval).

    This is intentionally simple and deterministic: we resample the *seed-level*
    scalar values with replacement and compute a percentile interval for the
    bootstrap distribution of the mean.

    Args:
        values: iterable of per-seed scalar values.
        alpha: significance level (0.05 => 95% CI).
        n_boot: number of bootstrap resamples.
        seed: RNG seed for determinism.

    Returns:
        (mean, lo, hi), where lo/hi are the two-sided percentile bounds.
    """
    v = np.asarray(list(values), dtype=float)
    if v.size == 0:
        raise ValueError("values must be non-empty")
    mean = float(np.mean(v))
    if v.size == 1:
        return mean, mean, mean

    rng = np.random.default_rng(int(seed))
    # Indices shape: (n_boot, n)
    idx = rng.integers(low=0, high=int(v.size), size=(int(n_boot), int(v.size)))
    boot_means = np.mean(v[idx], axis=1)
    lo = float(np.quantile(boot_means, alpha / 2.0))
    hi = float(np.quantile(boot_means, 1.0 - alpha / 2.0))
    return mean, lo, hi


def format_mean_ci(mean: float, half_width: float, *, digits: int = 4) -> str:
    """Pretty-print mean ± CI for LaTeX tables."""
    fmt = f"{{:.{digits}f}}"
    return f"{fmt.format(mean)}$\\pm${fmt.format(half_width)}"


def format_ci_interval(lo: float, hi: float, *, digits: int = 4) -> str:
    """Pretty-print a two-sided CI as a LaTeX-friendly interval."""
    fmt = f"{{:.{digits}f}}"
    return f"[{fmt.format(lo)},\,{fmt.format(hi)}]"


def median_iqr_scalar(values: Iterable[float]) -> Tuple[float, float, float, float]:
    """Return (median, q1, q3, iqr) for scalar values."""
    v = np.asarray(list(values), dtype=float)
    if v.size == 0:
        raise ValueError("values must be non-empty")
    q1 = float(np.quantile(v, 0.25))
    med = float(np.quantile(v, 0.50))
    q3 = float(np.quantile(v, 0.75))
    return med, q1, q3, float(q3 - q1)


def format_median_iqr(med: float, q1: float, q3: float, *, digits: int = 0) -> str:
    """Pretty-print median with IQR as 'med[q1,q3]' for LaTeX tables."""
    fmt = f"{{:.{digits}f}}"
    return f"{fmt.format(med)}[{fmt.format(q1)},{fmt.format(q3)}]"
