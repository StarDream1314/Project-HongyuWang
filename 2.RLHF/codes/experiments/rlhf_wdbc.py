"""Real dataset RLHF-style drift case study (WDBC).

This script reproduces the figures:
- figs/rlhf_real_calibration.pdf
- figs/rlhf_real_reward.pdf
- figs/rlhf_real_audit_tradeoff.pdf

and stores raw trajectories in:
- results/rlhf_wdbc/

The experiment is designed to match Section 7.1 of the paper.
"""

from __future__ import annotations

import warnings
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
from numpy.random import SeedSequence
import sklearn

# Pandas may emit a benign RuntimeWarning when converting NaNs during CSV export
# (depends on NumPy/Pandas versions). Suppress to keep artifact logs clean.
warnings.filterwarnings("ignore", message="invalid value encountered in cast", category=RuntimeWarning)

# Ensure a non-interactive backend for reproducibility in headless environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from src.aomp import aomp_step_simplex
from src.audit_schedules import Exp3AuditConfig, Exp3AuditScheduler
from src.metrics import RecoveryConfig, time_to_recovery_after_drift
from src.plotting import (
    MeanCI,
    align_and_stack,
    bootstrap_ci_scalar,
    format_ci_interval,
    format_median_iqr,
    format_mean_ci,
    median_iqr_scalar,
    mean_ci_scalar,
    savefig_atomic,
    set_matplotlib_style,
    t_confidence_interval,
)
from src.simplex import F_from_reward, gapS_simplex, kl_prox, moving_average


def _write_config(out_dir: Path, cfg: Config) -> None:
    """Write a lightweight provenance record for artifact reproducibility."""
    payload = {
        "config": asdict(cfg),
        "versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    (out_dir / "config.json").write_text(json.dumps(payload, indent=2, sort_keys=True))


@dataclass(frozen=True)
class Config:
    # Dataset / checkpoints
    seed_models: int = 0
    m: int = 20  # number of checkpoints (policies)

    # Outer loop
    T: int = 1500
    eta: float = 0.3
    tau: float = 0.005

    # Proxy judge drift
    drift_iter: int = 750
    beta0: float = 0.0
    beta1: float = 0.8
    sigma_judge: float = 0.06

    # Auditing
    n_mon: int = 20  # monitoring audits per iteration (held-out stream)
    delta: float = 0.05

    # Schedules
    n_cal_low: int = 5
    n_cal_high: int = 200
    warmup: int = 200
    burst_len: int = 120
    threshold_scale: float = 2.5

    # Plotting
    smooth_window: int = 25
    post_drift_window: int = 200

    # Exp3 audit controller (additional schedule)
    # Arms are {0, n_cal_low, n_cal_high} by default.
    exp3_rho: float = 0.25
    exp3_xi_bound: float = 100.0

    # Multi-seed evaluation (stochasticity in audits + judge noise)
    # We run 10 seeds to stabilize heavy-tailed metrics (e.g., time-to-recover)
    # and reduce uncertainty for adaptive schedules.
    seeds: Tuple[int, ...] = tuple(range(10))


def build_checkpoints(cfg: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train m logistic-regression checkpoints and return:
    - correct matrix of shape (m, n_test) with {0,1} correctness
    - r_true in [0,1]^m test accuracies
    - u spurious feature (standardized) in R^m

    Notes:
      We intentionally construct a spread in checkpoint quality by varying:
        - regularization C
        - training-set size
        - max_iter
    """
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # Fixed split (60/20/20)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.4, random_state=0, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=1, stratify=y_tmp
    )

    rng = np.random.default_rng(cfg.seed_models)

    models = []
    # 10 full-data models
    for C in np.logspace(-3, 3, 10):
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=float(C), solver="liblinear", max_iter=200, random_state=0),
        )
        clf.fit(X_train, y_train)
        models.append(clf)

    # 5 half-data models
    for C in np.logspace(-2, 2, 5):
        idx = rng.choice(len(X_train), size=int(0.5 * len(X_train)), replace=False)
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=float(C), solver="liblinear", max_iter=200, random_state=1),
        )
        clf.fit(X_train[idx], y_train[idx])
        models.append(clf)

    # 5 small-data, lower-iter models
    for C in np.logspace(-2, 2, 5):
        idx = rng.choice(len(X_train), size=int(0.2 * len(X_train)), replace=False)
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=float(C), solver="liblinear", max_iter=50, random_state=2),
        )
        clf.fit(X_train[idx], y_train[idx])
        models.append(clf)

    if len(models) != cfg.m:
        raise RuntimeError(f"Expected m={cfg.m} models, got {len(models)}")

    # Precompute correctness matrix on test set
    preds = np.vstack([m.predict(X_test) for m in models])  # (m, n_test)
    correct = (preds == y_test).astype(float)  # {0,1}

    r_true = correct.mean(axis=1)

    # Spurious feature u: coefficient L1 norm (standardized)
    u_raw = np.array(
        [np.linalg.norm(m.named_steps["logisticregression"].coef_.ravel(), ord=1) for m in models],
        dtype=float,
    )
    u = (u_raw - u_raw.mean()) / (u_raw.std() + 1e-12)

    return correct, r_true, u


def sample_audits(rng: np.random.Generator, correct: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Sample n audited labels: choose policy i ~ Unif[m], datapoint j ~ Unif[n_test]."""
    m, n_test = correct.shape
    if n <= 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=float)
    pol = rng.integers(low=0, high=m, size=n, endpoint=False)
    idx = rng.integers(low=0, high=n_test, size=n, endpoint=False)
    y = correct[pol, idx].astype(float)
    return pol, y


def run_schedule(
    name: str,
    cfg: Config,
    correct: np.ndarray,
    r_true: np.ndarray,
    u: np.ndarray,
    rng: np.random.Generator,
    adaptive: bool,
    n_cal_fixed: int = 0,
    scheduler: Exp3AuditScheduler | None = None,
) -> pd.DataFrame:
    """Run one schedule and return a trajectory DataFrame."""
    # rng is passed in so multi-seed runs can use independent, reproducible streams.

    m, n_test = correct.shape
    x_ref = np.ones(m, dtype=float) / m
    z = x_ref.copy()
    g_prev = np.zeros(m, dtype=float)

    # certificate diameter for simplex in l1
    R = 2.0

    # parameters for Freedman/Azuma-style slack in Theorem 5 (uniform importance sampling)
    b = float(m)              # bounded increment per coordinate
    sigma2 = float(m * m) / 4.0  # crude bound var <= m^2/4 for Bernoulli

    cert_hist: List[float] = []
    threshold: float | None = None
    burst_remaining = 0

    rows = []
    cum_aud = 0

    # Statefully carry forward the most recent calibration estimate.
    # This is important when the scheduler chooses n_cal=0: in realistic systems
    # we keep using the last known calibration until refreshed.
    beta_hat_state = 0.0

    for t in range(1, cfg.T + 1):
        beta_true = cfg.beta0 if t <= cfg.drift_iter else cfg.beta1

        # cheap judge score vector
        s = r_true + beta_true * u + rng.normal(0.0, cfg.sigma_judge, size=m)

        # monitoring stream for certificate (held-out)
        pol_mon, y_mon = sample_audits(rng, correct, cfg.n_mon)
        if cfg.n_mon > 0:
            # unbiased importance-sampled reward estimate (uniform => weight m)
            sums = np.bincount(pol_mon, weights=y_mon * m, minlength=m).astype(float)
            r_hat_mon = sums / float(cfg.n_mon)
        else:
            r_hat_mon = r_true.copy()

        # audited certificate at current z
        F_hat_z = F_from_reward(z, r_hat_mon, cfg.tau, x_ref)
        z_plus = kl_prox(z, cfg.eta, F_hat_z)
        residual = (1.0 / cfg.eta) * (np.log(z) - np.log(z_plus))
        cert = float(np.dot(F_hat_z, z - z_plus) + R * np.max(np.abs(residual)))

        # high-probability slack epsilon_n(delta) (scalar bound for simplicity)
        if cfg.n_mon > 0:
            log_term = np.log(2.0 * m / cfg.delta)
            eps = np.sqrt(2.0 * sigma2 * log_term / cfg.n_mon) + (2.0 * b * log_term) / (3.0 * cfg.n_mon)
        else:
            eps = 0.0
        cert_hp = cert + R * eps

        cert_hist.append(cert_hp)

        # choose calibration audits
        arm_idx: int | None = None
        p_arm: float | None = None
        if scheduler is not None:
            arm_idx, n_cal, p_arm = scheduler.sample(rng)
        else:
            if adaptive:
                if t <= cfg.warmup:
                    n_cal = cfg.n_cal_low
                else:
                    if threshold is None:
                        base = np.array(cert_hist[: cfg.warmup], dtype=float)
                        threshold = float(base.mean() + cfg.threshold_scale * base.std())
                    if burst_remaining > 0:
                        n_cal = cfg.n_cal_high
                        burst_remaining -= 1
                    else:
                        n_cal = cfg.n_cal_low
                        if cert_hp > threshold:
                            burst_remaining = cfg.burst_len - 1
                            n_cal = cfg.n_cal_high
            else:
                n_cal = n_cal_fixed

        # calibration audits (independent from monitoring)
        pol_cal, y_cal = sample_audits(rng, correct, n_cal)
        beta_hat = float(beta_hat_state)
        if n_cal > 0:
            u_cal = u[pol_cal]
            resp = s[pol_cal] - y_cal
            denom = float(np.dot(u_cal, u_cal) + 1e-12)
            beta_hat = float(np.dot(u_cal, resp) / denom)
            beta_hat_state = beta_hat

        # debiased reward estimate for the lookahead channel
        r_hat = s - beta_hat * u

        # A-OMP step: F_est(w) computed from r_hat
        def F_est(w: np.ndarray) -> np.ndarray:
            return F_from_reward(w, r_hat, cfg.tau, x_ref)

        z, w, g_prev = aomp_step_simplex(z=z, g_prev=g_prev, eta=cfg.eta, F_est=F_est)

        # Discrepancy proxy for audit-control (||g_t - F^{mon}(w_t)||_*^2).
        # We use the dual norm for the simplex (l_infty) to match the paper's
        # certificate geometry.
        F_mon_w = F_from_reward(w, r_hat_mon, cfg.tau, x_ref)
        diff = g_prev - F_mon_w
        xi_sq = float(np.max(np.abs(diff)) ** 2)

        loss: float | None = None
        if scheduler is not None:
            assert arm_idx is not None and p_arm is not None
            loss = scheduler.loss_from_proxy(xi_sq=xi_sq, n_cal=n_cal)
            scheduler.update(arm_idx=arm_idx, p_arm=p_arm, loss=loss)

        reward = float(np.dot(r_true, z))
        gap = gapS_simplex(z, F_from_reward(z, r_true, cfg.tau, x_ref))

        cum_aud += cfg.n_mon + n_cal

        rows.append(
            dict(
                schedule=name,
                t=t,
                reward=reward,
                gap=gap,
                beta_true=beta_true,
                beta_hat=beta_hat,
                cert=cert,
                cert_hp=cert_hp,
                n_cal=n_cal,
                n_mon=cfg.n_mon,
                cum_aud=cum_aud,
                xi_sq=xi_sq,
                exp3_arm=(arm_idx if arm_idx is not None else np.nan),
                exp3_p=(p_arm if p_arm is not None else np.nan),
                exp3_loss=(loss if loss is not None else np.nan),
            )
        )

    return pd.DataFrame(rows)


def compute_metrics(cfg: Config, df: pd.DataFrame) -> dict:
    """Metrics used in the paper tables.

    We emphasize robustness under drift, so we report:
      * min reward in a window immediately after drift
      * total calibration-audit counts
      * final reward
    """
    drift = int(cfg.drift_iter)
    win = int(cfg.post_drift_window)
    post = df[(df.t >= drift) & (df.t < drift + win)]

    # Time-to-recovery after drift: see src.metrics for the operational definition.
    t_rec = time_to_recovery_after_drift(
        df.reward.to_numpy(),
        cfg=RecoveryConfig(
            drift_iter=drift,
            post_window=win,
            pre_window=50,
            smooth_window=cfg.smooth_window,
            tol=0.0,
        ),
    )
    return dict(
        final_reward=float(df.reward.iloc[-1]),
        min_post_reward=float(post.reward.min()),
        recovery_time=int(t_rec),
        final_gap=float(df.gap.iloc[-1]),
        total_cal_audits=int(df.n_cal.sum()),
        total_mon_audits=int(df.n_mon.sum()),
        total_audits=int(df.cum_aud.iloc[-1]),
    )


def save_summary_multiseed(cfg: Config, dfs_by_seed: Dict[int, Dict[str, pd.DataFrame]], out_path: Path) -> None:
    """Write per-seed metrics and mean±CI summaries."""
    schedules = sorted(next(iter(dfs_by_seed.values())).keys())
    per_seed = {int(seed): {k: compute_metrics(cfg, v) for k, v in dfs.items()} for seed, dfs in dfs_by_seed.items()}

    # Aggregate mean ± 95% t-CI across seeds, and include a bootstrap CI
    # robustness check for the mean.
    agg: Dict[str, Dict[str, dict]] = {}
    for sched in schedules:
        agg[sched] = {}
        for field in ["final_reward", "min_post_reward", "recovery_time", "total_cal_audits"]:
            vals = [per_seed[int(seed)][sched][field] for seed in sorted(per_seed.keys())]
            mean, hw = mean_ci_scalar(vals)
            b_mean, b_lo, b_hi = bootstrap_ci_scalar(vals, seed=0)
            digits = 4 if "reward" in field else 0
            entry: Dict[str, object] = {
                "mean": mean,
                "ci_halfwidth": hw,
                "pretty": format_mean_ci(mean, hw, digits=digits),
                "bootstrap": {
                    "mean": b_mean,
                    "lo": b_lo,
                    "hi": b_hi,
                    "pretty_interval": format_ci_interval(b_lo, b_hi, digits=digits),
                },
            }
            # Heavy-tailed metric hygiene: for threshold-based recovery times we also
            # report a robust median/IQR summary.
            if field == "recovery_time":
                med, q1, q3, iqr = median_iqr_scalar(vals)
                entry.update(
                    {
                        "median": med,
                        "q1": q1,
                        "q3": q3,
                        "iqr": iqr,
                        "pretty_robust": format_median_iqr(med, q1, q3, digits=0),
                    }
                )
            agg[sched][field] = entry

    payload = {
        "config": asdict(cfg),
        "seeds": list(sorted(per_seed.keys())),
        "per_seed": per_seed,
        "aggregate": agg,
    }
    out_path.write_text(json.dumps(payload, indent=2))


def plot_and_save_multiseed(cfg: Config, dfs_by_seed: Dict[int, Dict[str, pd.DataFrame]], fig_dir: Path) -> None:
    """Generate ICML-style plots with mean ± 95% CI shading."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(set_matplotlib_style())

    seeds = sorted(dfs_by_seed.keys())
    example = dfs_by_seed[seeds[0]]
    t = example[next(iter(example))].t.values
    drift = int(cfg.drift_iter)

    def series_ci(schedule: str, field: str) -> MeanCI:
        curves = []
        for seed in seeds:
            arr = dfs_by_seed[seed][schedule][field].values
            curves.append(moving_average(arr, cfg.smooth_window))
        stacked = align_and_stack(curves)
        return t_confidence_interval(stacked)

    # Calibration plot (exclude monitoring-only which has no calibration)
    plt.figure(figsize=(6.5, 3.2))
    for name in ["fixed_low", "adaptive", "exp3", "high"]:
        ci = series_ci(name, "beta_hat")
        label = {
            "fixed_low": "fixed low",
            "adaptive": "adaptive bursts",
            "exp3": "Exp3 controller",
            "high": "high",
        }[name]
        line = plt.plot(t, ci.mean, label=label)[0]
        plt.fill_between(t, ci.lo, ci.hi, color=line.get_color(), alpha=0.22, linewidth=0)
    # True drift (deterministic)
    plt.plot(t, example["fixed_low"].beta_true.values, linestyle="--", linewidth=1.5, label="true drift")
    plt.axvline(drift, linestyle=":", linewidth=1.0)
    plt.xlabel("Outer-loop iteration t")
    plt.ylabel(r"Calibration coefficient $\hat\beta_t$")
    plt.legend(ncol=2, fontsize=8, loc="upper left")
    plt.tight_layout()
    savefig_atomic(fig_dir / "rlhf_real_calibration.pdf")
    plt.close()

    # Reward plot
    plt.figure(figsize=(6.5, 3.2))
    for name in ["monitor_only", "fixed_low", "adaptive", "exp3", "high"]:
        ci = series_ci(name, "reward")
        label = {
            "monitor_only": "monitoring only",
            "fixed_low": "fixed low",
            "adaptive": "adaptive bursts",
            "exp3": "Exp3 controller",
            "high": "high",
        }[name]
        line = plt.plot(t, ci.mean, label=label)[0]
        plt.fill_between(t, ci.lo, ci.hi, color=line.get_color(), alpha=0.22, linewidth=0)
    plt.axvline(drift, linestyle=":", linewidth=1.0)
    plt.xlabel("Outer-loop iteration t")
    plt.ylabel("True reward of mixture")
    plt.legend(ncol=2, fontsize=8, loc="upper left")
    plt.tight_layout()
    savefig_atomic(fig_dir / "rlhf_real_reward.pdf")
    plt.close()

    # Stationarity-gap plot (true certificate; conservative upper bound on Minty gap).
    # We use a symmetric-log y-scale to make the transient-to-floor transition visible.
    plt.figure(figsize=(6.5, 3.2))
    for name in ["monitor_only", "fixed_low", "adaptive", "exp3", "high"]:
        ci = series_ci(name, "gap")
        label = {
            "monitor_only": "monitoring only",
            "fixed_low": "fixed low",
            "adaptive": "adaptive bursts",
            "exp3": "Exp3 controller",
            "high": "high",
        }[name]
        line = plt.plot(t, ci.mean, label=label)[0]
        plt.fill_between(t, ci.lo, ci.hi, color=line.get_color(), alpha=0.22, linewidth=0)
    plt.axvline(drift, linestyle=":", linewidth=1.0)
    plt.xlabel("Outer-loop iteration t")
    plt.ylabel("Stationarity gap (true)")
    plt.yscale("symlog")
    plt.legend(ncol=2, fontsize=8, loc="upper left")
    plt.tight_layout()
    savefig_atomic(fig_dir / "rlhf_real_gap.pdf")
    plt.close()

    # Audit tradeoff plot with CI error bars
    plt.figure(figsize=(5.2, 3.2))
    order = ["monitor_only", "fixed_low", "adaptive", "exp3", "high"]
    labels = ["monitoring only", "fixed low", "adaptive", "Exp3", "high"]
    xs, xerr, ys, yerr = [], [], [], []
    for sched in order:
        m = [compute_metrics(cfg, dfs_by_seed[seed][sched]) for seed in seeds]
        # total calibration audits
        cal = [float(mm["total_cal_audits"]) for mm in m]
        # min reward post drift
        post = [float(mm["min_post_reward"]) for mm in m]
        x_ci = mean_ci_scalar(cal)
        y_ci = mean_ci_scalar(post)
        xs.append(x_ci[0])
        xerr.append(x_ci[1])
        ys.append(y_ci[0])
        yerr.append(y_ci[1])

    plt.errorbar(xs, ys, xerr=xerr, yerr=yerr, fmt="o", capsize=3)
    # Manual label offsets to avoid overlap in the high-audit region.
    y_off = {
        "monitoring only": 0.0000,
        "fixed low": 0.0004,
        "adaptive": -0.0004,
        "Exp3": -0.0008,
        "high": 0.0008,
    }
    for x, y, lab in zip(xs, ys, labels):
        plt.text(x, y + y_off.get(lab, 0.0), " " + lab, va="center", fontsize=8)
    plt.xscale("symlog")
    plt.xlabel("Total calibration audits")
    plt.ylabel(f"Min reward in {cfg.post_drift_window} iters after drift")
    plt.tight_layout()
    savefig_atomic(fig_dir / "rlhf_real_audit_tradeoff.pdf")
    plt.close()

    # Recovery-time tradeoff plot (post-drift time-to-recovery vs audit cost)
    plt.figure(figsize=(5.2, 3.2))
    xs, xerr, ys, yerr = [], [], [], []
    for sched in order:
        m = [compute_metrics(cfg, dfs_by_seed[seed][sched]) for seed in seeds]
        cal = [float(mm["total_cal_audits"]) for mm in m]
        rec = [float(mm["recovery_time"]) for mm in m]
        x_ci = mean_ci_scalar(cal)
        y_ci = mean_ci_scalar(rec)
        xs.append(x_ci[0])
        xerr.append(x_ci[1])
        ys.append(y_ci[0])
        yerr.append(y_ci[1])
    plt.errorbar(xs, ys, xerr=xerr, yerr=yerr, fmt="o", capsize=3)
    y_off2 = {
        "monitoring only": 15.0,
        "fixed low": -20.0,
        "adaptive": -35.0,
        "Exp3": 25.0,
        "high": -10.0,
    }
    for x, y, lab in zip(xs, ys, labels):
        plt.text(x, y + y_off2.get(lab, 0.0), " " + lab, va="center", fontsize=8)
    plt.xscale("symlog")
    plt.xlabel("Total calibration audits")
    plt.ylabel("Time-to-recovery after drift (iters)")
    plt.tight_layout()
    savefig_atomic(fig_dir / "rlhf_real_recovery_tradeoff.pdf")
    plt.close()


def main() -> None:
    cfg = Config()

    root = Path(__file__).resolve().parents[2]  # package root (contains figs/, results/, and codes/)
    fig_dir = root / "figs"
    res_dir = root / "results" / "rlhf_wdbc"
    res_dir.mkdir(parents=True, exist_ok=True)
    _write_config(res_dir, cfg)

    correct, r_true, u = build_checkpoints(cfg)

    def run_all_schedules(seed: int) -> Dict[str, pd.DataFrame]:
        # Independent, reproducible RNG streams per schedule (so varying n_cal does not
        # implicitly change the randomness used by other components).
        names = ["monitor_only", "fixed_low", "adaptive", "exp3", "high"]
        ss = SeedSequence(int(seed))
        rngs = {n: np.random.default_rng(child) for n, child in zip(names, ss.spawn(len(names)))}

        dfs_one: Dict[str, pd.DataFrame] = {}
        dfs_one["monitor_only"] = run_schedule(
            "monitor_only", cfg, correct, r_true, u, rngs["monitor_only"], adaptive=False, n_cal_fixed=0
        )
        dfs_one["fixed_low"] = run_schedule(
            "fixed_low", cfg, correct, r_true, u, rngs["fixed_low"], adaptive=False, n_cal_fixed=cfg.n_cal_low
        )
        dfs_one["adaptive"] = run_schedule(
            "adaptive", cfg, correct, r_true, u, rngs["adaptive"], adaptive=True
        )
        exp3 = Exp3AuditScheduler(
            T=cfg.T,
            cfg=Exp3AuditConfig(
                arms=[0, cfg.n_cal_low, cfg.n_cal_high],
                rho=cfg.exp3_rho,
                xi_bound=cfg.exp3_xi_bound,
            ),
        )
        dfs_one["exp3"] = run_schedule(
            "exp3", cfg, correct, r_true, u, rngs["exp3"], adaptive=False, n_cal_fixed=0, scheduler=exp3
        )
        dfs_one["high"] = run_schedule(
            "high", cfg, correct, r_true, u, rngs["high"], adaptive=False, n_cal_fixed=cfg.n_cal_high
        )
        return dfs_one

    # Incremental recomputation: if a per-seed set of trajectories already exists
    # on disk, load it and only run missing seeds. This keeps CPU time reasonable
    # when increasing seeds for high-variance adaptive schedules.
    schedule_names = ["monitor_only", "fixed_low", "adaptive", "exp3", "high"]

    def _traj_path(name: str, seed: int) -> Path:
        return res_dir / f"trajectory_{name}_seed{seed}.csv"

    dfs_by_seed: Dict[int, Dict[str, pd.DataFrame]] = {}
    for seed in cfg.seeds:
        sd = int(seed)
        if all(_traj_path(n, sd).exists() for n in schedule_names):
            dfs_by_seed[sd] = {n: pd.read_csv(_traj_path(n, sd)) for n in schedule_names}
        else:
            dfs_by_seed[sd] = run_all_schedules(sd)
            for name, df in dfs_by_seed[sd].items():
                df.to_csv(_traj_path(name, sd), index=False)

    save_summary_multiseed(cfg, dfs_by_seed, res_dir / "summary_multiseed.json")
    plot_and_save_multiseed(cfg, dfs_by_seed, fig_dir)

    print(f"[rlhf_wdbc] Wrote results to {res_dir}")
    print(f"[rlhf_wdbc] Wrote figures to {fig_dir}")


if __name__ == "__main__":
    main()
