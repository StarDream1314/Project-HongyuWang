"""A-OMP HumanEval benchmark validation experiment.

This script instantiates the two-channel A-OMP interface on a precomputed
HumanEval solution matrix:

  * Policies are precomputed code-generation strategies.
  * The cheap channel is a deterministic hint signal from a partial test mask.
  * The accurate channel is split into fixed-budget monitoring and variable-
    budget calibration via uniform problem sampling.

Outputs:
  * Figures in ../figs/: humaneval_reward.pdf, humaneval_audit_tradeoff.pdf,
    humaneval_discrepancy.pdf
  * Raw trajectories in results/rlhf_humaneval/
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib
import numpy as np
import pandas as pd
import sklearn
from numpy.random import Generator, SeedSequence

# Pandas may emit a benign RuntimeWarning when converting NaNs during CSV export.
warnings.filterwarnings("ignore", message="invalid value encountered in cast", category=RuntimeWarning)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.aomp import aomp_step_simplex
from src.audit_schedules import Exp3AuditConfig, Exp3AuditScheduler
from src.plotting import align_and_stack, savefig_atomic, set_matplotlib_style, t_confidence_interval
from src.simplex import F_from_reward, gapS_simplex, kl_prox


@dataclass(frozen=True)
class Config:
    T: int = 1000
    eta: float = 0.3
    tau: float = 0.01
    drift_iter: int = 600
    hint_frac: float = 0.2
    n_mon: int = 25
    n_cal_low: int = 8
    n_cal_high: int = 250
    warmup: int = 150
    burst_len: int = 120
    threshold_scale: float = 2.5
    delta: float = 0.05
    smooth_window: int = 25
    audit_sweep_n_cal: Tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128, 250)
    exp3_rho: float = 0.25
    exp3_xi_bound: float = 100.0
    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)


def _write_config(out_dir: Path, cfg: Config) -> None:
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
    (out_dir / "config.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_humaneval_dataset(data_path: Path) -> dict[str, np.ndarray]:
    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing dataset: {data_path}. Data preparation is out of scope; obtain humaneval_solutions.npz first."
        )

    with np.load(data_path) as data:
        required = {"pass_fail", "test_counts", "strategy_names", "problem_ids", "test_difficulties"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"Dataset missing required arrays: {sorted(missing)}")

        arrays = {name: np.array(data[name]) for name in required}

    pass_fail = arrays["pass_fail"]
    test_counts = arrays["test_counts"]
    strategy_names = arrays["strategy_names"]
    problem_ids = arrays["problem_ids"]
    test_difficulties = arrays["test_difficulties"]

    if pass_fail.ndim != 3:
        raise ValueError(f"pass_fail must be 3D, got shape {pass_fail.shape}")
    K, N, M_max = pass_fail.shape
    if test_counts.shape != (N,):
        raise ValueError(f"test_counts must have shape {(N,)}, got {test_counts.shape}")
    if test_difficulties.shape != (N, M_max):
        raise ValueError(f"test_difficulties must have shape {(N, M_max)}, got {test_difficulties.shape}")
    if strategy_names.shape != (K,):
        raise ValueError(f"strategy_names must have shape {(K,)}, got {strategy_names.shape}")
    if problem_ids.shape != (N,):
        raise ValueError(f"problem_ids must have shape {(N,)}, got {problem_ids.shape}")

    if pass_fail.dtype != np.uint8:
        raise ValueError(f"pass_fail must be uint8, got {pass_fail.dtype}")
    if test_counts.dtype.kind not in {"i", "u"}:
        raise ValueError(f"test_counts must be integer dtype, got {test_counts.dtype}")
    if strategy_names.dtype.kind != "U":
        raise ValueError(f"strategy_names must be fixed-width Unicode, got {strategy_names.dtype}")
    if problem_ids.dtype.kind != "U":
        raise ValueError(f"problem_ids must be fixed-width Unicode, got {problem_ids.dtype}")
    if test_difficulties.dtype.kind != "f":
        raise ValueError(f"test_difficulties must be float dtype, got {test_difficulties.dtype}")

    if not np.isin(pass_fail, [0, 1]).all():
        raise ValueError("pass_fail must only contain 0/1 values")
    if not ((test_counts > 0).all() and (test_counts <= M_max).all()):
        raise ValueError("test_counts values must satisfy 0 < test_counts[n] <= M_max")

    for n, valid in enumerate(test_counts.astype(int)):
        if valid < M_max:
            if not np.all(pass_fail[:, n, valid:] == 0):
                raise ValueError(f"pass_fail padding must be zero for problem index {n}")
            if not np.isnan(test_difficulties[n, valid:]).all():
                raise ValueError(f"test_difficulties padding must be NaN for problem index {n}")

    return arrays


def compute_problem_pass_rates(pass_fail: np.ndarray, test_counts: np.ndarray) -> np.ndarray:
    K, N, _ = pass_fail.shape
    out = np.empty((K, N), dtype=float)
    for n, valid in enumerate(test_counts.astype(int)):
        out[:, n] = pass_fail[:, n, :valid].mean(axis=1)
    return out


def compute_ground_truth_reward_vector(problem_pass_rates: np.ndarray) -> np.ndarray:
    if problem_pass_rates.ndim != 2:
        raise ValueError("problem_pass_rates must have shape [K, N]")
    return problem_pass_rates.mean(axis=1)


def compute_ground_truth_reward(z: np.ndarray, reward_vector: np.ndarray) -> float:
    return float(np.dot(np.asarray(z, dtype=float), np.asarray(reward_vector, dtype=float)))


def compute_hint_score(pass_fail: np.ndarray, test_counts: np.ndarray, hint_mask: np.ndarray) -> np.ndarray:
    K, N, M_max = pass_fail.shape
    if hint_mask.shape != (N, M_max):
        raise ValueError(f"hint_mask must have shape {(N, M_max)}, got {hint_mask.shape}")

    scores = np.empty(K, dtype=float)
    for k in range(K):
        problem_scores = np.empty(N, dtype=float)
        for n, valid in enumerate(test_counts.astype(int)):
            visible = hint_mask[n, :valid]
            if not np.any(visible):
                raise ValueError(f"Problem {n} has zero visible tests in hint_mask")
            problem_scores[n] = pass_fail[k, n, :valid][visible].mean()
        scores[k] = problem_scores.mean()
    return scores


def compute_monitoring_reward(problem_pass_rates: np.ndarray, n_mon: int, rng: Generator) -> np.ndarray:
    if n_mon <= 0:
        raise ValueError(f"n_mon must be positive, got {n_mon}")
    _, N = problem_pass_rates.shape
    sampled_idx = rng.choice(N, size=int(n_mon), replace=True)
    return problem_pass_rates[:, sampled_idx].mean(axis=1)


def compute_calibration_reward(problem_pass_rates: np.ndarray, n_cal: int, rng: Generator) -> np.ndarray:
    if n_cal <= 0:
        raise ValueError(f"n_cal must be positive, got {n_cal}")
    _, N = problem_pass_rates.shape
    sampled_idx = rng.choice(N, size=int(n_cal), replace=True)
    return problem_pass_rates[:, sampled_idx].mean(axis=1)


def construct_hint_mask(
    test_counts: np.ndarray,
    test_difficulties: np.ndarray,
    hint_frac: float,
    is_post_drift: bool,
    rng: Generator,
) -> np.ndarray:
    N, M_max = test_difficulties.shape
    hint_mask = np.zeros((N, M_max), dtype=bool)

    for n, valid in enumerate(test_counts.astype(int)):
        n_visible = max(1, int(hint_frac * valid))
        if is_post_drift:
            sorted_idx = np.argsort(test_difficulties[n, :valid])[::-1]
            hint_mask[n, sorted_idx[:n_visible]] = True
        else:
            visible_idx = rng.choice(valid, size=n_visible, replace=False)
            hint_mask[n, visible_idx] = True
    return hint_mask


def run_schedule(
    name: str,
    cfg: Config,
    pass_fail: np.ndarray,
    test_counts: np.ndarray,
    test_difficulties: np.ndarray,
    problem_pass_rates: np.ndarray,
    reward_true: np.ndarray,
    rng_hint: Generator,
    rng_mon: Generator,
    rng_cal: Generator,
    rng_exp3: Generator,
    adaptive: bool,
    n_cal_fixed: int = 0,
    scheduler: Exp3AuditScheduler | None = None,
) -> pd.DataFrame:
    if cfg.n_mon <= 0:
        raise ValueError("Config.n_mon must be positive for HumanEval monitoring")

    K = int(reward_true.shape[0])
    x_ref = np.ones(K, dtype=float) / float(K)
    z = x_ref.copy()
    g_prev = np.zeros(K, dtype=float)

    R = 2.0
    b = float(K)
    sigma2 = float(K * K) / 4.0

    cert_hist: list[float] = []
    threshold: float | None = None
    burst_remaining = 0

    T = int(cfg.T)
    ts = np.arange(1, T + 1, dtype=int)
    reward_arr = np.empty(T, dtype=float)
    gap_arr = np.empty(T, dtype=float)
    n_cal_arr = np.empty(T, dtype=int)
    n_mon_arr = np.full(T, int(cfg.n_mon), dtype=int)
    cum_aud_arr = np.empty(T, dtype=int)
    xi_sq_arr = np.empty(T, dtype=float)
    cert_arr = np.empty(T, dtype=float)
    cert_hp_arr = np.empty(T, dtype=float)

    cum_aud = 0

    for idx, t in enumerate(ts):
        hint_mask = construct_hint_mask(
            test_counts=test_counts,
            test_difficulties=test_difficulties,
            hint_frac=cfg.hint_frac,
            is_post_drift=bool(t >= cfg.drift_iter),
            rng=rng_hint,
        )
        s_t = compute_hint_score(pass_fail, test_counts, hint_mask)
        r_hat_mon = compute_monitoring_reward(problem_pass_rates, cfg.n_mon, rng_mon)

        F_hat_z = F_from_reward(z, r_hat_mon, cfg.tau, x_ref)
        z_plus = kl_prox(z, cfg.eta, F_hat_z)
        residual = (1.0 / cfg.eta) * (np.log(z) - np.log(z_plus))
        cert = float(np.dot(F_hat_z, z - z_plus) + R * np.max(np.abs(residual)))
        log_term = np.log(2.0 * K / cfg.delta)
        eps = np.sqrt(2.0 * sigma2 * log_term / cfg.n_mon) + (2.0 * b * log_term) / (3.0 * cfg.n_mon)
        cert_hp = cert + R * eps
        cert_hist.append(cert_hp)

        arm_idx: int | None = None
        p_arm: float | None = None
        if scheduler is not None:
            arm_idx, n_cal, p_arm = scheduler.sample(rng_exp3)
        elif adaptive:
            if t <= cfg.warmup:
                n_cal = cfg.n_cal_low
            else:
                if threshold is None:
                    base = np.asarray(cert_hist[: cfg.warmup], dtype=float)
                    threshold = float(base.mean() + cfg.threshold_scale * base.std())
                if burst_remaining > 0:
                    n_cal = cfg.n_cal_high
                    burst_remaining -= 1
                elif cert_hp > threshold:
                    n_cal = cfg.n_cal_high
                    burst_remaining = cfg.burst_len - 1
                else:
                    n_cal = cfg.n_cal_low
        else:
            n_cal = int(n_cal_fixed)

        r_est = compute_calibration_reward(problem_pass_rates, n_cal, rng_cal) if n_cal > 0 else s_t

        def F_est(w: np.ndarray) -> np.ndarray:
            return F_from_reward(w, r_est, cfg.tau, x_ref)

        z, w, g_prev = aomp_step_simplex(z=z, g_prev=g_prev, eta=cfg.eta, F_est=F_est)

        F_mon_w = F_from_reward(w, r_hat_mon, cfg.tau, x_ref)
        diff = g_prev - F_mon_w
        xi_sq = float(np.max(np.abs(diff)) ** 2)

        if scheduler is not None:
            assert arm_idx is not None and p_arm is not None
            loss = scheduler.loss_from_proxy(xi_sq=xi_sq, n_cal=n_cal)
            scheduler.update(arm_idx=arm_idx, p_arm=p_arm, loss=loss)

        reward_arr[idx] = compute_ground_truth_reward(z, reward_true)
        gap_arr[idx] = gapS_simplex(z, F_from_reward(z, reward_true, cfg.tau, x_ref))
        cert_arr[idx] = cert
        cert_hp_arr[idx] = cert_hp
        n_cal_arr[idx] = int(n_cal)
        xi_sq_arr[idx] = xi_sq
        cum_aud += cfg.n_mon + int(n_cal)
        cum_aud_arr[idx] = cum_aud

    return pd.DataFrame(
        {
            "schedule": [name] * T,
            "t": ts,
            "reward": reward_arr,
            "gap": gap_arr,
            "n_cal": n_cal_arr,
            "n_mon": n_mon_arr,
            "cum_aud": cum_aud_arr,
            "xi_sq": xi_sq_arr,
            "cert": cert_arr,
            "cert_hp": cert_hp_arr,
        }
    )


def _traj_path(res_dir: Path, schedule: str, seed: int) -> Path:
    return res_dir / f"trajectory_{schedule}_seed{seed}.csv"


def _load_or_run_schedule(
    *,
    res_dir: Path,
    cfg: Config,
    schedule_name: str,
    seed: int,
    pass_fail: np.ndarray,
    test_counts: np.ndarray,
    test_difficulties: np.ndarray,
    problem_pass_rates: np.ndarray,
    reward_true: np.ndarray,
    rng_hint: Generator,
    rng_mon: Generator,
    rng_cal: Generator,
    rng_exp3: Generator,
) -> pd.DataFrame:
    fp = _traj_path(res_dir, schedule_name, seed)
    if fp.exists():
        df_cached = pd.read_csv(fp)
        if int(len(df_cached)) == int(cfg.T):
            return df_cached

    scheduler = None
    adaptive = False
    n_cal_fixed = 0
    if schedule_name == "monitor_only":
        n_cal_fixed = 0
    elif schedule_name == "fixed_low":
        n_cal_fixed = cfg.n_cal_low
    elif schedule_name == "adaptive":
        adaptive = True
    elif schedule_name == "exp3":
        scheduler = Exp3AuditScheduler(
            T=cfg.T,
            cfg=Exp3AuditConfig(
                arms=[0, cfg.n_cal_low, cfg.n_cal_high],
                rho=cfg.exp3_rho,
                xi_bound=cfg.exp3_xi_bound,
            ),
        )
    elif schedule_name == "high":
        n_cal_fixed = cfg.n_cal_high
    else:
        raise ValueError(f"Unknown schedule: {schedule_name}")

    df = run_schedule(
        name=schedule_name,
        cfg=cfg,
        pass_fail=pass_fail,
        test_counts=test_counts,
        test_difficulties=test_difficulties,
        problem_pass_rates=problem_pass_rates,
        reward_true=reward_true,
        rng_hint=rng_hint,
        rng_mon=rng_mon,
        rng_cal=rng_cal,
        rng_exp3=rng_exp3,
        adaptive=adaptive,
        n_cal_fixed=n_cal_fixed,
        scheduler=scheduler,
    )
    df.to_csv(fp, index=False)
    return df


def _final_window_mean(df: pd.DataFrame, cfg: Config) -> float:
    window = min(200, int(cfg.T))
    return float(df["reward"].tail(window).mean())


def run_budget_sweep(
    cfg: Config,
    *,
    res_dir: Path,
    pass_fail: np.ndarray,
    test_counts: np.ndarray,
    test_difficulties: np.ndarray,
    problem_pass_rates: np.ndarray,
    reward_true: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for seed in cfg.seeds:
        ss_seed = SeedSequence(int(seed))
        n_vals = len(cfg.audit_sweep_n_cal)
        schedule_seeds = ss_seed.spawn(n_vals)
        for n_cal, schedule_seed in zip(cfg.audit_sweep_n_cal, schedule_seeds):
            rng_seeds = schedule_seed.spawn(4)
            df = run_schedule(
                name=f"fixed_{int(n_cal)}",
                cfg=cfg,
                pass_fail=pass_fail,
                test_counts=test_counts,
                test_difficulties=test_difficulties,
                problem_pass_rates=problem_pass_rates,
                reward_true=reward_true,
                rng_hint=np.random.default_rng(rng_seeds[0]),
                rng_mon=np.random.default_rng(rng_seeds[1]),
                rng_cal=np.random.default_rng(rng_seeds[2]),
                rng_exp3=np.random.default_rng(rng_seeds[3]),
                adaptive=False,
                n_cal_fixed=int(n_cal),
            )
            rows.append(
                {
                    "seed": int(seed),
                    "n_cal": int(n_cal),
                    "final_reward": _final_window_mean(df, cfg),
                    "T": int(cfg.T),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(res_dir / "sweep_results.csv", index=False)
    return out


def plot_reward_trajectories(dfs_by_seed: Dict[int, Dict[str, pd.DataFrame]], fig_dir: Path, cfg: Config) -> None:
    plt.rcParams.update(set_matplotlib_style())
    fig, ax = plt.subplots(figsize=(8, 5))
    schedules = ["monitor_only", "fixed_low", "adaptive", "exp3", "high"]
    for schedule in schedules:
        stacked = align_and_stack([dfs_by_seed[seed][schedule]["reward"].to_numpy() for seed in dfs_by_seed])
        ci = t_confidence_interval(stacked)
        ts = dfs_by_seed[next(iter(dfs_by_seed))][schedule]["t"].to_numpy()
        ax.plot(ts, ci.mean, label=schedule)
        ax.fill_between(ts, ci.lo, ci.hi, alpha=0.2)
    ax.axvline(cfg.drift_iter, color="black", linestyle="--", linewidth=1.0, label="drift")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Ground-truth reward")
    ax.legend()
    savefig_atomic(fig_dir / "humaneval_reward.pdf", fig=fig)
    plt.close(fig)


def plot_discrepancy(dfs_by_seed: Dict[int, Dict[str, pd.DataFrame]], fig_dir: Path, cfg: Config) -> None:
    plt.rcParams.update(set_matplotlib_style())
    fig, ax = plt.subplots(figsize=(8, 5))
    schedules = ["monitor_only", "fixed_low", "adaptive", "exp3", "high"]
    for schedule in schedules:
        stacked = align_and_stack([dfs_by_seed[seed][schedule]["xi_sq"].to_numpy() for seed in dfs_by_seed])
        ci = t_confidence_interval(stacked)
        ts = dfs_by_seed[next(iter(dfs_by_seed))][schedule]["t"].to_numpy()
        ax.plot(ts, ci.mean, label=schedule)
        ax.fill_between(ts, ci.lo, ci.hi, alpha=0.2)
    ax.axvline(cfg.drift_iter, color="black", linestyle="--", linewidth=1.0, label="drift")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("xi_sq")
    ax.legend()
    savefig_atomic(fig_dir / "humaneval_discrepancy.pdf", fig=fig)
    plt.close(fig)


def plot_tradeoff(sweep_df: pd.DataFrame, fig_dir: Path) -> None:
    plt.rcParams.update(set_matplotlib_style())
    fig, ax = plt.subplots(figsize=(7, 5))
    n_cals = sorted(int(v) for v in sweep_df["n_cal"].unique())
    means = []
    los = []
    his = []
    for n_cal in n_cals:
        values = sweep_df.loc[sweep_df["n_cal"] == n_cal, "final_reward"].to_numpy()
        ci = t_confidence_interval(values[:, None])
        means.append(float(ci.mean[0]))
        los.append(float(ci.lo[0]))
        his.append(float(ci.hi[0]))
    xs = np.asarray(n_cals, dtype=float)
    ax.plot(xs, means, marker="o")
    ax.fill_between(xs, los, his, alpha=0.2)
    ax.set_xscale("log")
    ax.set_xlabel("Calibration budget per iteration (n_cal)")
    ax.set_ylabel("Final-window reward")
    savefig_atomic(fig_dir / "humaneval_audit_tradeoff.pdf", fig=fig)
    plt.close(fig)


def plot_all(
    dfs_by_seed: Dict[int, Dict[str, pd.DataFrame]],
    sweep_df: pd.DataFrame,
    fig_dir: Path,
    cfg: Config,
) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_reward_trajectories(dfs_by_seed, fig_dir, cfg)
    plot_discrepancy(dfs_by_seed, fig_dir, cfg)
    plot_tradeoff(sweep_df, fig_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the HumanEval A-OMP benchmark.")
    parser.add_argument("--smoke", action="store_true", help="Run a short smoke configuration.")
    parser.add_argument("--data-path", type=Path, default=None, help="Path to humaneval_solutions.npz.")
    parser.add_argument("--results-dir", type=Path, default=None, help="Directory for trajectory CSV outputs.")
    parser.add_argument("--fig-dir", type=Path, default=None, help="Directory for generated figures.")
    args = parser.parse_args()

    cfg = Config(T=50, seeds=(0, 1)) if args.smoke else Config()
    experiment_root = Path(__file__).resolve().parents[2]
    res_dir = args.results_dir or experiment_root / "results" / "rlhf_humaneval"
    fig_dir = args.fig_dir or experiment_root / "figs"
    data_path = args.data_path or experiment_root / "results" / "data" / "humaneval_solutions.npz"

    arrays = load_humaneval_dataset(data_path)
    pass_fail = arrays["pass_fail"]
    test_counts = arrays["test_counts"]
    test_difficulties = arrays["test_difficulties"]

    problem_pass_rates = compute_problem_pass_rates(pass_fail, test_counts)
    reward_true = compute_ground_truth_reward_vector(problem_pass_rates)

    res_dir.mkdir(parents=True, exist_ok=True)
    _write_config(res_dir, cfg)

    schedule_names = ["monitor_only", "fixed_low", "adaptive", "exp3", "high"]
    dfs_by_seed: Dict[int, Dict[str, pd.DataFrame]] = {}

    for seed in cfg.seeds:
        ss_seed = SeedSequence(int(seed))
        schedule_seeds = ss_seed.spawn(len(schedule_names))
        dfs_one: Dict[str, pd.DataFrame] = {}
        for schedule_name, schedule_seed in zip(schedule_names, schedule_seeds):
            rng_seeds = schedule_seed.spawn(4)
            dfs_one[schedule_name] = _load_or_run_schedule(
                res_dir=res_dir,
                cfg=cfg,
                schedule_name=schedule_name,
                seed=int(seed),
                pass_fail=pass_fail,
                test_counts=test_counts,
                test_difficulties=test_difficulties,
                problem_pass_rates=problem_pass_rates,
                reward_true=reward_true,
                rng_hint=np.random.default_rng(rng_seeds[0]),
                rng_mon=np.random.default_rng(rng_seeds[1]),
                rng_cal=np.random.default_rng(rng_seeds[2]),
                rng_exp3=np.random.default_rng(rng_seeds[3]),
            )
        dfs_by_seed[int(seed)] = dfs_one

    sweep_df = run_budget_sweep(
        cfg,
        res_dir=res_dir,
        pass_fail=pass_fail,
        test_counts=test_counts,
        test_difficulties=test_difficulties,
        problem_pass_rates=problem_pass_rates,
        reward_true=reward_true,
    )
    plot_all(dfs_by_seed, sweep_df, fig_dir, cfg)


if __name__ == "__main__":
    main()
