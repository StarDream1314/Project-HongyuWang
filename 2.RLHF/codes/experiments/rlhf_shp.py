"""RLHF-style drift stress test on the Stanford Human Preferences (SHP) dataset.

This script instantiates the paper's two-channel interface on a preference dataset:

  * "Policies" are preference-model checkpoints (logistic regressions on TF-IDF
    features) that predict which of two candidate responses is preferred.
  * The outer loop optimizes a simplex mixture over these checkpoints using
    KL-geometry A-OMP.
  * The cheap channel is a drifting proxy judge that becomes biased toward a
    spurious feature (here: response-length preference).
  * The expensive channel is an audited stream of ground-truth preference labels
    used for (i) calibrating the proxy and (ii) high-probability monitoring.

Outputs:
  - Figures in figs/: rlhf_shp_*.pdf
  - Raw trajectories in results/rlhf_shp/

The experiment is designed to match the paper's Section 7 (real-data RLHF-style
case study) and to provide an ablation demonstrating that, under the same audit
budget, spending audits to improve lookahead fidelity is substantially more
effective than spending them on the hint channel.
"""

from __future__ import annotations

import warnings
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

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
    # Data
    data_dir: str = "codes/data"
    # Training domain for preference-model checkpoints.
    train_domain: str = "askphysics"  # shipped with the package (train/val/test)
    # Evaluation domain used for drift + monitoring (distribution shift slice).
    # Set eval_domain==train_domain for the in-domain experiment.
    eval_domain: str = "askphysics"  # for cross-domain: "askhistorians" (test split shipped)
    # Keep the defaults light enough to run on a laptop in a few minutes.
    max_features: int = 8000
    ngram_max: int = 1
    min_df: int = 2

    # Checkpoints
    seed_models: int = 0
    m: int = 20

    # Outer loop
    T: int = 1000
    eta: float = 0.3
    tau: float = 0.01

    # Proxy judge drift (length bias)
    drift_iter: int = 600
    beta0: float = 0.0
    beta1: float = 0.9
    sigma_judge: float = 0.08

    # Auditing
    n_mon: int = 25  # monitoring audits per iteration
    delta: float = 0.05

    # Calibration schedules
    n_cal_low: int = 8
    n_cal_high: int = 250
    warmup: int = 150
    burst_len: int = 120
    threshold_scale: float = 2.5

    # Audit-rate sweep for the scaling-law validation plot.
    # Interpreted as the number of calibration audits per iteration.
    audit_sweep_n_cal: Tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128, 250)

    # Ablation (fixed per-iter calibration budget split between hint vs lookahead)
    ablation_budget: int = 80

    # Plotting
    smooth_window: int = 25
    post_drift_window: int = 150

    # Exp3 audit controller (additional schedule)
    # Arms are {0, n_cal_low, n_cal_high} by default.
    exp3_rho: float = 0.25
    exp3_xi_bound: float = 100.0

    # Multi-seed evaluation (stochasticity in audits + judge noise)
    # We run 10 seeds to stabilize heavy-tailed metrics (e.g., time-to-recover)
    # and reduce uncertainty for adaptive schedules.
    seeds: Tuple[int, ...] = tuple(range(10))


def _load_jsonl(path: Path) -> List[dict]:
    out: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _resp_text(ex: dict, which: str) -> str:
    # Keep it deliberately simple and deterministic.
    # We concatenate the prompt/question context with the candidate response.
    if which not in {"A", "B"}:
        raise ValueError("which must be 'A' or 'B'")
    hist = ex.get("history", "")
    resp = ex.get(f"human_ref_{which}", "")
    return f"{hist}\n\n{resp}".strip()


def _word_count(s: str) -> int:
    return int(len(s.split()))


def _load_domain_split(data_dir: Path, domain: str, split: str) -> List[dict]:
    path = data_dir / f"{domain}_{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing SHP file: {path} (domain='{domain}', split='{split}')")
    return _load_jsonl(path)


def train_preference_checkpoints(cfg: Config) -> Tuple[TfidfVectorizer, List[LogisticRegression]]:
    """Train preference-model checkpoints on cfg.train_domain.

    The returned vectorizer is fit on the training domain and reused for
    evaluation under distribution shift (cfg.eval_domain).
    """
    root = Path(__file__).resolve().parents[2]
    data_dir = root / cfg.data_dir
    train = _load_domain_split(data_dir, cfg.train_domain, "train")
    val = _load_domain_split(data_dir, cfg.train_domain, "validation")

    pool = train + val
    rng = np.random.default_rng(cfg.seed_models)

    texts: List[str] = []
    for ex in pool:
        texts.append(_resp_text(ex, "A"))
        texts.append(_resp_text(ex, "B"))

    vectorizer = TfidfVectorizer(
        max_features=cfg.max_features,
        ngram_range=(1, cfg.ngram_max),
        min_df=cfg.min_df,
        stop_words="english",
    )
    vectorizer.fit(texts)

    def featurize(examples: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
        A_texts = [_resp_text(ex, "A") for ex in examples]
        B_texts = [_resp_text(ex, "B") for ex in examples]
        XA = vectorizer.transform(A_texts)
        XB = vectorizer.transform(B_texts)
        Xdiff = XA - XB
        y = np.array([int(ex["labels"]) for ex in examples], dtype=int)
        return Xdiff, y

    X_pool, y_pool = featurize(pool)

    # Construct m checkpoints.
    # 10 full-data models, 5 half-data, 5 small-data; vary C.
    Cs_full = np.logspace(-3, 3, 10)
    Cs_half = np.logspace(-2, 2, 5)
    Cs_small = np.logspace(-2, 2, 5)
    models: List[LogisticRegression] = []
    idx_full = np.arange(X_pool.shape[0])

    def fit_on_idx(idx: np.ndarray, C: float, seed: int) -> LogisticRegression:
        clf = LogisticRegression(C=float(C), solver="liblinear", max_iter=200, random_state=seed)
        clf.fit(X_pool[idx], y_pool[idx])
        return clf

    for k, C in enumerate(Cs_full):
        models.append(fit_on_idx(idx_full, float(C), seed=0 + k))
    for k, C in enumerate(Cs_half):
        idx = rng.choice(idx_full, size=int(0.5 * len(idx_full)), replace=False)
        models.append(fit_on_idx(idx, float(C), seed=100 + k))
    for k, C in enumerate(Cs_small):
        idx = rng.choice(idx_full, size=int(0.2 * len(idx_full)), replace=False)
        models.append(fit_on_idx(idx, float(C), seed=200 + k))

    if len(models) != cfg.m:
        raise RuntimeError(f"Expected m={cfg.m} models, got {len(models)}")

    return vectorizer, models


def evaluate_checkpoints(
    cfg: Config, vectorizer: TfidfVectorizer, models: List[LogisticRegression], eval_domain: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate checkpoints on eval_domain test split.

    Returns:
      - correct matrix of shape (m, n_test) with {0,1} correctness
      - r_true in [0,1]^m: test accuracies
      - u in R^m: spurious 'length preference' feature (standardized)
    """
    root = Path(__file__).resolve().parents[2]
    data_dir = root / cfg.data_dir
    test = _load_domain_split(data_dir, eval_domain, "test")

    A_texts = [_resp_text(ex, "A") for ex in test]
    B_texts = [_resp_text(ex, "B") for ex in test]
    XA = vectorizer.transform(A_texts)
    XB = vectorizer.transform(B_texts)
    X_test = XA - XB
    y_test = np.array([int(ex["labels"]) for ex in test], dtype=int)
    dlen = np.array([_word_count(ex["human_ref_A"]) - _word_count(ex["human_ref_B"]) for ex in test])
    len_sign_test = np.sign(dlen).astype(float)

    preds = np.vstack([m.predict(X_test) for m in models])
    correct = (preds == y_test[None, :]).astype(float)
    r_true = correct.mean(axis=1)

    # Spurious feature u: tendency to prefer longer response.
    pref_sign = (2.0 * preds.astype(float) - 1.0)
    u_raw = (pref_sign * len_sign_test[None, :]).mean(axis=1)
    u = (u_raw - u_raw.mean()) / (u_raw.std() + 1e-12)

    return correct, r_true, u


def build_checkpoints(cfg: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train preference checkpoints on cfg.train_domain and evaluate on cfg.eval_domain."""
    vectorizer, models = train_preference_checkpoints(cfg)
    return evaluate_checkpoints(cfg, vectorizer, models, cfg.eval_domain)


def sample_audits(
    rng: np.random.Generator, correct: np.ndarray, n: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample n audits: choose policy i ~ Unif[m], datapoint j ~ Unif[n_test]."""
    m, n_test = correct.shape
    if n <= 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=float)
    pol = rng.integers(low=0, high=m, size=n, endpoint=False)
    idx = rng.integers(low=0, high=n_test, size=n, endpoint=False)
    y = correct[pol, idx].astype(float)
    return pol, y


def _beta_hat_ls(u: np.ndarray, pol: np.ndarray, s: np.ndarray, y: np.ndarray) -> float:
    if len(pol) == 0:
        return 0.0
    u_cal = u[pol]
    resp = s[pol] - y
    denom = float(np.dot(u_cal, u_cal) + 1e-12)
    return float(np.dot(u_cal, resp) / denom)


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
    """Run one audit schedule (A-OMP) and return a trajectory DataFrame."""
    # rng is passed in so multi-seed runs can use independent, reproducible streams.
    m, _ = correct.shape
    x_ref = np.ones(m, dtype=float) / m
    z = x_ref.copy()
    g_prev = np.zeros(m, dtype=float)

    # certificate diameter for simplex in l1
    R = 2.0

    # Freedman/Azuma-style slack (uniform importance sampling)
    b = float(m)
    sigma2 = float(m * m) / 4.0

    cert_hist: List[float] = []
    threshold: float | None = None
    burst_remaining = 0

    T = int(cfg.T)
    # Preallocate arrays (substantially faster than appending dicts for long horizons).
    ts = np.arange(1, T + 1, dtype=int)
    reward_arr = np.empty(T, dtype=float)
    gap_arr = np.empty(T, dtype=float)
    beta_true_arr = np.empty(T, dtype=float)
    beta_hat_arr = np.empty(T, dtype=float)
    cert_arr = np.empty(T, dtype=float)
    cert_hp_arr = np.empty(T, dtype=float)
    n_cal_arr = np.empty(T, dtype=int)
    n_mon_arr = np.full(T, int(cfg.n_mon), dtype=int)
    cum_aud_arr = np.empty(T, dtype=int)
    xi_sq_arr = np.empty(T, dtype=float)
    exp3_arm_arr = np.full(T, np.nan, dtype=float)
    exp3_p_arr = np.full(T, np.nan, dtype=float)
    exp3_loss_arr = np.full(T, np.nan, dtype=float)

    cum_aud = 0

    # Stateful calibration: when the schedule chooses n_cal=0, we keep using the
    # last estimate rather than resetting (more realistic and avoids artifacts).
    beta_hat_state = 0.0

    for idx, t in enumerate(ts, start=0):
        beta_true = cfg.beta0 if t <= cfg.drift_iter else cfg.beta1

        # cheap judge score vector
        s = r_true + beta_true * u + rng.normal(0.0, cfg.sigma_judge, size=m)

        # monitoring stream for certificate (held-out)
        pol_mon, y_mon = sample_audits(rng, correct, cfg.n_mon)
        if cfg.n_mon > 0:
            sums = np.bincount(pol_mon, weights=y_mon * m, minlength=m).astype(float)
            r_hat_mon = sums / float(cfg.n_mon)
        else:
            r_hat_mon = r_true.copy()

        # audited prox certificate at current z
        F_hat_z = F_from_reward(z, r_hat_mon, cfg.tau, x_ref)
        z_plus = kl_prox(z, cfg.eta, F_hat_z)
        residual = (1.0 / cfg.eta) * (np.log(z) - np.log(z_plus))
        cert = float(np.dot(F_hat_z, z - z_plus) + R * np.max(np.abs(residual)))

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

        pol_cal, y_cal = sample_audits(rng, correct, n_cal)
        if n_cal > 0:
            beta_hat_state = _beta_hat_ls(u, pol_cal, s, y_cal)
        beta_hat = float(beta_hat_state)
        r_hat = s - beta_hat * u

        def F_est(w: np.ndarray) -> np.ndarray:
            return F_from_reward(w, r_hat, cfg.tau, x_ref)

        z, w, g_prev = aomp_step_simplex(z=z, g_prev=g_prev, eta=cfg.eta, F_est=F_est)

        # Discrepancy proxy for audit-control (||g_t - F^{mon}(w_t)||_*^2).
        # Dual norm for the simplex is l_infty; we square it to mimic xi^2.
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

        reward_arr[idx] = reward
        gap_arr[idx] = gap
        beta_true_arr[idx] = beta_true
        beta_hat_arr[idx] = beta_hat
        cert_arr[idx] = cert
        cert_hp_arr[idx] = cert_hp
        n_cal_arr[idx] = int(n_cal)
        cum_aud_arr[idx] = int(cum_aud)
        xi_sq_arr[idx] = xi_sq
        if arm_idx is not None:
            exp3_arm_arr[idx] = float(arm_idx)
        if p_arm is not None:
            exp3_p_arr[idx] = float(p_arm)
        if loss is not None:
            exp3_loss_arr[idx] = float(loss)

    return pd.DataFrame(
        dict(
            schedule=[name] * T,
            t=ts,
            reward=reward_arr,
            gap=gap_arr,
            beta_true=beta_true_arr,
            beta_hat=beta_hat_arr,
            cert=cert_arr,
            cert_hp=cert_hp_arr,
            n_cal=n_cal_arr,
            n_mon=n_mon_arr,
            cum_aud=cum_aud_arr,
            xi_sq=xi_sq_arr,
            exp3_arm=exp3_arm_arr,
            exp3_p=exp3_p_arr,
            exp3_loss=exp3_loss_arr,
        )
    )


def run_hint_vs_lookahead_ablation(
    cfg: Config,
    correct: np.ndarray,
    r_true: np.ndarray,
    u: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Ablation: fixed per-iteration calibration budget, allocated to hint vs lookahead.

    We run a two-evaluation KL-geometry Mirror-Prox update, where both evaluations
    depend on a debiased reward estimate. Audits are used to estimate the drift
    coefficient, but we can choose whether to spend them to debias the hint
    evaluation (at z) or the lookahead evaluation (at w).
    """
    # rng is passed in for multi-seed evaluation.
    m, _ = correct.shape
    x_ref = np.ones(m, dtype=float) / m

    allocations = {
        "hint_only": (cfg.ablation_budget, 0),
        "split": (cfg.ablation_budget // 2, cfg.ablation_budget - cfg.ablation_budget // 2),
        "lookahead_only": (0, cfg.ablation_budget),
    }

    T = int(cfg.T)
    out_frames: List[pd.DataFrame] = []
    for sched, (n_hint, n_look) in allocations.items():
        z = x_ref.copy()
        cum_aud = 0
        ts = np.arange(1, T + 1, dtype=int)
        reward_arr = np.empty(T, dtype=float)
        gap_arr = np.empty(T, dtype=float)
        beta_true_arr = np.empty(T, dtype=float)
        beta_hat_hint_arr = np.empty(T, dtype=float)
        beta_hat_look_arr = np.empty(T, dtype=float)
        cum_aud_arr = np.empty(T, dtype=int)
        for idx, t in enumerate(ts, start=0):
            beta_true = cfg.beta0 if t <= cfg.drift_iter else cfg.beta1
            s = r_true + beta_true * u + rng.normal(0.0, cfg.sigma_judge, size=m)

            # Separate audit pools for hint and lookahead debiasing.
            pol_h, y_h = sample_audits(rng, correct, n_hint)
            pol_l, y_l = sample_audits(rng, correct, n_look)
            beta_hat_h = _beta_hat_ls(u, pol_h, s, y_h)
            beta_hat_l = _beta_hat_ls(u, pol_l, s, y_l)
            r_hat_h = s - beta_hat_h * u
            r_hat_l = s - beta_hat_l * u

            # Hint evaluation at z, lookahead at w.
            F_hint = F_from_reward(z, r_hat_h if n_hint > 0 else s, cfg.tau, x_ref)
            w = kl_prox(z, cfg.eta, F_hint)
            F_look = F_from_reward(w, r_hat_l if n_look > 0 else s, cfg.tau, x_ref)
            z = kl_prox(z, cfg.eta, F_look)

            reward = float(np.dot(r_true, z))
            gap = gapS_simplex(z, F_from_reward(z, r_true, cfg.tau, x_ref))
            cum_aud += n_hint + n_look
            reward_arr[idx] = reward
            gap_arr[idx] = gap
            beta_true_arr[idx] = beta_true
            beta_hat_hint_arr[idx] = beta_hat_h
            beta_hat_look_arr[idx] = beta_hat_l
            cum_aud_arr[idx] = int(cum_aud)

        out_frames.append(
            pd.DataFrame(
                dict(
                    schedule=[sched] * T,
                    t=ts,
                    reward=reward_arr,
                    gap=gap_arr,
                    beta_true=beta_true_arr,
                    beta_hat_hint=beta_hat_hint_arr,
                    beta_hat_look=beta_hat_look_arr,
                    n_hint=int(n_hint),
                    n_look=int(n_look),
                    cum_aud=cum_aud_arr,
                )
            )
        )

    return pd.concat(out_frames, axis=0, ignore_index=True)


def compute_metrics(cfg: Config, df: pd.DataFrame) -> dict:
    """Metrics reported in the paper tables."""
    drift = int(cfg.drift_iter)
    win = int(cfg.post_drift_window)
    post = df[(df.t >= drift) & (df.t < drift + win)]
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
        total_cal_audits=int(df.n_cal.sum()) if "n_cal" in df.columns else int(df.cum_aud.iloc[-1]),
        total_mon_audits=int(df.n_mon.sum()) if "n_mon" in df.columns else 0,
        total_audits=int(df.cum_aud.iloc[-1]) if "cum_aud" in df.columns else int(df.cum_aud.iloc[-1]),
    )


def compute_post_drift_gap_floor(cfg: Config, df: pd.DataFrame) -> float:
    """A robust post-drift stationarity "floor" metric.

    We use the median gap after a short post-drift burn-in window to reduce
    sensitivity to transient spikes right at the drift time.
    """

    start_t = int(cfg.drift_iter + cfg.post_drift_window)
    tail = df[df.t >= start_t]
    if len(tail) == 0:
        tail = df
    return float(np.median(tail.gap.values))


def save_summary_multiseed(
    cfg: Config,
    dfs_by_seed: Dict[int, Dict[str, pd.DataFrame]],
    ablation_by_seed: Dict[int, pd.DataFrame] | None,
    out_path: Path,
) -> None:
    """Write per-seed metrics and mean±CI summaries."""
    schedules = sorted(next(iter(dfs_by_seed.values())).keys())
    seeds = sorted(dfs_by_seed.keys())

    per_seed = {int(seed): {k: compute_metrics(cfg, v) for k, v in dfs.items()} for seed, dfs in dfs_by_seed.items()}

    agg: Dict[str, Dict[str, dict]] = {}
    for sched in schedules:
        agg[sched] = {}
        for field in ["final_reward", "min_post_reward", "recovery_time", "total_cal_audits"]:
            vals = [per_seed[int(seed)][sched][field] for seed in seeds]
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

    ablation_payload = None
    if ablation_by_seed is not None:
        # Aggregate ablation metrics across seeds.
        ab_per_seed: Dict[int, Dict[str, dict]] = {}
        ab_seeds = sorted(ablation_by_seed.keys())
        for seed in ab_seeds:
            df = ablation_by_seed[int(seed)]
            ab_per_seed[int(seed)] = {}
            for sched in sorted(df.schedule.unique()):
                ab_per_seed[int(seed)][sched] = compute_metrics(cfg, df[df.schedule == sched])
        ab_agg: Dict[str, Dict[str, dict]] = {}
        for sched in sorted(next(iter(ab_per_seed.values())).keys()):
            ab_agg[sched] = {}
            for field in ["final_reward", "min_post_reward", "total_cal_audits"]:
                vals = [ab_per_seed[int(seed)][sched][field] for seed in ab_seeds]
                mean, hw = mean_ci_scalar(vals)
                ab_agg[sched][field] = {
                    "mean": mean,
                    "ci_halfwidth": hw,
                    "pretty": format_mean_ci(mean, hw, digits=(4 if "reward" in field else 0)),
                }
        ablation_payload = {"seeds": ab_seeds, "per_seed": ab_per_seed, "aggregate": ab_agg}

    payload = {
        "config": asdict(cfg),
        "seeds": seeds,
        "per_seed": per_seed,
        "aggregate": agg,
        "ablation": ablation_payload,
    }
    out_path.write_text(json.dumps(payload, indent=2))


def plot_and_save_multiseed(
    cfg: Config,
    dfs_by_seed: Dict[int, Dict[str, pd.DataFrame]],
    ablation_by_seed: Dict[int, pd.DataFrame] | None,
    fig_dir: Path,
    prefix: str = "rlhf_shp",
) -> None:
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

    # Calibration plot
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
    plt.plot(t, example["fixed_low"].beta_true.values, linestyle="--", linewidth=1.5, label="true drift")
    plt.axvline(drift, linestyle=":", linewidth=1.0)
    plt.xlabel("Outer-loop iteration t")
    plt.ylabel(r"Calibration coefficient $\hat\beta_t$")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    savefig_atomic(fig_dir / f"{prefix}_calibration.pdf")
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
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    savefig_atomic(fig_dir / f"{prefix}_reward.pdf")
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
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    savefig_atomic(fig_dir / f"{prefix}_gap.pdf")
    plt.close()

    # Audit-burst visualization (discrete per-iteration calibration audits).
    # We plot a representative seed to avoid visually averaging away the bursts.
    rep_seed = int(seeds[0])
    df_rep = dfs_by_seed[rep_seed]
    plt.figure(figsize=(6.5, 3.0))
    # Adaptive schedule: show the realized calibration-audit bursts.
    plt.plot(t, df_rep["adaptive"].n_cal.values, label="adaptive bursts")
    # Reference baselines (constant calibration rates).
    plt.plot(t, df_rep["fixed_low"].n_cal.values, linestyle="--", linewidth=1.5, label="fixed low")
    plt.plot(t, df_rep["high"].n_cal.values, linestyle=":", linewidth=1.8, label="high")
    plt.axvline(drift, linestyle=":", linewidth=1.0)
    plt.xlabel("Outer-loop iteration t")
    plt.ylabel("Calibration audits per iteration")
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    savefig_atomic(fig_dir / f"{prefix}_bursts.pdf")
    plt.close()

    # Audit tradeoff plot with CI error bars
    plt.figure(figsize=(5.2, 3.2))
    order = ["monitor_only", "fixed_low", "adaptive", "exp3", "high"]
    labels = ["monitoring only", "fixed low", "adaptive", "Exp3", "high"]
    xs, xerr, ys, yerr = [], [], [], []
    for sched in order:
        m = [compute_metrics(cfg, dfs_by_seed[seed][sched]) for seed in seeds]
        cal = [float(mm["total_cal_audits"]) for mm in m]
        post = [float(mm["min_post_reward"]) for mm in m]
        x_ci = mean_ci_scalar(cal)
        y_ci = mean_ci_scalar(post)
        xs.append(x_ci[0])
        xerr.append(x_ci[1])
        ys.append(y_ci[0])
        yerr.append(y_ci[1])

    plt.errorbar(xs, ys, xerr=xerr, yerr=yerr, fmt="o", capsize=3)
    y_off = {
        "monitoring only": 0.0000,
        "fixed low": 0.0003,
        "adaptive": -0.0005,
        "Exp3": -0.0008,
        "high": 0.0010,
    }
    for x, y, lab in zip(xs, ys, labels):
        plt.text(x, y + y_off.get(lab, 0.0), " " + lab, va="center", fontsize=8)
    plt.xscale("symlog")
    plt.xlabel("Total calibration audits")
    plt.ylabel(f"Min reward in {cfg.post_drift_window} iters after drift")
    plt.tight_layout()
    savefig_atomic(fig_dir / f"{prefix}_audit_tradeoff.pdf")
    plt.close()

    # Recovery-time tradeoff plot (time-to-recovery vs audit cost)
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
        "monitoring only": 12.0,
        "fixed low": -14.0,
        "adaptive": -20.0,
        "Exp3": 20.0,
        "high": -10.0,
    }
    for x, y, lab in zip(xs, ys, labels):
        plt.text(x, y + y_off2.get(lab, 0.0), " " + lab, va="center", fontsize=8)
    plt.xscale("symlog")
    plt.xlabel("Total calibration audits")
    plt.ylabel("Time-to-recovery after drift (iters)")
    plt.tight_layout()
    savefig_atomic(fig_dir / f"{prefix}_recovery_tradeoff.pdf")
    plt.close()

    if ablation_by_seed is not None:
        seeds_ab = sorted(ablation_by_seed.keys())
        # Hint vs lookahead ablation (CI-shaded)
        plt.figure(figsize=(6.5, 3.2))
        for sched in ["hint_only", "split", "lookahead_only"]:
            curves = []
            for seed in seeds_ab:
                df = ablation_by_seed[seed]
                sub = df[df.schedule == sched]
                curves.append(moving_average(sub.reward.values, cfg.smooth_window))
            stacked = align_and_stack(curves)
            ci = t_confidence_interval(stacked)
            line = plt.plot(t, ci.mean, label=sched.replace("_", " "))[0]
            plt.fill_between(t, ci.lo, ci.hi, color=line.get_color(), alpha=0.22, linewidth=0)
        plt.axvline(drift, linestyle=":", linewidth=1.0)
        plt.xlabel("Outer-loop iteration t")
        plt.ylabel("True reward of mixture")
        plt.legend(ncol=3, fontsize=8)
        plt.tight_layout()
        savefig_atomic(fig_dir / f"{prefix}_hint_vs_lookahead.pdf")
        plt.close()


def plot_gap_floor_vs_audit_rate(
    cfg: Config,
    floors_by_n_cal: Dict[int, list[float]],
    fig_dir: Path,
    *,
    prefix: str = "rlhf_shp",
) -> None:
    """Scaling-law ablation: post-drift gap floor vs audit rate.

    The two-channel A-OMP bound predicts that, all else equal, the stationary
    noise floor should decay proportionally to the lookahead MSE. In this SHP
    setup the dominant term is the calibration (audited) drift-correction error,
    which empirically behaves like O(1/n_cal). We therefore plot the measured
    post-drift gap floor vs the per-iteration calibration-audit count.
    """

    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(set_matplotlib_style())

    n_cals = sorted(floors_by_n_cal.keys())
    means, errs = [], []
    for n in n_cals:
        mean, hw = mean_ci_scalar(floors_by_n_cal[n])
        means.append(mean)
        errs.append(hw)

    plt.figure(figsize=(5.2, 3.2))
    plt.errorbar(n_cals, means, yerr=errs, fmt="o", capsize=3, label="measured")

    # Reference slope -1 guide line anchored at the highest-audit point.
    n_ref = float(n_cals[-1])
    y_ref = float(means[-1])
    ref = [y_ref * (n_ref / float(n)) for n in n_cals]
    plt.plot(n_cals, ref, linestyle="--", linewidth=1.5, label=r"$\propto 1/n_{\mathrm{cal}}$")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Calibration audits per iteration (audit rate)")
    plt.ylabel("Post-drift stationarity-gap floor")
    plt.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    # This plot has a long y-label on log scales; add a touch more padding to
    # avoid any renderer-dependent clipping in tight bounding boxes.
    savefig_atomic(fig_dir / f"{prefix}_floor_vs_audit.pdf", bbox_inches="tight", pad_inches=0.08)
    plt.close()


def main() -> None:
    cfg = Config()

    # NOTE: The in-domain SHP suite is the heaviest experiment in this package.
    # For reproducibility and iterative development, we support *incremental*
    # recomputation: if per-seed trajectories already exist on disk, we load them
    # and only run the missing seeds. This allows us to increase seeds for the
    # high-variance adaptive schedules without repeatedly paying the full cost of
    # checkpoint training + long outer-loop sweeps.

    root = Path(__file__).resolve().parents[2]  # package root (contains figs/, results/, and codes/)
    fig_dir = root / "figs"

    # Train checkpoints once (on cfg.train_domain), then evaluate them under
    # both in-domain and cross-domain test distributions.
    vectorizer, models = train_preference_checkpoints(cfg)

    # -----------------------------
    # In-domain experiment (multi-seed)
    # -----------------------------
    res_dir = root / "results" / "rlhf_shp"
    res_dir.mkdir(parents=True, exist_ok=True)
    _write_config(res_dir, cfg)
    correct, r_true, u = evaluate_checkpoints(cfg, vectorizer, models, cfg.eval_domain)

    schedules = ["monitor_only", "fixed_low", "adaptive", "exp3", "high"]
    seeds_all = [int(s) for s in cfg.seeds]
    seeds_ablation = seeds_all[:5]  # enough for this stable ablation plot

    def _traj_path(name: str, seed: int) -> Path:
        return res_dir / f"trajectory_{name}_seed{seed}.csv"

    def _load_or_run_schedule(seed: int, name: str, rng: np.random.Generator) -> pd.DataFrame:
        fp = _traj_path(name, seed)
        if fp.exists():
            return pd.read_csv(fp)

        if name == "monitor_only":
            df = run_schedule("monitor_only", cfg, correct, r_true, u, rng, adaptive=False, n_cal_fixed=0)
        elif name == "fixed_low":
            df = run_schedule("fixed_low", cfg, correct, r_true, u, rng, adaptive=False, n_cal_fixed=cfg.n_cal_low)
        elif name == "adaptive":
            df = run_schedule("adaptive", cfg, correct, r_true, u, rng, adaptive=True)
        elif name == "exp3":
            exp3 = Exp3AuditScheduler(
                T=cfg.T,
                cfg=Exp3AuditConfig(
                    arms=[0, cfg.n_cal_low, cfg.n_cal_high],
                    rho=cfg.exp3_rho,
                    xi_bound=cfg.exp3_xi_bound,
                ),
            )
            df = run_schedule("exp3", cfg, correct, r_true, u, rng, adaptive=False, n_cal_fixed=0, scheduler=exp3)
        elif name == "high":
            df = run_schedule("high", cfg, correct, r_true, u, rng, adaptive=False, n_cal_fixed=cfg.n_cal_high)
        else:
            raise ValueError(f"Unknown schedule: {name}")

        df.to_csv(fp, index=False)
        return df

    dfs_by_seed: Dict[int, Dict[str, pd.DataFrame]] = {}
    ablation_by_seed: Dict[int, pd.DataFrame] = {}
    for seed in seeds_all:
        # Deterministic per-seed RNG split matching the original implementation.
        names_for_rng = schedules + ["ablation"]
        ss = SeedSequence(int(seed))
        rngs = {n: np.random.default_rng(child) for n, child in zip(names_for_rng, ss.spawn(len(names_for_rng)))}

        dfs_one: Dict[str, pd.DataFrame] = {}
        for name in schedules:
            dfs_one[name] = _load_or_run_schedule(seed, name, rngs[name])
        dfs_by_seed[int(seed)] = dfs_one

        if seed in seeds_ablation:
            abl_fp = res_dir / f"trajectory_ablation_hint_vs_lookahead_seed{seed}.csv"
            if abl_fp.exists():
                abl = pd.read_csv(abl_fp)
            else:
                abl = run_hint_vs_lookahead_ablation(cfg, correct, r_true, u, rng=rngs["ablation"])
                abl.to_csv(abl_fp, index=False)
            ablation_by_seed[int(seed)] = abl

    # -----------------------------
    # Audit-rate sweep (fixed schedules) for the scaling-law validation plot.
    # -----------------------------
    sweep_path = res_dir / "audit_sweep_gap_floor.json"
    floors_by_n_cal: Dict[int, list[float]] = {int(n): [] for n in cfg.audit_sweep_n_cal}
    if sweep_path.exists():
        sweep = json.loads(sweep_path.read_text())
        per_seed = sweep.get("per_seed_gap_floor", {})
        for n_cal_i, vals in per_seed.items():
            floors_by_n_cal[int(n_cal_i)] = [float(v) for v in vals]
    else:
        # This sweep is expensive; we run it only on the first 5 seeds.
        for seed in seeds_ablation:
            ss = SeedSequence(int(seed) + 10_000)  # disjoint from the main schedule RNGs
            rngs = [np.random.default_rng(child) for child in ss.spawn(len(cfg.audit_sweep_n_cal))]
            for n_cal, rng in zip(cfg.audit_sweep_n_cal, rngs):
                n_cal_i = int(n_cal)
                df = run_schedule(
                    f"audit_sweep_ncal{n_cal_i}",
                    cfg,
                    correct,
                    r_true,
                    u,
                    rng,
                    adaptive=False,
                    n_cal_fixed=n_cal_i,
                )
                df.to_csv(res_dir / f"trajectory_audit_sweep_ncal{n_cal_i}_seed{seed}.csv", index=False)
                floors_by_n_cal[n_cal_i].append(compute_post_drift_gap_floor(cfg, df))

        sweep_agg: Dict[str, dict] = {}
        for n_cal_i in sorted(floors_by_n_cal.keys()):
            mean, hw = mean_ci_scalar(floors_by_n_cal[n_cal_i])
            sweep_agg[str(n_cal_i)] = {
                "mean": mean,
                "ci_halfwidth": hw,
                "pretty": format_mean_ci(mean, hw, digits=4),
            }
        sweep_path.write_text(
            json.dumps(
                {
                    "config": asdict(cfg),
                    "n_cal_list": list(cfg.audit_sweep_n_cal),
                    "per_seed_gap_floor": {str(k): v for k, v in floors_by_n_cal.items()},
                    "aggregate": sweep_agg,
                },
                indent=2,
            )
        )

    save_summary_multiseed(cfg, dfs_by_seed, ablation_by_seed, res_dir / "summary_multiseed.json")
    plot_and_save_multiseed(cfg, dfs_by_seed, ablation_by_seed, fig_dir, prefix="rlhf_shp")
    plot_gap_floor_vs_audit_rate(cfg, floors_by_n_cal, fig_dir, prefix="rlhf_shp")
    print(f"[rlhf_shp] Wrote results to {res_dir}")
    print(f"[rlhf_shp] Wrote figures to {fig_dir}")

    # -----------------------------
    # Cross-domain distribution shift slice (multi-seed)
    # -----------------------------
    print("[rlhf_shp_cross_domain] Starting...")
    cfg_cd = Config(train_domain=cfg.train_domain, eval_domain="askhistorians", seeds=tuple(seeds_ablation))
    res_dir_cd = root / "results" / "rlhf_shp_cross_domain"
    res_dir_cd.mkdir(parents=True, exist_ok=True)
    _write_config(res_dir_cd, cfg_cd)
    correct_cd, r_true_cd, u_cd = evaluate_checkpoints(cfg_cd, vectorizer, models, cfg_cd.eval_domain)

    dfs_by_seed_cd: Dict[int, Dict[str, pd.DataFrame]] = {}
    for seed in cfg_cd.seeds:
        names = ["monitor_only", "fixed_low", "adaptive", "exp3", "high"]
        ss = SeedSequence(int(seed))
        rngs = {n: np.random.default_rng(child) for n, child in zip(names, ss.spawn(len(names)))}
        dfs_one: Dict[str, pd.DataFrame] = {}
        # Incremental: load or recompute per-seed trajectories.
        def _cd_path(name: str) -> Path:
            return res_dir_cd / f"trajectory_{name}_seed{seed}.csv"

        def _load_or_run_cd(name: str) -> pd.DataFrame:
            fp = _cd_path(name)
            if fp.exists():
                return pd.read_csv(fp)
            if name == "monitor_only":
                df = run_schedule("monitor_only", cfg_cd, correct_cd, r_true_cd, u_cd, rngs["monitor_only"], adaptive=False, n_cal_fixed=0)
            elif name == "fixed_low":
                df = run_schedule("fixed_low", cfg_cd, correct_cd, r_true_cd, u_cd, rngs["fixed_low"], adaptive=False, n_cal_fixed=cfg_cd.n_cal_low)
            elif name == "adaptive":
                df = run_schedule("adaptive", cfg_cd, correct_cd, r_true_cd, u_cd, rngs["adaptive"], adaptive=True)
            elif name == "exp3":
                exp3_cd = Exp3AuditScheduler(
                    T=cfg_cd.T,
                    cfg=Exp3AuditConfig(
                        arms=[0, cfg_cd.n_cal_low, cfg_cd.n_cal_high],
                        rho=cfg_cd.exp3_rho,
                        xi_bound=cfg_cd.exp3_xi_bound,
                    ),
                )
                df = run_schedule("exp3", cfg_cd, correct_cd, r_true_cd, u_cd, rngs["exp3"], adaptive=False, n_cal_fixed=0, scheduler=exp3_cd)
            elif name == "high":
                df = run_schedule("high", cfg_cd, correct_cd, r_true_cd, u_cd, rngs["high"], adaptive=False, n_cal_fixed=cfg_cd.n_cal_high)
            else:
                raise ValueError(name)
            df.to_csv(fp, index=False)
            return df

        dfs_one["monitor_only"] = _load_or_run_cd("monitor_only")
        dfs_one["fixed_low"] = _load_or_run_cd("fixed_low")
        dfs_one["adaptive"] = _load_or_run_cd("adaptive")
        dfs_one["exp3"] = _load_or_run_cd("exp3")
        dfs_one["high"] = _load_or_run_cd("high")
        dfs_by_seed_cd[int(seed)] = dfs_one


    save_summary_multiseed(cfg_cd, dfs_by_seed_cd, ablation_by_seed=None, out_path=res_dir_cd / "summary_multiseed.json")
    plot_and_save_multiseed(cfg_cd, dfs_by_seed_cd, ablation_by_seed=None, fig_dir=fig_dir, prefix="rlhf_shp_cross_domain")
    print(f"[rlhf_shp_cross_domain] Wrote results to {res_dir_cd}")
    print(f"[rlhf_shp_cross_domain] Wrote figures to {fig_dir}")


if __name__ == "__main__":
    main()
