"""Tiny Transformer + DPO checkpoint mixture under drifting proxy judgment.

This experiment is designed as a *small-scale* but structurally faithful proxy
for modern LLM post-training pipelines:

  1) We train a tiny causal Transformer language model from scratch on
     prompt--response text (byte-level tokenization; no external downloads).
  2) We run a lightweight Direct Preference Optimization (DPO) loop
     (Rafailov et al., 2023) on SHP-style preference pairs, saving m checkpoints.
  3) Each checkpoint induces a preference score by comparing conditional
     log-likelihoods of two candidate responses given the prompt.
  4) The outer loop optimizes a simplex mixture over these checkpoints using
     KL-geometry Audited Optimistic Mirror--Prox (A-OMP).
  5) A cheap proxy judge undergoes a rubric shift that is *not* guaranteed to
     align with the true held-out preference signal (e.g., a sudden conciseness
     preference). Audits are used to (i) calibrate the drifting proxy and (ii)
     provide an independent monitoring certificate. Certificate-driven audit
     bursts recover robustness at lower audit cost.

Outputs:
  - Main-paper figures in figs/: rlhf_tinyllm_reward.pdf, rlhf_tinyllm_bursts.pdf
    (spurious conciseness drift)
  - Appendix figures in figs/: rlhf_tinyllm_aligned_reward.pdf,
    rlhf_tinyllm_aligned_gap.pdf (aligned verbosity drift)
  - Raw trajectories in results/rlhf_tinyllm/ and results/rlhf_tinyllm_aligned/

The goal is not to achieve state-of-the-art language modeling, but to provide a
reproducible, end-to-end preference-tuning stress test that is closer to LLM
practice than linear preference models.
"""

from __future__ import annotations

import json
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib

# Pandas may emit a benign RuntimeWarning when converting NaNs during CSV export.
warnings.filterwarnings("ignore", message="invalid value encountered in cast", category=RuntimeWarning)

# Ensure a non-interactive backend for reproducibility in headless environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from numpy.random import SeedSequence

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


def _write_config(out_dir: Path, cfg: "Config") -> None:
    payload = {
        "config": asdict(cfg),
        "versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
            "torch": torch.__version__,
        },
    }
    (out_dir / "config.json").write_text(json.dumps(payload, indent=2, sort_keys=True))


class ByteTokenizer:
    """Deterministic byte-level tokenizer (no external deps)."""

    BOS = 256
    EOS = 257
    SEP = 258
    PAD = 259
    VOCAB_SIZE = 260

    @staticmethod
    def encode_bytes(text: str, max_len: int) -> List[int]:
        b = text.encode("utf-8", errors="ignore")
        if max_len <= 0:
            return []
        return list(b[:max_len])


class TinyCausalTransformer(nn.Module):
    """A small GPT-like causal Transformer (decoder-only) built from Encoder blocks."""

    def __init__(
        self,
        *,
        vocab_size: int,
        max_seq_len: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.max_seq_len = int(max_seq_len)
        self.d_model = int(d_model)

        self.tok_emb = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_emb = nn.Embedding(self.max_seq_len, self.d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(n_heads),
            dim_feedforward=int(d_ff),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(n_layers))
        self.ln_f = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.d_model, self.vocab_size, bias=False)

        # Initialize similarly to common small-LLM defaults.
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return logits of shape (B, L, V)."""
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (B, L)")
        B, L = input_ids.shape
        if L > self.max_seq_len:
            raise ValueError(f"Sequence length {L} exceeds max_seq_len={self.max_seq_len}")

        pos = torch.arange(L, device=input_ids.device, dtype=torch.long)
        pos = pos.unsqueeze(0).expand(B, L)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)

        # Causal attention mask: True values are masked.
        # TransformerEncoderLayer expects float mask with -inf, or bool mask.
        causal_mask = torch.triu(torch.ones(L, L, device=input_ids.device, dtype=torch.bool), diagonal=1)
        x = self.encoder(x, mask=causal_mask)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits


@dataclass(frozen=True)
class Config:
    # Data
    data_dir: str = "codes/data"
    train_domain: str = "askphysics"
    eval_domain: str = "askphysics"

    # Tokenization / sequence
    max_prompt_bytes: int = 96
    max_resp_bytes: int = 96

    # Model
    d_model: int = 96
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 192
    dropout: float = 0.0

    # Training (kept deliberately small so the end-to-end experiment is CPU-runnable)
    seed_model: int = 0
    batch_size: int = 8
    lr: float = 3e-4
    weight_decay: float = 0.01
    pretrain_steps: int = 120
    dpo_steps: int = 240
    dpo_beta: float = 0.1
    # Save m checkpoints evenly across DPO steps.
    m: int = 12

    # Evaluation set size (None => full test split). We default to a
    # moderately sized subsample to keep the experiment fast while still
    # nontrivial.
    eval_subset: int | None = 128

    # Outer loop
    T: int = 1000
    eta: float = 0.3
    tau: float = 0.01

    # Proxy judge drift (length bias).
    #
    # We define u_i as the standardized tendency of checkpoint i to prefer the
    # *longer* completion on held-out pairs. A change in beta therefore models
    # a rubric shift toward verbosity (beta>0) or conciseness (beta<0).
    drift_iter: int = 600
    beta0: float = 0.0
    # Main-paper setting: conciseness drift (often spurious on SHP askphysics).
    beta1: float = -0.9
    sigma_judge: float = 0.08

    # Auditing
    n_mon: int = 25
    delta: float = 0.05

    # Calibration schedules
    n_cal_low: int = 8
    n_cal_high: int = 250
    warmup: int = 150
    burst_len: int = 120
    threshold_scale: float = 2.5

    # Exp3 audit controller
    exp3_rho: float = 0.25
    exp3_xi_bound: float = 100.0

    # Plotting
    smooth_window: int = 25
    post_drift_window: int = 150

    # Multi-seed evaluation (stochasticity in audits + judge noise)
    # TinyLLM checkpoint-pool training is the slowest CPU experiment in this package.
    # We therefore keep the default at 5 seeds; the main drift-case-study seeds are
    # increased for WDBC/SHP where the high-variance recovery metrics appear.
    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)

    @property
    def max_seq_len(self) -> int:
        # [BOS] + prompt + [SEP] + response + [EOS]
        return 1 + int(self.max_prompt_bytes) + 1 + int(self.max_resp_bytes) + 1


def _load_jsonl(path: Path) -> List[dict]:
    out: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _load_domain_split(data_dir: Path, domain: str, split: str) -> List[dict]:
    path = data_dir / f"{domain}_{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing SHP file: {path} (domain='{domain}', split='{split}')")
    return _load_jsonl(path)


def _resp_text(ex: dict, which: str) -> str:
    if which not in {"A", "B"}:
        raise ValueError("which must be 'A' or 'B'")
    hist = ex.get("history", "")
    resp = ex.get(f"human_ref_{which}", "")
    return f"{hist}\n\n{resp}".strip()


def _word_count(s: str) -> int:
    return int(len(s.split()))


def _batch_encode(
    cfg: Config,
    tok: ByteTokenizer,
    prompts: List[str],
    responses: List[str],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode prompt/response pairs into (input_ids, response_mask).

    response_mask has shape (B, L-1) and selects positions whose predicted
    tokens are part of the response+EOS (i.e., the conditional completion).
    """
    B = len(prompts)
    L = int(cfg.max_seq_len)
    ids = torch.full((B, L), fill_value=tok.PAD, dtype=torch.long)
    mask = torch.zeros((B, L - 1), dtype=torch.float32)

    for i, (p, r) in enumerate(zip(prompts, responses)):
        p_bytes = tok.encode_bytes(p, cfg.max_prompt_bytes)
        r_bytes = tok.encode_bytes(r, cfg.max_resp_bytes)
        seq = [tok.BOS] + p_bytes + [tok.SEP] + r_bytes + [tok.EOS]
        if len(seq) != L:
            # Defensive: ensure fixed length via truncation/padding.
            seq = seq[:L]
            if len(seq) < L:
                seq = seq + [tok.PAD] * (L - len(seq))
        ids[i] = torch.tensor(seq, dtype=torch.long)

        # Response tokens begin after BOS + prompt + SEP.
        start_resp = 1 + len(p_bytes) + 1
        # Targets correspond to positions 1..L-1. Mark targets at positions
        # >= start_resp and < start_resp+len(response)+1 (EOS).
        end_resp = min(L, start_resp + len(r_bytes) + 1)
        if start_resp < L:
            mask[i, (start_resp - 1) : (end_resp - 1)] = 1.0

    return ids.to(device), mask.to(device)

def _logprob_completion(
    model: nn.Module,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Return log p(response|prompt) for each row in the batch."""
    logits = model(input_ids)  # (B, L, V)
    # Predict tokens 1..L-1
    logits_next = logits[:, :-1, :]
    targets = input_ids[:, 1:]
    logp = F.log_softmax(logits_next, dim=-1)
    token_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (token_logp * response_mask).sum(dim=-1)


def _pretrain_lm(cfg: Config, tok: ByteTokenizer, model: nn.Module, train_ex: List[dict]) -> None:
    """Lightweight supervised LM pretraining on prompt+response text."""
    device = next(model.parameters()).device
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

    # Build a pool of (prompt, response) pairs using both candidates.
    prompts: List[str] = []
    responses: List[str] = []
    for ex in train_ex:
        prompts.append(ex.get("history", ""))
        responses.append(ex.get("human_ref_A", ""))
        prompts.append(ex.get("history", ""))
        responses.append(ex.get("human_ref_B", ""))

    rng = np.random.default_rng(int(cfg.seed_model))
    model.train()
    for step in range(int(cfg.pretrain_steps)):
        idx = rng.integers(low=0, high=len(prompts), size=int(cfg.batch_size))
        p_b = [prompts[i] for i in idx]
        r_b = [responses[i] for i in idx]
        input_ids, resp_mask = _batch_encode(cfg, tok, p_b, r_b, device)
        logits = model(input_ids)
        logits_next = logits[:, :-1, :]
        targets = input_ids[:, 1:]
        logp = F.log_softmax(logits_next, dim=-1)
        token_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        # Average NLL over completion tokens.
        denom = resp_mask.sum().clamp_min(1.0)
        loss = -((token_logp * resp_mask).sum() / denom)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


def _train_dpo_checkpoints(
    cfg: Config,
    tok: ByteTokenizer,
    base_model: nn.Module,
    train_ex: List[dict],
) -> List[Dict[str, torch.Tensor]]:
    """Run DPO starting from base_model and return m checkpoint state_dicts."""
    device = next(base_model.parameters()).device

    # Reference (frozen) model.
    ref = TinyCausalTransformer(
        vocab_size=tok.VOCAB_SIZE,
        max_seq_len=cfg.max_seq_len,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
    ).to(device)
    ref.load_state_dict({k: v.detach().clone() for k, v in base_model.state_dict().items()})
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    model = base_model
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

    # Build preference dataset.
    prompts: List[str] = []
    chosen: List[str] = []
    rejected: List[str] = []
    for ex in train_ex:
        p = ex.get("history", "")
        a = ex.get("human_ref_A", "")
        b = ex.get("human_ref_B", "")
        lab = int(ex.get("labels", 0))
        # labels==1 => A preferred, else B preferred (matches SHP preproc).
        if lab == 1:
            c, r = a, b
        else:
            c, r = b, a
        prompts.append(p)
        chosen.append(c)
        rejected.append(r)

    rng = np.random.default_rng(int(cfg.seed_model) + 123)

    total_steps = int(cfg.dpo_steps)
    m = int(cfg.m)
    if m <= 0:
        raise ValueError("m must be positive")
    save_every = max(1, total_steps // m)
    snapshots: List[Dict[str, torch.Tensor]] = []

    beta = float(cfg.dpo_beta)

    for step in range(1, total_steps + 1):
        idx = rng.integers(low=0, high=len(prompts), size=int(cfg.batch_size))
        p_b = [prompts[i] for i in idx]
        c_b = [chosen[i] for i in idx]
        r_b = [rejected[i] for i in idx]

        ids_c, mask_c = _batch_encode(cfg, tok, p_b, c_b, device)
        ids_r, mask_r = _batch_encode(cfg, tok, p_b, r_b, device)

        logp_c = _logprob_completion(model, ids_c, mask_c)
        logp_r = _logprob_completion(model, ids_r, mask_r)
        with torch.no_grad():
            logp_c_ref = _logprob_completion(ref, ids_c, mask_c)
            logp_r_ref = _logprob_completion(ref, ids_r, mask_r)

        pi_lr = logp_c - logp_r
        ref_lr = logp_c_ref - logp_r_ref
        adv = beta * (pi_lr - ref_lr)
        loss = -F.logsigmoid(adv).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (step % save_every == 0) or (step == total_steps):
            snapshots.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            if len(snapshots) >= m:
                break

    # Ensure exactly m snapshots (pad with last if needed).
    if len(snapshots) < m:
        last = snapshots[-1]
        while len(snapshots) < m:
            snapshots.append({k: v.clone() for k, v in last.items()})
    if len(snapshots) > m:
        snapshots = snapshots[:m]
    return snapshots

def build_checkpoints(
    cfg: Config, *, cache_path: Path | None = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train tiny LM checkpoints and evaluate them to obtain (correct, r_true, u).

    If cache_path is provided and exists, load (correct, r_true, u) from disk.
    This makes repeated outer-loop reruns cheap without retraining the checkpoint pool.
    """
    if cache_path is not None and cache_path.exists():
        data = np.load(cache_path)
        return data["correct"], data["r_true"], data["u"]
    torch.set_num_threads(1)
    device = torch.device("cpu")

    tok = ByteTokenizer()

    root = Path(__file__).resolve().parents[2]
    data_dir = root / cfg.data_dir
    train = _load_domain_split(data_dir, cfg.train_domain, "train")
    val = _load_domain_split(data_dir, cfg.train_domain, "validation")
    train_ex = train + val
    test = _load_domain_split(data_dir, cfg.eval_domain, "test")
    if cfg.eval_subset is not None:
        test = test[: int(cfg.eval_subset)]

    # Base model
    model = TinyCausalTransformer(
        vocab_size=tok.VOCAB_SIZE,
        max_seq_len=cfg.max_seq_len,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
    ).to(device)

    # Pretrain and DPO
    _pretrain_lm(cfg, tok, model, train_ex)
    snapshots = _train_dpo_checkpoints(cfg, tok, model, train_ex)
    m = int(cfg.m)

    # Pre-tokenize evaluation prompts/responses.
    prompts = [ex.get("history", "") for ex in test]
    A = [ex.get("human_ref_A", "") for ex in test]
    B = [ex.get("human_ref_B", "") for ex in test]
    y = np.array([int(ex.get("labels", 0)) for ex in test], dtype=int)

    dlen = np.array([_word_count(a) - _word_count(b) for a, b in zip(A, B)], dtype=float)
    len_sign = np.sign(dlen).astype(float)

    # Encode both candidates once; keep on CPU.
    ids_A, mask_A = _batch_encode(cfg, tok, prompts, A, device)
    ids_B, mask_B = _batch_encode(cfg, tok, prompts, B, device)

    # Evaluate each checkpoint.
    correct = np.zeros((m, len(test)), dtype=float)
    preds = np.zeros((m, len(test)), dtype=int)

    # Batched evaluation over test examples.
    bs = 32
    for i, sd in enumerate(snapshots):
        model_i = TinyCausalTransformer(
            vocab_size=tok.VOCAB_SIZE,
            max_seq_len=cfg.max_seq_len,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            d_ff=cfg.d_ff,
            dropout=cfg.dropout,
        ).to(device)
        model_i.load_state_dict(sd)
        model_i.eval()

        scores_A: List[np.ndarray] = []
        scores_B: List[np.ndarray] = []
        for j in range(0, len(test), bs):
            sl = slice(j, min(len(test), j + bs))
            with torch.no_grad():
                lpA = _logprob_completion(model_i, ids_A[sl], mask_A[sl]).cpu().numpy()
                lpB = _logprob_completion(model_i, ids_B[sl], mask_B[sl]).cpu().numpy()
            scores_A.append(lpA)
            scores_B.append(lpB)
        lpA_all = np.concatenate(scores_A, axis=0)
        lpB_all = np.concatenate(scores_B, axis=0)
        pred = (lpA_all > lpB_all).astype(int)
        preds[i] = pred
        correct[i] = (pred == y).astype(float)

    r_true = correct.mean(axis=1)

    # Spurious feature u: tendency to prefer the longer response.
    pref_sign = (2.0 * preds.astype(float) - 1.0)
    u_raw = (pref_sign * len_sign[None, :]).mean(axis=1)
    u = (u_raw - u_raw.mean()) / (u_raw.std() + 1e-12)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, correct=correct, r_true=r_true, u=u)

    return correct, r_true, u


def sample_audits(rng: np.random.Generator, correct: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
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
    m, _ = correct.shape
    x_ref = np.ones(m, dtype=float) / m
    z = x_ref.copy()
    g_prev = np.zeros(m, dtype=float)

    # Certificate diameter for simplex in l1.
    R = 2.0
    # Freedman/Azuma-style slack (uniform importance sampling).
    b = float(m)
    sigma2 = float(m * m) / 4.0

    cert_hist: List[float] = []
    threshold: float | None = None
    burst_remaining = 0
    beta_hat_state = 0.0

    T = int(cfg.T)
    ts = np.arange(1, T + 1, dtype=int)
    reward_arr = np.empty(T, dtype=float)
    gap_arr = np.empty(T, dtype=float)
    beta_true_arr = np.empty(T, dtype=float)
    beta_hat_arr = np.empty(T, dtype=float)
    cert_hp_arr = np.empty(T, dtype=float)
    n_cal_arr = np.empty(T, dtype=int)
    n_mon_arr = np.full(T, int(cfg.n_mon), dtype=int)
    cum_aud_arr = np.empty(T, dtype=int)
    xi_sq_arr = np.empty(T, dtype=float)
    exp3_arm_arr = np.full(T, np.nan, dtype=float)
    exp3_p_arr = np.full(T, np.nan, dtype=float)
    exp3_loss_arr = np.full(T, np.nan, dtype=float)

    cum_aud = 0
    for idx, t in enumerate(ts, start=0):
        beta_true = cfg.beta0 if t <= cfg.drift_iter else cfg.beta1

        # Cheap proxy score vector.
        s = r_true + beta_true * u + rng.normal(0.0, cfg.sigma_judge, size=m)

        # Monitoring stream for high-probability certificate (held-out).
        pol_mon, y_mon = sample_audits(rng, correct, cfg.n_mon)
        if cfg.n_mon > 0:
            sums = np.bincount(pol_mon, weights=y_mon * m, minlength=m).astype(float)
            r_hat_mon = sums / float(cfg.n_mon)
        else:
            r_hat_mon = r_true.copy()

        # Audited certificate at current z.
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
        cert_hp_arr[idx] = cert_hp

        # Choose calibration audits.
        arm_idx: int | None = None
        p_arm: float | None = None
        if scheduler is not None:
            arm_idx, n_cal, p_arm = scheduler.sample(rng)
        else:
            if adaptive:
                if t <= cfg.warmup:
                    n_cal = cfg.n_cal_low
                else:
                    # Initialize certificate threshold once we have a warmup window.
                    if threshold is None:
                        base = np.array(cert_hist[: cfg.warmup], dtype=float)
                        threshold = float(base.mean() + cfg.threshold_scale * base.std())
                    if burst_remaining > 0:
                        n_cal = cfg.n_cal_high
                        burst_remaining -= 1
                    else:
                        n_cal = cfg.n_cal_low
                        assert threshold is not None
                        if cert_hp > threshold:
                            burst_remaining = cfg.burst_len - 1
                            n_cal = cfg.n_cal_high
            else:
                n_cal = n_cal_fixed

        pol_cal, y_cal = sample_audits(rng, correct, n_cal)
        if n_cal > 0:
            beta_hat_state = _beta_hat_ls(u, pol_cal, s, y_cal)
        beta_hat = float(beta_hat_state)
        beta_hat_arr[idx] = beta_hat
        r_hat = s - beta_hat * u

        def F_est(w: np.ndarray) -> np.ndarray:
            return F_from_reward(w, r_hat, cfg.tau, x_ref)

        z, w, g_prev = aomp_step_simplex(z=z, g_prev=g_prev, eta=cfg.eta, F_est=F_est)

        # Discrepancy proxy for audit-control (||g_t - F^{mon}(w_t)||_*^2).
        F_mon_w = F_from_reward(w, r_hat_mon, cfg.tau, x_ref)
        diff = g_prev - F_mon_w
        xi_sq = float(np.max(np.abs(diff)) ** 2)
        xi_sq_arr[idx] = xi_sq

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
        n_cal_arr[idx] = int(n_cal)
        cum_aud_arr[idx] = int(cum_aud)
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


def compute_metrics(cfg: Config, df: pd.DataFrame) -> dict:
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
        total_cal_audits=int(df.n_cal.sum()),
        total_mon_audits=int(df.n_mon.sum()),
        total_audits=int(df.cum_aud.iloc[-1]),
    )


def save_summary_multiseed(cfg: Config, dfs_by_seed: Dict[int, Dict[str, pd.DataFrame]], out_path: Path) -> None:
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

    payload = {"config": asdict(cfg), "seeds": seeds, "per_seed": per_seed, "aggregate": agg}
    out_path.write_text(json.dumps(payload, indent=2))


def plot_and_save_multiseed(
    cfg: Config,
    dfs_by_seed: Dict[int, Dict[str, pd.DataFrame]],
    fig_dir: Path,
    prefix: str = "rlhf_tinyllm",
) -> None:
    """Generate ICML-style plots."""
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
    # This plot makes the transient-to-floor transition visible and is used
    # as a sanity-check that audit schedules are controlling the true stationarity
    # diagnostic (even when reward alone can be misleading under proxy drift).
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

    # Audit-burst visualization: make audit *escalation* feel inevitable by
    # plotting the fraction of runs that have entered high-audit mode at least
    # once since the drift.
    n_high = int(cfg.n_cal_high)

    def _ever_high_since_drift(n_cal: np.ndarray, *, drift_t: int) -> np.ndarray:
        """Indicator that becomes 1 once a run enters high-audit mode post-drift."""
        out = np.zeros_like(n_cal, dtype=float)
        triggered = False
        for i in range(len(n_cal)):
            t_i = i + 1
            if t_i >= drift_t:
                if n_cal[i] == n_high:
                    triggered = True
                out[i] = 1.0 if triggered else 0.0
        return out

    burst_cum = []
    for seed in seeds:
        n_cal = dfs_by_seed[int(seed)]["adaptive"].n_cal.values
        burst_cum.append(_ever_high_since_drift(n_cal, drift_t=int(cfg.drift_iter)))
    burst_stack = align_and_stack(burst_cum)
    ci_burst = t_confidence_interval(burst_stack)

    # Baselines for reference.
    fixed_low = np.zeros_like(t, dtype=float)
    always_high = (t >= int(cfg.drift_iter)).astype(float)

    plt.figure(figsize=(6.5, 3.0))
    line = plt.plot(t, ci_burst.mean, label="adaptive bursts")[0]
    plt.fill_between(t, ci_burst.lo, ci_burst.hi, color=line.get_color(), alpha=0.22, linewidth=0)
    plt.plot(t, fixed_low, linestyle="--", linewidth=1.5, label="fixed low")
    plt.plot(t, always_high, linestyle=":", linewidth=1.8, label="high")
    plt.axvline(drift, linestyle=":", linewidth=1.0)
    plt.xlabel("Outer-loop iteration t")
    plt.ylabel("Fraction of runs with a burst since drift")
    plt.ylim(-0.05, 1.05)
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    savefig_atomic(fig_dir / f"{prefix}_bursts.pdf")
    plt.close()

    # Optional: audit tradeoff plot (used in appendix / debugging)
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
    for x, y, lab in zip(xs, ys, labels):
        plt.text(x, y, " " + lab, va="center", fontsize=8)
    plt.xscale("symlog")
    plt.xlabel("Total calibration audits")
    plt.ylabel(f"Min reward in {cfg.post_drift_window} iters after drift")
    plt.tight_layout()
    savefig_atomic(fig_dir / f"{prefix}_audit_tradeoff.pdf")
    plt.close()


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--build_only",
        action="store_true",
        help="Only (re)build and cache the checkpoint pool, then exit.",
    )
    args = ap.parse_args()
    # Two complementary drift regimes:
    #  (A) main-paper spurious conciseness drift (beta1<0) -> reward degradation.
    #  (B) appendix aligned verbosity drift (beta1>0) -> reward may improve while
    #      stationarity certificates break.
    cfg_spurious = Config()
    cfg_aligned = Config(beta1=abs(float(cfg_spurious.beta1)))

    root = Path(__file__).resolve().parents[2]
    fig_dir = root / "figs"

    # Cache the (correct, r_true, u) arrays so repeated outer-loop reruns are cheap.
    pool_path = root / "results" / "rlhf_tinyllm" / "checkpoint_pool.npz"

    # Train the checkpoint pool once; the outer-loop drift regimes reuse (r_true,u).
    # This is the expensive part, so we keep the outer-loop runs incremental below.
    correct, r_true, u = build_checkpoints(cfg_spurious, cache_path=pool_path)
    if args.build_only:
        print(f"[rlhf_tinyllm] Cached checkpoint pool to {pool_path}")
        return

    def run_variant(cfg: Config, *, res_dir: Path, prefix: str) -> None:
        res_dir.mkdir(parents=True, exist_ok=True)
        _write_config(res_dir, cfg)

        def run_all_schedules(seed: int) -> Dict[str, pd.DataFrame]:
            names = ["monitor_only", "fixed_low", "adaptive", "exp3", "high"]
            ss = SeedSequence(int(seed))
            children = ss.spawn(len(names))
            rngs = {n: np.random.default_rng(child) for n, child in zip(names, children)}
            out: Dict[str, pd.DataFrame] = {}
            out["monitor_only"] = run_schedule(
                "monitor_only", cfg, correct, r_true, u, rngs["monitor_only"], adaptive=False, n_cal_fixed=0
            )
            out["fixed_low"] = run_schedule(
                "fixed_low", cfg, correct, r_true, u, rngs["fixed_low"], adaptive=False, n_cal_fixed=cfg.n_cal_low
            )
            out["adaptive"] = run_schedule("adaptive", cfg, correct, r_true, u, rngs["adaptive"], adaptive=True)
            exp3 = Exp3AuditScheduler(
                T=cfg.T,
                cfg=Exp3AuditConfig(
                    arms=[0, cfg.n_cal_low, cfg.n_cal_high], rho=cfg.exp3_rho, xi_bound=cfg.exp3_xi_bound
                ),
            )
            out["exp3"] = run_schedule(
                "exp3", cfg, correct, r_true, u, rngs["exp3"], adaptive=False, n_cal_fixed=0, scheduler=exp3
            )
            out["high"] = run_schedule(
                "high", cfg, correct, r_true, u, rngs["high"], adaptive=False, n_cal_fixed=cfg.n_cal_high
            )
            return out

        schedules = ["monitor_only", "fixed_low", "adaptive", "exp3", "high"]
        def _path(name: str, seed: int) -> Path:
            return res_dir / f"trajectory_{name}_seed{seed}.csv"

        dfs_by_seed: Dict[int, Dict[str, pd.DataFrame]] = {}
        for seed in cfg.seeds:
            seed_i = int(seed)
            dfs_one: Dict[str, pd.DataFrame] = {}
            missing = [n for n in schedules if not _path(n, seed_i).exists()]
            if missing:
                # Only run what is missing, but keep RNG splits consistent.
                dfs_new = run_all_schedules(seed_i)
                for n in schedules:
                    fp = _path(n, seed_i)
                    if fp.exists():
                        dfs_one[n] = pd.read_csv(fp)
                    else:
                        df = dfs_new[n]
                        df.to_csv(fp, index=False)
                        dfs_one[n] = df
            else:
                for n in schedules:
                    dfs_one[n] = pd.read_csv(_path(n, seed_i))
            dfs_by_seed[seed_i] = dfs_one

        save_summary_multiseed(cfg, dfs_by_seed, res_dir / "summary_multiseed.json")
        plot_and_save_multiseed(cfg, dfs_by_seed, fig_dir, prefix=prefix)
        print(f"[rlhf_tinyllm] Wrote results to {res_dir}")
        print(f"[rlhf_tinyllm] Wrote figures to {fig_dir} (prefix={prefix})")

    # Main-paper figures.
    run_variant(cfg_spurious, res_dir=root / "results" / "rlhf_tinyllm", prefix="rlhf_tinyllm")
    # Appendix drift-direction ablation figures.
    run_variant(
        cfg_aligned,
        res_dir=root / "results" / "rlhf_tinyllm_aligned",
        prefix="rlhf_tinyllm_aligned",
    )


if __name__ == "__main__":
    main()
