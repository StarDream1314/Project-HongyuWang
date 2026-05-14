#!/usr/bin/env python3
"""
Regenerate SHP figures from the saved per-seed trajectory CSVs.

This is useful if you have already run the experiments and only want to
re-render plots (e.g., after changing Matplotlib versions or style).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import pandas as pd

# Ensure `codes/` is on sys.path so we can import experiments.* when invoked from the package root.
CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from experiments.rlhf_shp import (  # noqa: E402
    Config,
    compute_post_drift_gap_floor,
    plot_and_save_multiseed,
    plot_gap_floor_vs_audit_rate,
)


SCHEDULES_MAIN = ["monitor_only", "fixed_low", "adaptive", "exp3", "high"]


def load_seed_bundle(res_dir: Path, seed: int, *, include_ablation: bool) -> tuple[Dict[str, pd.DataFrame], pd.DataFrame | None]:
    dfs_one: Dict[str, pd.DataFrame] = {}
    for sched in SCHEDULES_MAIN:
        p = res_dir / f"trajectory_{sched}_seed{seed}.csv"
        if not p.exists():
            raise FileNotFoundError(p)
        dfs_one[sched] = pd.read_csv(p)

    ablation_df = None
    if include_ablation:
        p_ab = res_dir / f"trajectory_ablation_hint_vs_lookahead_seed{seed}.csv"
        # The main SHP suite is reported over cfg.seeds (typically 10 seeds),
        # but the hint-vs-lookahead ablation can be run on a smaller stable
        # subset of seeds. For plot-only regeneration we therefore treat the
        # ablation trajectory as optional and only load it when present.
        if p_ab.exists():
            ablation_df = pd.read_csv(p_ab)

    return dfs_one, ablation_df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".", help="Package root (contains figs/ and results/)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    fig_dir = root / "figs"

    # In-domain SHP
    cfg = Config()
    res_dir = root / "results" / "rlhf_shp"
    dfs_by_seed: Dict[int, Dict[str, pd.DataFrame]] = {}
    ablation_by_seed: Dict[int, pd.DataFrame] = {}
    for seed in cfg.seeds:
        dfs_one, abl = load_seed_bundle(res_dir, int(seed), include_ablation=True)
        dfs_by_seed[int(seed)] = dfs_one
        if abl is not None:
            ablation_by_seed[int(seed)] = abl
    plot_and_save_multiseed(cfg, dfs_by_seed, ablation_by_seed or None, fig_dir, prefix="rlhf_shp")

    # Scaling-law ablation: gap floor vs audit rate (from precomputed sweep files).
    floors_by_n_cal: Dict[int, list[float]] = {}
    sweep_json = res_dir / "audit_sweep_gap_floor.json"
    if sweep_json.exists():
        payload = json.loads(sweep_json.read_text())
        per = payload.get("per_seed_gap_floor", {})
        floors_by_n_cal = {int(k): [float(x) for x in v] for k, v in per.items()}
    else:
        # Fall back to computing floors from the saved per-seed sweep trajectories.
        floors_by_n_cal = {int(n): [] for n in cfg.audit_sweep_n_cal}
        for seed in cfg.seeds:
            for n_cal in cfg.audit_sweep_n_cal:
                p = res_dir / f"trajectory_audit_sweep_ncal{int(n_cal)}_seed{seed}.csv"
                if not p.exists():
                    continue
                df = pd.read_csv(p)
                floors_by_n_cal[int(n_cal)].append(compute_post_drift_gap_floor(cfg, df))

    if floors_by_n_cal and all(len(v) > 0 for v in floors_by_n_cal.values()):
        plot_gap_floor_vs_audit_rate(cfg, floors_by_n_cal, fig_dir, prefix="rlhf_shp")
        print(f"[plot_shp_from_csv] Wrote audit-sweep scaling figure to {fig_dir}")
    print(f"[plot_shp_from_csv] Wrote figs with prefix rlhf_shp to {fig_dir}")

    # Cross-domain SHP slice
    res_dir_cd = root / "results" / "rlhf_shp_cross_domain"
    # Use the shipped config for the cross-domain slice if present (it typically
    # uses fewer seeds than the in-domain suite to keep the artifact compact).
    cfg_cd = Config(train_domain=cfg.train_domain, eval_domain="askhistorians", seeds=cfg.seeds)
    cfg_path_cd = res_dir_cd / "config.json"
    if cfg_path_cd.exists():
        payload_cd = json.loads(cfg_path_cd.read_text())
        cfg_cd = Config(**payload_cd.get("config", {}))
    dfs_by_seed_cd: Dict[int, Dict[str, pd.DataFrame]] = {}
    for seed in cfg_cd.seeds:
        dfs_one, _ = load_seed_bundle(res_dir_cd, int(seed), include_ablation=False)
        dfs_by_seed_cd[int(seed)] = dfs_one
    plot_and_save_multiseed(cfg_cd, dfs_by_seed_cd, ablation_by_seed=None, fig_dir=fig_dir, prefix="rlhf_shp_cross_domain")
    print(f"[plot_shp_from_csv] Wrote figs with prefix rlhf_shp_cross_domain to {fig_dir}")


if __name__ == "__main__":
    main()
