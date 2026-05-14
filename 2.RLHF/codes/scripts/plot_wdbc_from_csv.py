"""Regenerate WDBC RLHF-style drift figures from saved CSV trajectories.

Usage:
  python codes/scripts/plot_wdbc_from_csv.py --root .

This script is plot-only: it reads results/rlhf_wdbc/trajectory_*.csv and
recreates the paper figures under figs/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=".", help="Project root (contains codes/, results/, and figs/).")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    code_dir = root / "codes"
    sys.path.append(str(code_dir))

    from experiments.rlhf_wdbc import Config, plot_and_save_multiseed

    res_dir = root / "results" / "rlhf_wdbc"
    summary_path = res_dir / "summary_multiseed.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")

    summary = json.loads(summary_path.read_text())
    cfg_dict = summary.get("config", {})
    seeds = list(cfg_dict.get("seeds", summary.get("seeds", [0, 1, 2, 3, 4])))

    fields = getattr(Config, "__dataclass_fields__", {})
    filtered = {k: v for k, v in cfg_dict.items() if k in fields}
    cfg = Config(**filtered)

    schedules = ["monitor_only", "fixed_low", "adaptive", "exp3", "high"]

    dfs_by_seed = {}
    for seed in seeds:
        per_seed = {}
        for sched in schedules:
            p = res_dir / f"trajectory_{sched}_seed{seed}.csv"
            if not p.exists():
                raise FileNotFoundError(f"Missing trajectory CSV: {p}")
            per_seed[sched] = pd.read_csv(p)
        dfs_by_seed[int(seed)] = per_seed

    fig_dir = root / "figs"
    plot_and_save_multiseed(cfg, dfs_by_seed, fig_dir)
    print(f"Wrote WDBC figures to: {fig_dir}")


if __name__ == "__main__":
    main()
