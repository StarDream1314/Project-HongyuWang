"""Regenerate TinyLLM figures from saved CSV trajectories.

Usage:
  python codes/scripts/plot_tinyllm_from_csv.py --root .

This script is plot-only: it reads

  - results/rlhf_tinyllm/trajectory_*.csv (main paper: spurious conciseness drift)
  - results/rlhf_tinyllm_aligned/trajectory_*.csv (appendix: aligned verbosity drift)

and recreates the corresponding paper figures under figs/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".", help="Package root (contains figs/ and results/)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    code_dir = root / "codes"
    sys.path.insert(0, str(code_dir))

    from experiments.rlhf_tinyllm import Config, plot_and_save_multiseed  # noqa: E402

    def _plot_from_resdir(res_dir: Path, *, prefix: str) -> None:
        summary_path = res_dir / "summary_multiseed.json"
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)
        summary = json.loads(summary_path.read_text())
        cfg_dict = summary.get("config", {})
        seeds = list(cfg_dict.get("seeds", summary.get("seeds", [0, 1, 2, 3, 4])))

        # Reconstruct the config used to generate the CSVs (robust to extra keys).
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
                    raise FileNotFoundError(p)
                per_seed[sched] = pd.read_csv(p)
            dfs_by_seed[int(seed)] = per_seed

        fig_dir = root / "figs"
        plot_and_save_multiseed(cfg, dfs_by_seed, fig_dir, prefix=prefix)
        print(f"[plot_tinyllm_from_csv] Wrote figs with prefix {prefix} to {fig_dir}")

    _plot_from_resdir(root / "results" / "rlhf_tinyllm", prefix="rlhf_tinyllm")
    # Optional appendix variant (only if the results are present).
    aligned = root / "results" / "rlhf_tinyllm_aligned"
    if aligned.exists():
        _plot_from_resdir(aligned, prefix="rlhf_tinyllm_aligned")


if __name__ == "__main__":
    main()
