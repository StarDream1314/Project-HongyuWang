"""Generate the SHP bootstrap CI sanity-check table (Appendix).

We show mean±tCI alongside a nonparametric bootstrap percentile CI for the mean,
and add a robust median[IQR] summary for the heavy-tailed time-to-recover metric.

Reads:
  - results/rlhf_shp/summary_multiseed.json

Writes:
  - tables/bootstrap_shp_table.tex

Usage:
  python codes/scripts/make_bootstrap_shp_table.py --root .
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _pct_pretty(field: dict, *, digits: int = 2) -> str:
    mean = float(field["mean"]) * 100.0
    hw = float(field["ci_halfwidth"]) * 100.0
    return f"{mean:.{digits}f}$\\pm${hw:.{digits}f}"


def _pct_boot(field: dict, *, digits: int = 2) -> str:
    lo = float(field["bootstrap"]["lo"]) * 100.0
    hi = float(field["bootstrap"]["hi"]) * 100.0
    return f"$[{lo:.{digits}f},\\,{hi:.{digits}f}]$"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    summ = _load_json(root / "results" / "rlhf_shp" / "summary_multiseed.json")
    agg = summ.get("aggregate", {})

    schedules = [
        ("monitor_only", "Monitoring only"),
        ("fixed_low", "Fixed low"),
        ("adaptive", "Adaptive bursts"),
        ("exp3", "Exp3 controller"),
        ("high", "High"),
    ]

    lines: list[str] = []
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    lines.append(
        r"Schedule & min. post-drift reward (t-CI) & bootstrap CI & time-to-recover (t-CI) & bootstrap CI & median[IQR]\\"
    )
    lines.append(r"\midrule")

    for key, name in schedules:
        s = agg[key]
        min_post = s["min_post_reward"]
        rec = s["recovery_time"]

        row = [
            name,
            _pct_pretty(min_post, digits=2),
            _pct_boot(min_post, digits=2),
            str(rec.get("pretty", "")),
            f"${rec['bootstrap']['pretty_interval']}$",
            str(rec.get("pretty_robust", "")),
        ]
        lines.append(" & ".join(row) + r"\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    out_dir = root / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bootstrap_shp_table.tex"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[make_bootstrap_shp_table] Wrote {out_path}")


if __name__ == "__main__":
    main()
