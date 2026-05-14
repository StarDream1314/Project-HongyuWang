"""Generate the SHP cross-domain drift table (Appendix).

Reads:
  - results/rlhf_shp_cross_domain/summary_multiseed.json

Writes:
  - tables/shp_cross_domain_table.tex

The table reports mean±95% t-CI over seeds. Rewards are printed in percent to
match the rest of the paper's tables; time-to-recover is shown as mean±tCI with
median[IQR] in a two-line cell.

Usage:
  python codes/scripts/make_shp_cross_domain_table.py --root .
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _fmt_pct(field: dict, *, digits: int = 2) -> str:
    mean = float(field["mean"]) * 100.0
    hw = float(field["ci_halfwidth"]) * 100.0
    return f"{mean:.{digits}f}$\\pm${hw:.{digits}f}"


def _recovery_cell(field: dict) -> str:
    mean_ci = str(field.get("pretty", ""))
    robust = str(field.get("pretty_robust", ""))
    if mean_ci and robust:
        return r"\shortstack{" + mean_ci + r"\\" + robust + r"}"
    return mean_ci


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    summ = _load_json(root / "results" / "rlhf_shp_cross_domain" / "summary_multiseed.json")
    agg = summ.get("aggregate", {})

    schedules = [
        ("monitor_only", "Monitoring only"),
        ("fixed_low", "Fixed low"),
        ("adaptive", "Adaptive bursts"),
        ("exp3", "Exp3 controller"),
        ("high", "High"),
    ]

    lines: list[str] = []
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Schedule & total cal.\ audits & min.\ post-drift reward & time-to-recover & final reward\\")
    lines.append(r"\midrule")
    for key, name in schedules:
        s = agg[key]
        row = [
            name,
            str(s["total_cal_audits"].get("pretty", "")),
            _fmt_pct(s["min_post_reward"], digits=2),
            _recovery_cell(s["recovery_time"]),
            _fmt_pct(s["final_reward"], digits=2),
        ]
        lines.append(" & ".join(row) + r"\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    out_dir = root / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "shp_cross_domain_table.tex"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[make_shp_cross_domain_table] Wrote {out_path}")


if __name__ == "__main__":
    main()
