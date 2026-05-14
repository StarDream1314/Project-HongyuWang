"""Generate a compact LaTeX table for the TinyLLM drift-direction ablation.

This table is intended to make the "reward vs certifiability" message
immediately visible: depending on the sign of the judge drift, reward can
either degrade (spurious drift) or even improve (aligned drift) while the
stationarity certificate (Minty gap) can still fail without calibration.

Reads:
  - results/rlhf_tinyllm/summary_multiseed.json
  - results/rlhf_tinyllm_aligned/summary_multiseed.json

Writes:
  - tables/tinyllm_drift_reward_cert_table.tex

Usage:
  python codes/scripts/make_tinyllm_drift_table.py --root .
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_summary(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _mean_ci(values: list[float]) -> tuple[float, float]:
    # Import locally to keep this script lightweight.
    from src.plotting import mean_ci_scalar

    m, hw = mean_ci_scalar(values)
    return float(m), float(hw)


def _fmt(mean: float, hw: float, *, digits: int) -> str:
    from src.plotting import format_mean_ci

    return format_mean_ci(float(mean), float(hw), digits=int(digits))


def _collect(summary: dict, schedule: str, field: str) -> list[float]:
    per_seed = summary.get("per_seed", {})
    out: list[float] = []
    for seed, d in per_seed.items():
        if schedule not in d:
            raise KeyError(f"Missing schedule '{schedule}' in per_seed[{seed}]")
        if field not in d[schedule]:
            raise KeyError(f"Missing field '{field}' for schedule '{schedule}' (seed={seed})")
        out.append(float(d[schedule][field]))
    if len(out) == 0:
        raise ValueError(f"No seeds found in summary for schedule='{schedule}'")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".", help="Package root (contains codes/, results/, and tables/)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    code_dir = root / "codes"
    # Make imports (src.plotting) work.
    import sys

    sys.path.insert(0, str(code_dir))

    spurious = _load_summary(root / "results" / "rlhf_tinyllm" / "summary_multiseed.json")
    aligned = _load_summary(root / "results" / "rlhf_tinyllm_aligned" / "summary_multiseed.json")

    schedules = ["monitor_only", "adaptive"]
    # Rewards are in [0,1] in the saved summaries; convert to percent (without '%').
    digits_reward = 2
    digits_gap = 4

    rows = []
    for drift_label, summ in [
        (r"$\beta<0$ (concise)", spurious),
        (r"$\beta>0$ (verbose)", aligned),
    ]:
        for sched in schedules:
            r_vals = _collect(summ, sched, "final_reward")
            g_vals = _collect(summ, sched, "final_gap")
            r_m, r_hw = _mean_ci(r_vals)
            g_m, g_hw = _mean_ci(g_vals)
            rows.append(
                dict(
                    drift=drift_label,
                    schedule=sched,
                    reward=_fmt(100.0 * r_m, 100.0 * r_hw, digits=digits_reward),
                    gap=_fmt(g_m, g_hw, digits=digits_gap),
                )
            )

    # Pretty schedule names for LaTeX.
    sched_name = {
        "monitor_only": "Monitoring only",
        "adaptive": "Adaptive bursts",
    }

    lines: list[str] = []
    lines.append(r"\begin{tabular}{l l c c}")
    lines.append(r"\toprule")
    lines.append(r"Drift sign & Schedule & final reward $\uparrow$ & final stationarity gap $\downarrow$\\")
    lines.append(r"\midrule")

    # Group by drift sign (two blocks).
    for drift_label in [r"$\beta<0$ (concise)", r"$\beta>0$ (verbose)"]:
        for row in [rr for rr in rows if rr["drift"] == drift_label]:
            # NOTE: LaTeX row termination needs a literal "\\".
            lines.append(
                f"{row['drift']} & {sched_name[row['schedule']]} & {row['reward']} & {row['gap']}\\\\"
            )
        lines.append(r"\midrule")

    # Replace last midrule with bottomrule.
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")

    out_dir = root / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tinyllm_drift_reward_cert_table.tex"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[make_tinyllm_drift_table] Wrote {out_path}")


if __name__ == "__main__":
    main()
