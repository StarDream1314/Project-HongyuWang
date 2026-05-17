"""Summarize ALFWorld validation results.

The script reads frozen trajectory and budget CSV files under ``results/``
and writes a compact CSV/Markdown summary for acceptance checks.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


EXPECTED_DATASET_NAME = "alfworld"


class DatasetMismatchError(ValueError):
    """Raised when a result directory contains non-ALFWorld trajectories."""


@dataclass(frozen=True)
class ResultSpec:
    category: str
    prompt_mode: str
    method: str
    result_dir: str
    schedule: str


RESULT_SPECS = [
    ResultSpec("main", "with_hints", "adaptive_ole", "adaptive", "adaptive"),
    ResultSpec("main", "with_hints", "fixed_low", "fixed_low", "fixed_low"),
    ResultSpec("main", "with_hints", "cheap_only", "cheap_only", "cheap_only"),
    ResultSpec("main", "with_hints", "exp3", "exp3", "exp3"),
    ResultSpec("main", "with_hints", "oracle_high", "oracle_high", "oracle_high"),
    ResultSpec("prompt_ablation", "reduced", "adaptive_ole", "adaptive-ablation study", "adaptive"),
    ResultSpec("prompt_ablation", "reduced", "cheap_only", "cheap_only-ablation study", "cheap_only"),
    ResultSpec("prompt_ablation", "reduced", "fixed_low", "fixed_low-ablation study", "fixed_low"),
    ResultSpec("prompt_ablation", "reduced", "exp3", "exp3-ablation study", "exp3"),
    ResultSpec("prompt_ablation", "reduced", "oracle_high", "oracle_high-ablation study", "oracle_high"),
]


def _read_one_csv(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows[0]


def _as_float(value: str | None, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    return float(value)


def _trajectory_rows(run_dir: Path, schedule: str, seed: str) -> list[dict[str, str]]:
    candidates = sorted(run_dir.glob(f"trajectory_{schedule}_seed{seed}.csv"))
    if not candidates:
        candidates = sorted(run_dir.glob("trajectory_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"missing trajectory CSV under {run_dir}")
    with candidates[0].open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _looks_like_alfworld(rows: list[dict[str, str]]) -> bool:
    if not rows:
        return False
    dataset_names = {str(row.get("dataset_name", "")).strip().lower() for row in rows}
    dataset_names.discard("")
    if dataset_names:
        return dataset_names == {EXPECTED_DATASET_NAME}
    return all(str(row.get("task_id", "")).startswith("alfworld-") for row in rows)


def summarize_one(root: Path, spec: ResultSpec) -> dict[str, object]:
    result_root = root / "results" / spec.result_dir
    if not result_root.exists():
        result_root = root / spec.result_dir
    if not result_root.exists():
        raise FileNotFoundError(result_root)

    budget_paths = sorted(result_root.rglob("budget_analysis.csv"))
    if not budget_paths:
        raise FileNotFoundError(f"no budget_analysis.csv under {result_root}")

    all_trajectory_rows: list[dict[str, str]] = []
    total_calls = 0.0
    seeds: list[str] = []
    for budget_path in budget_paths:
        row = _read_one_csv(budget_path)
        if row.get("schedule") != spec.schedule:
            continue
        dataset_name = str(row.get("dataset_name", "")).strip().lower()
        if dataset_name and dataset_name != EXPECTED_DATASET_NAME:
            continue
        seed = str(row.get("seed", ""))
        trajectory_rows = _trajectory_rows(budget_path.parent, spec.schedule, seed)
        if not _looks_like_alfworld(trajectory_rows):
            continue
        seeds.append(seed)
        total_calls += _as_float(row.get("budget_actual"), _as_float(row.get("total_cost")))
        all_trajectory_rows.extend(trajectory_rows)

    if not all_trajectory_rows:
        raise DatasetMismatchError(
            f"no ALFWorld rows for schedule={spec.schedule} in {result_root}"
        )

    progress_values = [_as_float(row.get("progress")) for row in all_trajectory_rows]
    success_values = [str(row.get("success", "")).strip().lower() in {"true", "1", "yes"} for row in all_trajectory_rows]
    partial_rows = sum(1 for value in progress_values if value < 1.0)
    min_progress = min(progress_values)
    avg_progress = sum(progress_values) / len(progress_values)
    success_rate = sum(1 for value in success_values if value) / len(success_values)

    return {
        "category": spec.category,
        "prompt_mode": spec.prompt_mode,
        "method": spec.method,
        "schedule": spec.schedule,
        "result_dir": spec.result_dir,
        "seeds": " ".join(sorted(seeds, key=int)),
        "rows": len(all_trajectory_rows),
        "success_rate": f"{success_rate:.4f}",
        "avg_progress": f"{avg_progress:.4f}",
        "min_progress": f"{min_progress:.4f}",
        "partial_rows": partial_rows,
        "runtime_llm_calls": int(round(total_calls)),
    }


def _write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("no summary rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "category",
        "prompt_mode",
        "method",
        "rows",
        "success_rate",
        "avg_progress",
        "min_progress",
        "partial_rows",
        "runtime_llm_calls",
    ]
    lines = ["# ALFWorld Final Result Summary", ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row[key]) for key in headers) + " |")
    lines.append("")
    lines.append("Note: this table reports task performance only; OLE certificate coverage is stored with the certificate artifact.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize ALFWorld results.")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--csv", default="results/selected_alfworld_summary.csv")
    parser.add_argument("--md", default="results/selected_alfworld_summary.md")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    rows = []
    for spec in RESULT_SPECS:
        try:
            rows.append(summarize_one(root, spec))
        except (FileNotFoundError, DatasetMismatchError):
            if not args.allow_missing:
                raise
    _write_csv(root / args.csv, rows)
    _write_markdown(root / args.md, rows)
    print(f"wrote {root / args.csv}")
    print(f"wrote {root / args.md}")


if __name__ == "__main__":
    main()
