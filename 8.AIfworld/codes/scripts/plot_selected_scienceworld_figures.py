"""Plot selected ALFWorld final-result figures.

Reads ``results/selected_alfworld_summary.csv`` and writes four PDF
figures under ``figs/``:

- Figure 1: Avg Progress by Scheduler
- Figure 2: Partial Rows by Scheduler
- Figure 3: Prompt Ablation: With Hints vs Reduced
- Figure 4: LLM Calls vs Avg Progress
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHOD_LABELS = {
    "adaptive_ole": "Adaptive OLE",
    "fixed_low": "Fixed Low",
    "cheap_only": "Cheap Only",
    "exp3": "EXP3",
    "oracle_high": "Oracle High",
}

METHOD_COLORS = {
    "adaptive_ole": "#2E7D32",
    "fixed_low": "#4E79A7",
    "cheap_only": "#9C755F",
    "exp3": "#E15759",
    "oracle_high": "#59A14F",
}

MAIN_ORDER = ["adaptive_ole", "fixed_low", "cheap_only", "exp3", "oracle_high"]
ABLATION_ORDER = ["adaptive_ole", "fixed_low", "cheap_only", "exp3", "oracle_high"]
PROMPT_ABLATION_FIGURE = "Figure3-Prompt Ablation-With Hints vs Reduced.pdf"


Row = dict[str, str]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_rows(path: Path) -> list[Row]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _as_float(row: Row, key: str) -> float:
    return float(str(row[key]).strip())


def _as_int(row: Row, key: str) -> int:
    return int(round(float(str(row[key]).strip())))


def _select(rows: Iterable[Row], *, category: str, prompt_mode: str | None = None) -> list[Row]:
    selected = [row for row in rows if row.get("category") == category]
    if prompt_mode is not None:
        selected = [row for row in selected if row.get("prompt_mode") == prompt_mode]
    return selected


def _order_rows(rows: Iterable[Row], order: list[str]) -> list[Row]:
    by_method = {row["method"]: row for row in rows}
    missing = [method for method in order if method not in by_method]
    if missing:
        raise ValueError(f"Missing expected methods in summary CSV: {', '.join(missing)}")
    return [by_method[method] for method in order]


def _has_methods(rows: Iterable[Row], order: list[str]) -> bool:
    methods = {row.get("method") for row in rows}
    return all(method in methods for method in order)


def _padded_limits(values: list[float], *, lower_bound: float = 0.0, upper_bound: float | None = None) -> tuple[float, float]:
    if not values:
        return lower_bound, upper_bound if upper_bound is not None else 1.0
    min_value = min(values)
    max_value = max(values)
    span = max(max_value - min_value, 0.05)
    low = max(lower_bound, min_value - span * 0.2)
    high = max_value + span * 0.2
    if upper_bound is not None:
        high = min(upper_bound, high)
    if low >= high:
        high = low + 0.05
    return low, high


def _style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_avg_progress(rows: list[Row], output_dir: Path) -> Path:
    main_rows = _order_rows(_select(rows, category="main", prompt_mode="with_hints"), MAIN_ORDER)
    labels = [METHOD_LABELS[row["method"]] for row in main_rows]
    values = [_as_float(row, "avg_progress") for row in main_rows]
    colors = [METHOD_COLORS[row["method"]] for row in main_rows]

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    bars = ax.bar(labels, values, color=colors, width=0.64)
    ax.set_title("Figure 1: Avg Progress by Scheduler", fontsize=14, pad=14)
    ax.set_ylabel("Avg Progress")
    ax.set_ylim(*_padded_limits(values, upper_bound=1.05))
    ax.tick_params(axis="x", rotation=20)
    _style_axis(ax)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.006,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    path = output_dir / "Figure1-Avg Progress by Scheduler.pdf"
    _save(fig, path)
    return path


def plot_partial_rows(rows: list[Row], output_dir: Path) -> Path:
    main_rows = _order_rows(_select(rows, category="main", prompt_mode="with_hints"), MAIN_ORDER)
    labels = [METHOD_LABELS[row["method"]] for row in main_rows]
    values = [_as_int(row, "partial_rows") for row in main_rows]
    colors = [METHOD_COLORS[row["method"]] for row in main_rows]

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    bars = ax.bar(labels, values, color=colors, width=0.64)
    ax.set_title("Figure 2: Partial Rows by Scheduler", fontsize=14, pad=14)
    ax.set_ylabel("Rows with Progress < 1.0")
    ax.set_ylim(0, max(values) + 6)
    ax.tick_params(axis="x", rotation=20)
    _style_axis(ax)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.5,
            str(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    path = output_dir / "Figure2-Partial Rows by Scheduler.pdf"
    _save(fig, path)
    return path


def plot_prompt_ablation(rows: list[Row], output_dir: Path) -> Path:
    main_candidates = _select(rows, category="main", prompt_mode="with_hints")
    reduced_candidates = _select(rows, category="prompt_ablation", prompt_mode="reduced")
    if not _has_methods(main_candidates, ABLATION_ORDER) or not _has_methods(reduced_candidates, ABLATION_ORDER):
        raise ValueError("Missing ALFWorld reduced prompt-ablation rows in summary CSV")
    main_rows = _order_rows(main_candidates, ABLATION_ORDER)
    reduced_rows = _order_rows(reduced_candidates, ABLATION_ORDER)

    labels = [METHOD_LABELS[row["method"]] for row in main_rows]
    with_hints = [_as_float(row, "avg_progress") for row in main_rows]
    reduced = [_as_float(row, "avg_progress") for row in reduced_rows]
    x = list(range(len(labels)))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    bars_a = ax.bar([pos - width / 2 for pos in x], with_hints, width, label="With hints", color="#4E79A7")
    bars_b = ax.bar([pos + width / 2 for pos in x], reduced, width, label="Reduced", color="#F28E2B")
    ax.set_title("Figure 3: Prompt Ablation: With Hints vs Reduced", fontsize=14, pad=24)
    ax.set_ylabel("Avg Progress")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylim(*_padded_limits(with_hints + reduced, upper_bound=1.08))
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=2,
        borderaxespad=0.0,
    )
    _style_axis(ax)

    for bars in (bars_a, bars_b):
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.006,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    path = output_dir / PROMPT_ABLATION_FIGURE
    _save(fig, path)
    return path


def plot_llm_calls_vs_progress(rows: list[Row], output_dir: Path) -> Path:
    main_rows = _order_rows(_select(rows, category="main", prompt_mode="with_hints"), MAIN_ORDER)
    fig, ax = plt.subplots(figsize=(7.4, 5.2))

    for row in main_rows:
        method = row["method"]
        x_value = _as_int(row, "runtime_llm_calls")
        y_value = _as_float(row, "avg_progress")
        ax.scatter(
            x_value,
            y_value,
            s=95,
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
        )
        label = METHOD_LABELS[method]
        offset = (6, 6)
        if method == "oracle_high":
            offset = (6, -14)
        if method == "adaptive_ole":
            offset = (6, 8)
        ax.annotate(
            label,
            (x_value, y_value),
            textcoords="offset points",
            xytext=offset,
            fontsize=9,
        )

    ax.set_title("Figure 4: LLM Calls vs Avg Progress", fontsize=14, pad=14)
    ax.set_xlabel("Runtime LLM Calls")
    ax.set_ylabel("Avg Progress")
    x_values = [_as_int(row, "runtime_llm_calls") for row in main_rows]
    y_values = [_as_float(row, "avg_progress") for row in main_rows]
    x_low, x_high = _padded_limits([float(value) for value in x_values])
    ax.set_xlim(max(0, x_low), x_high)
    ax.set_ylim(*_padded_limits(y_values, upper_bound=1.05))
    _style_axis(ax)

    path = output_dir / "Figure4-LLM Calls vs Avg Progress.pdf"
    _save(fig, path)
    return path

def main() -> None:
    root = _repo_root()
    parser = argparse.ArgumentParser(description="Plot selected ALFWorld final-result figures.")
    parser.add_argument("--summary", default=str(root / "results" / "selected_alfworld_summary.csv"))
    parser.add_argument("--output-dir", default=str(root / "figs"))
    args = parser.parse_args()

    summary_path = Path(args.summary)
    output_dir = Path(args.output_dir)
    rows = _read_rows(summary_path)

    paths = [
        plot_avg_progress(rows, output_dir),
        plot_partial_rows(rows, output_dir),
        plot_llm_calls_vs_progress(rows, output_dir),
    ]
    try:
        paths.append(plot_prompt_ablation(rows, output_dir))
    except ValueError as exc:
        stale_path = output_dir / PROMPT_ABLATION_FIGURE
        if stale_path.exists():
            stale_path.unlink()
        print(f"skipped prompt ablation figure: {exc}")
    for path in paths:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
