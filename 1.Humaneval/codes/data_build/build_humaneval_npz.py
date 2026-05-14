from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from data_build.prompts import load_strategies


def code_root() -> Path:
    return Path(__file__).resolve().parents[1]


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_evalplus_path() -> Path:
    return project_root() / "test.jsonl"


def default_eval_dir() -> Path:
    return code_root() / "intermediate" / "humaneval_eval"


def default_output_path() -> Path:
    return code_root() / "data" / "humaneval_solutions.npz"


def load_evalplus_tasks(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_strategy_eval(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_ordered_strategy_payloads(eval_dir: Path, expected_strategy_names: list[str]) -> list[dict[str, Any]]:
    found = {path.stem: path for path in eval_dir.glob("*.json")}
    missing = [name for name in expected_strategy_names if name not in found]
    if missing:
        raise ValueError(
            f"Incomplete evaluated artifacts under {eval_dir}: missing strategy files {missing}"
        )
    return [load_strategy_eval(found[name]) for name in expected_strategy_names]


def build_npz_from_eval(evalplus_tasks: list[dict[str, Any]], strategy_payloads: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    problem_ids = np.array([task["task_id"] for task in evalplus_tasks], dtype="U64")
    strategy_names = np.array([payload["strategy_name"] for payload in strategy_payloads], dtype="U64")
    test_counts_list = []
    by_strategy = []

    for payload in strategy_payloads:
        rows = payload["results"]
        if len(rows) != len(problem_ids):
            raise ValueError(f"Strategy {payload['strategy_name']} has {len(rows)} tasks, expected {len(problem_ids)}")
        row_map = {row["task_id"]: row["pass_fail"] for row in rows}
        ordered_rows = []
        for task_id in problem_ids:
            if task_id not in row_map:
                raise ValueError(f"Strategy {payload['strategy_name']} is missing task {task_id}")
            ordered_rows.append(row_map[str(task_id)])
        by_strategy.append(ordered_rows)

    for problem_idx in range(len(problem_ids)):
        test_count = len(by_strategy[0][problem_idx])
        test_counts_list.append(test_count)
        for strat_idx in range(1, len(by_strategy)):
            if len(by_strategy[strat_idx][problem_idx]) != test_count:
                raise ValueError(f"Problem {problem_ids[problem_idx]} has inconsistent test counts across strategies")

    test_counts = np.array(test_counts_list, dtype=np.int32)
    m_max = int(test_counts.max())
    pass_fail = np.zeros((len(strategy_payloads), len(problem_ids), m_max), dtype=np.uint8)
    test_difficulties = np.full((len(problem_ids), m_max), np.nan, dtype=np.float32)

    for strat_idx, ordered_rows in enumerate(by_strategy):
        for problem_idx, row in enumerate(ordered_rows):
            valid = len(row)
            pass_fail[strat_idx, problem_idx, :valid] = np.asarray(row, dtype=np.uint8)

    for problem_idx, valid in enumerate(test_counts.astype(int)):
        test_difficulties[problem_idx, :valid] = pass_fail[:, problem_idx, :valid].mean(axis=0, dtype=np.float64)

    return {
        "pass_fail": pass_fail,
        "test_counts": test_counts,
        "strategy_names": strategy_names,
        "problem_ids": problem_ids,
        "test_difficulties": test_difficulties,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build humaneval_solutions.npz from evaluated strategy outputs.")
    parser.add_argument("--evalplus-path", type=Path, default=default_evalplus_path())
    parser.add_argument("--eval-dir", type=Path, default=default_eval_dir())
    parser.add_argument("--output-path", type=Path, default=default_output_path())
    args = parser.parse_args()

    evalplus_tasks = load_evalplus_tasks(args.evalplus_path)
    strategy_names = [str(strategy["name"]) for strategy in load_strategies()]
    strategy_payloads = load_ordered_strategy_payloads(args.eval_dir, strategy_names)
    arrays = build_npz_from_eval(evalplus_tasks, strategy_payloads)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_path, **arrays)


if __name__ == "__main__":
    main()