from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any, Callable

from data_build.api_client import DeepSeekClient
from data_build.prompts import load_strategies, render_messages


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def code_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_humaneval_path() -> Path:
    return project_root() / "HumanEval.jsonl.gz"


def default_candidates_dir() -> Path:
    return code_root() / "intermediate" / "humaneval_candidates"


def load_humaneval_tasks(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_existing_task_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as handle:
        return {json.loads(line)["task_id"] for line in handle if line.strip()}


def generate_for_strategy(
    strategy: dict[str, Any],
    tasks: list[dict[str, Any]],
    out_path: Path,
    complete_fn: Callable[[list[dict[str, str]], float], str],
) -> None:
    done = read_existing_task_ids(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as handle:
        for task in tasks:
            task_id = task["task_id"]
            if task_id in done:
                continue
            completion = complete_fn(render_messages(task, strategy), float(strategy["temperature"]))
            record = {"task_id": task_id, "completion": completion}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate HumanEval candidates for all approved strategies.")
    parser.add_argument("--humaneval-path", type=Path, default=default_humaneval_path())
    parser.add_argument("--strategies-path", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=default_candidates_dir())
    parser.add_argument("--strategy", type=str, default=None, help="Run only one named strategy.")
    parser.add_argument("--limit", type=int, default=None, help="Limit to the first N HumanEval tasks.")
    args = parser.parse_args()

    tasks = load_humaneval_tasks(args.humaneval_path)
    if args.limit is not None:
        tasks = tasks[: max(0, int(args.limit))]

    strategies = load_strategies(args.strategies_path)
    if args.strategy is not None:
        strategies = [strategy for strategy in strategies if strategy["name"] == args.strategy]
        if not strategies:
            raise ValueError(f"Unknown strategy: {args.strategy}")

    client = DeepSeekClient.from_env()
    for strategy in strategies:
        out_path = args.out_dir / f"{strategy['name']}.jsonl"
        generate_for_strategy(strategy, tasks, out_path, client.complete)


if __name__ == "__main__":
    main()
