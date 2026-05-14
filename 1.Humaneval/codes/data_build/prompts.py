from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_SYSTEM_PROMPTS = {
    "direct": (
        "You solve Python programming tasks. Return only valid Python code for the target function. "
        "Do not include markdown fences, explanations, or tests."
    ),
    "careful": (
        "You solve Python programming tasks with strong attention to corner cases, types, and exact behavior. "
        "Return only valid Python code for the target function. Do not include markdown fences or explanations."
    ),
    "reasoned": (
        "You may reason internally before answering, but your visible response must contain only the final valid "
        "Python code for the target function. Do not include markdown fences, explanations, or tests."
    ),
    "minimal": (
        "You solve Python programming tasks by preferring the simplest correct implementation. Return only valid "
        "Python code for the target function. Do not include markdown fences, explanations, or tests."
    ),
}


def default_strategies_path() -> Path:
    return Path(__file__).with_name("strategies.json")


def load_strategies(path: Path | None = None) -> list[dict[str, Any]]:
    strategy_path = default_strategies_path() if path is None else Path(path)
    strategies = json.loads(strategy_path.read_text(encoding="utf-8"))
    if len(strategies) != 20:
        raise ValueError(f"Expected 20 strategies, found {len(strategies)}")
    return strategies


def render_messages(task: dict[str, Any], strategy: dict[str, Any]) -> list[dict[str, str]]:
    family = strategy["prompt_family"]
    if family not in _SYSTEM_PROMPTS:
        raise ValueError(f"Unknown prompt family: {family}")

    prompt = task["prompt"].rstrip()
    user = (
        "Complete the following HumanEval task.\n"
        "Return only the Python function implementation that solves the task.\n\n"
        f"{prompt}\n"
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPTS[family]},
        {"role": "user", "content": user},
    ]
