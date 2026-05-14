from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from data_build.prompts import load_strategies


def _disable_int_str_digit_limit() -> None:
    setter = getattr(sys, "set_int_max_str_digits", None)
    if setter is not None:
        setter(0)


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def code_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_evalplus_path() -> Path:
    return project_root() / "test.jsonl"


def default_candidates_dir() -> Path:
    return code_root() / "intermediate" / "humaneval_candidates"


def default_eval_dir() -> Path:
    return code_root() / "intermediate" / "humaneval_eval"


def load_evalplus_tasks(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_candidate_file(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as handle:
        return {record["task_id"]: record["completion"] for record in map(json.loads, handle) if record}


def resolve_candidate_paths(candidates_dir: Path, expected_strategy_names: list[str]) -> list[Path]:
    found = {path.stem: path for path in candidates_dir.glob("*.jsonl")}
    missing = [name for name in expected_strategy_names if name not in found]
    if missing:
        raise ValueError(
            f"Incomplete candidate outputs under {candidates_dir}: missing strategy files {missing}"
        )
    return [found[name] for name in expected_strategy_names]


def _literal_expr(node: ast.AST) -> Any:
    return ast.literal_eval(node)


def _compute_results_via_ref_func(test_code: str, inputs: list[Any]) -> list[Any]:
    namespace: dict[str, Any] = {}
    exec(test_code, namespace, namespace)
    ref_func = namespace.get("ref_func")
    if not callable(ref_func):
        raise ValueError("Expanded test code is missing results assignments and ref_func")
    return [ref_func(*inp) for inp in inputs]


def parse_expanded_cases(test_code: str) -> dict[str, Any]:
    module = ast.parse(test_code)
    check_fn = None
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == "check":
            check_fn = node
            break
    if check_fn is None:
        raise ValueError("Expanded test code does not define check(candidate)")

    inputs = None
    results = None
    atol = 1e-6
    for node in ast.walk(check_fn):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "inputs":
                    inputs = _literal_expr(node.value)
                elif isinstance(target, ast.Name) and target.id == "results":
                    results = _literal_expr(node.value)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "assertion":
            if len(node.args) >= 3:
                atol = float(_literal_expr(node.args[2]))

    if inputs is None:
        raise ValueError("Expanded test code is missing inputs/results assignments")
    if results is None:
        results = _compute_results_via_ref_func(test_code, inputs)
    if len(inputs) != len(results):
        raise ValueError("Expanded test code has mismatched inputs/results lengths")
    return {"inputs": inputs, "results": results, "atol": atol}


def _worker_script() -> str:
    return """
import json
import sys

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)

import numpy as np

payload = json.loads(sys.stdin.read())
namespace = {}
exec(payload["candidate_code"], namespace, namespace)
candidate = namespace[payload["entry_point"]]

def is_floats(x):
    if isinstance(x, float):
        return True
    if isinstance(x, (list, tuple)):
        return all(isinstance(i, float) for i in x)
    if isinstance(x, np.ndarray):
        return x.dtype == np.float64 or x.dtype == np.float32
    return False

def assertion(out, exp, atol):
    exact_match = out == exp
    if atol == 0 and is_floats(exp):
        atol = 1e-6
    if not exact_match and atol != 0:
        np.testing.assert_allclose(out, exp, atol=atol)
    else:
        assert exact_match

passed = []
for inp, exp in zip(payload["inputs"], payload["results"]):
    try:
        result = candidate(*inp)
        assertion(result, exp, payload["atol"])
        passed.append(1)
    except Exception:
        passed.append(0)
print(json.dumps({"pass_fail": passed}))
"""


def evaluate_completion(entry_point: str, completion: str, cases: dict[str, Any], timeout_s: float = 10.0) -> list[int]:
    _disable_int_str_digit_limit()
    payload = {
        "entry_point": entry_point,
        "candidate_code": completion,
        "inputs": cases["inputs"],
        "results": cases["results"],
        "atol": cases["atol"],
    }
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _worker_script()],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return [0] * len(cases["inputs"])
    if proc.returncode != 0:
        return [0] * len(cases["inputs"])
    return json.loads(proc.stdout)["pass_fail"]


def evaluate_strategy(
    candidate_path: Path, evalplus_tasks: list[dict[str, Any]], out_path: Path, timeout_s: float = 10.0
) -> None:
    completions = load_candidate_file(candidate_path)
    expected_task_ids = [str(task["task_id"]) for task in evalplus_tasks]
    missing_task_ids = [task_id for task_id in expected_task_ids if task_id not in completions]
    if missing_task_ids:
        raise ValueError(
            f"Candidate file {candidate_path} is incomplete: missing {len(missing_task_ids)} task_ids; "
            f"first missing task_id is {missing_task_ids[0]}"
        )

    results = []
    for task in evalplus_tasks:
        task_id = str(task["task_id"])
        cases = parse_expanded_cases(task["test"])
        passed = evaluate_completion(task["entry_point"], completions[task_id], cases, timeout_s=timeout_s)
        results.append({"task_id": task_id, "pass_fail": passed})

    _disable_int_str_digit_limit()
    payload = {"strategy_name": candidate_path.stem, "results": results}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HumanEval candidates on expanded tests.")
    parser.add_argument("--evalplus-path", type=Path, default=default_evalplus_path())
    parser.add_argument("--candidates-dir", type=Path, default=default_candidates_dir())
    parser.add_argument("--out-dir", type=Path, default=default_eval_dir())
    parser.add_argument("--timeout-s", type=float, default=10.0)
    args = parser.parse_args()

    evalplus_tasks = load_evalplus_tasks(args.evalplus_path)
    strategy_names = [str(strategy["name"]) for strategy in load_strategies()]
    for candidate_path in resolve_candidate_paths(args.candidates_dir, strategy_names):
        out_path = args.out_dir / f"{candidate_path.stem}.json"
        evaluate_strategy(candidate_path, evalplus_tasks, out_path, timeout_s=args.timeout_s)


if __name__ == "__main__":
    main()


