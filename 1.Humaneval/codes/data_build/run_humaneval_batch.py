from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from data_build.prompts import load_strategies

REQUIRED_ENV_VARS = (
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_MODEL_ID",
)


@dataclass
class StrategyState:
    name: str
    count: int
    total: int
    consecutive_failures: int = 0
    backoff_until: float = 0.0
    last_progress_at: float = 0.0

    @property
    def remaining(self) -> int:
        return max(0, self.total - self.count)

    @property
    def complete(self) -> bool:
        return self.count >= self.total


@dataclass
class ActiveAttempt:
    strategy: str
    start_count: int
    target_count: int
    process: subprocess.Popen[str]
    started_at: float
    command: list[str]


def code_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_candidates_dir(code_dir: Path) -> Path:
    return code_dir / "intermediate" / "humaneval_candidates"


def default_logs_dir(code_dir: Path) -> Path:
    return code_dir / "intermediate" / "logs"


def load_local_env() -> None:
    env_path = Path(__file__).resolve().parents[3] / ".env.local"
    if not env_path.exists():
        raise RuntimeError(f"Missing local environment file: {env_path}")
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise RuntimeError("Missing dependency: python-dotenv. Run `python -m pip install -r requirements.txt`.") from exc
    load_dotenv(env_path, override=False)


def load_env_var(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if value:
        return value
    raise RuntimeError(f"Missing required variable in .env.local: {name}")


def build_child_env() -> dict[str, str]:
    load_local_env()
    env = os.environ.copy()
    for name in REQUIRED_ENV_VARS:
        env[name] = load_env_var(name)
    return env


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def load_strategy_names(strategies_path: Path | None = None) -> list[str]:
    return [strategy["name"] for strategy in load_strategies(strategies_path)]


def build_state_map(code_dir: Path, strategy_names: list[str], total: int) -> dict[str, StrategyState]:
    base = default_candidates_dir(code_dir)
    return {
        name: StrategyState(name=name, count=count_lines(base / f"{name}.jsonl"), total=total)
        for name in strategy_names
    }


def refresh_state_counts(states: dict[str, StrategyState], code_dir: Path, now: float) -> list[tuple[str, int, int]]:
    updates: list[tuple[str, int, int]] = []
    base = default_candidates_dir(code_dir)
    for state in states.values():
        current = count_lines(base / f"{state.name}.jsonl")
        if current > state.count:
            delta = current - state.count
            state.count = current
            state.consecutive_failures = 0
            state.backoff_until = 0.0
            state.last_progress_at = now
            updates.append((state.name, delta, current))
        elif current != state.count:
            state.count = current
    return updates


def select_runnable_strategies(
    states: dict[str, StrategyState],
    active_names: set[str],
    max_parallel: int,
    now: float,
) -> list[StrategyState]:
    if max_parallel <= len(active_names):
        return []
    candidates = [
        state
        for state in states.values()
        if not state.complete and state.name not in active_names and now >= state.backoff_until
    ]
    candidates.sort(key=lambda state: (-state.remaining, state.count, state.name))
    return candidates[: max_parallel - len(active_names)]


def compute_target_count(state: StrategyState, step_size: int) -> int:
    return min(state.count + step_size, state.total)


def iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def total_progress(states: dict[str, StrategyState]) -> tuple[int, int]:
    done = sum(state.count for state in states.values())
    target = sum(state.total for state in states.values())
    return done, target


def format_incomplete_counts(states: dict[str, StrategyState]) -> str:
    incomplete = [state for state in states.values() if not state.complete]
    incomplete.sort(key=lambda state: (-state.remaining, state.count, state.name))
    if not incomplete:
        return "none"
    return ", ".join(f"{state.name}={state.count}/{state.total}" for state in incomplete)


def format_progress_line(
    *,
    timestamp: str,
    states: dict[str, StrategyState],
    active_attempts: dict[str, ActiveAttempt],
) -> str:
    done, target = total_progress(states)
    completed = sum(1 for state in states.values() if state.complete)
    active = ", ".join(
        f"{name}->{attempt.target_count}"
        for name, attempt in sorted(active_attempts.items(), key=lambda item: item[0])
    )
    if not active:
        active = "none"
    return (
        f"{timestamp} | total={done}/{target} | complete={completed}/{len(states)} | "
        f"active={len(active_attempts)} [{active}] | incomplete={format_incomplete_counts(states)}"
    )


def append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def append_text(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def write_event(path: Path, event: str, **payload: object) -> None:
    record = {"event": event, "timestamp": iso_now()}
    record.update(payload)
    append_jsonl(path, record)


def write_summary(
    *,
    json_log: Path,
    progress_log: Path,
    states: dict[str, StrategyState],
    active_attempts: dict[str, ActiveAttempt],
    reason: str,
) -> None:
    line = format_progress_line(timestamp=iso_now(), states=states, active_attempts=active_attempts)
    append_text(progress_log, line)
    write_event(
        json_log,
        "summary_tick",
        reason=reason,
        counts={name: state.count for name, state in states.items()},
        active_targets={name: attempt.target_count for name, attempt in active_attempts.items()},
    )


def acquire_lock(lock_path: Path, force: bool) -> None:
    if lock_path.exists():
        if not force:
            raise RuntimeError(
                f"Lock file already exists: {lock_path}. "
                "Use --force-lock if you are sure no other runner instance is active."
            )
        lock_path.unlink()
    append_jsonl(lock_path, {"pid": os.getpid(), "argv": sys.argv, "timestamp": iso_now()})


def release_lock(lock_path: Path) -> None:
    if lock_path.exists():
        lock_path.unlink()


def build_command(python_exe: str, strategy: str, target_count: int) -> list[str]:
    return [
        python_exe,
        "-m",
        "data_build.generate_humaneval_candidates",
        "--strategy",
        strategy,
        "--limit",
        str(target_count),
    ]


def launch_attempt(
    *,
    code_dir: Path,
    python_exe: str,
    state: StrategyState,
    step_size: int,
    env: dict[str, str],
) -> ActiveAttempt:
    target_count = compute_target_count(state, step_size)
    command = build_command(python_exe, state.name, target_count)
    process = subprocess.Popen(
        command,
        cwd=code_dir,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return ActiveAttempt(
        strategy=state.name,
        start_count=state.count,
        target_count=target_count,
        process=process,
        started_at=time.monotonic(),
        command=command,
    )


def finalize_attempt(
    *,
    attempt: ActiveAttempt,
    states: dict[str, StrategyState],
    code_dir: Path,
    max_retries: int,
    stall_seconds: int,
    stdout: str,
    stderr: str,
    detail: str,
    timed_out: bool,
    json_log: Path,
) -> tuple[str, int]:
    now = time.monotonic()
    state = states[attempt.strategy]
    updated = count_lines(default_candidates_dir(code_dir) / f"{attempt.strategy}.jsonl")
    delta = updated - attempt.start_count
    state.count = updated

    if delta > 0:
        state.consecutive_failures = 0
        state.backoff_until = 0.0
        state.last_progress_at = now
        write_event(
            json_log,
            "task_finish",
            strategy=attempt.strategy,
            status="progress",
            start_count=attempt.start_count,
            target_count=attempt.target_count,
            end_count=updated,
            delta=delta,
            returncode=attempt.process.returncode,
        )
        return "progress", delta

    state.consecutive_failures += 1
    stalled = state.consecutive_failures >= max_retries
    backoff_until = 0.0
    if stalled:
        backoff_until = now + stall_seconds
        state.backoff_until = backoff_until
        state.consecutive_failures = 0

    write_event(
        json_log,
        "task_finish",
        strategy=attempt.strategy,
        status="timeout" if timed_out else "no_progress",
        start_count=attempt.start_count,
        target_count=attempt.target_count,
        end_count=updated,
        delta=0,
        returncode=attempt.process.returncode,
        detail=detail,
        stderr=stderr.strip()[:400],
        stdout=stdout.strip()[:400],
        stalled=stalled,
        backoff_until=backoff_until or None,
    )
    return "stalled" if stalled else "retry", 0


def terminate_process(process: subprocess.Popen[str]) -> tuple[str, str]:
    process.terminate()
    try:
        return process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        return process.communicate(timeout=5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume incomplete HumanEval candidate strategies with bounded parallelism and progress logs."
    )
    parser.add_argument("--code-dir", type=Path, default=code_root())
    parser.add_argument("--python", dest="python_exe", default=sys.executable)
    parser.add_argument("--total", type=int, default=164)
    parser.add_argument("--step-size", type=int, default=1)
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--summary-seconds", type=int, default=60)
    parser.add_argument("--attempt-timeout", type=int, default=900)
    parser.add_argument("--stall-seconds", type=int, default=180)
    parser.add_argument("--strategies-path", type=Path, default=None)
    parser.add_argument("--strategy", action="append", dest="strategies", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--force-lock", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.total <= 0:
        raise ValueError("--total must be positive")
    if args.step_size <= 0:
        raise ValueError("--step-size must be positive")
    if args.max_parallel <= 0:
        raise ValueError("--max-parallel must be positive")
    if args.max_retries <= 0:
        raise ValueError("--max-retries must be positive")
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")
    if args.summary_seconds <= 0:
        raise ValueError("--summary-seconds must be positive")
    if args.attempt_timeout <= 0:
        raise ValueError("--attempt-timeout must be positive")
    if args.stall_seconds < 0:
        raise ValueError("--stall-seconds must be non-negative")
    if not args.code_dir.exists():
        raise FileNotFoundError(f"Code directory not found: {args.code_dir}")
    generator = args.code_dir / "data_build" / "generate_humaneval_candidates.py"
    if not generator.exists():
        raise FileNotFoundError(f"Generator script not found: {generator}")


def resolve_strategy_names(args: argparse.Namespace) -> list[str]:
    configured = load_strategy_names(args.strategies_path)
    if args.strategies is None:
        return configured
    unknown = [name for name in args.strategies if name not in configured]
    if unknown:
        raise ValueError(f"Unknown strategy names: {', '.join(unknown)}")
    seen: set[str] = set()
    ordered: list[str] = []
    for name in args.strategies:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def print_banner(
    *,
    args: argparse.Namespace,
    strategy_names: list[str],
    states: dict[str, StrategyState],
    json_log: Path,
    progress_log: Path,
) -> None:
    print(f"code_dir={args.code_dir}")
    print(f"python={args.python_exe}")
    print(f"strategies={', '.join(strategy_names)}")
    print(
        "settings="
        f"total:{args.total}, step_size:{args.step_size}, max_parallel:{args.max_parallel}, "
        f"max_retries:{args.max_retries}, timeout:{args.attempt_timeout}s"
    )
    print(f"json_log={json_log}")
    print(f"progress_log={progress_log}")
    print(format_progress_line(timestamp=iso_now(), states=states, active_attempts={}))


def main() -> int:
    args = parse_args()
    validate_args(args)
    strategy_names = resolve_strategy_names(args)
    states = build_state_map(args.code_dir, strategy_names, args.total)

    logs_dir = default_logs_dir(args.code_dir)
    json_log = logs_dir / "humaneval_runner.jsonl"
    progress_log = logs_dir / "humaneval_progress.log"
    lock_path = logs_dir / "humaneval_runner.lock"

    print_banner(
        args=args,
        strategy_names=strategy_names,
        states=states,
        json_log=json_log,
        progress_log=progress_log,
    )

    if args.dry_run:
        logs_dir.mkdir(parents=True, exist_ok=True)
        write_summary(
            json_log=json_log,
            progress_log=progress_log,
            states=states,
            active_attempts={},
            reason="dry_run",
        )
        write_event(json_log, "dry_run", counts={name: state.count for name, state in states.items()})
        for state in select_runnable_strategies(states, set(), len(states), time.monotonic()):
            target = compute_target_count(state, args.step_size)
            print(f"[dry-run] {state.name}: {state.count}/{state.total} -> limit {target}")
        return 0

    env = build_child_env()
    logs_dir.mkdir(parents=True, exist_ok=True)
    acquire_lock(lock_path, args.force_lock)
    write_event(
        json_log,
        "scan_start",
        strategies=strategy_names,
        counts={name: state.count for name, state in states.items()},
    )

    active_attempts: dict[str, ActiveAttempt] = {}
    last_summary = 0.0
    launched_once: set[str] = set()

    try:
        while True:
            now = time.monotonic()
            progress_updates = refresh_state_counts(states, args.code_dir, now)
            for name, delta, current in progress_updates:
                write_event(
                    json_log,
                    "external_progress",
                    strategy=name,
                    delta=delta,
                    current=current,
                )

            finished_names: list[str] = []
            for name, attempt in list(active_attempts.items()):
                process = attempt.process
                elapsed = now - attempt.started_at
                if process.poll() is None and elapsed <= args.attempt_timeout:
                    continue

                timed_out = process.poll() is None
                if timed_out:
                    stdout, stderr = terminate_process(process)
                    detail = f"timeout after {args.attempt_timeout}s"
                else:
                    stdout, stderr = process.communicate()
                    detail = stderr.strip() or stdout.strip() or f"exit code {process.returncode}"

                status, delta = finalize_attempt(
                    attempt=attempt,
                    states=states,
                    code_dir=args.code_dir,
                    max_retries=args.max_retries,
                    stall_seconds=args.stall_seconds,
                    stdout=stdout,
                    stderr=stderr,
                    detail=detail,
                    timed_out=timed_out,
                    json_log=json_log,
                )
                if status == "progress":
                    print(f"[progress] {name}: +{delta}, now {states[name].count}/{states[name].total}")
                elif status == "stalled":
                    print(f"[backoff] {name}: no durable progress, backing off for {args.stall_seconds}s")
                else:
                    print(f"[retry] {name}: no durable progress")
                finished_names.append(name)

            for name in finished_names:
                active_attempts.pop(name, None)

            if all(state.complete for state in states.values()):
                write_summary(
                    json_log=json_log,
                    progress_log=progress_log,
                    states=states,
                    active_attempts=active_attempts,
                    reason="all_done",
                )
                write_event(json_log, "all_done", counts={name: state.count for name, state in states.items()})
                print("all strategies complete")
                return 0

            runnable = select_runnable_strategies(states, set(active_attempts), args.max_parallel, now)
            if args.once:
                runnable = [state for state in runnable if state.name not in launched_once]

            for state in runnable:
                attempt = launch_attempt(
                    code_dir=args.code_dir,
                    python_exe=args.python_exe,
                    state=state,
                    step_size=args.step_size,
                    env=env,
                )
                active_attempts[state.name] = attempt
                launched_once.add(state.name)
                print(f"[launch] {state.name}: {state.count}/{state.total} -> limit {attempt.target_count}")
                write_event(
                    json_log,
                    "task_launch",
                    strategy=state.name,
                    start_count=state.count,
                    target_count=attempt.target_count,
                    pid=attempt.process.pid,
                    command=attempt.command,
                )

            if now - last_summary >= args.summary_seconds:
                write_summary(
                    json_log=json_log,
                    progress_log=progress_log,
                    states=states,
                    active_attempts=active_attempts,
                    reason="tick",
                )
                last_summary = now

            if args.once and not active_attempts:
                write_summary(
                    json_log=json_log,
                    progress_log=progress_log,
                    states=states,
                    active_attempts=active_attempts,
                    reason="once_complete",
                )
                return 0

            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        write_summary(
            json_log=json_log,
            progress_log=progress_log,
            states=states,
            active_attempts=active_attempts,
            reason="keyboard_interrupt",
        )
        write_event(json_log, "interrupted", counts={name: state.count for name, state in states.items()})
        print("interrupted")
        return 130
    finally:
        release_lock(lock_path)


if __name__ == "__main__":
    raise SystemExit(main())

