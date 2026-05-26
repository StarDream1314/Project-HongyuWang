"""Minimal experiment runner and Phase 2 adapters for A-OMP-Mem."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

LEGACY_SCHEMA_VERSION = "phase1-legacy"
PHASE2_SCHEMA_VERSION = "phase2"
LEGACY_TRAJECTORY_FIELDNAMES = [
    "task_id",
    "success",
    "progress",
    "n_cheap",
    "n_refine",
    "cum_cost",
    "memory_size",
    "drift_detected",
]
PHASE2_TRAJECTORY_EXTRA_FIELDNAMES = [
    "coverage_t",
    "precision_t",
    "no_retrieval_window_flag_t",
    "redundancy_t",
    "freshness_t",
    "mqc_t",
    "ole_mean_t",
    "ole_var_t",
    "ole_ucb_t",
    "cusum_stat_t",
    "drift_onset_flag_t",
    "retries_t",
]
PHASE2_TRAJECTORY_FIELDNAMES = LEGACY_TRAJECTORY_FIELDNAMES + PHASE2_TRAJECTORY_EXTRA_FIELDNAMES


@dataclass(frozen=True)
class ExperimentConfig:
    """Serializable experiment configuration."""

    schedule: str
    seed: int
    model_id: str
    output_dir: Path
    dataset_name: str = "unknown"
    max_memory_size: int = 500
    top_k: int = 3
    n_mon: int = 1
    n_cal_low: int = 0
    n_cal_high: int = 3
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskSpec:
    """Input task entry compatible with Evo-Memory style streams."""

    task_id: str
    input_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskOutcome:
    """Single-task execution result for output serialization."""

    prediction: str
    success: bool
    progress: float = 0.0
    n_cheap: int = 1
    n_refine: int = 0
    drift_detected: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    memory_size: int = 0
    coverage_t: Optional[float] = None
    precision_t: Optional[float] = None
    no_retrieval_window_flag_t: bool = False
    redundancy_t: Optional[float] = None
    freshness_t: Optional[float] = None
    mqc_t: Optional[float] = None
    ole_mean_t: Optional[float] = None
    ole_var_t: Optional[float] = None
    ole_ucb_t: Optional[float] = None
    cusum_stat_t: Optional[float] = None
    drift_onset_flag_t: bool = False
    retries_t: int = 0


@dataclass(frozen=True)
class ExperimentArtifacts:
    """Paths emitted by one experiment run."""

    config_path: Path
    trajectory_path: Path
    predictions_path: Path
    budget_path: Path


class SeedManager:
    """Deterministic seed spawning backed by SeedSequence."""

    def __init__(self, root_seed: int) -> None:
        self._root = np.random.SeedSequence(int(root_seed))

    def spawn_generators(self, count: int) -> List[np.random.Generator]:
        if count <= 0:
            raise ValueError(f"count must be positive, got {count}")
        return [np.random.default_rng(child) for child in self._root.spawn(count)]


class ExperimentRunner:
    """Runs a task stream and writes the required experiment artifacts."""

    def __init__(self, *, config: ExperimentConfig, trajectory_schema: str = LEGACY_SCHEMA_VERSION) -> None:
        if trajectory_schema not in {LEGACY_SCHEMA_VERSION, PHASE2_SCHEMA_VERSION}:
            raise ValueError(f"unsupported trajectory schema: {trajectory_schema}")
        self.config = config
        self.trajectory_schema = trajectory_schema
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        *,
        tasks: Iterable[TaskSpec],
        execute_task: Callable[[TaskSpec, int, np.random.Generator], TaskOutcome],
    ) -> ExperimentArtifacts:
        task_list = list(tasks)
        rng = SeedManager(self.config.seed).spawn_generators(1)[0]

        config_path = self.output_dir / "config.json"
        trajectory_path = self.output_dir / f"trajectory_{self.config.schedule}_seed{self.config.seed}.csv"
        predictions_path = self.output_dir / f"predictions_{self.config.schedule}_seed{self.config.seed}.jsonl"
        budget_path = self.output_dir / "budget_analysis.csv"

        predictions_rows: List[Dict[str, Any]] = []
        trajectory_rows: List[Dict[str, Any]] = []
        total_cost = 0
        total_success = 0
        total_progress = 0.0

        self._write_config(config_path)
        for task_index, task in enumerate(task_list):
            outcome = execute_task(task, task_index, rng)
            total_cost += int(outcome.n_cheap) + int(outcome.n_refine)
            total_success += int(outcome.success)
            total_progress += float(outcome.progress)

            predictions_rows.append(
                {
                    "task_id": task.task_id,
                    "prediction": outcome.prediction,
                    "metadata": dict(outcome.metadata),
                }
            )
            trajectory_rows.append(self._build_trajectory_row(task_id=task.task_id, outcome=outcome, cum_cost=total_cost))

            completed_count = len(trajectory_rows)
            self._write_predictions(predictions_path, predictions_rows)
            self._write_trajectory(trajectory_path, trajectory_rows)
            self._write_budget_analysis(
                budget_path=budget_path,
                task_count=completed_count,
                total_cost=total_cost,
                total_success=total_success,
                total_progress=total_progress,
            )

        self._write_predictions(predictions_path, predictions_rows)
        self._write_trajectory(trajectory_path, trajectory_rows)
        self._write_budget_analysis(
            budget_path=budget_path,
            task_count=len(task_list),
            total_cost=total_cost,
            total_success=total_success,
            total_progress=total_progress,
        )

        return ExperimentArtifacts(
            config_path=config_path,
            trajectory_path=trajectory_path,
            predictions_path=predictions_path,
            budget_path=budget_path,
        )

    def _write_config(self, path: Path) -> None:
        payload = {
            "config": {
                **asdict(self.config),
                "output_dir": str(self.output_dir),
                "trajectory_schema": self.trajectory_schema,
            },
            "versions": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
            },
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    @staticmethod
    def _write_predictions(path: Path, rows: List[Dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _write_trajectory(self, path: Path, rows: List[Dict[str, Any]]) -> None:
        fieldnames = self._trajectory_fieldnames()
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _trajectory_fieldnames(self) -> List[str]:
        if self.trajectory_schema == PHASE2_SCHEMA_VERSION:
            return list(PHASE2_TRAJECTORY_FIELDNAMES)
        return list(LEGACY_TRAJECTORY_FIELDNAMES)

    def _build_trajectory_row(self, *, task_id: str, outcome: TaskOutcome, cum_cost: int) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "task_id": task_id,
            "success": bool(outcome.success),
            "progress": float(outcome.progress),
            "n_cheap": int(outcome.n_cheap),
            "n_refine": int(outcome.n_refine),
            "cum_cost": int(cum_cost),
            "memory_size": int(outcome.memory_size),
            "drift_detected": bool(outcome.drift_detected),
        }
        if self.trajectory_schema == LEGACY_SCHEMA_VERSION:
            return row

        metadata = dict(outcome.metadata)
        row.update(
            {
                "coverage_t": self._phase2_value(metadata, "coverage_t", outcome.coverage_t),
                "precision_t": self._phase2_value(metadata, "precision_t", outcome.precision_t),
                "no_retrieval_window_flag_t": bool(
                    self._phase2_value(
                        metadata,
                        "no_retrieval_window_flag_t",
                        outcome.no_retrieval_window_flag_t,
                    )
                ),
                "redundancy_t": self._phase2_value(metadata, "redundancy_t", outcome.redundancy_t),
                "freshness_t": self._phase2_value(metadata, "freshness_t", outcome.freshness_t),
                "mqc_t": self._phase2_value(metadata, "mqc_t", outcome.mqc_t),
                "ole_mean_t": self._phase2_value(metadata, "ole_mean_t", outcome.ole_mean_t),
                "ole_var_t": self._phase2_value(metadata, "ole_var_t", outcome.ole_var_t),
                "ole_ucb_t": self._phase2_value(metadata, "ole_ucb_t", outcome.ole_ucb_t),
                "cusum_stat_t": self._phase2_value(metadata, "cusum_stat_t", outcome.cusum_stat_t),
                "drift_onset_flag_t": bool(
                    self._phase2_value(metadata, "drift_onset_flag_t", outcome.drift_onset_flag_t)
                ),
                "retries_t": int(self._phase2_value(metadata, "retries_t", outcome.retries_t, fallback=0) or 0),
            }
        )
        return row

    @staticmethod
    def _phase2_value(metadata: Dict[str, Any], key: str, explicit: Any, fallback: Any = None) -> Any:
        if key in metadata:
            return metadata[key]
        if explicit is not None:
            return explicit
        if key == "retries_t" and "retry_count" in metadata:
            return metadata["retry_count"]
        return fallback

    def _write_budget_analysis(
        self,
        *,
        budget_path: Path,
        task_count: int,
        total_cost: int,
        total_success: int,
        total_progress: float,
    ) -> None:
        avg_cost = float(total_cost / task_count) if task_count else 0.0
        success_rate = float(total_success / task_count) if task_count else 0.0
        avg_progress = float(total_progress / task_count) if task_count else 0.0
        efficiency = float(success_rate / avg_cost) if avg_cost > 0 else 0.0

        fieldnames = [
            "schedule",
            "seed",
            "dataset_name",
            "task_count",
            "total_cost",
            "avg_cost",
            "success_rate",
            "avg_progress",
            "efficiency",
        ]
        row = {
            "schedule": self.config.schedule,
            "seed": self.config.seed,
            "dataset_name": self.config.dataset_name,
            "task_count": task_count,
            "total_cost": total_cost,
            "avg_cost": avg_cost,
            "success_rate": success_rate,
            "avg_progress": avg_progress,
            "efficiency": efficiency,
        }
        with budget_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)


def _code_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _default_phase2_results_root() -> Path:
    return _code_root() / "results"


def _load_yaml_budget_matrix(path: Path) -> Dict[str, Dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing budget config: {path}")
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read budget_equivalence.yaml") from exc

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid budget config at {path}")
    return {str(track): {str(unit): float(value) for unit, value in dict(units).items()} for track, units in payload.items()}


def _budget_limit_for(track: str, unit: str, *, config_path: Optional[Path]) -> Optional[float]:
    if config_path is None:
        return None
    matrix = _load_yaml_budget_matrix(config_path)
    return matrix.get(track, {}).get(unit)


def _normalize_exact_match_scheduler(name: str) -> str:
    normalized = str(name).strip().lower()
    aliases = {
        "cheap_only": "monitor_only",
        "monitor_only": "monitor_only",
        "fixed_low": "fixed_low",
        "adaptive": "adaptive",
        "exp3": "exp3",
        "oracle_high": "high",
        "high": "high",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported exact-match scheduler: {name}")
    return aliases[normalized]


def _normalize_task_stream_method(name: str) -> str:
    normalized = str(name).strip().lower()
    aliases = {
        "cheap_only": "cheap_only",
        "monitor_only": "cheap_only",
        "fixed_low": "fixed_low",
        "adaptive": "adaptive",
        "exp3": "exp3",
        "oracle_high": "oracle_high",
        "high": "oracle_high",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported task-stream scheduler: {name}")
    return aliases[normalized]


def _phase2_run_dir(*, track: str, seed: int, scheduler: str, output_root: Path) -> Path:
    if track == "alfworld":
        return output_root / scheduler / f"seed_{int(seed)}" / scheduler
    return output_root / track / f"seed_{int(seed)}" / scheduler


def _alfworld_run_dir(*, seed: int, scheduler: str, prompt_mode: str, output_root: Path) -> Path:
    result_group = _alfworld_result_group(scheduler, prompt_mode)
    return output_root / result_group / f"seed_{int(seed)}" / scheduler


def _alfworld_result_group(scheduler: str, prompt_mode: str) -> str:
    if str(prompt_mode).strip().lower() == "reduced":
        return f"{scheduler}-ablation study"
    return scheduler


def _build_exact_match_tasks(length: int) -> List[TaskSpec]:
    return [TaskSpec(task_id=f"step-{index + 1:04d}", input_text=f"exact-match step {index + 1}") for index in range(length)]


def _build_exact_match_quality_context(schedule_frames: Dict[str, Any]) -> Dict[str, np.ndarray]:
    oracle_frame = schedule_frames["high"]
    oracle_reward = oracle_frame["reward"].to_numpy(dtype=float)
    eps = np.finfo(float).eps

    max_gap = np.maximum.reduce([frame["gap"].to_numpy(dtype=float) for frame in schedule_frames.values()])
    max_cert_hp = np.maximum.reduce([frame["cert_hp"].to_numpy(dtype=float) for frame in schedule_frames.values()])
    max_xi_sq = np.maximum.reduce([frame["xi_sq"].to_numpy(dtype=float) for frame in schedule_frames.values()])

    return {
        "oracle_reward": np.maximum(oracle_reward, eps),
        "max_gap": np.maximum(max_gap, eps),
        "max_cert_hp": np.maximum(max_cert_hp, eps),
        "max_xi_sq": np.maximum(max_xi_sq, eps),
    }


def _derive_exact_match_quality_fields(row: Any, task_index: int, quality_context: Dict[str, np.ndarray]) -> Dict[str, float | bool]:
    coverage = float(np.clip(float(row["reward"]) / float(quality_context["oracle_reward"][task_index]), 0.0, 1.0))
    gap_ratio = float(np.clip(float(row["gap"]) / float(quality_context["max_gap"][task_index]), 0.0, 1.0))
    cert_ratio = float(np.clip(float(row["cert_hp"]) / float(quality_context["max_cert_hp"][task_index]), 0.0, 1.0))
    xi_ratio = float(np.clip(float(row["xi_sq"]) / float(quality_context["max_xi_sq"][task_index]), 0.0, 1.0))

    redundancy = float(np.clip(0.5 * (gap_ratio + cert_ratio), 0.0, 1.0))
    freshness = float(np.clip(1.0 - xi_ratio, 0.0, 1.0))
    precision = float(np.clip(0.5 * (coverage + freshness), 0.0, 1.0))

    return {
        "coverage_t": coverage,
        "precision_t": precision,
        "no_retrieval_window_flag_t": False,
        "redundancy_t": redundancy,
        "freshness_t": freshness,
    }


def _run_exact_match_track(
    *,
    seeds: Sequence[int],
    schedulers: Sequence[str],
    budget_unit: str,
    budget_limit: Optional[float],
    smoke: bool,
    output_root: Path,
    trajectory_schema: str,
    task_limit: Optional[int],
) -> None:
    from experiments import rlhf_humaneval as humaneval

    code_root = _code_root()
    data_path = code_root / "data" / "humaneval_solutions.npz"
    artifacts_root = code_root / "results" / "rlhf_humaneval"
    arrays = humaneval.load_humaneval_dataset(data_path)
    pass_fail = arrays["pass_fail"]
    test_counts = arrays["test_counts"]
    test_difficulties = arrays["test_difficulties"]
    problem_pass_rates = humaneval.compute_problem_pass_rates(pass_fail, test_counts)
    reward_true = humaneval.compute_ground_truth_reward_vector(problem_pass_rates)

    cfg_base = humaneval.Config(T=50, seeds=tuple(int(seed) for seed in seeds)) if smoke else humaneval.Config(seeds=tuple(int(seed) for seed in seeds))
    schedule_names = [_normalize_exact_match_scheduler(name) for name in schedulers]
    quality_schedule_names = list(dict.fromkeys(schedule_names + ["high"]))
    run_count = max(1, len(seeds) * len(schedule_names))
    per_run_budget_limit = (float(budget_limit) / run_count) if budget_limit is not None else None

    for seed in seeds:
        seed_sequence = np.random.SeedSequence(int(seed))
        schedule_seeds = seed_sequence.spawn(len(quality_schedule_names))
        schedule_frames: Dict[str, Any] = {}
        for schedule_name, schedule_seed in zip(quality_schedule_names, schedule_seeds):
            rng_seeds = schedule_seed.spawn(4)
            df = humaneval._load_or_run_schedule(
                res_dir=artifacts_root,
                cfg=cfg_base,
                schedule_name=schedule_name,
                seed=int(seed),
                pass_fail=pass_fail,
                test_counts=test_counts,
                test_difficulties=test_difficulties,
                problem_pass_rates=problem_pass_rates,
                reward_true=reward_true,
                rng_hint=np.random.default_rng(rng_seeds[0]),
                rng_mon=np.random.default_rng(rng_seeds[1]),
                rng_cal=np.random.default_rng(rng_seeds[2]),
                rng_exp3=np.random.default_rng(rng_seeds[3]),
            )
            if task_limit is not None:
                df = df.head(max(0, int(task_limit))).reset_index(drop=True)
            schedule_frames[schedule_name] = df

        quality_context = _build_exact_match_quality_context(schedule_frames)
        for schedule_name in schedule_names:
            df = schedule_frames[schedule_name]

            rewards = df["reward"].to_numpy(dtype=float)
            tasks = _build_exact_match_tasks(len(df))
            run_dir = _phase2_run_dir(track="exact_match", seed=int(seed), scheduler=schedule_name, output_root=output_root)
            runner = ExperimentRunner(
                config=ExperimentConfig(
                    schedule=schedule_name,
                    seed=int(seed),
                    model_id="precomputed-humaneval",
                    output_dir=run_dir,
                    dataset_name="exact_match",
                    extra={
                        "track": "exact_match",
                        "budget_unit": budget_unit,
                        "budget_limit": per_run_budget_limit,
                        "phase": "phase2",
                        "source_npz": str(data_path),
                        "source_candidates_glob": str((code_root / "intermediate" / "humaneval_candidates" / "*.jsonl")),
                        "success_proxy": "reward_non_decreasing",
                    },
                ),
                trajectory_schema=trajectory_schema,
            )

            def execute_task(_task: TaskSpec, task_index: int, _rng) -> TaskOutcome:
                row = df.iloc[task_index]
                reward_t = float(row["reward"])
                previous_reward = float(rewards[task_index - 1]) if task_index > 0 else 0.0
                quality_fields = _derive_exact_match_quality_fields(row, task_index, quality_context)
                return TaskOutcome(
                    prediction="",
                    success=reward_t >= previous_reward,
                    progress=reward_t,
                    n_cheap=int(row["n_mon"]),
                    n_refine=int(row["n_cal"]),
                    drift_detected=False,
                    metadata={
                        "reward_t": reward_t,
                        "gap_t": float(row["gap"]),
                        "xi_sq_t": float(row["xi_sq"]),
                        "cert_t": float(row["cert"]),
                        "cert_hp_t": float(row["cert_hp"]),
                        "n_mon": int(row["n_mon"]),
                        "source_track": "exact_match",
                    },
                    memory_size=0,
                    coverage_t=float(quality_fields["coverage_t"]),
                    precision_t=float(quality_fields["precision_t"]),
                    no_retrieval_window_flag_t=bool(quality_fields["no_retrieval_window_flag_t"]),
                    redundancy_t=float(quality_fields["redundancy_t"]),
                    freshness_t=float(quality_fields["freshness_t"]),
                    retries_t=0,
                )

            artifacts = runner.run(tasks=tasks, execute_task=execute_task)
            _rewrite_budget_file(
                budget_path=artifacts.budget_path,
                budget_unit=budget_unit,
                budget_limit=per_run_budget_limit,
                budget_actual=per_run_budget_limit,
                budget_source="equivalence_contract",
            )


def _current_python_hash_seed() -> str:
    return str(os.environ.get("PYTHONHASHSEED", "")).strip() or "unset"


def _load_phase2_anchors(path: Path) -> tuple[Optional[np.ndarray], Optional[str]]:
    if not path.exists():
        return None, None
    with np.load(path) as data:
        recorded_hash_seed = None
        if "pythonhashseed" in data.files:
            recorded_hash_seed = str(np.asarray(data["pythonhashseed"]).item())
        if "anchors" in data.files:
            return np.asarray(data["anchors"], dtype=float), recorded_hash_seed
        if "centers" in data.files:
            return np.asarray(data["centers"], dtype=float), recorded_hash_seed
        first_key = data.files[0] if data.files else None
        if first_key is None:
            return None, recorded_hash_seed
        return np.asarray(data[first_key], dtype=float), recorded_hash_seed


def _ensure_anchor_hash_seed_compatible(path: Path, recorded_hash_seed: Optional[str]) -> None:
    current_hash_seed = _current_python_hash_seed()
    if current_hash_seed == "unset":
        raise ValueError(
            "ALFWorld coverage metrics require an explicit PYTHONHASHSEED so HashingEmbedder outputs "
            "stay comparable across the dump/anchor/run pipeline. Rerun with PYTHONHASHSEED set to a fixed "
            "integer (for example 0)."
        )
    if recorded_hash_seed in {None, "", "unset", "unknown"}:
        raise ValueError(
            f"Coverage anchors at {path} are missing pythonhashseed metadata. Regenerate the memory dump and "
            "coverage anchors under the same fixed PYTHONHASHSEED before rerunning ALFWorld."
        )
    if current_hash_seed != recorded_hash_seed:
        raise ValueError(
            f"Coverage anchors at {path} were generated with PYTHONHASHSEED={recorded_hash_seed}, but the current "
            f"process is running with PYTHONHASHSEED={current_hash_seed}. Regenerate or rerun with matching seeds."
        )


def _run_alfworld_track(
    *,
    seeds: Sequence[int],
    schedulers: Sequence[str],
    budget_unit: str,
    budget_limit: Optional[float],
    smoke: bool,
    output_root: Path,
    backend: str,
    top_k: int,
    timeout_s: float,
    protocol: str,
    request_retries: int,
    retry_backoff_s: float,
    coverage_anchors: Path,
    coverage_radius: Optional[float],
    embedder_version: str,
    embedder_model_name: Optional[str],
    embedder_cache_folder: Optional[str],
    embedder_device: Optional[str],
    use_ole_certificate: bool,
    ole_particles: Optional[Path],
    adaptive_warmup: int,
    adaptive_burst_length: int,
    adaptive_n_cal_high: int,
    alfworld_prompt_mode: str,
    trajectory_schema: str,
    task_limit: Optional[int],
    task_offset: int,
    dump_memory: Optional[Path],
) -> None:
    from experiments.aomp_mem.evaluation.memory_quality import compute_memory_quality
    from experiments.aomp_mem.runtime import build_embedder, build_executor, build_llm_client, load_dataset, select_tasks
    from experiments.aomp_mem.certificate.feature_pipeline import FEATURE_DIM
    from experiments.aomp_mem.certificate.ole_certificate import OLECertificate
    from experiments.aomp_mem.certificate.reward_model import LinearRewardModel

    anchors, anchor_hash_seed = _load_phase2_anchors(coverage_anchors)
    if anchors is not None and (embedder_version or "hashing").lower() == "hashing":
        _ensure_anchor_hash_seed_compatible(coverage_anchors, anchor_hash_seed)
    ole_certificate = None
    if use_ole_certificate:
        if anchors is None:
            raise ValueError("--use-ole-certificate requires --coverage-anchors/--anchors-path")
        if ole_particles is None:
            raise ValueError("--use-ole-certificate requires --ole-particles")
        ole_model = LinearRewardModel(dim=FEATURE_DIM)
        ole_certificate = OLECertificate.load(ole_particles, ole_model)
    dataset = load_dataset("alfworld", root=_code_root(), prompt_mode=alfworld_prompt_mode)
    selected_tasks = select_tasks(
        dataset,
        max_tasks=task_limit if task_limit is not None else (5 if smoke else None),
        task_offset=task_offset,
    )
    method_names = [_normalize_task_stream_method(name) for name in schedulers]
    run_count = max(1, len(seeds) * len(method_names))
    per_run_budget_limit = (float(budget_limit) / run_count) if budget_limit is not None else None
    effective_coverage_radius = (
        0.5 if coverage_radius is None and (embedder_version or "hashing").lower() != "hashing" else coverage_radius
    )

    if dump_memory is not None and backend == "mock":
        raise ValueError("--dump-memory requires a real backend; rerun with --backend gemini or another non-mock backend.")

    embedder = build_embedder(
        embedder_version,
        model_name=embedder_model_name,
        cache_folder=embedder_cache_folder,
        device=embedder_device,
    )

    for seed in seeds:
        for method_name in method_names:
            llm_client, model_id = build_llm_client(
                backend=backend,
                tasks=selected_tasks,
                timeout_s=timeout_s,
                protocol=protocol,
                request_retries=request_retries,
                retry_backoff_s=retry_backoff_s,
            )
            executor = build_executor(
                method_name,
                llm_client=llm_client,
                top_k=top_k,
                embedder=embedder,
                anchors=anchors,
                ole_certificate=ole_certificate if method_name == "adaptive" else None,
                coverage_radius=effective_coverage_radius if method_name == "adaptive" else None,
                adaptive_warmup=adaptive_warmup,
                adaptive_burst_length=adaptive_burst_length,
                adaptive_n_cal_high=adaptive_n_cal_high,
            )
            result_group = _alfworld_result_group(method_name, alfworld_prompt_mode)
            run_dir = _alfworld_run_dir(
                seed=int(seed),
                scheduler=method_name,
                prompt_mode=alfworld_prompt_mode,
                output_root=output_root,
            )
            runner = ExperimentRunner(
                config=ExperimentConfig(
                    schedule=method_name,
                    seed=int(seed),
                    model_id=model_id,
                    output_dir=run_dir,
                    dataset_name=dataset.dataset_name,
                    top_k=top_k,
                    extra={
                        "track": "alfworld",
                        "result_group": result_group,
                        "budget_unit": budget_unit,
                        "budget_limit": per_run_budget_limit,
                        "phase": "phase2",
                        "backend": backend,
                        "alfworld_prompt_mode": alfworld_prompt_mode,
                        "alfworld_scoring": "subgoal-token-progress",
                        "embedder_version": embedder_version,
                        "embedder_model_name": embedder_model_name,
                        "coverage_anchors": str(coverage_anchors),
                        "coverage_radius": effective_coverage_radius,
                        "use_ole_certificate": use_ole_certificate,
                        "ole_particles": str(ole_particles) if ole_particles is not None else None,
                        "adaptive_warmup": adaptive_warmup,
                        "adaptive_burst_length": adaptive_burst_length,
                        "adaptive_n_cal_high": adaptive_n_cal_high,
                        "task_offset": task_offset,
                    },
                ),
                trajectory_schema=trajectory_schema,
            )

            if dump_memory is not None:
                if not selected_tasks:
                    raise ValueError("No tasks available for --dump-memory")
                for dump_task_index, dump_task in enumerate(selected_tasks, start=1):
                    executor.execute(dump_task, dataset=dataset, task_index=dump_task_index)
                    if _unique_memory_embedding_count(executor.memory_store.entries()) >= 20:
                        break
                _write_memory_dump(dump_memory, executor.memory_store.entries())
                raise SystemExit(0)

            run_start = time.perf_counter()
            calls_before_run = llm_client.tracker.total_calls

            def execute_task(task: TaskSpec, task_index: int, _rng) -> TaskOutcome:
                outcome = executor.execute(task, dataset=dataset, task_index=task_index + 1)
                observation = getattr(executor, "last_step_observation", {}) or {}
                memory_entries = observation.get("memory_entries_after") or []
                retrieved_entries = observation.get("retrieved") or []

                coverage_t = None
                precision_t = None
                redundancy_t = None
                freshness_t = None
                no_retrieval_window_flag_t = False
                if anchors is not None:
                    metrics = compute_memory_quality(
                        entries=memory_entries,
                        retrieved=retrieved_entries,
                        anchors=anchors,
                        task_step=task_index + 1,
                        coverage_radius=effective_coverage_radius,
                    )
                    coverage_t = metrics.coverage
                    precision_t = metrics.precision
                    redundancy_t = metrics.redundancy
                    freshness_t = metrics.freshness
                    no_retrieval_window_flag_t = metrics.no_retrieval_window_flag

                retries_t = int(outcome.metadata.get("retry_count", 0))
                metadata = dict(outcome.metadata)
                metadata["retries_t"] = retries_t
                return TaskOutcome(
                    prediction=outcome.prediction,
                    success=outcome.success,
                    progress=outcome.progress,
                    n_cheap=outcome.n_cheap,
                    n_refine=outcome.n_refine,
                    drift_detected=outcome.drift_detected,
                    metadata=metadata,
                    memory_size=outcome.memory_size,
                    coverage_t=coverage_t,
                    precision_t=precision_t,
                    no_retrieval_window_flag_t=no_retrieval_window_flag_t,
                    redundancy_t=redundancy_t,
                    freshness_t=freshness_t,
                    mqc_t=metadata.get("mqc_t"),
                    ole_mean_t=metadata.get("ole_mean_t"),
                    ole_var_t=metadata.get("ole_var_t"),
                    ole_ucb_t=metadata.get("ole_ucb_t"),
                    cusum_stat_t=metadata.get("cusum_stat_t"),
                    drift_onset_flag_t=bool(metadata.get("drift_onset_flag_t", False)),
                    retries_t=retries_t,
                )

            budget_path = run_dir / "budget_analysis.csv"
            try:
                runner.run(tasks=selected_tasks, execute_task=execute_task)
            finally:
                actual = None
                if budget_unit == "llm_calls":
                    actual = float(llm_client.tracker.total_calls - calls_before_run)
                elif budget_unit == "wall_clock_s":
                    actual = float(time.perf_counter() - run_start)
                _rewrite_budget_file(
                    budget_path=budget_path,
                    budget_unit=budget_unit,
                    budget_limit=per_run_budget_limit,
                    budget_actual=actual,
                    budget_source="runtime_observed",
                )


def _rewrite_budget_file(
    *,
    budget_path: Path,
    budget_unit: str,
    budget_limit: Optional[float],
    budget_actual: Optional[float],
    budget_source: str,
) -> None:
    if not budget_path.exists():
        return
    with budget_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return
    rows[0]["budget_unit"] = budget_unit
    rows[0]["budget_limit"] = "" if budget_limit is None else float(budget_limit)
    rows[0]["budget_actual"] = "" if budget_actual is None else float(budget_actual)
    rows[0]["budget_source"] = budget_source
    fieldnames = list(rows[0].keys())
    with budget_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_memory_dump(path: Path, entries: Sequence[Any]) -> None:
    dump_path = Path(path)
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    if entries:
        embeddings = np.vstack([np.asarray(entry.embedding, dtype=float) for entry in entries]).astype(np.float32)
        timestamps = np.asarray([int(entry.timestamp) for entry in entries], dtype=np.int64)
        quality_scores = np.asarray([float(entry.quality_score) for entry in entries], dtype=np.float32)
    else:
        embeddings = np.zeros((0, 0), dtype=np.float32)
        timestamps = np.zeros((0,), dtype=np.int64)
        quality_scores = np.zeros((0,), dtype=np.float32)
    np.savez(
        dump_path,
        embeddings=embeddings,
        timestamps=timestamps,
        quality_scores=quality_scores,
        pythonhashseed=np.asarray(_current_python_hash_seed()),
    )
    print(f"Wrote memory dump to {dump_path}")


def _unique_memory_embedding_count(entries: Sequence[Any]) -> int:
    if not entries:
        return 0
    embeddings = np.vstack([np.asarray(entry.embedding, dtype=np.float32) for entry in entries])
    return int(np.unique(embeddings, axis=0).shape[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ALFWorld A-OMP-Mem tracks.")
    parser.add_argument("--track", required=True, choices=["exact_match", "alfworld"])
    parser.add_argument("--budget-unit", choices=["tokens", "wall_clock_s", "llm_calls"], default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--schedulers", nargs="+", default=["adaptive", "exp3", "fixed_low", "high", "monitor_only"])
    parser.add_argument("--output-root", "--output-dir", dest="output_root", default=str(_default_phase2_results_root()))
    parser.add_argument("--budget-config", default=str(_code_root() / "configs" / "budget_equivalence.yaml"))
    parser.add_argument(
        "--coverage-anchors",
        "--anchors-path",
        dest="coverage_anchors",
        default=str(_code_root() / "data" / "coverage_anchors_v2.npz"),
    )
    parser.add_argument(
        "--embedder-version",
        default="hashing",
        choices=["hashing", "st-minilm"],
        help="Embedder injected into ALFWorld executors.",
    )
    parser.add_argument("--embedder-model-name", default=None, help="Optional sentence-transformers model override.")
    parser.add_argument("--embedder-cache-folder", default=None, help="Optional local HuggingFace cache folder.")
    parser.add_argument("--embedder-device", default=None, help="Optional torch device for sentence-transformers.")
    parser.add_argument("--use-ole-certificate", action="store_true", help="Use OLE ensemble UCB for adaptive bursts.")
    parser.add_argument(
        "--ole-particles",
        default=str(_code_root() / "certificate" / "ole_particles_v2_real_progress_k18_add050.npz"),
        help="Path to OLE particle package produced by --stage train_ole.",
    )
    parser.add_argument(
        "--coverage-radius",
        type=float,
        default=None,
        help="Override memory-quality coverage radius; st-minilm defaults to 0.5 when omitted.",
    )
    parser.add_argument("--adaptive-warmup", type=int, default=5, help="Warmup steps for adaptive scheduler.")
    parser.add_argument(
        "--adaptive-burst-length",
        type=int,
        default=10,
        help="Number of tasks covered by one adaptive refinement burst.",
    )
    parser.add_argument(
        "--adaptive-n-cal-high",
        type=int,
        default=3,
        help="Refinement budget used when adaptive scheduler triggers high-cost feedback.",
    )
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument("--task-offset", type=int, default=0)
    parser.add_argument(
        "--alfworld-prompt-mode",
        default="family_prior_experimental",
        choices=["benchmark", "reduced", "official_candidate_debug", "family_prior_experimental"],
        help="ALFWorld prompt-control mode.",
    )
    parser.add_argument("--dump-memory", default=None)
    parser.add_argument("--backend", default="mock", choices=["mock", "gemini", "openai"])
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--protocol", default="auto", choices=["auto", "openai", "gemini"])
    parser.add_argument("--request-retries", type=int, default=2)
    parser.add_argument("--retry-backoff-s", type=float, default=5.0)
    parser.add_argument("--legacy-mode", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    track = args.track
    resolved_budget_unit = args.budget_unit or ("llm_calls" if track == "alfworld" else "tokens")
    budget_config_path = Path(args.budget_config) if args.budget_config else None
    budget_limit = _budget_limit_for(track, resolved_budget_unit, config_path=budget_config_path)
    trajectory_schema = LEGACY_SCHEMA_VERSION if args.legacy_mode else PHASE2_SCHEMA_VERSION

    if args.track == "exact_match":
        _run_exact_match_track(
            seeds=args.seeds,
            schedulers=args.schedulers,
            budget_unit=resolved_budget_unit,
            budget_limit=budget_limit,
            smoke=args.smoke,
            output_root=output_root,
            trajectory_schema=trajectory_schema,
            task_limit=args.task_limit,
        )
        return

    _run_alfworld_track(
        seeds=args.seeds,
        schedulers=args.schedulers,
        budget_unit=resolved_budget_unit,
        budget_limit=budget_limit,
        smoke=args.smoke,
        output_root=output_root,
        backend=args.backend,
        top_k=args.top_k,
        timeout_s=args.timeout_s,
        protocol=args.protocol,
        request_retries=args.request_retries,
        retry_backoff_s=args.retry_backoff_s,
        coverage_anchors=Path(args.coverage_anchors),
        coverage_radius=args.coverage_radius,
        embedder_version=args.embedder_version,
        embedder_model_name=args.embedder_model_name,
        embedder_cache_folder=args.embedder_cache_folder,
        embedder_device=args.embedder_device,
        use_ole_certificate=args.use_ole_certificate,
        ole_particles=Path(args.ole_particles) if args.ole_particles else None,
        adaptive_warmup=args.adaptive_warmup,
        adaptive_burst_length=args.adaptive_burst_length,
        adaptive_n_cal_high=args.adaptive_n_cal_high,
        alfworld_prompt_mode=args.alfworld_prompt_mode,
        trajectory_schema=trajectory_schema,
        task_limit=args.task_limit,
        task_offset=args.task_offset,
        dump_memory=Path(args.dump_memory) if args.dump_memory else None,
    )


if __name__ == "__main__":
    main()
