"""Shared runtime helpers for A-OMP-Mem experiment entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from experiments.aomp_mem.core import (
    CallTracker,
    EmbedderProtocol,
    HashingEmbedder,
    SentenceTransformerEmbedder,
    build_gemini_client_from_env,
    build_openai_client_from_env,
)
from experiments.aomp_mem.core.executor import AOMPMemExecutor
from experiments.aomp_mem.core.llm_interface import BaseLLMClient, GenerationResult
from experiments.aomp_mem.datasets import (
    BaseTaskStreamDataset,
    ScienceWorldDataset,
)
from experiments.aomp_mem.schedulers import (
    AdaptiveScheduler,
    CheapOnlyScheduler,
    Exp3Scheduler,
    FixedLowScheduler,
    OracleHighScheduler,
)


@dataclass(frozen=True)
class DatasetDefinition:
    """Metadata required to construct the ScienceWorld dataset adapter."""

    dataset_name: str
    relative_path: Path
    loader: type[BaseTaskStreamDataset]


class StageAwareTaskStreamMockLLMClient(BaseLLMClient):
    """Stage-aware mock backend that emits gold answers for cheap/act stages."""

    def __init__(self, answers: list[str], tracker: Optional[CallTracker] = None) -> None:
        super().__init__(tracker=tracker)
        self.answers = list(answers)
        self.answer_index = 0

    def generate(self, prompt: str, *, metadata: Optional[dict] = None) -> GenerationResult:
        metadata = dict(metadata or {})
        stage = metadata.get("stage", "unknown")
        self.tracker.record(prompt)

        if stage in {"cheap", "act"}:
            answer = self.answers[min(self.answer_index, len(self.answers) - 1)] if self.answers else ""
            self.answer_index += 1
            return GenerationResult(text=answer, metadata={"stage": stage, "mock": True})

        if stage == "think":
            return GenerationResult(
                text="Identify reusable patterns from prior tasks and note the required adaptation.",
                metadata={"stage": stage, "mock": True},
            )
        if stage in {"refine", "refine_iter"}:
            return GenerationResult(
                text="Keep successful reusable patterns, merge duplicates, and drop clearly failed entries.",
                metadata={"stage": stage, "mock": True},
            )
        return GenerationResult(text="mock-response", metadata={"stage": stage, "mock": True})


DATASET_DEFINITIONS: dict[str, DatasetDefinition] = {
    "scienceworld": DatasetDefinition(
        dataset_name="scienceworld",
        relative_path=Path("data/aomp_mem/processed/scienceworld.jsonl"),
        loader=ScienceWorldDataset,
    ),
}


def code_root() -> Path:
    """Return the migrated ScienceWorld root for data and results path resolution."""

    return Path(__file__).resolve().parents[3]


def available_dataset_names() -> list[str]:
    return list(DATASET_DEFINITIONS.keys())


def load_dataset(
    dataset_name: str,
    *,
    root: Optional[Path] = None,
    scienceworld_prompt_mode: str = ScienceWorldDataset.PROMPT_MODE_BENCHMARK,
) -> BaseTaskStreamDataset:
    """Instantiate one dataset adapter from the registry."""

    if dataset_name not in DATASET_DEFINITIONS:
        supported = ", ".join(sorted(DATASET_DEFINITIONS))
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Supported datasets: {supported}")

    definition = DATASET_DEFINITIONS[dataset_name]
    base_root = Path(root) if root is not None else code_root()
    dataset_path = base_root / definition.relative_path
    if dataset_name == "scienceworld":
        return definition.loader(dataset_path, prompt_mode=scienceworld_prompt_mode)
    return definition.loader(dataset_path)


def extract_expected_answers(tasks: Iterable) -> list[str]:
    """Collect expected answers for stage-aware mock execution."""

    answers: list[str] = []
    for task in tasks:
        answers.append(str(task.metadata.get("expected_output", task.metadata.get("answer", ""))))
    return answers


def build_llm_client(
    *,
    backend: str,
    tasks,
    timeout_s: float,
    protocol: str,
    request_retries: int,
    retry_backoff_s: float,
):
    """Create one concrete LLM backend plus its model identifier."""

    if backend == "mock":
        return StageAwareTaskStreamMockLLMClient(extract_expected_answers(tasks)), "stage-aware-task-stream-mock"
    if backend == "openai":
        client = build_openai_client_from_env(
            timeout_s=timeout_s,
            max_retries=request_retries,
            retry_backoff_s=retry_backoff_s,
        )
        return client, client.model_id

    client = build_gemini_client_from_env(
        timeout_s=timeout_s,
        protocol=protocol,
        max_retries=request_retries,
        retry_backoff_s=retry_backoff_s,
    )
    return client, client.model_id


def build_embedder(
    version: str = "hashing",
    *,
    dim: int = 128,
    model_name: Optional[str] = None,
    cache_folder: Optional[str] = None,
    device: Optional[str] = None,
) -> EmbedderProtocol:
    """Construct one embedder by version label.

    ``version="hashing"`` keeps the legacy 128-dim deterministic embedder used
    in Phase 1 and plan-3 reproductions; ``version="st-minilm"`` returns a
    real sentence-transformers semantic embedder (default
    ``all-MiniLM-L6-v2``, dim=384) for plan-3.5 v2 trajectories.

    Either choice is structurally compatible with ``EmbedderProtocol`` so
    downstream code (executor, retrieval, memory_quality) accepts them
    interchangeably without touching AC-2 locked files.
    """

    normalized = (version or "hashing").lower()
    if normalized == "hashing":
        return HashingEmbedder(dim=int(dim))
    if normalized in {"st-minilm", "minilm", "sentence-transformers"}:
        return SentenceTransformerEmbedder(
            model_name=model_name,
            cache_folder=cache_folder,
            device=device,
        )
    raise ValueError(
        f"Unsupported embedder version '{version}'. Supported: hashing, st-minilm."
    )


def build_executor(
    method: str,
    llm_client: BaseLLMClient,
    top_k: int,
    *,
    embedder: Optional[EmbedderProtocol] = None,
    anchors: Optional[np.ndarray] = None,
    ole_certificate: Optional[Any] = None,
    coverage_radius: Optional[float] = None,
):
    """Build one ScienceWorld experiment executor for a named scheduler.

    The optional ``embedder`` parameter accepts any ``EmbedderProtocol``
    implementation; when omitted it falls back to the legacy
    ``HashingEmbedder(dim=128)``.
    """

    if embedder is None:
        embedder = build_embedder("hashing", dim=128)
    if method == "cheap_only":
        return AOMPMemExecutor(
            llm_client=llm_client,
            embedder=embedder,
            scheduler=CheapOnlyScheduler(),
            top_k=top_k,
        )
    if method == "fixed_low":
        return AOMPMemExecutor(
            llm_client=llm_client,
            embedder=embedder,
            scheduler=FixedLowScheduler(interval=5, n_cal_high=3),
            top_k=top_k,
        )
    if method == "adaptive":
        return AOMPMemExecutor(
            llm_client=llm_client,
            embedder=embedder,
            scheduler=AdaptiveScheduler(
                warmup=5,
                drift_window=5,
                drift_threshold=0.2,
                burst_length=10,
                n_cal_high=3,
                anchors=anchors,
                ole_certificate=ole_certificate,
                coverage_radius=coverage_radius,
            ),
            top_k=top_k,
        )
    if method == "exp3":
        return AOMPMemExecutor(
            llm_client=llm_client,
            embedder=embedder,
            scheduler=Exp3Scheduler(
                n_cal_low=0,
                n_cal_high=3,
                horizon=100,
                lambda_cost=0.25,
                rng=np.random.default_rng(0),
            ),
            top_k=top_k,
        )
    if method == "oracle_high":
        return AOMPMemExecutor(
            llm_client=llm_client,
            embedder=embedder,
            scheduler=OracleHighScheduler(n_cal_high=3),
            top_k=top_k,
        )
    raise ValueError(f"Unsupported method: {method}")


def available_method_names() -> list[str]:
    """Return the schedulers used by the ScienceWorld main experiments."""

    return ["cheap_only", "fixed_low", "adaptive", "exp3", "oracle_high"]


def select_tasks(
    dataset: BaseTaskStreamDataset,
    *,
    max_tasks: Optional[int],
    task_ids: Optional[Iterable[str]] = None,
    task_offset: int = 0,
):
    """Select a contiguous task slice or explicit task subset from one dataset."""

    tasks = dataset.tasks()
    if task_ids:
        selected_ids = {str(task_id) for task_id in task_ids}
        tasks = [task for task in tasks if task.task_id in selected_ids]
    offset = max(0, int(task_offset))
    if offset:
        tasks = tasks[offset:]
    if max_tasks is None:
        return tasks
    return tasks[: max(0, int(max_tasks))]
