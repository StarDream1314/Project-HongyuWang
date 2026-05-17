"""Dataset abstractions and JSONL loading helpers for A-OMP-Mem."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable, List, Sequence

from experiments.aomp_mem.core.memory import FeedbackRecord
from experiments.aomp_mem.core.retrieval import RetrievalResult
from experiments.aomp_mem.evaluation.runner import TaskSpec


class BaseTaskStreamDataset(ABC):
    """Base interface for Evo-Memory style task streams."""

    dataset_name: str

    @abstractmethod
    def tasks(self) -> List[TaskSpec]:
        """Return the ordered task stream."""

    @abstractmethod
    def evaluate(
        self,
        task: TaskSpec,
        prediction: str,
        *,
        n_refine: int,
        retrieved: Sequence[RetrievalResult],
    ) -> FeedbackRecord:
        """Evaluate a prediction and convert it into a FeedbackRecord."""

    def answer_instruction(self, *, stage: str) -> str:
        """Return dataset-specific output formatting instruction for one generation stage."""

        return "Return the final answer."

    def stage_instruction(self, *, stage: str) -> str:
        """Return dataset-specific instruction text for one generation stage."""

        if stage in {"cheap", "act"}:
            return self.answer_instruction(stage=stage)
        if stage == "think":
            return "Return a concise reasoning trace with reusable patterns and required adaptations."
        if stage == "refine":
            return "Summarize which entries to keep, merge, or drop, and extract one reusable template."
        if stage == "refine_iter":
            return "Return a tighter reusable memory template."
        return "Return a concise useful output."

    def prompt_context(self, task: TaskSpec, *, stage: str) -> str:
        """Return optional task-specific prompt context for one generation stage."""

        return ""

    def postprocess_prediction(self, prediction: str, *, stage: str) -> str:
        """Normalize raw model output into the dataset's expected answer surface form."""

        return prediction.strip()


class JsonlTaskDataset(BaseTaskStreamDataset):
    """Generic JSONL adapter for single-turn and simplified multi-turn datasets."""

    dataset_name = "jsonl"

    def __init__(self, *, path: Path, dataset_name: str, input_field: str = "input") -> None:
        self.path = Path(path)
        self.dataset_name = dataset_name
        self.input_field = input_field
        self._tasks = self._load_tasks()

    def tasks(self) -> List[TaskSpec]:
        return list(self._tasks)

    def evaluate(
        self,
        task: TaskSpec,
        prediction: str,
        *,
        n_refine: int,
        retrieved: Sequence[RetrievalResult],
    ) -> FeedbackRecord:
        metadata = task.metadata
        expected = metadata.get("expected_output", metadata.get("answer"))
        if expected is None:
            raise ValueError(f"Task {task.task_id} is missing expected_output/answer metadata")
        normalized_prediction = self.postprocess_prediction(prediction, stage="evaluate")
        success = self._normalize(normalized_prediction) == self._normalize(str(expected))
        progress = 1.0 if success else float(metadata.get("progress_if_failure", 0.0))
        return FeedbackRecord(success=success, progress=progress, correct_answer=str(expected))

    def _load_tasks(self) -> List[TaskSpec]:
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.path}")

        tasks: List[TaskSpec] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                payload = json.loads(line)
                if "task_id" not in payload:
                    raise ValueError(f"Line {line_number} missing task_id")
                if self.input_field not in payload:
                    raise ValueError(f"Line {line_number} missing {self.input_field}")
                task_metadata = dict(payload.get("metadata", {}))
                for key, value in payload.items():
                    if key not in {"task_id", self.input_field, "metadata"}:
                        task_metadata.setdefault(key, value)
                tasks.append(
                    TaskSpec(
                        task_id=str(payload["task_id"]),
                        input_text=str(payload[self.input_field]),
                        metadata=task_metadata,
                    )
                )
        return tasks

    @staticmethod
    def _normalize(value: str) -> str:
        return " ".join(value.strip().lower().split())
