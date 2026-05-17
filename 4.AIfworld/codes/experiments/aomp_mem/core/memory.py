"""Memory data structures for A-OMP-Mem."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


@dataclass
class FeedbackRecord:
    """Feedback emitted by a dataset evaluator."""

    success: bool
    progress: Optional[float] = None
    correct_answer: Optional[str] = None
    trajectory: Optional[List[str]] = None


@dataclass
class MemoryEntry:
    """Stored experience entry used for retrieval and refinement."""

    task_input: str
    task_output: str
    feedback: FeedbackRecord
    embedding: np.ndarray
    quality_score: float
    usage_count: int
    timestamp: int
    success_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        task_input: str,
        task_output: str,
        feedback: FeedbackRecord,
        embedding: np.ndarray,
        timestamp: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "MemoryEntry":
        success_count = 1 if feedback.success else 0
        usage_count = 0
        quality_score = cls.compute_quality_score(success_count=success_count, usage_count=usage_count)
        return cls(
            task_input=task_input,
            task_output=task_output,
            feedback=feedback,
            embedding=np.asarray(embedding, dtype=float),
            quality_score=quality_score,
            usage_count=usage_count,
            timestamp=timestamp,
            success_count=success_count,
            metadata=dict(metadata or {}),
        )

    @staticmethod
    def compute_quality_score(*, success_count: int, usage_count: int) -> float:
        return float((success_count + 1) / (usage_count + 2))

    def touch_usage(self, *, success: Optional[bool] = None) -> None:
        self.usage_count += 1
        if success is True:
            self.success_count += 1
        self.quality_score = self.compute_quality_score(
            success_count=self.success_count,
            usage_count=self.usage_count,
        )


class MemoryStore:
    """Mutable in-memory experience buffer with bounded capacity."""

    def __init__(self, *, max_size: int = 500) -> None:
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")
        self.max_size = int(max_size)
        self._entries: List[MemoryEntry] = []
        self._timestamp_counter = 0

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterable[MemoryEntry]:
        return iter(self._entries)

    def entries(self) -> List[MemoryEntry]:
        return list(self._entries)

    def next_timestamp(self) -> int:
        self._timestamp_counter += 1
        return self._timestamp_counter

    def add(self, entry: MemoryEntry) -> None:
        self._entries.append(entry)
        self._prune_if_needed()

    def append(
        self,
        *,
        task_input: str,
        task_output: str,
        feedback: FeedbackRecord,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryEntry:
        entry = MemoryEntry.create(
            task_input=task_input,
            task_output=task_output,
            feedback=feedback,
            embedding=embedding,
            timestamp=self.next_timestamp(),
            metadata=metadata,
        )
        self.add(entry)
        return entry

    def bulk_add(self, entries: Iterable[MemoryEntry]) -> None:
        for entry in entries:
            self.add(entry)

    def replace(self, entries: Iterable[MemoryEntry]) -> None:
        self._entries = []
        self.bulk_add(entries)

    def _prune_if_needed(self) -> None:
        while len(self._entries) > self.max_size:
            prune_idx = min(
                range(len(self._entries)),
                key=lambda idx: (self._entries[idx].quality_score, self._entries[idx].timestamp),
            )
            del self._entries[prune_idx]
