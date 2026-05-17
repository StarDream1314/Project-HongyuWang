"""Scheduler abstractions for A-OMP-Mem."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np

from experiments.aomp_mem.core.embedding import EmbedderProtocol
from experiments.aomp_mem.core.memory import MemoryEntry


@dataclass
class SchedulerContext:
    """Mutable context available when a schedule makes a decision."""

    task_index: int
    success_history: Sequence[bool]
    query_embedding: Optional[np.ndarray] = None
    history_embeddings: Sequence[np.ndarray] = field(default_factory=list)
    memory_entries: Optional[Sequence[MemoryEntry]] = None
    retrieved: Optional[Sequence[MemoryEntry]] = None
    embedder: Optional[EmbedderProtocol] = None


@dataclass
class SchedulerDecision:
    n_refine: int
    drift_detected: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseScheduler(ABC):
    @abstractmethod
    def decide(self, context: SchedulerContext) -> SchedulerDecision:
        """Return the refinement budget for a task."""
