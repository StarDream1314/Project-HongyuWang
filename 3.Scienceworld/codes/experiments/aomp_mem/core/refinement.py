"""Deep refinement pipeline for the accurate A-OMP-Mem channel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np

from experiments.aomp_mem.core.llm_interface import BaseLLMClient
from experiments.aomp_mem.core.memory import FeedbackRecord, MemoryEntry, MemoryStore
from experiments.aomp_mem.core.retrieval import HashingEmbedder, RetrievalResult, cosine_similarity
from experiments.aomp_mem.datasets.base import BaseTaskStreamDataset


@dataclass
class RefinementResult:
    """Outputs produced by a full Think-Refine-Act pass."""

    prediction: str
    reasoning: str
    summary: str
    llm_calls: int
    memory_size: int
    feedback: FeedbackRecord


class DeepRefinementEngine:
    """Accurate channel that actively refines memory before acting."""

    def __init__(
        self,
        *,
        llm_client: BaseLLMClient,
        embedder: HashingEmbedder,
        max_refine_steps: int = 2,
        redundancy_threshold: float = 0.92,
    ) -> None:
        self.llm_client = llm_client
        self.embedder = embedder
        self.max_refine_steps = max(0, int(max_refine_steps))
        self.redundancy_threshold = float(redundancy_threshold)

    def run(
        self,
        *,
        task,
        dataset: BaseTaskStreamDataset,
        memory_store: MemoryStore,
        retrieved: Sequence[RetrievalResult],
        feedback: Optional[FeedbackRecord] = None,
        feedback_resolver: Optional[Callable[[str], FeedbackRecord]] = None,
        extra_refine_steps: int = 0,
    ) -> RefinementResult:
        calls_before = self.llm_client.tracker.total_calls
        query_embedding = self.embedder.encode(task.input_text)
        self.dataset = dataset

        reasoning = self.think(task=task, retrieved=retrieved)
        refinement_notes = [self.refine(task=task, reasoning=reasoning, retrieved=retrieved)]
        extra_steps = min(max(0, int(extra_refine_steps)), self.max_refine_steps)
        for step in range(extra_steps):
            note = self._iterate_refine(
                task=task,
                reasoning=reasoning,
                retrieved=retrieved,
                previous_note=refinement_notes[-1],
                step_index=step,
            )
            refinement_notes.append(note)
        summary = refinement_notes[-1]

        refined_entries = self._build_refined_entries(
            entries=memory_store.entries(),
            query_embedding=query_embedding,
            summary=summary,
        )
        memory_store.replace(refined_entries)

        prediction = self.act(
            task=task,
            dataset=dataset,
            reasoning=reasoning,
            summary=summary,
            refined_entries=memory_store.entries(),
        )
        if feedback is None and feedback_resolver is not None:
            feedback = feedback_resolver(prediction)
        if feedback is None:
            raise ValueError("feedback or feedback_resolver must be provided for deep refinement")
        memory_store.append(
            task_input=task.input_text,
            task_output=prediction,
            feedback=feedback,
            embedding=query_embedding,
            metadata={"channel": "accurate", "reasoning": reasoning, "summary": summary},
        )

        calls_after = self.llm_client.tracker.total_calls
        return RefinementResult(
            prediction=prediction,
            reasoning=reasoning,
            summary=summary,
            llm_calls=calls_after - calls_before,
            memory_size=len(memory_store),
            feedback=feedback,
        )

    def think(self, *, task, retrieved: Sequence[RetrievalResult]) -> str:
        prompt = (
            "Think about reusable patterns for the current task.\n"
            f"Task: {task.input_text}\n"
            f"{self.dataset.prompt_context(task, stage='think')}"
            f"Retrieved experiences:\n{self._format_retrieved(retrieved)}\n"
            f"{self.dataset.stage_instruction(stage='think')}"
        )
        return self.llm_client.generate(prompt, metadata={"stage": "think"}).text

    def refine(self, *, task, reasoning: str, retrieved: Sequence[RetrievalResult]) -> str:
        prompt = (
            "Refine memory for future tasks.\n"
            f"Task: {task.input_text}\n"
            f"{self.dataset.prompt_context(task, stage='refine')}"
            f"Reasoning: {reasoning}\n"
            f"Retrieved experiences:\n{self._format_retrieved(retrieved)}\n"
            f"{self.dataset.stage_instruction(stage='refine')}"
        )
        return self.llm_client.generate(prompt, metadata={"stage": "refine"}).text

    def act(
        self,
        *,
        task,
        dataset: BaseTaskStreamDataset,
        reasoning: str,
        summary: str,
        refined_entries: Sequence[MemoryEntry],
    ) -> str:
        prompt = (
            "Solve the task using refined memory.\n"
            f"Task: {task.input_text}\n"
            f"{dataset.prompt_context(task, stage='act')}"
            f"Reasoning: {reasoning}\n"
            f"Refined summary: {summary}\n"
            f"Refined memory:\n{self._format_entries(refined_entries)}\n"
            f"{dataset.stage_instruction(stage='act')}"
        )
        raw_prediction = self.llm_client.generate(prompt, metadata={"stage": "act"}).text
        return dataset.postprocess_prediction(raw_prediction, stage="act")

    def _iterate_refine(
        self,
        *,
        task,
        reasoning: str,
        retrieved: Sequence[RetrievalResult],
        previous_note: str,
        step_index: int,
    ) -> str:
        prompt = (
            "Improve the refinement summary.\n"
            f"Task: {task.input_text}\n"
            f"{self.dataset.prompt_context(task, stage='refine_iter')}"
            f"Reasoning: {reasoning}\n"
            f"Previous refinement: {previous_note}\n"
            f"Retrieved experiences:\n{self._format_retrieved(retrieved)}\n"
            f"Iteration: {step_index + 1}\n"
            f"{self.dataset.stage_instruction(stage='refine_iter')}"
        )
        return self.llm_client.generate(prompt, metadata={"stage": "refine_iter"}).text

    def _build_refined_entries(
        self,
        *,
        entries: Sequence[MemoryEntry],
        query_embedding: np.ndarray,
        summary: str,
    ) -> List[MemoryEntry]:
        ranked = sorted(
            entries,
            key=lambda entry: (
                entry.feedback.success,
                entry.quality_score,
                entry.usage_count,
                -entry.timestamp,
            ),
            reverse=True,
        )

        selected: List[MemoryEntry] = []
        for entry in ranked:
            if not entry.feedback.success and entry.quality_score <= 0.5:
                continue
            if any(
                cosine_similarity(entry.embedding, existing.embedding) >= self.redundancy_threshold
                for existing in selected
            ):
                continue
            selected.append(entry)

        if summary.strip():
            summary_embedding = self._summary_embedding(query_embedding, selected)
            selected.insert(
                0,
                MemoryEntry.create(
                    task_input="refined-template",
                    task_output=summary.strip(),
                    feedback=FeedbackRecord(success=True),
                    embedding=summary_embedding,
                    timestamp=max((entry.timestamp for entry in entries), default=0) + 1,
                    metadata={"kind": "summary"},
                ),
            )
        return selected

    @staticmethod
    def _format_retrieved(retrieved: Sequence[RetrievalResult]) -> str:
        if not retrieved:
            return "- none"
        return "\n".join(
            (
                f"- score={result.score:.3f} "
                f"task={DeepRefinementEngine._shorten(result.entry.task_input, max_chars=120)} "
                f"output={DeepRefinementEngine._shorten(result.entry.task_output, max_chars=80)}"
            )
            for result in retrieved
        )

    @staticmethod
    def _format_entries(entries: Sequence[MemoryEntry]) -> str:
        if not entries:
            return "- none"
        return "\n".join(
            (
                f"- task={DeepRefinementEngine._shorten(entry.task_input, max_chars=90)} "
                f"quality={entry.quality_score:.3f} "
                f"output={DeepRefinementEngine._shorten(entry.task_output, max_chars=80)}"
            )
            for entry in entries
        )

    @staticmethod
    def _shorten(text: str, *, max_chars: int) -> str:
        compact = " ".join(str(text).split())
        if len(compact) <= max_chars:
            return compact
        return compact[: max_chars - 3] + "..."

    @staticmethod
    def _summary_embedding(query_embedding: np.ndarray, selected: Iterable[MemoryEntry]) -> np.ndarray:
        vectors = [query_embedding]
        vectors.extend(entry.embedding for entry in selected)
        mean_vector = np.mean(np.vstack(vectors), axis=0)
        norm = np.linalg.norm(mean_vector)
        if norm > 0:
            mean_vector = mean_vector / norm
        return mean_vector
