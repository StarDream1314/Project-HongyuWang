"""Stateful task executor that wires retrieval, scheduling, and refinement together."""

from __future__ import annotations

from typing import List, Sequence

from experiments.aomp_mem.core.embedding import EmbedderProtocol
from experiments.aomp_mem.core.llm_interface import BaseLLMClient
from experiments.aomp_mem.core.memory import MemoryStore
from experiments.aomp_mem.core.refinement import DeepRefinementEngine
from experiments.aomp_mem.core.retrieval import RetrievalResult, retrieve_top_k
from experiments.aomp_mem.datasets.base import BaseTaskStreamDataset
from experiments.aomp_mem.evaluation.precision_signal import record_hit
from experiments.aomp_mem.evaluation.runner import TaskOutcome, TaskSpec
from experiments.aomp_mem.schedulers.base import BaseScheduler, SchedulerContext


class AOMPMemExecutor:
    """Executes a task stream under one scheduler with persistent memory."""

    def __init__(
        self,
        *,
        llm_client: BaseLLMClient,
        embedder: EmbedderProtocol,
        scheduler: BaseScheduler,
        memory_store: MemoryStore | None = None,
        top_k: int = 3,
        max_refine_steps: int = 2,
    ) -> None:
        self.llm_client = llm_client
        self.embedder = embedder
        self.scheduler = scheduler
        self.memory_store = memory_store or MemoryStore()
        self.top_k = int(top_k)
        self.refiner = DeepRefinementEngine(
            llm_client=llm_client,
            embedder=embedder,
            max_refine_steps=max_refine_steps,
        )
        self.success_history: List[bool] = []
        self.last_step_observation = {}

    def execute(self, task: TaskSpec, *, dataset: BaseTaskStreamDataset, task_index: int) -> TaskOutcome:
        query_embedding = self.embedder.encode(task.input_text)
        retrieved = retrieve_top_k(
            query=task.input_text,
            entries=self.memory_store.entries(),
            embedder=self.embedder,
            k=self.top_k,
        )
        record_hit(self.memory_store.entries(), retrieved)
        decision = self.scheduler.decide(
            SchedulerContext(
                task_index=task_index,
                success_history=list(self.success_history),
                query_embedding=query_embedding,
                history_embeddings=[entry.embedding for entry in self.memory_store.entries()],
                memory_entries=self.memory_store.entries(),
                retrieved=[result.entry for result in retrieved],
                embedder=self.embedder,
            )
        )

        if decision.n_refine > 0:
            refinement = self.refiner.run(
                task=task,
                dataset=dataset,
                memory_store=self.memory_store,
                retrieved=retrieved,
                feedback_resolver=lambda prediction: dataset.evaluate(
                    task,
                    prediction,
                    n_refine=decision.n_refine,
                    retrieved=retrieved,
                ),
                extra_refine_steps=max(0, decision.n_refine - 3),
            )
            prediction = refinement.prediction
            feedback = refinement.feedback
            llm_calls = refinement.llm_calls
        else:
            prediction = self._run_cheap_channel(task=task, dataset=dataset, retrieved=retrieved)
            feedback = dataset.evaluate(task, prediction, n_refine=0, retrieved=retrieved)
            self.memory_store.append(
                task_input=task.input_text,
                task_output=prediction,
                feedback=feedback,
                embedding=query_embedding,
                metadata={"channel": "cheap", "retrieved_count": len(retrieved)},
            )
            llm_calls = 1

        self.success_history.append(feedback.success)
        if hasattr(self.scheduler, "observe"):
            arm_index = decision.metadata.get("arm_index")
            try:
                self.scheduler.observe(success=feedback.success, cost=llm_calls, arm_index=arm_index)
            except TypeError:
                self.scheduler.observe(success=feedback.success, cost=llm_calls)

        metadata = {
            "schedule": decision.metadata.get("schedule"),
            "llm_calls": llm_calls,
            "retrieved_count": len(retrieved),
            **decision.metadata,
        }
        self.last_step_observation = {
            "task_index": task_index,
            "retrieved": [result.entry for result in retrieved],
            "memory_entries_after": self.memory_store.entries(),
            "query_embedding": query_embedding,
        }
        return TaskOutcome(
            prediction=prediction,
            success=feedback.success,
            progress=float(feedback.progress or (1.0 if feedback.success else 0.0)),
            n_cheap=1,
            n_refine=int(decision.n_refine),
            drift_detected=bool(decision.drift_detected),
            metadata=metadata,
            memory_size=len(self.memory_store),
        )

    def _run_cheap_channel(
        self,
        *,
        task: TaskSpec,
        dataset: BaseTaskStreamDataset,
        retrieved: Sequence[RetrievalResult],
    ) -> str:
        prompt = (
            "Solve the task with lightweight retrieval.\n"
            f"Task: {task.input_text}\n"
            f"{dataset.prompt_context(task, stage='cheap')}"
            f"Retrieved experiences:\n{self._format_retrieved(retrieved)}\n"
            f"{dataset.stage_instruction(stage='cheap')}"
        )
        raw_prediction = self.llm_client.generate(prompt, metadata={"stage": "cheap"}).text
        return dataset.postprocess_prediction(raw_prediction, stage="cheap")

    @staticmethod
    def _format_retrieved(retrieved: Sequence[RetrievalResult]) -> str:
        if not retrieved:
            return "- none"
        return "\n".join(
            (
                f"- score={result.score:.3f} "
                f"task={AOMPMemExecutor._shorten(result.entry.task_input, max_chars=120)} "
                f"output={AOMPMemExecutor._shorten(result.entry.task_output, max_chars=80)}"
            )
            for result in retrieved
        )

    @staticmethod
    def _shorten(text: str, *, max_chars: int) -> str:
        compact = " ".join(str(text).split())
        if len(compact) <= max_chars:
            return compact
        return compact[: max_chars - 3] + "..."
