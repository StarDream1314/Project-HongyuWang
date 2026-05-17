"""ALFWorld dataset adapter for subgoal-plan evaluation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

from experiments.aomp_mem.core.memory import FeedbackRecord
from experiments.aomp_mem.core.retrieval import RetrievalResult
from experiments.aomp_mem.datasets.base import JsonlTaskDataset
from experiments.aomp_mem.evaluation.runner import TaskSpec


class AlfWorldDataset(JsonlTaskDataset):
    dataset_name = "alfworld"
    success_progress_threshold = 2.0 / 3.0
    PROMPT_MODE_BENCHMARK = "benchmark"
    PROMPT_MODE_REDUCED = "reduced"
    PROMPT_MODE_OFFICIAL_CANDIDATE_DEBUG = "official_candidate_debug"
    PROMPT_MODE_FAMILY_PRIOR_EXPERIMENTAL = "family_prior_experimental"

    _STOPWORDS = {
        "a",
        "an",
        "and",
        "at",
        "in",
        "into",
        "of",
        "on",
        "the",
        "to",
        "you",
    }
    _SYNONYMS = {
        "see": {"find", "locate", "look", "inspect", "observe"},
        "pick": {"grab", "take", "collect", "hold"},
        "put": {"place", "move", "insert", "drop"},
        "look": {"inspect", "examine", "find", "see"},
        "heat": {"warm", "microwave", "boil"},
        "cool": {"chill", "refrigerate", "freeze"},
        "clean": {"wash", "rinse", "sanitize"},
    }

    def __init__(self, path: Path, *, prompt_mode: str = PROMPT_MODE_BENCHMARK) -> None:
        super().__init__(path=path, dataset_name=self.dataset_name, input_field="input")
        if prompt_mode not in {
            self.PROMPT_MODE_BENCHMARK,
            self.PROMPT_MODE_REDUCED,
            self.PROMPT_MODE_OFFICIAL_CANDIDATE_DEBUG,
            self.PROMPT_MODE_FAMILY_PRIOR_EXPERIMENTAL,
        }:
            raise ValueError(
                f"Unsupported ALFWorld prompt_mode '{prompt_mode}'. "
                f"Supported modes: {self.PROMPT_MODE_BENCHMARK}, "
                f"{self.PROMPT_MODE_REDUCED}, "
                f"{self.PROMPT_MODE_OFFICIAL_CANDIDATE_DEBUG}, "
                f"{self.PROMPT_MODE_FAMILY_PRIOR_EXPERIMENTAL}"
            )
        self.prompt_mode = prompt_mode

    def answer_instruction(self, *, stage: str) -> str:
        if stage in {"cheap", "act"}:
            return (
                "Return only a concrete ALFWorld subgoal plan. "
                "Use one line per step in the format 'Subgoal N: <action>'. "
                "Use exact object and receptacle names from the task when possible. "
                "Prefer concrete verbs such as look at, pick up, put in/on, heat, cool, clean, or open. "
                "Do not add explanation."
            )
        return super().answer_instruction(stage=stage)

    def stage_instruction(self, *, stage: str) -> str:
        if stage == "think":
            return "Return at most 2 short reusable household-action hints. No full plan."
        if stage == "refine":
            return "Return a tighter reusable ALFWorld subgoal template with one step per line."
        if stage == "refine_iter":
            return "Return one shorter reusable ALFWorld subgoal template."
        return super().stage_instruction(stage=stage)

    def prompt_context(self, task: TaskSpec, *, stage: str) -> str:
        if stage not in {"cheap", "act"}:
            return ""
        if self.prompt_mode == self.PROMPT_MODE_REDUCED:
            return "\n".join(
                [
                    "Reduced prompt-control action style:",
                    "Return short ALFWorld subgoals only.",
                    "Use one executable step per line and avoid explanations or alternatives.",
                ]
            ) + "\n"
        if self.prompt_mode == self.PROMPT_MODE_OFFICIAL_CANDIDATE_DEBUG:
            return "\n".join(
                [
                    "Official debug guidance for this ALFWorld task:",
                    "Use the task text to derive a compact step-by-step plan.",
                    "Do not paraphrase object names or invent extra branches.",
                ]
            ) + "\n"
        if self.prompt_mode == self.PROMPT_MODE_FAMILY_PRIOR_EXPERIMENTAL:
            return "\n".join(
                [
                    "Experimental family-prior action template:",
                    "Use conservative object-centric subgoals derived from the task text.",
                    "Return the final plan only, one step per line.",
                ]
            ) + "\n"
        task_text = str(task.input_text)
        lines = [
            "Benchmark-safe ALFWorld subgoal hints:",
            "Prefer short concrete steps aligned with the task text.",
            "Use one subgoal per line and keep exact object names when possible.",
        ]
        if "look at" in task_text.lower():
            lines.append("Start by finding the named object and then inspect or pick it up if required.")
        if "put" in task_text.lower():
            lines.append("For put tasks, keep the same object referent from pickup through placement.")
        if "cool" in task_text.lower() or "heat" in task_text.lower() or "clean" in task_text.lower():
            lines.append("For state-change tasks, be explicit about the tool/container used for the transformation.")
        return "\n".join(lines) + "\n"

    def postprocess_prediction(self, prediction: str, *, stage: str) -> str:
        return "\n".join(
            f"Subgoal {index}: {line}"
            for index, line in enumerate(self._extract_subgoal_lines(prediction), start=1)
        )

    def evaluate(
        self,
        task: TaskSpec,
        prediction: str,
        *,
        n_refine: int,
        retrieved: Sequence[RetrievalResult],
    ) -> FeedbackRecord:
        expected_lines = self._extract_subgoal_lines(str(task.metadata.get("subgoals", "")))
        predicted_lines = self._extract_subgoal_lines(prediction)
        if not expected_lines:
            return super().evaluate(task, prediction, n_refine=n_refine, retrieved=retrieved)

        matched = sum(
            1
            for expected_line in expected_lines
            if self._matches_any_expected_step(expected_line, predicted_lines)
        )
        progress = matched / len(expected_lines)
        return FeedbackRecord(
            success=progress >= self.success_progress_threshold,
            progress=progress,
            correct_answer=str(task.metadata.get("subgoals", "")),
        )

    @classmethod
    def _extract_subgoal_lines(cls, text: str) -> list[str]:
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*|\s*```$", "", str(text).strip(), flags=re.MULTILINE)
        lines = []
        for raw_line in text.replace("\r", "\n").splitlines():
            cleaned = raw_line.strip().strip("`").strip("\"'")
            if not cleaned:
                continue
            cleaned = re.sub(r"^subgoal\s*\d+\s*:\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"^(?:step|action)\s*\d+\s*[:.)-]\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"^\d+\s*[\).:-]\s*", "", cleaned)
            cleaned = re.sub(r"^[-*]\s*", "", cleaned)
            cleaned = cleaned.strip().strip("`").strip("\"'").rstrip(".")
            if cleaned:
                lines.append(cleaned)
        if not lines and text.strip():
            lines.append(text.strip())
        return lines

    @classmethod
    def _tokenize(cls, text: str) -> set[str]:
        tokens = set(re.findall(r"[a-z]+", text.lower()))
        return {token for token in tokens if token not in cls._STOPWORDS}

    @classmethod
    def _expand_token_set(cls, tokens: set[str]) -> set[str]:
        expanded = set(tokens)
        for token in list(tokens):
            expanded.update(cls._SYNONYMS.get(token, set()))
        return expanded

    @classmethod
    def _matches_any_expected_step(cls, expected_line: str, predicted_lines: Sequence[str]) -> bool:
        expected_tokens = cls._tokenize(expected_line)
        if not expected_tokens:
            return False
        expected_expanded = cls._expand_token_set(expected_tokens)

        for predicted_line in predicted_lines:
            predicted_tokens = cls._tokenize(predicted_line)
            predicted_expanded = cls._expand_token_set(predicted_tokens)
            overlap = expected_tokens & predicted_expanded
            if len(overlap) >= max(1, min(2, len(expected_tokens))):
                return True
            if expected_expanded <= predicted_expanded:
                return True
        return False
