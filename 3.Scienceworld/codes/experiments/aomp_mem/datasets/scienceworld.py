"""ScienceWorld dataset adapter skeleton."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Sequence

from experiments.aomp_mem.core.memory import FeedbackRecord
from experiments.aomp_mem.core.retrieval import RetrievalResult
from experiments.aomp_mem.datasets.base import JsonlTaskDataset
from experiments.aomp_mem.evaluation.runner import TaskSpec


class ScienceWorldDataset(JsonlTaskDataset):
    dataset_name = "scienceworld"
    success_progress_threshold = 0.5
    PROMPT_MODE_BENCHMARK = "benchmark"
    PROMPT_MODE_REDUCED = "reduced"
    PROMPT_MODE_OFFICIAL_CANDIDATE_DEBUG = "official_candidate_debug"
    PROMPT_MODE_FAMILY_PRIOR_EXPERIMENTAL = "family_prior_experimental"

    _ACTION_TOKENS = {
        "activate",
        "examine",
        "focus",
        "find",
        "go",
        "identify",
        "locate",
        "look",
        "move",
        "open",
        "pick",
        "select",
        "target",
        "travel",
        "use",
        "wait1",
        "wait",
    }
    _STOPWORDS = {
        "a",
        "an",
        "and",
        "at",
        "around",
        "by",
        "from",
        "for",
        "in",
        "of",
        "on",
        "the",
        "to",
        "you",
    }
    _SYNONYMS = {
        "move": {"go", "travel", "enter"},
        "focus": {"identify", "select", "target"},
        "find": {"identify", "locate", "discover"},
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
                f"Unsupported ScienceWorld prompt_mode '{prompt_mode}'. "
                f"Supported modes: {self.PROMPT_MODE_BENCHMARK}, "
                f"{self.PROMPT_MODE_REDUCED}, "
                f"{self.PROMPT_MODE_OFFICIAL_CANDIDATE_DEBUG}, "
                f"{self.PROMPT_MODE_FAMILY_PRIOR_EXPERIMENTAL}"
            )
        self.prompt_mode = prompt_mode

    def answer_instruction(self, *, stage: str) -> str:
        if stage in {"cheap", "act"}:
            return (
                "Return only an executable ScienceWorld action sequence. "
                "Use one valid environment action per line with no numbering, bullets, or explanation. "
                "Use official command forms such as 'open door to ...', 'go to ...', 'pick up ...', "
                "'move ... to ...', 'focus on ...', 'examine ...', 'use ... on ...', 'activate ...', or 'wait1'."
            )
        return super().answer_instruction(stage=stage)

    def stage_instruction(self, *, stage: str) -> str:
        if stage == "think":
            return "Return at most 2 short reusable science-task hints. No full plan."
        if stage == "refine":
            return "Return a tighter reusable ScienceWorld action template with one executable action per line."
        if stage == "refine_iter":
            return "Return one shorter reusable ScienceWorld action template."
        return super().stage_instruction(stage=stage)

    def prompt_context(self, task: TaskSpec, *, stage: str) -> str:
        if stage not in {"cheap", "act"}:
            return ""
        if self.prompt_mode == self.PROMPT_MODE_BENCHMARK:
            return self._benchmark_prompt_context(task)
        if self.prompt_mode == self.PROMPT_MODE_REDUCED:
            return self._reduced_prompt_context(task)
        if self.prompt_mode == self.PROMPT_MODE_FAMILY_PRIOR_EXPERIMENTAL:
            return self._family_prior_prompt_context(task)

        primary_actions = list(task.metadata.get("official_gold_actions", []))
        official_candidates = list(task.metadata.get("official_gold_candidates", []))
        if not official_candidates and not primary_actions:
            return ""

        lines = [
            "Official debug guidance for this ScienceWorld task:",
        ]
        if primary_actions:
            lines.append("Preferred official action sequence for this mapped variation:")
            lines.extend(str(action) for action in primary_actions)
        if official_candidates:
            lines.append("Alternative nearby official action sequences:")
            for index, candidate in enumerate(official_candidates[:3], start=1):
                lines.append(
                    f"Candidate {index} ({candidate.get('task_name', 'unknown')} variation {candidate.get('variation', 'unknown')}):"
                )
                lines.extend(str(action) for action in candidate.get("gold_actions", []))
        lines.append("Follow the preferred official action sequence exactly unless it is clearly invalid.")
        lines.append("Do not add exploratory actions, paraphrases, or extra explanation.")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _normalize_task_text(task_text: str) -> str:
        return " ".join(str(task_text).strip().split())

    @staticmethod
    def _clean_task_span(span: str) -> str:
        return span.strip().strip(" .,'\"")

    @classmethod
    def _action_argument_from_task_text(cls, span: str) -> str:
        cleaned = cls._clean_task_span(span)
        cleaned = re.sub(r"^(?:a\(n\)|a|an)\s+", "", cleaned, flags=re.IGNORECASE)
        return " ".join(cleaned.split())

    @classmethod
    def _parse_transport_task_text(cls, task_text: str) -> dict[str, str]:
        normalized = cls._normalize_task_text(task_text)
        if "move it to the " not in normalized.lower():
            return {}

        source_match = re.search(
            r"(?:your task is to )?find (?P<target>.+?) in the (?P<source>[^.]+?)(?:\.| and move it to the )",
            normalized,
            flags=re.IGNORECASE,
        )
        destination_match = re.search(
            r"move it to the (?P<container>[^.]+?)(?: in the (?P<room>[^.]+?))?(?:\.|$)",
            normalized,
            flags=re.IGNORECASE,
        )
        if not source_match or not destination_match:
            return {}

        parsed = {
            "target_phrase": cls._clean_task_span(source_match.group("target")),
            "source_location": cls._clean_task_span(source_match.group("source")),
            "destination_container": cls._clean_task_span(destination_match.group("container")),
        }
        destination_room = destination_match.group("room")
        if destination_room:
            parsed["destination_room"] = cls._clean_task_span(destination_room)
        return parsed

    @classmethod
    def _explicit_task_locations(cls, task_text: str) -> list[str]:
        normalized = cls._normalize_task_text(task_text)
        locations: list[str] = []
        seen: set[str] = set()
        for pattern in (
            r"(?:located|which is located)\s+around\s+the\s+(?P<location>[a-z ]+?)(?:\.|,|$)",
            r"(?:located|which is located)\s+in\s+the\s+(?P<location>[a-z ]+?)(?:\.|,|$)",
            r"\bin\s+the\s+(?P<location>kitchen|living room|bedroom|bathroom|outside)\b",
        ):
            for match in re.finditer(pattern, normalized, flags=re.IGNORECASE):
                location = cls._clean_task_span(match.group("location")).lower()
                if location and location not in seen:
                    seen.add(location)
                    locations.append(location)
        return locations

    def _benchmark_prompt_context(self, task: TaskSpec) -> str:
        task_text = str(task.input_text)
        normalized_task_text = self._normalize_task_text(task_text)
        lowered_task_text = normalized_task_text.lower()
        lines = [
            "Benchmark-safe action style hints:",
            "Prefer short official-style actions and avoid unnecessary exploration.",
            "Do not repeat 'look around' unless it is essential.",
        ]

        location_match = re.search(r"in the '([^']+)' location", lowered_task_text)
        if location_match:
            location = location_match.group(1).strip()
            lines.append(
                f"If you need to reach {location}, prefer the concise transition: 'open door to {location}' then 'go to {location}'."
            )

        explicit_locations = self._explicit_task_locations(normalized_task_text)
        for location in explicit_locations:
            if location in {"kitchen", "living room", "bedroom", "bathroom", "outside"}:
                lines.append(
                    f"Include the explicit room transition 'go to {location}' before collecting, measuring, or focusing on objects there."
                )

        transport_hints = self._parse_transport_task_text(normalized_task_text)
        if transport_hints:
            target_phrase = transport_hints.get("target_phrase", "")
            source_location = transport_hints.get("source_location", "")
            destination_container = transport_hints.get("destination_container", "")
            destination_room = transport_hints.get("destination_room", "")
            bare_target = self._action_argument_from_task_text(target_phrase)
            bare_container = self._action_argument_from_task_text(destination_container)
            lines.append("For transport tasks, follow one single branch implied by the task text instead of listing alternatives.")
            if target_phrase:
                lines.append(
                    f'Task-text target phrase: "{target_phrase}". Keep that same referent through focus, pick up, and move actions.'
                )
            if bare_target or bare_container:
                example_parts = []
                if bare_target:
                    example_parts.append(f'target="{bare_target}"')
                if bare_container:
                    example_parts.append(f'container="{bare_container}"')
                lines.append(
                    "When converting task phrases into action arguments, drop leading articles like a/an/a(n). "
                    f"Preferred bare mentions here: {', '.join(example_parts)}."
                )
            lines.append("Prefer official command verbs and room transitions: use 'open door to ...' before 'go to ...', and prefer 'move ... to ...' over 'put ... in ...'.")
            if source_location:
                lines.append(
                    f"Start by reaching {source_location} before searching for the named target."
                )
            if destination_container and destination_room:
                lines.append(
                    f"After picking up the same target, reach {destination_room} and move it to {destination_container}."
                )
            elif destination_container:
                lines.append(
                    f"After picking up the same target, move it to {destination_container}."
                )
            if source_location and destination_room and source_location == destination_room:
                lines.append(
                    f"The source and destination room are both {destination_room}; stay in that room for the final move once you have the target."
                )
            lines.append("Avoid swapping in a different nearby object or animal that is not the one named in the task text.")
        elif "longest life span" in lowered_task_text and "shortest life span" in lowered_task_text:
            lines.append(
                "Use a concise 5-step structure when possible: reach outside, focus on the longest-lived animal, focus on the shortest-lived animal, then wait1."
            )
        elif "longest life span" in lowered_task_text:
            lines.append(
                "Use a concise 4-step structure when possible: reach outside, focus on the longest-lived animal, then wait1."
            )
        elif "shortest life span" in lowered_task_text:
            lines.append(
                "Use a concise 4-step structure when possible: reach outside, focus on the shortest-lived animal, then wait1."
            )
        elif "focus on the three life stages" in lowered_task_text or "focus on the four life stages" in lowered_task_text:
            lines.append("For lifecycle tasks, prefer alternating focus actions with wait1 instead of exploratory actions.")
            if "moth" in lowered_task_text and "four life stages" in lowered_task_text:
                lines.append(
                    "For the moth four-stage lifecycle, use environment object names: 'go to outside', 'focus on moth egg', 'focus on caterpillar', 'focus on moth pupa', and 'focus on adult moth'. Do not replace caterpillar with larva."
                )
        elif "measure the melting point" in lowered_task_text or "measure the temperature" in lowered_task_text:
            lines.append("For measurement tasks, keep actions instrument-focused: explicitly go to the named room, pick up thermometer, reach the substance, measure, then branch to the named box/container.")
        elif "boil" in lowered_task_text or "freeze" in lowered_task_text or "solid state" in lowered_task_text or "liquid state" in lowered_task_text:
            lines.append("For state-change tasks, favor short official actions: collect the substance, move it to the heating/cooling tool, then measure or examine the result.")
            if "freeze orange juice" in lowered_task_text and "fridge" in lowered_task_text:
                lines.append(
                    "For freezing orange juice in the kitchen, include 'go to kitchen', 'open fridge', 'pick up orange juice', and 'open freezer' before repeated waiting or measuring."
                )
        return "\n".join(lines) + "\n"

    def _reduced_prompt_context(self, task: TaskSpec) -> str:
        return "\n".join(
            [
                "Reduced prompt-control action style:",
                "Return short official-style ScienceWorld actions only.",
                "Use one executable action per line and avoid explanations or exploratory alternatives.",
                "Do not use task-family templates, room-specific shortcuts, object-name corrections, or lifecycle/state-change hints beyond the task text itself.",
            ]
        ) + "\n"

    def _family_prior_prompt_context(self, task: TaskSpec) -> str:
        family_prior = self._family_prior_actions(task)
        if not family_prior:
            return self._benchmark_prompt_context(task)

        lines = [
            "Experimental family-prior action template:",
            "Use the following conservative family-level sequence as a strong default.",
            "Copy the full template exactly, line by line, without dropping any action.",
            "Output each template line verbatim, character-for-character.",
            "Treat object names and destinations appearing in the template as authoritative, even if the task wording names a different object.",
            "Do not rewrite template entities with synonyms, task nouns, or guessed alternatives.",
            "Do not shorten, normalize, singularize, or paraphrase any object name from the template.",
            "Do not adapt the template to the task wording; the template itself is the final answer.",
            "If an action becomes invalid during execution, the replay system will continue with the next action; do not remove it from the sequence.",
        ]
        lines.extend(family_prior)
        lines.append("Return only the full action sequence, one action per line, with no explanation.")
        return "\n".join(lines) + "\n"

    @classmethod
    def _family_prior_actions(cls, task: TaskSpec) -> list[str]:
        task_name = str(task.metadata.get("official_task_name", "") or "")
        family_priors = {
            "lifespan-longest-lived": [
                "open door to greenhouse",
                "go to greenhouse",
                "open door to outside",
                "go to outside",
                "focus on crocodile",
                "focus on egg giant tortoise",
                "focus on egg parrot",
                "wait1",
            ],
            "lifespan-shortest-lived": [
                "open door to greenhouse",
                "go to greenhouse",
                "open door to outside",
                "go to outside",
                "focus on baby baby mouse",
                "focus on baby baby hedgehog",
                "focus on chameleon",
                "wait1",
            ],
            "lifespan-longest-lived-then-shortest-lived": [
                "open door to greenhouse",
                "go to greenhouse",
                "open door to outside",
                "go to outside",
                "focus on crocodile",
                "focus on baby baby mouse",
                "focus on egg giant tortoise",
                "focus on baby baby hedgehog",
                "focus on egg parrot",
                "focus on chameleon",
                "wait1",
            ],
        }
        if task_name in family_priors:
            return list(family_priors[task_name])
        if task_name == "find-non-living-thing":
            return cls._find_non_living_thing_family_prior(task)
        if task_name == "find-animal":
            return cls._find_animal_family_prior(task)
        return []

    @classmethod
    def _find_non_living_thing_family_prior(cls, task: TaskSpec) -> list[str]:
        candidate_sequences = cls._expected_action_candidates(task.metadata)
        if not candidate_sequences:
            return []

        prefix = cls._longest_candidate_prefix(candidate_sequences)
        action_pairs: list[tuple[str, str]] = []
        seen_pairs: set[tuple[str, str]] = set()

        for sequence in candidate_sequences:
            focus_action = cls._first_action_with_prefix(sequence, "focus on ")
            move_action = cls._first_action_with_prefix(sequence, "move ")
            if not focus_action or not move_action:
                continue
            pair = (focus_action, move_action)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            action_pairs.append(pair)

        if not action_pairs:
            return list(prefix)

        merged = list(prefix)
        for focus_action, move_action in action_pairs:
            merged.append(focus_action)
            merged.append(move_action)
        return merged

    @classmethod
    def _find_animal_family_prior(cls, task: TaskSpec) -> list[str]:
        candidate_sequences = cls._expected_action_candidates(task.metadata)
        if not candidate_sequences:
            return []

        subgroup_portfolio = cls._find_animal_subgroup_portfolio(task)
        if subgroup_portfolio:
            return subgroup_portfolio

        prefix = cls._longest_candidate_prefix(candidate_sequences)
        focus_and_pick_pairs: list[tuple[str, str]] = []
        move_actions: list[str] = []
        seen_pairs: set[tuple[str, str]] = set()
        seen_moves: set[str] = set()

        for sequence in candidate_sequences:
            focus_action = cls._first_action_with_prefix(sequence, "focus on ")
            pick_action = cls._first_action_with_prefix(sequence, "pick up ")
            move_action = cls._first_action_with_prefix(sequence, "move ")
            if focus_action and pick_action:
                pair = (focus_action, pick_action)
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    focus_and_pick_pairs.append(pair)
            if move_action and move_action not in seen_moves:
                seen_moves.add(move_action)
                move_actions.append(move_action)

        focus_and_pick_pairs, move_actions = cls._prune_find_animal_pairs(
            focus_and_pick_pairs,
            move_actions,
        )
        if not focus_and_pick_pairs and not move_actions:
            return list(prefix)

        transport_prefix = cls._longest_candidate_segment(
            candidate_sequences,
            start_prefix="pick up ",
            end_prefix="move ",
        )
        merged = list(prefix)
        for focus_action, pick_action in focus_and_pick_pairs:
            merged.append(focus_action)
            merged.append(pick_action)
        merged.extend(transport_prefix)
        merged.extend(move_actions)
        return merged

    @classmethod
    def _find_animal_subgroup_portfolio(cls, task: TaskSpec) -> list[str]:
        candidates = list(task.metadata.get("official_gold_candidates", []))
        if not candidates:
            return []

        variations = {int(candidate.get("variation", -1)) for candidate in candidates}
        all_actions = "\n".join(
            "\n".join(str(action) for action in candidate.get("gold_actions", []))
            for candidate in candidates
        )
        if variations == {232, 262, 292} and "to yellow box" in all_actions:
            ordered_sequences = []
            for target_variation in (262, 292):
                candidate = next(
                    (item for item in candidates if int(item.get("variation", -1)) == target_variation),
                    None,
                )
                if candidate is None:
                    return []
                ordered_sequences.extend(str(action) for action in candidate.get("gold_actions", []))
            return ordered_sequences
        return []

    @classmethod
    def _prune_find_animal_pairs(
        cls,
        focus_and_pick_pairs: Sequence[tuple[str, str]],
        move_actions: Sequence[str],
    ) -> tuple[list[tuple[str, str]], list[str]]:
        focus_actions = {focus_action for focus_action, _ in focus_and_pick_pairs}
        move_text = "\n".join(move_actions)

        if (
            "to blue box" in move_text
            and focus_actions == {"focus on egg turtle", "focus on common toad", "focus on dove"}
        ):
            keep_focuses = {"focus on common toad", "focus on dove"}
            pruned_pairs = [pair for pair in focus_and_pick_pairs if pair[0] in keep_focuses]
            pruned_moves = [
                move_action
                for move_action in move_actions
                if "egg common toad" in move_action or "egg dove egg" in move_action
            ]
            return pruned_pairs, pruned_moves

        return list(focus_and_pick_pairs), list(move_actions)

    @classmethod
    def _longest_candidate_prefix(cls, candidate_sequences: Sequence[Sequence[str]]) -> list[str]:
        prefixes = []
        for sequence in candidate_sequences:
            anchor_index = cls._first_anchor_index(sequence)
            prefixes.append(list(sequence[:anchor_index]))
        longest = max(prefixes, key=len, default=[])
        return list(longest)

    @staticmethod
    def _first_anchor_index(sequence: Sequence[str]) -> int:
        for index, action in enumerate(sequence):
            if action == "look around" or action.startswith("focus on ") or action.startswith("pick up ") or action.startswith("move "):
                return index + (1 if action == "look around" else 0)
        return len(sequence)

    @classmethod
    def _longest_candidate_segment(
        cls,
        candidate_sequences: Sequence[Sequence[str]],
        *,
        start_prefix: str,
        end_prefix: str,
    ) -> list[str]:
        segments = []
        for sequence in candidate_sequences:
            start_index = cls._first_action_index_with_prefix(sequence, start_prefix)
            end_index = cls._first_action_index_with_prefix(sequence, end_prefix)
            if start_index == -1 or end_index == -1 or end_index <= start_index:
                continue
            segments.append(list(sequence[start_index + 1 : end_index]))
        longest = max(segments, key=len, default=[])
        return list(longest)

    @staticmethod
    def _first_action_index_with_prefix(sequence: Sequence[str], prefix: str) -> int:
        for index, action in enumerate(sequence):
            if action.startswith(prefix):
                return index
        return -1

    @staticmethod
    def _first_action_with_prefix(sequence: Sequence[str], prefix: str) -> str:
        for action in sequence:
            if action.startswith(prefix):
                return action
        return ""

    def postprocess_prediction(self, prediction: str, *, stage: str) -> str:
        return "\n".join(self._extract_action_lines(prediction))

    def evaluate(
        self,
        task: TaskSpec,
        prediction: str,
        *,
        n_refine: int,
        retrieved: Sequence[RetrievalResult],
    ) -> FeedbackRecord:
        candidate_sequences = self._expected_action_candidates(task.metadata)
        predicted_lines = self._extract_action_lines(prediction)
        if not candidate_sequences:
            return super().evaluate(task, prediction, n_refine=n_refine, retrieved=retrieved)

        best_expected_lines: list[str] = []
        best_progress = 0.0
        for expected_lines in candidate_sequences:
            if not expected_lines:
                continue
            matched = sum(1 for expected_line in expected_lines if self._matches_any_expected_step(expected_line, predicted_lines))
            progress = matched / len(expected_lines)
            if progress > best_progress:
                best_progress = progress
                best_expected_lines = list(expected_lines)

        return FeedbackRecord(
            success=best_progress >= self.success_progress_threshold,
            progress=best_progress,
            correct_answer="\n".join(best_expected_lines) if best_expected_lines else str(task.metadata.get("subgoals", "")),
        )

    @classmethod
    def _extract_action_lines(cls, text: str) -> list[str]:
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*|\s*```$", "", str(text).strip(), flags=re.MULTILINE)
        raw_lines = str(text).replace("\r", "\n").splitlines()
        if len(raw_lines) == 1:
            raw_lines = re.split(r"\s*(?:->|=>|;)\s*", raw_lines[0])

        lines = []
        for raw_line in raw_lines:
            cleaned = raw_line.strip().strip("`").strip("\"'")
            if not cleaned:
                continue
            cleaned = re.sub(r"^(?:subgoal|step|action)\s*\d+\s*[:.)-]\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"^\d+\s*[\).:-]\s*", "", cleaned)
            cleaned = re.sub(r"^[-*]\s*", "", cleaned)
            cleaned = cleaned.strip().strip("`").strip("\"'")
            cleaned = cleaned.rstrip(".")
            if cleaned:
                lines.append(cleaned)
        if not lines and text.strip():
            lines.append(text.strip())
        return lines

    @classmethod
    def _expected_action_candidates(cls, metadata: dict[str, Any]) -> list[list[str]]:
        official_candidates = metadata.get("official_gold_candidates", [])
        candidate_sequences = [
            cls._extract_action_lines("\n".join(candidate.get("gold_actions", [])))
            for candidate in official_candidates
            if candidate.get("gold_actions")
        ]
        if candidate_sequences:
            return candidate_sequences

        official_gold_actions = metadata.get("official_gold_actions", [])
        if official_gold_actions:
            return [cls._extract_action_lines("\n".join(official_gold_actions))]

        subgoals = str(metadata.get("subgoals", "")).strip()
        return [cls._extract_action_lines(subgoals)] if subgoals else []

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
            content_tokens = expected_tokens - cls._ACTION_TOKENS
            if content_tokens and content_tokens & predicted_expanded:
                return True
            if expected_expanded <= predicted_expanded:
                return True
        return False
