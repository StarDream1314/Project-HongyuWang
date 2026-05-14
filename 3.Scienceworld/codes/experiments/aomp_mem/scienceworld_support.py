"""Helpers for running ScienceWorld reliably on Windows user installs."""

from __future__ import annotations

import inspect
import json
import re
import shutil
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class ScienceWorldRuntimeConfig:
    """Resolved local runtime assets required to start ScienceWorld."""

    runtime_dir: Path
    java_path: Path
    py4j_jar_path: Path
    scienceworld_jar_path: Path


def code_root() -> Path:
    return Path(__file__).resolve().parents[3]


_DESCRIPTION_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "around",
    "be",
    "done",
    "first",
    "for",
    "if",
    "in",
    "is",
    "it",
    "its",
    "located",
    "of",
    "on",
    "or",
    "the",
    "then",
    "to",
    "when",
    "which",
    "you",
    "your",
}

_ANIMAL_KEYWORDS = {
    "animal",
    "ant",
    "bee",
    "bird",
    "blue",
    "bluejay",
    "bluejayegg",
    "blue jay",
    "butterfly",
    "butterflyegg",
    "butterfly egg",
    "chameleon",
    "common",
    "crocodile",
    "dove",
    "doveegg",
    "dove egg",
    "dragonfly",
    "egg",
    "frog",
    "hedgehog",
    "jay",
    "lizard",
    "mouse",
    "parrot",
    "toad",
    "tortoise",
    "turtle",
    "wolf",
}
_PLANT_KEYWORDS = {
    "apple tree",
    "banana tree",
    "flower",
    "grass",
    "moss",
    "plant",
    "tree",
}
_NON_LIVING_KEYWORDS = {
    "book",
    "closet",
    "counter",
    "fork",
    "painting",
    "picture",
    "table",
}
_BOX_COLORS = {"red", "green", "blue", "orange", "yellow", "purple"}
_ROOM_NAMES = {"bedroom", "bathroom", "kitchen", "living room", "workshop", "art studio", "greenhouse", "outside"}


def normalize_scienceworld_text(text: str) -> str:
    """Collapse one ScienceWorld description/action string for matching."""

    normalized = re.sub(r"\s+", " ", str(text).strip().lower())
    normalized = normalized.replace(" then the shortest life span. focus on ", " then the shortest life span. first, focus on ")
    normalized = normalized.replace(" you also need to focus on ", " then, focus on ")
    normalized = normalized.replace(" and, focus on ", " then, focus on ")
    normalized = normalized.replace(" focus on it. then, move it to ", " focus on the thing. then, move it to ")
    return normalized.strip(" .")


def _description_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", normalize_scienceworld_text(text))
        if token not in _DESCRIPTION_STOPWORDS
    }


def _focus_targets_from_subgoals(text: str) -> list[set[str]]:
    targets: list[set[str]] = []
    for raw_line in str(text).splitlines():
        line = normalize_scienceworld_text(re.sub(r"^subgoal\s*\d+\s*:\s*", "", raw_line, flags=re.IGNORECASE))
        match = re.search(r"focus on (.+)$", line)
        if not match:
            continue
        tokens = _description_tokens(match.group(1))
        if tokens:
            targets.append(tokens)
    return targets


def _focus_targets_from_actions(actions: Iterable[str]) -> list[set[str]]:
    targets: list[set[str]] = []
    for action in actions:
        match = re.search(r"focus on (.+)$", normalize_scienceworld_text(action))
        if not match:
            continue
        tokens = _description_tokens(match.group(1))
        if tokens:
            targets.append(tokens)
    return targets


def _goal_match_score(goal_text: str, description: str) -> float:
    goal_normalized = normalize_scienceworld_text(goal_text)
    description_normalized = normalize_scienceworld_text(description)
    if goal_normalized == description_normalized:
        return 1.0

    goal_tokens = _description_tokens(goal_text)
    description_tokens = _description_tokens(description)
    token_overlap = (
        len(goal_tokens & description_tokens) / len(goal_tokens | description_tokens)
        if goal_tokens or description_tokens
        else 0.0
    )
    sequence_score = SequenceMatcher(None, goal_normalized, description_normalized).ratio()
    return (0.65 * sequence_score) + (0.35 * token_overlap)


def _extract_find_move_goal_fields(goal_text: str) -> dict[str, str] | None:
    match = re.search(
        r"your task is to find\s+(?P<object>.+?)\.\s*first,\s*focus on the thing\.\s*then,\s*move it to the\s+"
        r"(?P<color>red|green|blue|orange|yellow|purple)\s+box in the\s+(?P<room>[a-z ]+)$",
        normalize_scienceworld_text(goal_text),
    )
    if not match:
        return None
    object_phrase = match.group("object").strip()
    object_phrase = re.sub(r"^(?:a\(n\)\s+|a\s+|an\s+)", "", object_phrase).strip()
    room = match.group("room").strip()
    color = match.group("color").strip()
    return {"object_phrase": object_phrase, "color": color, "room": room}


def _infer_find_goal_family(object_phrase: str) -> str | None:
    normalized_object = normalize_scienceworld_text(object_phrase)
    if any(keyword in normalized_object for keyword in sorted(_PLANT_KEYWORDS, key=len, reverse=True)):
        return "find-plant"
    if any(keyword in normalized_object for keyword in sorted(_NON_LIVING_KEYWORDS, key=len, reverse=True)):
        return "find-non-living-thing"
    if any(keyword in normalized_object for keyword in sorted(_ANIMAL_KEYWORDS, key=len, reverse=True)):
        return "find-animal"
    return None


def _render_find_goal_template(*, family_task_name: str, color: str, room: str) -> str:
    family_label = {
        "find-animal": "a(n) animal",
        "find-living-thing": "a(n) living thing",
        "find-non-living-thing": "a(n) non-living thing",
        "find-plant": "a(n) plant",
    }[family_task_name]
    return f"Your task is to find {family_label}. First, focus on the thing. Then, move it to the {color} box in the {room}."


def resolve_template_family_match(
    *,
    goal_text: str,
    catalog_rows: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    parsed = _extract_find_move_goal_fields(goal_text)
    if not parsed:
        return {"match_type": "none", "primary": None, "candidates": []}

    family_task_name = _infer_find_goal_family(parsed["object_phrase"])
    if not family_task_name:
        return {"match_type": "none", "primary": None, "candidates": []}

    template_description = normalize_scienceworld_text(
        _render_find_goal_template(
            family_task_name=family_task_name,
            color=parsed["color"],
            room=parsed["room"],
        )
    )
    candidates = [
        row
        for row in catalog_rows
        if str(row.get("task_name", "")) == family_task_name
        and normalize_scienceworld_text(str(row.get("description", ""))) == template_description
    ]
    if not candidates:
        return {"match_type": "none", "primary": None, "candidates": []}

    sorted_candidates = sorted(candidates, key=lambda row: int(row.get("variation", -1)))
    return {
        "match_type": "template_family",
        "primary": sorted_candidates[0],
        "candidates": sorted_candidates,
    }


def _candidate_disambiguation_score(subgoals_text: str, gold_actions: Iterable[str]) -> int:
    expected_focus_targets = _focus_targets_from_subgoals(subgoals_text)
    action_focus_targets = _focus_targets_from_actions(gold_actions)
    if not expected_focus_targets or not action_focus_targets:
        return 0

    score = 0
    for expected_target, action_target in zip(expected_focus_targets, action_focus_targets):
        overlap = len(expected_target & action_target)
        if overlap:
            score += overlap * 10
        elif expected_target == action_target:
            score += 25
    return score


def load_goldpath_catalog(catalog_path: Path) -> list[dict[str, Any]]:
    """Load one exported ScienceWorld official gold-path catalog."""

    resolved_path = Path(catalog_path)
    if not resolved_path.exists():
        return []
    with resolved_path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve_goldpath_match(
    *,
    goal_text: str,
    subgoals_text: str,
    catalog_rows: Iterable[dict[str, Any]],
    min_similarity: float = 0.72,
) -> dict[str, Any]:
    """Match one raw ScienceWorld task to official gold-path candidates."""

    resolved_rows = list(catalog_rows)
    if not resolved_rows:
        return {"match_type": "none", "primary": None, "candidates": []}

    template_family_match = resolve_template_family_match(
        goal_text=goal_text,
        catalog_rows=resolved_rows,
    )
    if template_family_match["primary"] is not None:
        return template_family_match

    exact_candidates = [
        row
        for row in resolved_rows
        if normalize_scienceworld_text(row.get("description", "")) == normalize_scienceworld_text(goal_text)
    ]
    if exact_candidates:
        sorted_candidates = sorted(
            exact_candidates,
            key=lambda row: (
                -_candidate_disambiguation_score(subgoals_text, row.get("gold_actions", [])),
                int(row.get("variation", -1)),
            ),
        )
        return {
            "match_type": "exact",
            "primary": sorted_candidates[0],
            "candidates": sorted_candidates,
        }

    scored_candidates = [
        (_goal_match_score(goal_text, str(row.get("description", ""))), row)
        for row in resolved_rows
    ]
    best_score = max((score for score, _ in scored_candidates), default=0.0)
    if best_score < min_similarity:
        return {"match_type": "none", "primary": None, "candidates": []}

    shortlisted = [
        row
        for score, row in scored_candidates
        if abs(score - best_score) <= 1e-9
    ]
    sorted_candidates = sorted(
        shortlisted,
        key=lambda row: (
            -_candidate_disambiguation_score(subgoals_text, row.get("gold_actions", [])),
            int(row.get("variation", -1)),
        ),
    )
    return {
        "match_type": "similarity",
        "primary": sorted_candidates[0],
        "candidates": sorted_candidates,
    }


def _candidate_java_paths() -> list[Path]:
    candidates: list[Path] = []

    import os

    java_home_value = os.environ.get("JAVA_HOME")
    if java_home_value:
        candidates.append(Path(java_home_value) / "bin" / "java.exe")
        candidates.append(Path(java_home_value) / "bin" / "java")

    which_java = shutil.which("java")
    if which_java:
        candidates.append(Path(which_java))

    candidates.extend(
        [
            Path(r"C:\Program Files\Android\Android Studio\jbr\bin\java.exe"),
            Path(r"C:\Program Files\Android\jdk\jdk-8.0.302.8-hotspot\jdk8u302-b08\bin\java.exe"),
            Path(r"C:\Program Files\Eclipse Adoptium\jre-17.0.0.35-hotspot\bin\java.exe"),
            Path(r"C:\Program Files\IBM\SPSS\Statistics\27\JRE\bin\java.exe"),
        ]
    )
    return candidates


def find_java_path() -> Path:
    """Return one usable Java executable path."""

    for candidate in _candidate_java_paths():
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate a Java runtime. Set JAVA_HOME or install a JRE/JDK so ScienceWorld can start."
    )


def _resolve_py4j_jar_path() -> Path:
    import py4j.java_gateway as java_gateway

    discovered = java_gateway.find_jar_path()
    if discovered:
        discovered_path = Path(discovered)
        if discovered_path.exists():
            return discovered_path

    package_file = Path(inspect.getfile(java_gateway)).resolve()
    candidate_paths = [
        package_file.parents[3] / "share" / "py4j" / f"py4j{java_gateway.__version__}.jar",
        package_file.parent / f"py4j{java_gateway.__version__}.jar",
    ]
    for candidate in candidate_paths:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not locate the py4j jar required by ScienceWorld.")


def prepare_scienceworld_runtime(runtime_dir: Optional[Path] = None) -> ScienceWorldRuntimeConfig:
    """Copy ScienceWorld and py4j jars into an ASCII-safe workspace directory."""

    import scienceworld.scienceworld as scienceworld_module

    resolved_runtime_dir = Path(runtime_dir) if runtime_dir is not None else code_root() / "third_party" / "scienceworld_runtime"
    resolved_runtime_dir.mkdir(parents=True, exist_ok=True)

    py4j_jar_src = _resolve_py4j_jar_path()
    scienceworld_jar_src = Path(scienceworld_module.JAR_PATH)
    py4j_jar_dst = resolved_runtime_dir / py4j_jar_src.name
    scienceworld_jar_dst = resolved_runtime_dir / scienceworld_jar_src.name

    shutil.copy2(py4j_jar_src, py4j_jar_dst)
    shutil.copy2(scienceworld_jar_src, scienceworld_jar_dst)

    return ScienceWorldRuntimeConfig(
        runtime_dir=resolved_runtime_dir,
        java_path=find_java_path(),
        py4j_jar_path=py4j_jar_dst,
        scienceworld_jar_path=scienceworld_jar_dst,
    )


def create_scienceworld_env(runtime_dir: Optional[Path] = None):
    """Create one ScienceWorldEnv instance with Windows-safe runtime patching."""

    import py4j.java_gateway as java_gateway
    import scienceworld.scienceworld as scienceworld_module
    from scienceworld import ScienceWorldEnv

    runtime = prepare_scienceworld_runtime(runtime_dir=runtime_dir)

    def patched_find_jar_path() -> str:
        return str(runtime.py4j_jar_path)

    def patched_launch_gateway(*args, **kwargs):
        kwargs["jarpath"] = str(runtime.py4j_jar_path)
        kwargs["cwd"] = str(runtime.runtime_dir)
        kwargs["java_path"] = str(runtime.java_path)
        return java_gateway.launch_gateway(*args, **kwargs)

    java_gateway.find_jar_path = patched_find_jar_path
    scienceworld_module.launch_gateway = patched_launch_gateway
    scienceworld_module.JAR_PATH = str(runtime.scienceworld_jar_path)
    scienceworld_module.BASEPATH = str(runtime.runtime_dir)

    return ScienceWorldEnv(serverPath=str(runtime.scienceworld_jar_path))


def export_goldpath_catalog(
    output_path: Path,
    *,
    task_names: Optional[Iterable[str]] = None,
    max_test_variations: Optional[int] = None,
    runtime_dir: Optional[Path] = None,
) -> dict[str, int]:
    """Export a ScienceWorld task-description and gold-action catalog."""

    env = create_scienceworld_env(runtime_dir=runtime_dir)
    resolved_output = Path(output_path)
    resolved_output.parent.mkdir(parents=True, exist_ok=True)

    try:
        task_name_list = list(task_names) if task_names is not None else list(env.get_task_names())
        task_count = 0
        variation_count = 0

        with resolved_output.open("w", encoding="utf-8") as handle:
            for task_name in task_name_list:
                env.load(task_name, 0)
                test_variations = list(env.get_variations_test())
                if max_test_variations is not None:
                    test_variations = test_variations[: max(0, int(max_test_variations))]

                for variation in test_variations:
                    env.load(task_name, int(variation), generateGoldPath=True)
                    row = {
                        "task_name": task_name,
                        "variation": int(variation),
                        "description": env.get_task_description(),
                        "gold_actions": list(env.get_gold_action_sequence()),
                    }
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    variation_count += 1
                task_count += 1
    finally:
        env.close()

    return {"task_count": task_count, "variation_count": variation_count}
