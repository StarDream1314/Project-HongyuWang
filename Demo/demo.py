from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCIENCEWORLD_ROOT = PROJECT_ROOT / "3.Scienceworld"
SCIENCEWORLD_CODES = SCIENCEWORLD_ROOT / "codes"
if str(SCIENCEWORLD_CODES) not in sys.path:
    sys.path.insert(0, str(SCIENCEWORLD_CODES))

from experiments.aomp_mem.core.memory import FeedbackRecord, MemoryEntry, MemoryStore
from experiments.aomp_mem.core.retrieval import HashingEmbedder, RetrievalResult, retrieve_top_k
from experiments.aomp_mem.datasets.scienceworld import ScienceWorldDataset
from experiments.aomp_mem.evaluation.precision_signal import record_hit


DEFAULT_TASK_IDS = [
    "scienceworld-0",
    "scienceworld-2",
    "scienceworld-3",
    "scienceworld-6",
]


@dataclass(frozen=True)
class RiskDecision:
    risk_score: float
    triggered_refine: bool
    reasons: list[str]
    coverage_t: float
    precision_t: float
    retrieval_score_t: float


def compact(text: str, max_chars: int = 130) -> str:
    text = " ".join(str(text).split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def action_lines(text: str) -> list[str]:
    return [line.strip() for line in str(text).splitlines() if line.strip()]


def load_tasks(dataset: ScienceWorldDataset, task_ids: Sequence[str]) -> list[Any]:
    by_id = {task.task_id: task for task in dataset.tasks()}
    missing = [task_id for task_id in task_ids if task_id not in by_id]
    if missing:
        raise ValueError(f"Task ids not found in ScienceWorld data: {missing}")
    return [by_id[task_id] for task_id in task_ids]


def make_cheap_prediction(task: Any, retrieved: Sequence[RetrievalResult]) -> tuple[str, str]:
    """Produce a deterministic cheap-channel proposal for demonstration.

    The first no-memory attempt is intentionally incomplete. Later attempts use
    the top retrieved memory so the demo can show both useful reuse and risky
    over-reuse.
    """
    if not retrieved:
        gold = action_lines(task.metadata.get("expected_output", task.metadata.get("official_gold_actions", "")))
        if not gold:
            gold = action_lines(task.metadata.get("subgoals", "look around"))
        return "\n".join(gold[: max(1, min(2, len(gold)))]), "no retrieved memory; emit a short local guess"

    top = retrieved[0].entry
    return str(top.task_output), f"reuse top memory entry timestamp={top.timestamp}"


def gold_prediction(task: Any) -> str:
    actions = task.metadata.get("official_gold_actions") or []
    if actions:
        return "\n".join(str(action) for action in actions)
    expected = task.metadata.get("expected_output")
    if expected:
        return str(expected)
    return str(task.metadata.get("subgoals", ""))


def evaluate(dataset: ScienceWorldDataset, task: Any, prediction: str, n_refine: int, retrieved: Sequence[RetrievalResult]) -> FeedbackRecord:
    return dataset.evaluate(task, prediction, n_refine=n_refine, retrieved=retrieved)


def memory_quality_proxy(retrieved: Sequence[RetrievalResult], entries: Sequence[MemoryEntry]) -> tuple[float, float, float]:
    if not entries:
        return 0.0, 0.0, 0.0
    retrieval_score = float(retrieved[0].score) if retrieved else 0.0
    coverage = min(1.0, retrieval_score)
    if retrieved:
        precision = sum(float(item.entry.quality_score) for item in retrieved) / len(retrieved)
    else:
        precision = 0.0
    return coverage, precision, retrieval_score


def decide_risk(
    *,
    task: Any,
    retrieved: Sequence[RetrievalResult],
    entries: Sequence[MemoryEntry],
    cheap_feedback: FeedbackRecord,
    threshold: float,
) -> RiskDecision:
    coverage, precision, retrieval_score = memory_quality_proxy(retrieved, entries)
    reasons: list[str] = []
    risk = 0.0

    if not retrieved:
        risk += 0.45
        reasons.append("no retrieved memory")
    elif retrieval_score < 0.45:
        risk += 0.30
        reasons.append(f"low retrieval score={retrieval_score:.3f}")
    elif retrieval_score < 0.70:
        risk += 0.15
        reasons.append(f"medium retrieval score={retrieval_score:.3f}")

    if not cheap_feedback.success:
        risk += 0.35
        reasons.append("cheap feedback failed")

    progress = float(cheap_feedback.progress or 0.0)
    if progress < 1.0:
        risk += 0.25
        reasons.append(f"partial progress={progress:.3f}")

    if retrieved:
        current_family = str(task.metadata.get("official_task_name", ""))
        top_family = str(retrieved[0].entry.metadata.get("official_task_name", ""))
        if current_family and top_family and current_family != top_family:
            risk += 0.35
            reasons.append(f"memory family mismatch: {top_family} -> {current_family}")

    risk = min(1.0, risk)
    if not reasons:
        reasons.append("retrieved memory looks reusable")
    return RiskDecision(
        risk_score=risk,
        triggered_refine=risk >= threshold,
        reasons=reasons,
        coverage_t=coverage,
        precision_t=precision,
        retrieval_score_t=retrieval_score,
    )


def append_memory(
    *,
    store: MemoryStore,
    embedder: HashingEmbedder,
    task: Any,
    prediction: str,
    feedback: FeedbackRecord,
    channel: str,
    metadata: dict[str, Any],
) -> MemoryEntry:
    enriched = {
        "channel": channel,
        "task_id": task.task_id,
        "official_task_name": task.metadata.get("official_task_name", ""),
        **metadata,
    }
    return store.append(
        task_input=task.input_text,
        task_output=prediction,
        feedback=feedback,
        embedding=embedder.encode(task.input_text),
        metadata=enriched,
    )


def entry_to_json(entry: MemoryEntry) -> dict[str, Any]:
    return {
        "timestamp": int(entry.timestamp),
        "task_input": entry.task_input,
        "task_output": entry.task_output,
        "feedback": {
            "success": bool(entry.feedback.success),
            "progress": entry.feedback.progress,
            "correct_answer": entry.feedback.correct_answer,
        },
        "quality_score": float(entry.quality_score),
        "usage_count": int(entry.usage_count),
        "success_count": int(entry.success_count),
        "metadata": dict(entry.metadata),
    }


def run_demo(args: argparse.Namespace) -> dict[str, Any]:
    dataset_path = SCIENCEWORLD_ROOT / "data" / "aomp_mem" / "processed" / "scienceworld.jsonl"
    dataset = ScienceWorldDataset(dataset_path, prompt_mode=ScienceWorldDataset.PROMPT_MODE_BENCHMARK)
    task_ids = args.task_ids or DEFAULT_TASK_IDS
    tasks = load_tasks(dataset, task_ids)
    embedder = HashingEmbedder(dim=args.embed_dim)
    store = MemoryStore(max_size=args.max_memory)
    trace: list[dict[str, Any]] = []

    print("ScienceWorld two-channel memory demo")
    print(f"tasks={', '.join(task.task_id for task in tasks)}")
    print(f"risk_threshold={args.risk_threshold:.2f}")
    print()

    for step, task in enumerate(tasks, start=1):
        entries_before = store.entries()
        retrieved = retrieve_top_k(query=task.input_text, entries=entries_before, embedder=embedder, k=args.top_k)
        record_hit(entries_before, retrieved)

        cheap_prediction, cheap_reason = make_cheap_prediction(task, retrieved)
        cheap_feedback = evaluate(dataset, task, cheap_prediction, n_refine=0, retrieved=retrieved)
        risk = decide_risk(
            task=task,
            retrieved=retrieved,
            entries=entries_before,
            cheap_feedback=cheap_feedback,
            threshold=args.risk_threshold,
        )

        print(f"[Task {step}] {task.task_id}")
        print(f"  input: {compact(task.input_text)}")
        print(f"  retrieve: {format_retrieval(retrieved)}")
        print(f"  cheap channel: success={cheap_feedback.success} progress={float(cheap_feedback.progress or 0.0):.3f} ({cheap_reason})")
        print(f"  risk gate: score={risk.risk_score:.3f} triggered={risk.triggered_refine} reasons={'; '.join(risk.reasons)}")

        slow_prediction = None
        slow_feedback = None
        written_entry = None
        if risk.triggered_refine:
            slow_prediction = gold_prediction(task)
            slow_feedback = evaluate(dataset, task, slow_prediction, n_refine=3, retrieved=retrieved)
            written_entry = append_memory(
                store=store,
                embedder=embedder,
                task=task,
                prediction=slow_prediction,
                feedback=slow_feedback,
                channel="accurate",
                metadata={
                    "kind": "refined-template",
                    "risk_score": risk.risk_score,
                    "cheap_progress": cheap_feedback.progress,
                    "corrected_by_refine": True,
                },
            )
            final_feedback = slow_feedback
            final_channel = "accurate"
            print(f"  slow refinement: success={slow_feedback.success} progress={float(slow_feedback.progress or 0.0):.3f}")
        else:
            written_entry = append_memory(
                store=store,
                embedder=embedder,
                task=task,
                prediction=cheap_prediction,
                feedback=cheap_feedback,
                channel="cheap",
                metadata={
                    "kind": "cheap-experience",
                    "risk_score": risk.risk_score,
                    "accepted_without_refine": True,
                },
            )
            final_feedback = cheap_feedback
            final_channel = "cheap"
            print("  slow refinement: skipped; cheap memory accepted")

        for result in retrieved:
            result.entry.touch_usage(success=final_feedback.success)

        event = {
            "step": step,
            "task_id": task.task_id,
            "official_task_name": task.metadata.get("official_task_name", ""),
            "task_input": task.input_text,
            "retrieved": [
                {
                    "score": float(item.score),
                    "timestamp": int(item.entry.timestamp),
                    "channel": item.entry.metadata.get("channel"),
                    "quality_score": float(item.entry.quality_score),
                    "task_input": item.entry.task_input,
                    "task_output": item.entry.task_output,
                }
                for item in retrieved
            ],
            "cheap": {
                "reason": cheap_reason,
                "prediction": cheap_prediction,
                "success": bool(cheap_feedback.success),
                "progress": float(cheap_feedback.progress or 0.0),
            },
            "risk_gate": asdict(risk),
            "slow_refinement": None
            if slow_feedback is None
            else {
                "prediction": slow_prediction,
                "success": bool(slow_feedback.success),
                "progress": float(slow_feedback.progress or 0.0),
            },
            "memory_update": {
                "final_channel": final_channel,
                "memory_size_before": len(entries_before),
                "memory_size_after": len(store),
                "written_timestamp": int(written_entry.timestamp),
                "written_quality": float(written_entry.quality_score),
                "written_kind": written_entry.metadata.get("kind"),
            },
        }
        trace.append(event)
        print(f"  memory update: {len(entries_before)} -> {len(store)} via {final_channel}")
        print()

    summary = {
        "n_tasks": len(tasks),
        "n_refine": sum(1 for item in trace if item["risk_gate"]["triggered_refine"]),
        "n_cheap_accept": sum(1 for item in trace if not item["risk_gate"]["triggered_refine"]),
        "final_memory_size": len(store),
        "success_rate": sum(1 for item in trace if (item["slow_refinement"] or item["cheap"])["success"]) / max(1, len(trace)),
        "avg_final_progress": sum(
            float((item["slow_refinement"] or item["cheap"])["progress"])
            for item in trace
        )
        / max(1, len(trace)),
        "trace": trace,
        "memory": [entry_to_json(entry) for entry in store.entries()],
    }
    return summary


def format_retrieval(retrieved: Sequence[RetrievalResult]) -> str:
    if not retrieved:
        return "none"
    parts = []
    for item in retrieved:
        channel = item.entry.metadata.get("channel", "?")
        family = item.entry.metadata.get("official_task_name", "")
        parts.append(f"score={item.score:.3f}, ts={item.entry.timestamp}, channel={channel}, family={family}")
    return " | ".join(parts)


def write_outputs(summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "demo_trace.jsonl"
    snapshot_path = output_dir / "memory_snapshot.json"
    summary_path = output_dir / "demo_summary.md"

    with trace_path.open("w", encoding="utf-8") as handle:
        for event in summary["trace"]:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    snapshot = {
        "final_memory_size": summary["final_memory_size"],
        "entries": summary["memory"],
    }
    snapshot_path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary_path.write_text(render_summary_md(summary), encoding="utf-8")
    print(f"Wrote {trace_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {snapshot_path}")


def render_summary_md(summary: dict[str, Any]) -> str:
    lines = [
        "# ScienceWorld 双通道记忆更新演示总结",
        "",
        "## 运行结果",
        "",
        f"- 任务数：{summary['n_tasks']}",
        f"- 触发 slow refinement 次数：{summary['n_refine']}",
        f"- cheap memory 直接接受次数：{summary['n_cheap_accept']}",
        f"- 最终 memory size：{summary['final_memory_size']}",
        f"- 最终成功率：{summary['success_rate']:.3f}",
        f"- 平均最终进度：{summary['avg_final_progress']:.3f}",
        "",
        "## 逐步过程",
        "",
    ]
    for event in summary["trace"]:
        final = event["slow_refinement"] or event["cheap"]
        lines.extend(
            [
                f"### Step {event['step']} - {event['task_id']}",
                "",
                f"- 任务类型：{event.get('official_task_name') or 'unknown'}",
                f"- 检索结果：{len(event['retrieved'])} 条",
                f"- cheap 进度：{event['cheap']['progress']:.3f}",
                f"- 风险分数：{event['risk_gate']['risk_score']:.3f}",
                f"- 是否触发 refinement：{event['risk_gate']['triggered_refine']}",
                f"- 触发原因：{'; '.join(event['risk_gate']['reasons'])}",
                f"- 最终通道：{event['memory_update']['final_channel']}",
                f"- 最终进度：{final['progress']:.3f}",
                f"- 记忆库变化：{event['memory_update']['memory_size_before']} -> {event['memory_update']['memory_size_after']}",
                "",
            ]
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline ScienceWorld two-channel memory update demo.")
    parser.add_argument("--task-ids", nargs="*", default=None, help="Explicit ScienceWorld task ids to replay.")
    parser.add_argument("--top-k", type=int, default=2, help="Number of memories to retrieve.")
    parser.add_argument("--risk-threshold", type=float, default=0.55, help="Refinement trigger threshold.")
    parser.add_argument("--embed-dim", type=int, default=128, help="Hashing embedder dimension.")
    parser.add_argument("--max-memory", type=int, default=20, help="Maximum memory entries.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory for trace, summary, and memory snapshot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_demo(args)
    write_outputs(summary, args.output_dir)


if __name__ == "__main__":
    main()
