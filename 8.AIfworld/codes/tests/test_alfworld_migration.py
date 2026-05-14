from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.aomp_mem.datasets import AlfWorldDataset
from experiments.aomp_mem.runtime import available_dataset_names, load_dataset
from scripts.summarize_scienceworld_results import (
    DatasetMismatchError,
    ResultSpec,
    summarize_one,
)


class TestAlfWorldMigration(unittest.TestCase):
    def test_runtime_loads_alfworld_by_default(self) -> None:
        self.assertIn("alfworld", available_dataset_names())
        dataset = load_dataset("alfworld")

        self.assertEqual(dataset.dataset_name, "alfworld")
        self.assertGreater(len(dataset.tasks()), 0)

    def test_alfworld_evaluate_assigns_partial_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "alfworld.jsonl"
            payload = {
                "task_id": "alf-1",
                "input": "look at bowl under the desklamp.",
                "expected_output": (
                    "Subgoal 1: ^(?=.* you see)(?=.*a bowl \\d+)\n"
                    "Subgoal 2: You pick up the bowl \\d+\n"
                    "Subgoal 3: ^(?=.* you see)(?=.*a desklamp)"
                ),
                "metadata": {
                    "subgoals": (
                        "Subgoal 1: ^(?=.* you see)(?=.*a bowl \\d+)\n"
                        "Subgoal 2: You pick up the bowl \\d+\n"
                        "Subgoal 3: ^(?=.* you see)(?=.*a desklamp)"
                    )
                },
            }
            dataset_path.write_text(json.dumps(payload), encoding="utf-8")

            dataset = AlfWorldDataset(dataset_path)
            task = dataset.tasks()[0]
            feedback = dataset.evaluate(
                task,
                "find desklamp\ngo to desklamp\nfind bowl\nlook at bowl",
                n_refine=0,
                retrieved=[],
            )

            self.assertFalse(feedback.success)
            self.assertGreater(feedback.progress, 0.0)

    def test_alfworld_prompt_modes_are_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "alfworld.jsonl"
            dataset_path.write_text(
                json.dumps(
                    {
                        "task_id": "alf-1",
                        "input": "put a cool tomato in microwave.",
                        "expected_output": "Subgoal 1: You pick up the tomato",
                        "metadata": {"subgoals": "Subgoal 1: You pick up the tomato"},
                    }
                ),
                encoding="utf-8",
            )

            benchmark = AlfWorldDataset(dataset_path, prompt_mode="benchmark")
            reduced = AlfWorldDataset(dataset_path, prompt_mode="reduced")
            task = benchmark.tasks()[0]

            self.assertIn("Benchmark-safe ALFWorld", benchmark.prompt_context(task, stage="cheap"))
            self.assertIn("Reduced prompt-control", reduced.prompt_context(task, stage="cheap"))

    def test_summary_ignores_scienceworld_leftovers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            alf_run = root / "results" / "adaptive" / "seed_42" / "adaptive"
            alf_run.mkdir(parents=True)
            (alf_run / "budget_analysis.csv").write_text(
                "\n".join(
                    [
                        "schedule,seed,dataset_name,task_count,total_cost,avg_cost,success_rate,avg_progress,efficiency,budget_unit,budget_limit,budget_actual,budget_source",
                        "adaptive,42,alfworld,2,8,4.0,0.5,0.75,0.125,llm_calls,10,6,runtime_observed",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (alf_run / "trajectory_adaptive_seed42.csv").write_text(
                "\n".join(
                    [
                        "task_id,success,progress,n_cheap,n_refine,cum_cost,memory_size,drift_detected",
                        "alfworld-0,True,1.0,1,3,4,1,False",
                        "alfworld-1,False,0.5,1,3,8,2,False",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            old_run = root / "results" / "adaptive-ablation study" / "seed_42" / "adaptive"
            old_run.mkdir(parents=True)
            (old_run / "budget_analysis.csv").write_text(
                "\n".join(
                    [
                        "schedule,seed,dataset_name,task_count,total_cost,avg_cost,success_rate,avg_progress,efficiency,budget_unit,budget_limit,budget_actual,budget_source",
                        "adaptive,42,scienceworld,1,3,3.0,1.0,1.0,0.33,llm_calls,10,3,runtime_observed",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (old_run / "trajectory_adaptive_seed42.csv").write_text(
                "\n".join(
                    [
                        "task_id,success,progress,n_cheap,n_refine,cum_cost,memory_size,drift_detected",
                        "scienceworld-20,True,1.0,1,2,3,1,False",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = summarize_one(
                root,
                ResultSpec("main", "with_hints", "adaptive_ole", "adaptive", "adaptive"),
            )
            self.assertEqual(summary["rows"], 2)
            self.assertEqual(summary["seeds"], "42")

            with self.assertRaises(DatasetMismatchError):
                summarize_one(
                    root,
                    ResultSpec(
                        "prompt_ablation",
                        "reduced",
                        "adaptive_ole",
                        "adaptive-ablation study",
                        "adaptive",
                    ),
                )


if __name__ == "__main__":
    unittest.main()
