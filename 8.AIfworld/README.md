# 8.AIfworld

This directory is the ALFWorld version of the A-OMP-Mem validation package.
It keeps the original scheduling architecture intact:

- `adaptive`
- `fixed_low`
- `cheap_only`
- `exp3`
- `oracle_high`

The main change is the task stream. The package now loads
`data/aomp_mem/processed/alfworld.jsonl` by default, evaluates subgoal plans
with the `AlfWorldDataset` adapter, and writes the same experiment artifacts as
the previous ScienceWorld package.

## Layout

```text
codes/          runner, dataset adapters, scheduler, memory, OLE certificate
configs/        budget configuration
data/           ALFWorld processed data, coverage anchors, OLE training data
certificate/    OLE particle packages
results/        trajectory CSV, predictions JSONL, budget CSV, summaries
figs/           summary figures
```

## Environment

```powershell
python -m pip install -r requirements.txt
$env:PYTHONPATH="$PWD\codes"
```

Real backend credentials are read from environment variables or `.env.local` by
`codes/experiments/aomp_mem/core/llm_interface.py`.

## Local Smoke

```powershell
$env:PYTHONPATH="$PWD\codes"
$env:PYTHONHASHSEED="0"
python codes\scripts\regenerate_alfworld_ole.py --stage anchors --embedder-version hashing --output results\_smoke_check\coverage_anchors_test.npz --n-source-texts 10 --k 3
python -m experiments.aomp_mem.evaluation.runner --track alfworld --backend mock --schedulers cheap_only adaptive fixed_low exp3 oracle_high --seeds 1 --task-limit 3 --output-root results\_smoke_check\alfworld_results --coverage-anchors results\_smoke_check\coverage_anchors_test.npz
```

Each scheduler run writes:

- `config.json`
- `trajectory_<schedule>_seed<seed>.csv`
- `predictions_<schedule>_seed<seed>.jsonl`
- `budget_analysis.csv`

## ALFWorld OLE Regeneration

Generate ALFWorld anchors:

```powershell
$env:PYTHONHASHSEED="0"
python codes\scripts\regenerate_alfworld_ole.py --stage anchors --embedder-version hashing --output data\coverage_anchors_v2.npz --n-source-texts 134 --k 20
```

Run real OpenAI trajectories without OLE first to collect ALFWorld training
data:

```powershell
$env:PYTHONPATH="$PWD\codes"
$env:PYTHONHASHSEED="0"
python -m experiments.aomp_mem.evaluation.runner --track alfworld --backend openai --protocol openai --schedulers cheap_only fixed_low exp3 oracle_high adaptive --seeds 42 123 456 --coverage-anchors data\coverage_anchors_v2.npz
```

Build the OLE training set and train ALFWorld OLE:

```powershell
python codes\scripts\regenerate_alfworld_ole.py --stage prepare_training_set --input-dir results --output data\ole_training_set_v2.npz --label-source progress
python codes\scripts\regenerate_alfworld_ole.py --stage train_ole --training-set data\ole_training_set_v2.npz --output certificate\ole_particles_v2_real_progress_k18_add050.npz --n-particles 20 --n-iters 1000
```

Rerun adaptive with the trained certificate:

```powershell
$env:PYTHONHASHSEED="0"
python -m experiments.aomp_mem.evaluation.runner --track alfworld --backend openai --protocol openai --schedulers adaptive --seeds 42 123 456 --coverage-anchors data\coverage_anchors_v2.npz --use-ole-certificate --ole-particles certificate\ole_particles_v2_real_progress_k18_add050.npz
```

## Summaries And Figures

```powershell
python codes\scripts\summarize_scienceworld_results.py --allow-missing
python codes\scripts\plot_selected_scienceworld_figures.py
```

The script names are kept for compatibility, but their default output is now:

- `results/selected_alfworld_summary.csv`
- `results/selected_alfworld_summary.md`
- `figs/Figure*.pdf`

Use `--allow-missing` when reduced prompt-ablation ALFWorld runs have not been
generated. The summary script ignores old ScienceWorld result leftovers instead
of treating them as ALFWorld rows.
