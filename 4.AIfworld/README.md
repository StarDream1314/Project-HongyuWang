# 8.AIfworld

这是一个面向 ALFWorld 的 A-OMP-Mem 调度实验包。项目在同一条ALFWorld 任务流上比较五种调度策略：

- `adaptive`
- `fixed_low`
- `cheap_only`
- `exp3`
- `oracle_high`

当前主数据集是 `data/aomp_mem/processed/alfworld.jsonl`。

## 目录结构

```text
codes/          Python 包、runner、数据集适配器、调度器、OLE 逻辑
configs/        预算配置
data/           ALFWorld 数据、coverage anchors、OLE 训练集
certificate/    训练好的 OLE particle 包
results/        trajectory、prediction、budget CSV、汇总表
figs/           生成的 PDF 图
requirements.txt
```

`data/`、`certificate/`、`results/` 和 `figs/` 中的中间产物会保留，用于复现和检查最终结果。

## 环境准备

建议使用 Python 3.10 或更高版本。

```powershell
conda create -n alfworld python=3.10 -y
conda activate alfworld
python -m pip install -r requirements.txt
$env:PYTHONPATH="$PWD\codes"
```

真实 LLM 后端会从进程环境变量或上层 `.env.local` 读取配置：

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_MODEL`

## 快速检查

运行测试：

```powershell
$env:PYTHONPATH="$PWD\codes"
$env:PYTHONDONTWRITEBYTECODE="1"
python -m unittest discover -s codes\tests
```

## 主实验

当前最终结果已经保存在 `results/` 下。

运行主实验对比：

```powershell
$env:PYTHONPATH="$PWD\codes"
$env:PYTHONHASHSEED="0"
python -m experiments.aomp_mem.evaluation.runner --track alfworld --backend openai --protocol openai --schedulers cheap_only fixed_low exp3 oracle_high adaptive --seeds 42 123 456 --coverage-anchors data\coverage_anchors_v2.npz --use-ole-certificate --ole-particles certificate\ole_particles_v2_real_progress_k18_add050.npz
```

运行 reduced prompt ablation：

```powershell
$env:PYTHONPATH="$PWD\codes"
$env:PYTHONHASHSEED="0"
python -m experiments.aomp_mem.evaluation.runner --track alfworld --backend openai --protocol openai --schedulers cheap_only fixed_low exp3 oracle_high adaptive --seeds 42 123 456 --coverage-anchors data\coverage_anchors_v2.npz --use-ole-certificate --ole-particles certificate\ole_particles_v2_real_progress_k18_add050.npz --alfworld-prompt-mode reduced
```

每个 run 会写出：

- `config.json`
- `trajectory_<schedule>_seed<seed>.csv`
- `predictions_<schedule>_seed<seed>.jsonl`
- `budget_analysis.csv`

## OLE 相关产物

当前使用的 OLE certificate：

```text
certificate/ole_particles_v2_real_progress_k18_add050.npz
```

OLE 训练来源和中间产物：

```text
results/alfworld_ole_training_runs/
data/ole_training_set_v2.npz
data/coverage_anchors_v2.npz
```

重新生成 hashing coverage anchors：

```powershell
$env:PYTHONPATH="$PWD\codes"
$env:PYTHONHASHSEED="0"
python codes\scripts\regenerate_alfworld_ole.py --stage anchors --embedder-version hashing --output data\coverage_anchors_v2.npz --n-source-texts 134 --k 20
```

从轨迹重建 OLE 训练集并训练 OLE：

```powershell
$env:PYTHONPATH="$PWD\codes"
python codes\scripts\regenerate_alfworld_ole.py --stage prepare_training_set --input-dir results\alfworld_ole_training_runs --output data\ole_training_set_v2.npz --label-source progress
python codes\scripts\regenerate_alfworld_ole.py --stage train_ole --training-set data\ole_training_set_v2.npz --output certificate\ole_particles_v2_real_progress_k18_add050.npz --n-particles 20 --n-iters 1000
```

## 汇总和作图

```powershell
python codes\scripts\summarize_alfworld_results.py
python codes\scripts\plot_selected_alfworld_figures.py
```

输出文件：

- `results/selected_alfworld_summary.csv`
- `results/selected_alfworld_summary.md`
- `figs/Figure1-Avg Progress by Scheduler.pdf`
- `figs/Figure2-Partial Rows by Scheduler.pdf`
- `figs/Figure3-Prompt Ablation-With Hints vs Reduced.pdf`
- `figs/Figure4-LLM Calls vs Avg Progress.pdf`

