# 4.AIfworld

这是面向 ALFWorld 长期任务流的 A-OMP-Mem 双通道记忆调度实验。adaptive在减少高质量反馈调用的同时接近 oracle_high 效果。

比较的调度策略：

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
certificate/    OLE particle artifact
results/        trajectory、prediction、budget CSV、汇总表
figs/           生成的 PDF 图
requirements.txt
```

## 环境准备

```powershell
conda create -n alfworld python=3.10 -y
conda activate alfworld
python -m pip install -r requirements.txt
$env:PYTHONPATH="$PWD\codes"
```

真实 LLM 后端从环境变量或项目外层 `.env.local` 读取：

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_MODEL`

## 快速检查

```powershell
$env:PYTHONPATH="$PWD\codes"
$env:PYTHONDONTWRITEBYTECODE="1"
python -m unittest discover -s codes\tests
```

## 实验结果

实验结果目录：

```text
results/rerun_tau995_b2_n2_main/
results/rerun_tau995_b2_n2_reduced/
```

实验核心参数：

```text
certificate/ole_particles_v2_real_progress_tau995_auto.npz
adaptive_burst_length = 2
adaptive_n_cal_high = 2
coverage_anchors = data/coverage_anchors_v2.npz
seeds = 42, 123, 456
tasks = 134 per seed
```

主实验命令：

```powershell
$env:PYTHONPATH="$PWD\codes"
$env:PYTHONHASHSEED="0"
python -m experiments.aomp_mem.evaluation.runner `
  --track alfworld `
  --backend openai `
  --protocol openai `
  --schedulers adaptive exp3 fixed_low oracle_high cheap_only `
  --seeds 42 123 456 `
  --coverage-anchors data\coverage_anchors_v2.npz `
  --use-ole-certificate `
  --ole-particles certificate\ole_particles_v2_real_progress_tau995_auto.npz `
  --adaptive-burst-length 2 `
  --adaptive-n-cal-high 2 `
  --request-retries 10 `
  --retry-backoff-s 5 `
  --output-root results\rerun_tau995_b2_n2_main
```

Reduced prompt ablation 命令：

```powershell
$env:PYTHONPATH="$PWD\codes"
$env:PYTHONHASHSEED="0"
python -m experiments.aomp_mem.evaluation.runner `
  --track alfworld `
  --backend openai `
  --protocol openai `
  --schedulers adaptive exp3 fixed_low oracle_high cheap_only `
  --seeds 42 123 456 `
  --coverage-anchors data\coverage_anchors_v2.npz `
  --use-ole-certificate `
  --ole-particles certificate\ole_particles_v2_real_progress_tau995_auto.npz `
  --adaptive-burst-length 2 `
  --adaptive-n-cal-high 2 `
  --alfworld-prompt-mode reduced `
  --request-retries 10 `
  --retry-backoff-s 5 `
  --output-root results\rerun_tau995_b2_n2_reduced
```

每个 run 写出：

- `config.json`
- `trajectory_<schedule>_seed<seed>.csv`
- `predictions_<schedule>_seed<seed>.jsonl`
- `budget_analysis.csv`

## 汇总和作图

汇总入口：

```powershell
python codes\scripts\summarize_alfworld_results.py
python codes\scripts\plot_selected_alfworld_figures.py
```

实验专用汇总入口：

```powershell
python codes\scripts\summarize_tau995_rerun.py
```

输出文件：

- `results/selected_alfworld_summary.csv`
- `results/selected_alfworld_summary.md`
- `results/rerun_tau995_b2_n2_summary_comparison.csv`
- `results/rerun_tau995_b2_n2_summary_conclusion.md`
- `figs/Figure1-Avg Progress by Scheduler.pdf`
- `figs/Figure2-Partial Rows by Scheduler.pdf`
- `figs/Figure3-Prompt Ablation-With Hints vs Reduced.pdf`
- `figs/Figure4-LLM Calls vs Avg Progress.pdf`

## 当前结果

主实验：

| method | rows | success_rate | avg_progress | partial_rows | runtime_llm_calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| `adaptive` | 402 | 0.8856 | 0.8002 | 202 | 788 |
| `oracle_high` | 402 | 0.9229 | 0.8375 | 170 | 1608 |
| `cheap_only` | 402 | 0.7736 | 0.7040 | 289 | 402 |
| `fixed_low` | 402 | 0.8333 | 0.7172 | 298 | 636 |
| `exp3` | 402 | 0.7935 | 0.6468 | 352 | 738 |

Reduced prompt ablation：

| method | rows | success_rate | avg_progress | partial_rows | runtime_llm_calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| `adaptive` | 402 | 0.8905 | 0.7861 | 210 | 798 |
| `oracle_high` | 402 | 0.7985 | 0.7446 | 232 | 1608 |
| `cheap_only` | 402 | 0.8184 | 0.7156 | 294 | 402 |
| `fixed_low` | 402 | 0.8930 | 0.7807 | 235 | 636 |
| `exp3` | 402 | 0.8383 | 0.7172 | 295 | 690 |

## 结论

主实验中，`adaptive` 用 `788` 次 LLM 调用达到 `0.8002` 平均进度，调用数比 `oracle_high` 的 `1608` 少约 51%，平均进度只低 `0.0373`。相比 `cheap_only`，adaptive 平均进度提升 `0.0962`；相比最强非 oracle baseline，adaptive 仍高 `0.0829`。

Reduced prompt ablation 中，`adaptive` 用 `798` 次调用达到 `0.7861` 平均进度，高于 `oracle_high` 的 `0.7446`，并与 `fixed_low` 接近。这说明高质量反馈不是越密越好，关键在于何时触发高成本反馈以及如何把纠偏结果沉淀进记忆。

当前项目结论：系统通过双通道反馈判断何时相信低成本经验、何时触发高质量纠偏，从而在显著降低调用成本的同时保持接近 oracle 的任务执行效果。
