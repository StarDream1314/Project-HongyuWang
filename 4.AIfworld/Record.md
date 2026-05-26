# ALFWorld 新实验记录

## 1. 实验目标

本项目评估 A-OMP-Mem 在 ALFWorld 长期任务执行中的双通道记忆纠偏调度能力。核心问题是：在高质量反馈预算有限的情况下，adaptive scheduler 能否判断什么时候需要触发 slow/high feedback 来修正记忆，并在低于 oracle_high 调用成本的条件下保持接近的任务执行效果。

## 2. 方法设置

| 方法 | 调度器 | 说明 |
| --- | --- | --- |
| `adaptive` | `adaptive` | 使用 OLE certificate 的 UCB 风险分数触发短 burst refinement。 |
| `cheap_only` | `cheap_only` | 只走 cheap channel，不触发 refinement。 |
| `fixed_low` | `fixed_low` | 固定频率触发 refinement。 |
| `exp3` | `exp3` | 使用 EXP3 bandit 选择 refinement 强度。 |
| `oracle_high` | `oracle_high` | 每个任务都使用高强度 refinement，作为高成本上界。 |

关键参数：

```text
dataset = data/aomp_mem/processed/alfworld.jsonl
coverage_anchors = data/coverage_anchors_v2.npz
ole_particles = certificate/ole_particles_v2_real_progress_tau995_auto.npz
adaptive_burst_length = 2
adaptive_n_cal_high = 2
seeds = 42, 123, 456
tasks_per_seed = 134
```

## 3. 代码入口

| 路径 | 作用 |
| --- | --- |
| `codes/experiments/aomp_mem/evaluation/runner.py` | 实验入口，负责参数解析、seed/scheduler 循环、结果落盘和 budget 文件回写。 |
| `codes/experiments/aomp_mem/runtime.py` | 构建 dataset、LLM client、embedder、executor，并向 adaptive scheduler 传入新参数。 |
| `codes/experiments/aomp_mem/core/executor.py` | 串联 retrieval、scheduler、cheap/refine channel 和 memory store。 |
| `codes/experiments/aomp_mem/datasets/alfworld.py` | ALFWorld prompt、subgoal 后处理和 progress/success 评估。 |
| `codes/experiments/aomp_mem/schedulers/` | 五类调度策略实现。 |
| `codes/scripts/summarize_tau995_rerun.py` | 新实验专用汇总脚本。 |
| `codes/scripts/summarize_alfworld_results.py` | 默认汇总入口，已覆盖为读取新实验目录。 |
| `codes/scripts/plot_selected_alfworld_figures.py` | 根据新汇总结果生成论文/PPT 图。 |

## 4. 复现实验命令

主实验：

```powershell
cd "D:\学习\大学生活\科研\Project-HongyuWang\4.AIfworld"
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

Reduced prompt ablation：

```powershell
cd "D:\学习\大学生活\科研\Project-HongyuWang\4.AIfworld"
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

汇总与作图：

```powershell
python codes\scripts\summarize_alfworld_results.py
python codes\scripts\summarize_tau995_rerun.py
python codes\scripts\plot_selected_alfworld_figures.py
```

## 5. 结果文件

当前最终结果目录：

```text
results/rerun_tau995_b2_n2_main/
results/rerun_tau995_b2_n2_reduced/
```

当前最终汇总文件：

```text
results/selected_alfworld_summary.csv
results/selected_alfworld_summary.md
results/rerun_tau995_b2_n2_summary_comparison.csv
results/rerun_tau995_b2_n2_summary_conclusion.md
```

当前图表：

```text
figs/Figure1-Avg Progress by Scheduler.pdf
figs/Figure2-Partial Rows by Scheduler.pdf
figs/Figure3-Prompt Ablation-With Hints vs Reduced.pdf
figs/Figure4-LLM Calls vs Avg Progress.pdf
```

## 6. 当前结果

主实验：134 tasks x 3 seeds。

| method | rows | success_rate | avg_progress | min_progress | partial_rows | runtime_llm_calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `adaptive` | 402 | 0.8856 | 0.8002 | 0.0000 | 202 | 788 |
| `oracle_high` | 402 | 0.9229 | 0.8375 | 0.3333 | 170 | 1608 |
| `cheap_only` | 402 | 0.7736 | 0.7040 | 0.3333 | 289 | 402 |
| `fixed_low` | 402 | 0.8333 | 0.7172 | 0.3333 | 298 | 636 |
| `exp3` | 402 | 0.7935 | 0.6468 | 0.0000 | 352 | 738 |

Reduced prompt ablation：134 tasks x 3 seeds。

| method | rows | success_rate | avg_progress | min_progress | partial_rows | runtime_llm_calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `adaptive` | 402 | 0.8905 | 0.7861 | 0.0000 | 210 | 798 |
| `oracle_high` | 402 | 0.7985 | 0.7446 | 0.0000 | 232 | 1608 |
| `cheap_only` | 402 | 0.8184 | 0.7156 | 0.3333 | 294 | 402 |
| `fixed_low` | 402 | 0.8930 | 0.7807 | 0.3333 | 235 | 636 |
| `exp3` | 402 | 0.8383 | 0.7172 | 0.0000 | 295 | 690 |

## 7. 主要结论

1. 主实验中，`adaptive` 的 `avg_progress=0.8002`，接近 `oracle_high` 的 `0.8375`，但 LLM 调用数从 `1608` 降到 `788`，节省约 51.00%。
2. 主实验中，`adaptive` 相比 `cheap_only` 的平均进度提升 `0.0962`，相比最强非 oracle baseline 的平均进度仍高 `0.0829`。
3. Reduced prompt ablation 中，`adaptive` 的 `avg_progress=0.7861` 高于 `oracle_high` 的 `0.7446`，说明更密集的高质量 feedback 不一定更优，调度和记忆沉淀本身是关键。
