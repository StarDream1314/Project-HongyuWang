# ScienceWorld实验

本仓库是一个面向 ScienceWorld 任务流的 AOMP-mem 实验。项目重点比较不同慢反馈调度策略在多步交互任务中的表现，验证的问题是：在 ScienceWorld 多步任务流中，使用 OLE certificate 驱动的 adaptive slow-feedback scheduling 是否能在有限 LLM 调用预算下维持较好的任务进度。

当前主实验包含 5 个调度方法：

- `cheap_only`：只使用 cheap channel，不触发 refinement。
- `adaptive`：使用 OLE certificate 的 adaptive burst 调度。
- `oracle_high`：每个任务都使用高强度 refinement。
- `exp3`：使用 EXP3 bandit 调度 refinement。
- `fixed_low`：按固定间隔触发 refinement。

当前消融实验使用 reduced prompt，包含同样 5 个方法，用于比较 prompt hints 对结果的影响。

## 目录结构

```text
.
├── certificate/                         # OLE 粒子和证书产物
├── codes/
│   ├── experiments/aomp_mem/
│   │   ├── core/                        # LLM、memory、retrieval、executor、refinement
│   │   ├── datasets/                    # ScienceWorld 数据集适配器
│   │   ├── schedulers/                  # cheap_only / adaptive / exp3 / fixed_low / oracle_high
│   │   ├── certificate/                 # OLE、MQC、SGLD、feature pipeline
│   │   ├── evaluation/                  # runner、trajectory schema、quality metrics
│   │   ├── runtime.py                   # 构建 dataset、LLM client、embedder、executor
│   │   └── scienceworld_support.py      # ScienceWorld 环境和 gold path 工具
│   └── scripts/
│       ├── summarize_scienceworld_results.py
│       └── plot_selected_scienceworld_figures.py
├── configs/
│   └── budget_equivalence.yaml          # 预算配置
├── data/
│   ├── coverage_anchors_v2.npz          # coverage 指标 anchor
│   └── aomp_mem/
│       ├── raw/scienceworld/test.jsonl
│       └── processed/scienceworld.jsonl
├── figs/                                # 生成的 PDF 图
├── results/                             # 实验输出和汇总结果
└── README.md
```

## 环境准备

建议使用 Python 3.10+：

```powershell
conda create -n scienceworld python=3.10 -y
conda activate scienceworld
cd "\Project-HongyuWang\3.Scienceworld"
python -m pip install -r requirements.txt
```

运行前设置：

```powershell
$env:PYTHONPATH="$PWD\codes"
$env:PYTHONHASHSEED="0"
```

## API 配置

API 配置放在：

```text
\Project-HongyuWang\.env.local
```

OpenAI示例：

```env
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=your_base_url
OPENAI_MODEL=your_model_id
```

## 复现主实验：ScienceWorld 全量 90 任务

主实验会运行 5 个方法、3 个 seed、每个 seed 90 个任务：

```text
5 methods × 3 seeds × 90 tasks = 1350 task rows
```

完整命令：

```powershell
cd ".\3.Scienceworld"
$env:PYTHONPATH="$PWD\codes"
$env:PYTHONHASHSEED="0"

python -m experiments.aomp_mem.evaluation.runner `
  --track scienceworld_family_prior `
  --budget-unit llm_calls `
  --seeds 42 123 456 `
  --schedulers adaptive exp3 fixed_low oracle_high cheap_only `
  --backend openai `
  --protocol openai `
  --request-retries 10 `
  --retry-backoff-s 5 `
  --task-offset 0 `
  --scienceworld-prompt-mode family_prior_experimental `
  --embedder-version st-minilm `
  --coverage-anchors data\coverage_anchors_v2.npz `
  --coverage-radius 0.5 `
  --use-ole-certificate `
  --ole-particles certificate\ole_particles_v2_real_progress_k18_add050.npz `
  --output-root results
```

## Reduced prompt 消融实验

Reduced prompt 消融实验使用：

```powershell
--scienceworld-prompt-mode reduced
--task-offset 20
--task-limit 20
```

分别运行 5 个方法，输出到：

```text
results/adaptive-ablation study
results/cheap_only-ablation study
results/fixed_low-ablation study
results/exp3-ablation study
results/oracle_high-ablation study
```

示例：运行 `exp3` 消融：

```powershell
$env:PYTHONPATH="$PWD\codes"
$env:PYTHONHASHSEED="0"

python -m experiments.aomp_mem.evaluation.runner `
  --track scienceworld_family_prior `
  --budget-unit llm_calls `
  --seeds 42 123 456 `
  --schedulers exp3 `
  --backend openai `
  --protocol openai `
  --request-retries 10 `
  --retry-backoff-s 5 `
  --task-offset 20 `
  --task-limit 20 `
  --scienceworld-prompt-mode reduced `
  --embedder-version st-minilm `
  --coverage-anchors data\coverage_anchors_v2.npz `
  --coverage-radius 0.5 `
  --output-root "results\exp3-ablation study"
```

对 `fixed_low`、`oracle_high`、`cheap_only` 类似替换 `--schedulers` 和 `--output-root`。`adaptive` 消融需要额外加：

```powershell
--use-ole-certificate `
--ole-particles certificate\ole_particles_v2_real_progress_k18_add050.npz
```

## 汇总结果

生成 CSV 和 Markdown 汇总：

```powershell
python codes\scripts\summarize_scienceworld_results.py
```

输出：

```text
results/selected_scienceworld_summary.csv
results/selected_scienceworld_summary.md
```

当前主实验汇总为 90 任务 × 3 seeds：

| method | rows | success_rate | avg_progress | min_progress | partial_rows | runtime_llm_calls |
|---|---:|---:|---:|---:|---:|---:|
| adaptive_ole | 270 | 1.0000 | 0.9931 | 0.6000 | 10 | 390 |
| fixed_low | 270 | 1.0000 | 0.9868 | 0.5000 | 17 | 378 |
| cheap_only | 270 | 1.0000 | 0.9928 | 0.6000 | 13 | 270 |
| exp3 | 270 | 1.0000 | 0.9869 | 0.5000 | 14 | 426 |
| oracle_high | 270 | 1.0000 | 0.9932 | 0.7000 | 12 | 810 |

## 绘图

根据 summary 重新生成图：

```powershell
python codes\scripts\plot_selected_scienceworld_figures.py
```

输出：

```text
figs/Figure1-Avg Progress by Scheduler.pdf
figs/Figure2-Partial Rows by Scheduler.pdf
figs/Figure3-Prompt Ablation-With Hints vs Reduced.pdf
figs/Figure4-LLM Calls vs Avg Progress.pdf
```

## 结果文件说明

每个 run 的输出目录包含：

```text
config.json
trajectory_<scheduler>_seed<seed>.csv
predictions_<scheduler>_seed<seed>.jsonl
budget_analysis.csv
```

其中：

- `config.json`：记录运行配置、backend、模型、seed、prompt mode、OLE 是否启用等。
- `trajectory_*.csv`：逐任务结果，包含 `success`、`progress`、`n_cheap`、`n_refine`、`coverage_t`、`precision_t`、`ole_var_t` 等。
- `predictions_*.jsonl`：逐任务模型输出动作序列和调度 metadata。
- `budget_analysis.csv`：该 run 的总成本、平均成本、成功率、平均进度和预算记录。
