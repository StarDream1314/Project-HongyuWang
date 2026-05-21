# Project-HongyuWang

本仓库整理了四组围绕 A-OMP / A-OMP-Mem、双通道反馈和慢反馈调度的实验。整体结构按实验对象拆分，每个子目录都保留了可复现实验脚本、输入数据或中间产物、运行结果、图表和单独的 `README.md`，仓库配置便于理解的`PPT`和`Video`。

## 项目结构

| 路径 | 内容 |
| --- | --- |
| `1.Humaneval/` | HumanEval 代码生成实验。包含候选代码生成、EvalPlus 评测、`humaneval_solutions.npz` 打包，以及 A-OMP 审计调度对比实验。 |
| `2.RLHF/` | RLHF 风格实验集合。包含 WDBC、SHP、SHP 跨域、TinyLLM 和 TinyLLM aligned 等实验，用于比较不同 audit schedule 在 cheap feedback 漂移下的表现。 |
| `3.Scienceworld/` | ScienceWorld 多步交互任务上的 A-OMP-Mem 实验。包含 processed 数据、OLE certificate、调度器、runner、summary 和 PDF 图表。 |
| `4.AIfworld/` | ALFWorld 任务上的 A-OMP-Mem 实验。结构与 ScienceWorld 类似，额外包含迁移测试和 OLE 重建脚本。 |

每个子项目通常包含：

- `codes/`：实验代码、runner、脚本和共享模块。
- `results/`：轨迹 CSV、预测 JSONL、summary、budget analysis 等输出。
- `figs/`：论文或报告使用的 PDF 图。
- `requirements.txt`：该子项目的 Python 依赖。
- `README.md` / `Record.md`：子项目级说明和实验记录。

## 环境准备

建议使用 Python 3.10+，并为不同子项目分别创建环境，避免 `torch`、`sentence-transformers` 等依赖互相影响。

通用流程：

```powershell
cd .\1.Humaneval
conda create -n humaneval python=3.10 -y
conda activate humaneval
python -m pip install -r requirements.txt
```

其他子项目把目录和环境名换成对应名称即可：

```powershell
cd .\2.RLHF
python -m pip install -r requirements.txt

cd ..\3.Scienceworld
python -m pip install -r requirements.txt

cd ..\4.AIfworld
python -m pip install -r requirements.txt
```

运行模块式命令时通常需要设置 `PYTHONPATH`：

```powershell
$env:PYTHONPATH="$PWD\codes"
```

ScienceWorld 和 ALFWorld 的可复现实验建议固定 hash seed：

```powershell
$env:PYTHONHASHSEED="0"
```

## API 与本地配置

根目录的 `.env.local` 用于保存本地 API 配置，已被 `.gitignore` 排除，不应提交真实密钥。

HumanEval 重新生成 LLM 候选代码时会读取：

```env
DEEPSEEK_API_KEY=your_key
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL_ID=deepseek-reasoner
```

ScienceWorld / ALFWorld 使用 live LLM backend 时会读取 OpenAI 或 Gemini 配置：

```env
OPENAI_API_KEY=your_key
OPENAI_BASE_URL=your_base_url
OPENAI_MODEL=your_model
OPENAI_WIRE_API=chat_completions

GEMINI_API_KEY=your_key
GEMINI_BASE_URL=your_base_url
GEMINI_MODEL=gemini-2.5-flash
GEMINI_API_PROTOCOL=auto
```

只复用仓库中已有的离线结果、summary 和图表时，通常不需要 API key。

## 快速运行

### 1. HumanEval

已有输入数据位于 `1.Humaneval/results/data/humaneval_solutions.npz`。复现实验：

```powershell
cd .\1.Humaneval\codes
$env:PYTHONPATH="$PWD"
python -m experiments.rlhf_humaneval `
  --data-path ..\results\data\humaneval_solutions.npz `
  --results-dir ..\results\rlhf_humaneval `
  --fig-dir ..\figs
```

常用 smoke run：

```powershell
python -m experiments.rlhf_humaneval --smoke
```

如需从零重建 HumanEval 候选与评测矩阵，请按 `1.Humaneval/README.md` 中的流程运行：

- `data_build.generate_humaneval_candidates`
- `data_build.evaluate_humaneval_candidates`
- `data_build.build_humaneval_npz`

### 2. RLHF

从 `2.RLHF` 目录运行。该子项目包含 WDBC、SHP、SHP 跨域和 TinyLLM 实验。

```powershell
cd .\2.RLHF
$env:PYTHONPATH="$PWD\codes"

python codes\experiments\rlhf_wdbc.py
python codes\experiments\rlhf_shp.py
python codes\experiments\rlhf_tinyllm.py
```

TinyLLM 只构建 checkpoint pool：

```powershell
python codes\experiments\rlhf_tinyllm.py --build_only
```

重画图表和生成 LaTeX 表：

```powershell
python codes\scripts\plot_wdbc_from_csv.py --root .
python codes\scripts\plot_shp_from_csv.py --root .
python codes\scripts\plot_tinyllm_from_csv.py --root .

python codes\scripts\make_bootstrap_shp_table.py
python codes\scripts\make_shp_cross_domain_table.py
python codes\scripts\make_tinyllm_drift_table.py
```

### 3. ScienceWorld

主实验比较 `adaptive`、`exp3`、`fixed_low`、`oracle_high` 和 `cheap_only` 五类调度方法。当前 summary 已保存在 `3.Scienceworld/results/selected_scienceworld_summary.md`。

```powershell
cd .\3.Scienceworld
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
  --scienceworld-prompt-mode family_prior_experimental `
  --embedder-version st-minilm `
  --coverage-anchors data\coverage_anchors_v2.npz `
  --coverage-radius 0.5 `
  --use-ole-certificate `
  --ole-particles certificate\ole_particles_v2_real_progress_k18_add050.npz `
  --output-root results
```

汇总和作图：

```powershell
python codes\scripts\summarize_scienceworld_results.py
python codes\scripts\plot_selected_scienceworld_figures.py
```

### 4. ALFWorld / AIfworld

当前数据集入口为 `4.AIfworld/data/aomp_mem/processed/alfworld.jsonl`，summary 已保存在 `4.AIfworld/results/selected_alfworld_summary.md`。

```powershell
cd .\4.AIfworld
$env:PYTHONPATH="$PWD\codes"
$env:PYTHONHASHSEED="0"

python -m experiments.aomp_mem.evaluation.runner `
  --track alfworld `
  --budget-unit llm_calls `
  --seeds 42 123 456 `
  --schedulers cheap_only fixed_low exp3 oracle_high adaptive `
  --backend openai `
  --protocol openai `
  --coverage-anchors data\coverage_anchors_v2.npz `
  --use-ole-certificate `
  --ole-particles certificate\ole_particles_v2_real_progress_k18_add050.npz
```

运行 reduced prompt ablation 时增加：

```powershell
--alfworld-prompt-mode reduced
```

汇总、作图和测试：

```powershell
python codes\scripts\summarize_alfworld_results.py
python codes\scripts\plot_selected_alfworld_figures.py

$env:PYTHONDONTWRITEBYTECODE="1"
python -m unittest discover -s codes\tests
```

## 调度方法

四个实验线共同关注 cheap feedback 与 audited / slow feedback 的调度问题。常见方法包括：

| 方法 | 含义 |
| --- | --- |
| `cheap_only` / `monitor_only` | 只使用 cheap channel 或 monitoring，不做高成本校准/refinement。 |
| `fixed_low` | 按固定低预算或固定间隔触发校准/refinement。 |
| `adaptive` / `adaptive_ole` | 基于 certificate 或风险信号触发高预算 burst。 |
| `exp3` | 将不同审计预算视作 bandit arms，在线选择。 |
| `oracle_high` / `high` | 每轮使用高预算 audited feedback/refinement，通常成本最高。 |

## 结果文件

主要输出类型：

- `trajectory_<schedule>_seed<seed>.csv`：逐轮或逐任务轨迹。
- `summary_multiseed.json` / `selected_*_summary.md`：多 seed 或最终选择结果汇总。
- `predictions_<schedule>_seed<seed>.jsonl`：ScienceWorld / ALFWorld 的逐任务输出和调度 metadata。
- `budget_analysis.csv`：LLM calls、预算、成功率和平均进度等统计。
- `figs/*.pdf`：根据结果生成的图。
- `tables/*.tex`：RLHF 子项目生成的 LaTeX 表。
- `*.npz`：离线数据矩阵、coverage anchors、OLE particles 或 checkpoint pool。

当前仓库已经包含多数实验结果与图表，可以先阅读 summary 与 PDF。
