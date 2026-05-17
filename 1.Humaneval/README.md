# HumanEval实验

本仓库是一个基于 HumanEval 的实验，用来验证 two-channel feedback 场景下不同 audit / calibration 预算调度策略对 A-OMP 更新的影响。

实验先离线构造一组固定的 LLM 代码生成策略，评测这些策略在 HumanEval 题目上的通过情况，并打包成 `pass/fail` 张量。主实验阶段不再调用大模型，也不再重新执行候选代码，而是在固定的离线评测矩阵上模拟不同反馈和审计调度策略。

## 项目结构

```text
.
|-- requirements.txt
|-- codes/
|   |-- experiments/
|   |   `-- rlhf_humaneval.py          # 主实验入口
|   |-- src/
|   |   |-- aomp.py                    # A-OMP 更新
|   |   |-- audit_schedules.py         # Exp3 audit 调度器
|   |   |-- plotting.py                # 绘图和置信区间工具
|   |   `-- simplex.py                 # KL prox / simplex 工具
|   |-- data_build/
|   |   |-- generate_humaneval_candidates.py
|   |   |-- evaluate_humaneval_candidates.py
|   |   |-- build_humaneval_npz.py
|   |   |-- prompts.py
|   |   `-- strategies.json
|   `-- evalplus/                      # EvalPlus HumanEval 辅助代码
|-- results/
|   |-- data/
|   |   |-- HumanEval.jsonl.gz
|   |   `-- humaneval_solutions.npz    # 主实验输入
|   |-- intermediate/
|   |   |-- humaneval_candidates/      # LLM 生成的候选代码
|   |   `-- humaneval_eval/            # 每个策略的 pass/fail 评测结果
|   `-- rlhf_humaneval/                # 主实验 CSV 输出
`-- figs/
    |-- humaneval_reward.pdf
    |-- humaneval_discrepancy.pdf
    `-- humaneval_audit_tradeoff.pdf
```

## 实验简介

策略库共有 20 个固定策略，由两组因素交叉得到：

```text
4 个 prompt family x 5 个 temperature
= {direct, careful, reasoned, minimal} x {0.0, 0.2, 0.4, 0.6, 0.8}
```

候选代码生成和评测完成后，会被打包成：

```text
results/data/humaneval_solutions.npz
```

主实验比较 5 种 schedule：

```text
monitor_only   # cheap-only 更新，n_cal = 0
fixed_low      # 固定低 calibration 预算，n_cal = 8
high           # 固定高 calibration 预算，n_cal = 250
adaptive       # 根据 cert_hp 阈值在 8 和 250 之间切换
exp3           # 使用 Exp3 从 [0, 8, 250] 中在线选择 n_cal
```

默认 full run 配置为：

```text
T = 1000
seeds = (0, 1, 2, 3, 4)
n_mon = 25
n_cal_low = 8
n_cal_high = 250
warmup = 150
burst_len = 120
drift_iter = 600
```

## 环境安装

推荐 Python 3.10+：

```powershell
conda create -n humaneval python=3.10 -y
conda activate humaneval
python -m pip install -r requirements.txt
```

## 数据构建

如果只复现实验结果，通常不需要执行本节；直接使用已有的 `humaneval_solutions.npz` 即可。

如果要从零重新生成候选代码和评测矩阵，需要经历：

```text
HumanEval.jsonl.gz
-> 20 个策略生成候选代码
-> EvalPlus 展开测试评测候选代码
-> 打包成 humaneval_solutions.npz
-> 运行主实验
```

### 1. 配置 DeepSeek API

候选代码生成会调用外部 LLM API。运行前需要设置：

```text
DEEPSEEK_API_KEY
DEEPSEEK_BASE_URL
DEEPSEEK_MODEL_ID
```

主实验本身不需要 API key；只有重新生成候选代码时才需要。

### 2. 生成候选代码

从 `codes/` 目录运行：

```powershell
cd codes
python -m data_build.generate_humaneval_candidates `
  --humaneval-path ..\results\data\HumanEval.jsonl.gz `
  --out-dir ..\results\intermediate\humaneval_candidates
```

输出路径：

```text
results/intermediate/humaneval_candidates/<strategy>.jsonl
```

### 3. 评测候选代码

评测脚本需要 EvalPlus 展开后的 `test.jsonl`：

```powershell
cd codes
python -m data_build.evaluate_humaneval_candidates `
  --evalplus-path path\to\test.jsonl `
  --candidates-dir ..\results\intermediate\humaneval_candidates `
  --out-dir ..\results\intermediate\humaneval_eval
```

输出路径：

```text
results/intermediate/humaneval_eval/<strategy>.json
```

每个 JSON 文件保存该策略在每道题上的 `pass_fail` 列表。

### 4. 打包 NPZ

```powershell
cd codes
python -m data_build.build_humaneval_npz `
  --evalplus-path path\to\test.jsonl `
  --eval-dir ..\results\intermediate\humaneval_eval `
  --output-path ..\results\data\humaneval_solutions.npz
```

生成新的 `humaneval_solutions.npz` 后，即可重新运行主实验。

## 复现主实验

如果已经存在：

```text
results/data/humaneval_solutions.npz
```

则可以直接运行主实验：

```powershell
cd codes
python -m experiments.rlhf_humaneval `
  --data-path ..\results\data\humaneval_solutions.npz `
  --results-dir ..\results\rlhf_humaneval `
  --fig-dir ..\figs
```

运行后会生成：

```text
results/rlhf_humaneval/config.json
results/rlhf_humaneval/trajectory_<schedule>_seed<seed>.csv
results/rlhf_humaneval/sweep_results.csv
figs/humaneval_reward.pdf
figs/humaneval_discrepancy.pdf
figs/humaneval_audit_tradeoff.pdf
```

## 输出文件说明

每个 `trajectory_<schedule>_seed<seed>.csv` 表示某个 schedule 在某个 seed 下的完整轨迹，每一行对应一轮迭代：

```text
schedule,t,reward,gap,n_cal,n_mon,cum_aud,xi_sq,cert,cert_hp
```

主要字段含义：

```text
reward    # 当前混合策略 z_t 在完整离线数据上的真实 reward
gap       # simplex gap，用于诊断当前 z_t 是否接近稳定点
n_cal     # 当前轮 calibration 预算
n_mon     # 当前轮 monitoring 预算
cum_aud   # 累计 monitoring + calibration 审计成本
xi_sq     # 实际更新方向和 monitoring 方向之间的 discrepancy
cert_hp   # adaptive 中用于判断是否触发高预算 burst 的风险证书
```

三张图含义：

```text
humaneval_reward.pdf          # 不同 schedule 的 reward 曲线
humaneval_discrepancy.pdf     # 不同 schedule 的 xi_sq / discrepancy 曲线
humaneval_audit_tradeoff.pdf  # 固定 n_cal 与 final-window reward 的成本-性能权衡
```

