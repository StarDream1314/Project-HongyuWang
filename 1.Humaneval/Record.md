# HumanEval 实验记录

## 1. 实验目的

本实验在 HumanEval 离线评测矩阵上验证 two-channel feedback 场景下，不同 audit / calibration 预算调度策略对 A-OMP 更新轨迹的影响。

主实验阶段读取预先构造好的 `pass/fail` 张量，在固定随机种子下模拟 cheap hint、monitoring reward 和 calibration reward。

## 2. 代码与环境快照

### 2.1 核心代码

| 功能 | 路径 |
|---|---|
| 主实验 | `codes/experiments/rlhf_humaneval.py` |
| A-OMP 单步更新 | `codes/src/aomp.py` |
| KL prox / simplex 工具 | `codes/src/simplex.py` |
| Exp3 audit 调度器 | `codes/src/audit_schedules.py` |
| 绘图和置信区间工具 | `codes/src/plotting.py` |
| 候选代码生成 | `codes/data_build/generate_humaneval_candidates.py` |
| 候选代码评测 | `codes/data_build/evaluate_humaneval_candidates.py` |
| NPZ 数据打包 | `codes/data_build/build_humaneval_npz.py` |

### 2.2 依赖配置

依赖文件：`requirements.txt`

```text
numpy>=2.0,<3
pandas>=2.0,<3
matplotlib>=3.8,<4
scipy>=1.11,<2
scikit-learn>=1.4,<2
tqdm>=4.66,<5
rich>=13,<14
tempdir>=0.7,<1
python-dotenv>=1.0,<2
```

## 3. 数据与记忆快照

### 3.1 原始与离线数据

| 文件 | 用途 |
|---|---|
| `results/data/HumanEval.jsonl.gz` | HumanEval 原始任务 |
| `results/data/humaneval_solutions.npz` | 主实验离线 pass/fail 张量 |

`humaneval_solutions.npz` 内部数组：

| 数组 | shape | 含义 |
|---|---:|---|
| `pass_fail` | `(20, 164, 1100)` | 20 个策略在 164 道题、最多 1100 个测试上的通过情况 |
| `test_counts` | `(164,)` | 每道题有效测试数 |
| `strategy_names` | `(20,)` | 策略名称 |
| `problem_ids` | `(164,)` | HumanEval 题号 |
| `test_difficulties` | `(164, 1100)` | 每个测试在 20 个策略上的平均通过率 |

测试数统计：最少 12，最多 1100，平均 762.4207。题目范围为 `HumanEval/0` 到 `HumanEval/163`。

### 3.2 策略库

策略库共 20 个固定 LLM 代码生成策略：

```text
{direct, careful, reasoned, minimal} x {0.0, 0.2, 0.4, 0.6, 0.8}
```

实际策略名：

```text
direct_t00, direct_t02, direct_t04, direct_t06, direct_t08,
careful_t00, careful_t02, careful_t04, careful_t06, careful_t08,
reasoned_t00, reasoned_t02, reasoned_t04, reasoned_t06, reasoned_t08,
minimal_t00, minimal_t02, minimal_t04, minimal_t06, minimal_t08
```

离线单策略平均通过率最高的 5 个策略：

| 策略 | 平均通过率 |
|---|---:|
| `careful_t02` | 0.923557 |
| `reasoned_t04` | 0.922628 |
| `reasoned_t00` | 0.916091 |
| `reasoned_t02` | 0.914135 |
| `direct_t06` | 0.903356 |

最低的 5 个策略：

| 策略 | 平均通过率 |
|---|---:|
| `careful_t08` | 0.869045 |
| `minimal_t04` | 0.868757 |
| `direct_t08` | 0.867404 |
| `minimal_t06` | 0.867029 |
| `minimal_t00` | 0.860438 |

### 3.3 预计算记忆快照

主实验开始前已有完整离线记忆：

| 目录 | 文件数 | 每个文件内容 |
|---|---:|---|
| `results/intermediate/humaneval_candidates/` | 20 个 `.jsonl` | 每个策略 164 条候选代码 |
| `results/intermediate/humaneval_eval/` | 20 个 `.json` | 每个策略 164 道题的 pass/fail 列表 |
| `results/data/humaneval_solutions.npz` | 1 个 | 打包后的主实验输入矩阵 |

## 4. 数据划分与采样机制

164 道 HumanEval 题全部用于构造离线评测矩阵和计算 ground-truth reward。

每轮迭代中的随机性来自以下采样：

| 通道 | 代码函数 | 采样方式 |
|---|---|---|
| hint / cheap channel | `construct_hint_mask` + `compute_hint_score` | 每题暴露 `hint_frac=0.2` 的测试；drift 前随机抽可见测试，drift 后选更难测试 |
| monitoring channel | `compute_monitoring_reward` | 从 164 道题中有放回均匀采样 `n_mon=25` 道题 |
| calibration channel | `compute_calibration_reward` | 从 164 道题中有放回均匀采样 `n_cal` 道题 |

概念漂移设置：

```text
drift_iter = 600
```

当 `t >= 600` 时，hint mask 改为优先暴露 `test_difficulties` 较高的测试，即更偏向困难测试。

## 5. 实验配置

配置文件：`results/rlhf_humaneval/config.json`  

主要配置：

| 参数 | 值 |
|---|---:|
| `T` | 1000 |
| `eta` | 0.3 |
| `tau` | 0.01 |
| `drift_iter` | 600 |
| `hint_frac` | 0.2 |
| `n_mon` | 25 |
| `n_cal_low` | 8 |
| `n_cal_high` | 250 |
| `warmup` | 150 |
| `burst_len` | 120 |
| `threshold_scale` | 2.5 |
| `delta` | 0.05 |
| `smooth_window` | 25 |
| `audit_sweep_n_cal` | `(2, 4, 8, 16, 32, 64, 128, 250)` |
| `exp3_rho` | 0.25 |
| `exp3_xi_bound` | 100.0 |
| `seeds` | `(0, 1, 2, 3, 4)` |

对比的 schedule：

| schedule | 定义 |
|---|---|
| `monitor_only` | 只使用 monitoring，`n_cal=0` |
| `fixed_low` | 固定低 calibration，`n_cal=8` |
| `high` | 固定高 calibration，`n_cal=250` |
| `adaptive` | warmup 后基于 `cert_hp` 阈值在 8 和 250 之间切换 |
| `exp3` | Exp3 在 `[0, 8, 250]` 三个 calibration arm 中在线选择 |

## 6. 随机种子与可复现性

顶层随机种子：

```text
seeds = (0, 1, 2, 3, 4)
```

主实验使用 `numpy.random.SeedSequence`：

1. 对每个 `seed` 创建 `SeedSequence(seed)`。
2. 对 5 个 schedule 产生 `schedule_seeds = ss_seed.spawn(5)`。
3. 对每个 schedule 再产生 4 个 RNG：
   - `rng_hint`
   - `rng_mon`
   - `rng_cal`
   - `rng_exp3`

因此，只要代码、配置、输入 NPZ一致，轨迹 CSV 应可复现。

## 7. 复现命令

### 7.1 复现主实验

已有 `results/data/humaneval_solutions.npz` 时，直接运行：

```powershell
cd \Project-HongyuWang\1.Humaneval\codes
python -m experiments.rlhf_humaneval `
  --data-path ..\results\data\humaneval_solutions.npz `
  --results-dir ..\results\rlhf_humaneval `
  --fig-dir ..\figs
```

输出：

```text
results/rlhf_humaneval/config.json
results/rlhf_humaneval/trajectory_<schedule>_seed<seed>.csv
results/rlhf_humaneval/sweep_results.csv
figs/humaneval_reward.pdf
figs/humaneval_discrepancy.pdf
figs/humaneval_audit_tradeoff.pdf
```

### 7.2 从零重建离线数据

只有重建候选代码时才需要 DeepSeek API：

```text
DEEPSEEK_API_KEY
DEEPSEEK_BASE_URL
DEEPSEEK_MODEL_ID
```

推荐显式传路径：

```powershell
cd \Project-HongyuWang\1.Humaneval\codes

python -m data_build.generate_humaneval_candidates `
  --humaneval-path ..\results\data\HumanEval.jsonl.gz `
  --out-dir ..\results\intermediate\humaneval_candidates

python -m data_build.evaluate_humaneval_candidates `
  --evalplus-path path\to\test.jsonl `
  --candidates-dir ..\results\intermediate\humaneval_candidates `
  --out-dir ..\results\intermediate\humaneval_eval

python -m data_build.build_humaneval_npz `
  --evalplus-path path\to\test.jsonl `
  --eval-dir ..\results\intermediate\humaneval_eval `
  --output-path ..\results\data\humaneval_solutions.npz
```

## 8. 指标定义

轨迹文件列：

```text
schedule,t,reward,gap,n_cal,n_mon,cum_aud,xi_sq,cert,cert_hp
```

指标定义：

| 指标 | 定义 |
|---|---|
| `reward` | 当前混合策略 `z_t` 在完整离线评测矩阵上的 ground-truth reward |
| `gap` | simplex Stampacchia gap，用于诊断 `z_t` 是否接近稳定点 |
| `n_cal` | 当前轮 calibration audit 数 |
| `n_mon` | 当前轮 monitoring audit 数，本实验固定为 25 |
| `cum_aud` | 累计审计成本，按 `n_mon + n_cal` 累加 |
| `xi_sq` | calibration/lookahead 更新方向和 monitoring 方向之间的 discrepancy proxy |
| `cert` | 基于 monitoring 的风险证书 |
| `cert_hp` | 加入 high-probability 修正项后的风险证书，adaptive schedule 用它触发高预算 burst |

主要汇总指标：

```text
final_window_reward = 最后 200 轮 reward 的平均值
final_reward        = 第 1000 轮 reward
final_cum_aud       = 第 1000 轮累计审计成本
mean_n_cal          = 1000 轮平均 calibration 数
```

## 9. 日志与输出统计

### 9.1 主轨迹输出

轨迹文件数：25，即 5 个 schedule x 5 个 seed。每个轨迹 1000 行。

| schedule | final-window reward mean | std | final reward mean | final cum_aud mean | mean n_cal | n_cal range | final gap mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| `monitor_only` | 0.915149 | 0.000001 | 0.915266 | 25,000 | 0.000 | 0-0 | 0.002060 |
| `fixed_low` | 0.915474 | 0.001019 | 0.915722 | 33,000 | 8.000 | 8-8 | 0.004611 |
| `exp3` | 0.915497 | 0.000247 | 0.915587 | 62,998 | 37.998 | 0-250 | 0.002078 |
| `adaptive` | 0.915761 | 0.000325 | 0.916011 | 183,137 | 158.137 | 8-250 | 0.001384 |
| `high` | 0.915808 | 0.000063 | 0.916032 | 275,000 | 250.000 | 250-250 | 0.000580 |

逐 seed 摘要：

| schedule | seed | final-window reward | final reward | final cum_aud | mean n_cal |
|---|---:|---:|---:|---:|---:|
| `adaptive` | 0 | 0.915551 | 0.915862 | 171,182 | 146.182 |
| `adaptive` | 1 | 0.915944 | 0.916019 | 178,200 | 153.200 |
| `adaptive` | 2 | 0.915293 | 0.915787 | 201,674 | 176.674 |
| `adaptive` | 3 | 0.916055 | 0.916275 | 184,734 | 159.734 |
| `adaptive` | 4 | 0.915960 | 0.916111 | 179,894 | 154.894 |
| `exp3` | 0 | 0.915428 | 0.915589 | 62,328 | 37.328 |
| `exp3` | 1 | 0.915826 | 0.915674 | 63,578 | 38.578 |
| `exp3` | 2 | 0.915452 | 0.915619 | 63,942 | 38.942 |
| `exp3` | 3 | 0.915621 | 0.915961 | 62,096 | 37.096 |
| `exp3` | 4 | 0.915159 | 0.915094 | 63,046 | 38.046 |
| `fixed_low` | 0 | 0.914491 | 0.914842 | 33,000 | 8.000 |
| `fixed_low` | 1 | 0.915305 | 0.915221 | 33,000 | 8.000 |
| `fixed_low` | 2 | 0.914669 | 0.915273 | 33,000 | 8.000 |
| `fixed_low` | 3 | 0.915909 | 0.915719 | 33,000 | 8.000 |
| `fixed_low` | 4 | 0.916999 | 0.917553 | 33,000 | 8.000 |
| `high` | 0 | 0.915743 | 0.916055 | 275,000 | 250.000 |
| `high` | 1 | 0.915786 | 0.915992 | 275,000 | 250.000 |
| `high` | 2 | 0.915833 | 0.916050 | 275,000 | 250.000 |
| `high` | 3 | 0.915906 | 0.916055 | 275,000 | 250.000 |
| `high` | 4 | 0.915773 | 0.916008 | 275,000 | 250.000 |
| `monitor_only` | 0 | 0.915148 | 0.915265 | 25,000 | 0.000 |
| `monitor_only` | 1 | 0.915152 | 0.915268 | 25,000 | 0.000 |
| `monitor_only` | 2 | 0.915150 | 0.915267 | 25,000 | 0.000 |
| `monitor_only` | 3 | 0.915149 | 0.915266 | 25,000 | 0.000 |
| `monitor_only` | 4 | 0.915148 | 0.915265 | 25,000 | 0.000 |

### 9.2 Budget sweep 输出

文件：`results/rlhf_humaneval/sweep_results.csv`  
行数：40，即 8 个 `n_cal` x 5 个 seed。

| n_cal | final reward mean | std | min | max |
|---:|---:|---:|---:|---:|
| 2 | 0.914657 | 0.001064 | 0.913202 | 0.915766 |
| 4 | 0.915963 | 0.001000 | 0.914444 | 0.916857 |
| 8 | 0.915800 | 0.000272 | 0.915509 | 0.916248 |
| 16 | 0.915834 | 0.000488 | 0.915425 | 0.916588 |
| 32 | 0.915697 | 0.000346 | 0.915220 | 0.916122 |
| 64 | 0.915670 | 0.000238 | 0.915382 | 0.916030 |
| 128 | 0.915759 | 0.000097 | 0.915658 | 0.915885 |
| 250 | 0.915809 | 0.000090 | 0.915714 | 0.915952 |

### 9.3 图文件

| 文件 | SHA256 |
|---|---|
| `figs/humaneval_reward.pdf` | `8d51d81ef6ccae3f5d32f213877b3a4bd6a98e6bf23aef79e4b8581a27a713b9` |
| `figs/humaneval_discrepancy.pdf` | `8ea68a8ac13589add9bf6b272f685d46cfa81339bdfb5e791e3c2f8e2d724350` |
| `figs/humaneval_audit_tradeoff.pdf` | `187679167c971722ced12b002b6ebcc23e4487a7dc7b01c2830a5b609b5eab1f` |

## 10. 主要结论

1. `high` 的 final-window reward 最高，为 0.915808，但累计审计成本也最高，为 275,000。
2. `adaptive` 的 final-window reward 为 0.915761，几乎达到 `high`，但平均累计审计成本约 183,137，约为 `high` 的 66.6%。
3. `exp3` 的审计成本明显更低，累计成本约 62,998，约为 `high` 的 22.9%；其 final-window reward 为 0.915497，低于 `adaptive/high`，但高于 `monitor_only`。
4. `fixed_low` 成本最低的 calibration 方案为 33,000，final-window reward 为 0.915474；相比 `monitor_only` 有小幅提升，但 seed 间方差更大。
5. `monitor_only` 不使用 calibration，成本最低为 25,000，final-window reward 为 0.915149；作为 cheap-only baseline，性能略低于带 calibration 的调度。
6. 所有 schedule 的 reward 差异都在约 `1e-3` 量级以内，结论更适合表述为“成本-稳定性-收益 trade-off”，不宜夸大绝对 reward 提升。
7. 在 budget sweep 中，`n_cal=250` 的均值最高且方差较小；`n_cal=4` 的均值也较高，但方差更大，说明少量 calibration 可能有较好性价比，但稳定性不足。

