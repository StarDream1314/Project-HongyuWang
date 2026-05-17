# ALFWorld 实验

## 1. 实验目标

本项目评估 AOMP-mem 在 ALFWorld 家居环境多步任务流上的慢反馈调度策略。核心问题是：在有限 LLM 调用预算下，使用 OLE certificate 驱动的 adaptive slow-feedback scheduling，是否能以低于 oracle high refinement 的调用成本，维持更高或接近的任务成功率与子目标完成进度。

比较的调度器：

| 方法 | 代码调度器 | 说明 |
| --- | --- | --- |
| `adaptive_ole` | `adaptive` | 使用 OLE certificate 的 UCB 风险分数触发 refinement burst。 |
| `cheap_only` | `cheap_only` | 只走 cheap channel，不触发 refinement。 |
| `fixed_low` | `fixed_low` | 固定间隔触发 refinement。 |
| `exp3` | `exp3` | 使用 EXP3 bandit 选择 refinement 强度。 |
| `oracle_high` | `oracle_high` | 每个任务都使用高强度 refinement，作为高成本上界。 |

## 2. 代码与依赖

主要入口：

```powershell
python -m experiments.aomp_mem.evaluation.runner
```

关键代码目录：

| 路径 | 作用 |
| --- | --- |
| `codes/experiments/aomp_mem/evaluation/runner.py` | 实验入口，负责参数解析、seed/scheduler 循环、结果落盘和 budget 文件回写。 |
| `codes/experiments/aomp_mem/runtime.py` | 构建 dataset、LLM client、embedder、executor。 |
| `codes/experiments/aomp_mem/core/executor.py` | 串联 retrieval、scheduler、cheap/refine channel 和 memory store。 |
| `codes/experiments/aomp_mem/datasets/alfworld.py` | ALFWorld prompt、subgoal 后处理和 progress/success 评估。 |
| `codes/experiments/aomp_mem/schedulers/` | 5 个调度策略实现。 |
| `codes/experiments/aomp_mem/certificate/` | OLE、MQC、SGLD、feature pipeline 等证书逻辑。 |
| `codes/scripts/summarize_alfworld_results.py` | 汇总 `results/` 到 CSV/Markdown。 |
| `codes/scripts/plot_selected_alfworld_figures.py` | 根据汇总结果生成论文图。 |
| `codes/scripts/regenerate_alfworld_ole.py` | 重新生成 coverage anchors、OLE 训练集和 OLE particles。 |
| `codes/tests/test_alfworld_migration.py` | ALFWorld 数据加载、prompt mode、partial progress 和结果汇总测试。 |

依赖文件：

```text
requirements.txt
```

核心依赖范围：

```text
numpy>=2.0,<3
scipy>=1.11,<2
scikit-learn>=1.4,<2
PyYAML>=6.0,<7
matplotlib>=3.8,<4
pandas>=2.0,<3
statsmodels>=0.14,<1
```

`sentence-transformers` 仅在使用 `--embedder-version st-minilm` 时需要；当前落盘 ALFWorld 结果使用默认 `hashing` embedder。

复现前建议设置：

```powershell
$env:PYTHONPATH="$PWD\codes"
$env:PYTHONHASHSEED="0"
```

API 运行时需要通过环境变量或项目外层 `.env.local` 提供 OpenAI 兼容接口配置，例如：

```env
OPENAI_API_KEY=...
OPENAI_BASE_URL=...
OPENAI_MODEL=...
```

当前落盘结果的 `config.json` 中记录模型为 `gpt-5.2`，后端为 `openai`。

## 3. 固定配置与校验文件

预算配置：

```yaml
exact_match:
  tokens: 450000
  wall_clock_s: 7200
alfworld:
  llm_calls: 6750
```

本次 ALFWorld track 使用 `llm_calls` 作为预算单位。runner 会按当次命令中的 `seed 数 × scheduler 数` 分摊 `alfworld.llm_calls=6750`。当前多数落盘 run 的 `budget_limit` 为 `450.0`；部分主实验 `adaptive` run 记录为 `2250.0`，说明该批次可能曾按单方法命令单独运行。最终汇总中的 `runtime_llm_calls` 使用 `budget_analysis.csv` 里的 `budget_actual`，即 LLM client tracker 观测到的实际调用数。

关键 artifact 哈希：

| 文件 | 大小 bytes | SHA256 |
| --- | ---: | --- |
| `data/aomp_mem/processed/alfworld.jsonl` | 54700 | `8E48107F980F33D984F87B1948E18AA275C700B08C7C02927EA48A617B9618D1` |
| `data/coverage_anchors_v2.npz` | 12708 | `34AB176D5734A7ABD158075F3367A76EEFA76993B300B6865DD48C66EB71CEE5` |
| `data/ole_training_set_v2.npz` | 767812 | `E67247D38040ED769FF439077659E35E23976E4FA7D46C6D675115D4389EEB98` |
| `certificate/ole_particles_v2_real_progress_k18_add050.npz` | 14918 | `8F656F835DF8DD521AB82A7C0206B53687F72103E62B299A97379CFC06AB5D80` |
| `configs/budget_equivalence.yaml` | 79 | `1E52F592C05073042FDB8AD0C1F56A0FD90398A1BC591F1F399E54E5764E0CDE` |
| `configs/core_hashes.json` | 340 | `DAECF856592AFDB07986C34FE5F596BD7B2DF202AFF348C7161C4D633392DA8F` |

`configs/core_hashes.json` 中记录的核心代码哈希：

| 代码文件 | SHA256 |
| --- | --- |
| `experiments/aomp_mem/core/memory.py` | `20a7a5ceed786147f69c398417b5e38aa1c3cac1b2ec82645ed4e9672b9bbbe8` |
| `experiments/aomp_mem/core/refinement.py` | `0542600d5ea0e18ea79afadef02e1df31ce15ba59dc9487676c849ac948aef9a` |
| `experiments/aomp_mem/core/retrieval.py` | `b894c35d24b4157813965fb14bc0297e4f704e2c418c89028ae812c0d8b061e0` |

## 4. 数据划分

数据源：

```text
data/aomp_mem/processed/alfworld.jsonl
```

全量处理后数据：

| 字段 | 值 |
| --- | --- |
| 样本数 | 134 |
| task_id 范围 | `alfworld-0` 到 `alfworld-133` |
| difficulty 分布 | `hard=110`, `easy=24` |
| 起始动作分布 | `put=68`, `clean=16`, `heat=13`, `cool=13`, `examine=12`, `find=6`, `look=6` |

数据字段：

| 字段 | 说明 |
| --- | --- |
| `task_id` | ALFWorld 任务 ID。 |
| `input` | 自然语言任务描述。 |
| `expected_output` | gold subgoal 序列，包含正则式匹配片段。 |
| `metadata.source_id` | 原始样本序号。 |
| `metadata.difficulty` | 任务难度。 |
| `metadata.subgoals` | 用于 progress/success 评估的 gold subgoal。 |

实验划分：

| 实验 | prompt mode | task_offset | task_limit | 实际任务 | 每 seed 行数 | 总行数 |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| 主实验 | `family_prior_experimental` | 0 | 无 | `alfworld-0` 到 `alfworld-133` | 134 | 402 |
| Prompt ablation | `reduced` | 0 | 无 | `alfworld-0` 到 `alfworld-133` | 134 | 402 |

当前 `results/alfworld_ole_training_runs/` 下还保留了 5 个方法 × 3 个 seed 的 OLE 训练轨迹。这些文件用于构建 `data/ole_training_set_v2.npz` 和 `certificate/ole_particles_v2_real_progress_k18_add050.npz`，不属于最终主实验/ablation 汇总表。

## 5. 随机种子

所有已落盘最终实验使用相同 seed 集合：

```text
42, 123, 456
```

seed 在 `ExperimentConfig.seed` 中记录，并由 `SeedManager` 基于 `numpy.random.SeedSequence` 派生运行内随机数。EXP3 调度器内部构造时使用 `np.random.default_rng(0)` 作为 arm sampling RNG。

## 6. 复现实验命令

主实验命令：

```powershell
cd "\Project-HongyuWang\4.AIfworld"
$env:PYTHONPATH="$PWD\codes"
$env:PYTHONHASHSEED="0"

python -m experiments.aomp_mem.evaluation.runner `
  --track alfworld `
  --budget-unit llm_calls `
  --seeds 42 123 456 `
  --schedulers cheap_only fixed_low exp3 oracle_high adaptive `
  --backend openai `
  --protocol openai `
  --request-retries 2 `
  --retry-backoff-s 5 `
  --alfworld-prompt-mode family_prior_experimental `
  --embedder-version hashing `
  --coverage-anchors data\coverage_anchors_v2.npz `
  --use-ole-certificate `
  --ole-particles certificate\ole_particles_v2_real_progress_k18_add050.npz `
  --output-root results
```

Reduced prompt ablation 命令：

```powershell
cd "\Project-HongyuWang\4.AIfworld"
$env:PYTHONPATH="$PWD\codes"
$env:PYTHONHASHSEED="0"

python -m experiments.aomp_mem.evaluation.runner `
  --track alfworld `
  --budget-unit llm_calls `
  --seeds 42 123 456 `
  --schedulers cheap_only fixed_low exp3 oracle_high adaptive `
  --backend openai `
  --protocol openai `
  --request-retries 2 `
  --retry-backoff-s 5 `
  --alfworld-prompt-mode reduced `
  --embedder-version hashing `
  --coverage-anchors data\coverage_anchors_v2.npz `
  --use-ole-certificate `
  --ole-particles certificate\ole_particles_v2_real_progress_k18_add050.npz `
  --output-root results
```

重新生成 hashing coverage anchors：

```powershell
$env:PYTHONPATH="$PWD\codes"
$env:PYTHONHASHSEED="0"

python codes\scripts\regenerate_alfworld_ole.py `
  --stage anchors `
  --embedder-version hashing `
  --output data\coverage_anchors_v2.npz `
  --n-source-texts 134 `
  --k 20
```

从训练轨迹重建 OLE 训练集并训练 OLE：

```powershell
$env:PYTHONPATH="$PWD\codes"

python codes\scripts\regenerate_alfworld_ole.py `
  --stage prepare_training_set `
  --input-dir results\alfworld_ole_training_runs `
  --output data\ole_training_set_v2.npz `
  --label-source progress

python codes\scripts\regenerate_alfworld_ole.py `
  --stage train_ole `
  --training-set data\ole_training_set_v2.npz `
  --output certificate\ole_particles_v2_real_progress_k18_add050.npz `
  --n-particles 20 `
  --n-iters 1000
```

结果汇总：

```powershell
python codes\scripts\summarize_alfworld_results.py
```

生成图：

```powershell
python codes\scripts\plot_selected_alfworld_figures.py
```

快速测试：

```powershell
$env:PYTHONPATH="$PWD\codes"
$env:PYTHONDONTWRITEBYTECODE="1"
python -m unittest discover -s codes\tests
```

## 7. 结果文件与日志统计

每个 run 的输出目录结构：

```text
results/<method-or-ablation-dir>/seed_<seed>/<scheduler>/
  config.json
  trajectory_<scheduler>_seed<seed>.csv
  predictions_<scheduler>_seed<seed>.jsonl
  budget_analysis.csv
```

最终主实验和 reduced prompt ablation 的落盘 artifact 数量：

| 文件类型 | 数量 | 总大小 bytes |
| --- | ---: | ---: |
| `config.json` | 30 | 37946 |
| `budget_analysis.csv` | 30 | 8585 |
| `trajectory_*.csv` | 30 | 563629 |
| `predictions_*.jsonl` | 30 | 2077027 |

如果把 `results/alfworld_ole_training_runs/` 也计入，则 `results/` 下共有 45 组 run artifact。最终结果表只使用主实验和 prompt ablation 的 30 组 run。

日志字段说明：

| 文件 | 关键字段 |
| --- | --- |
| `config.json` | `schedule`, `seed`, `model_id`, `dataset_name`, `top_k`, `trajectory_schema`, `extra.backend`, `extra.alfworld_prompt_mode`, `extra.embedder_version`, `extra.coverage_anchors`, `extra.use_ole_certificate`, `extra.ole_particles`, `extra.task_offset` |
| `trajectory_*.csv` | `task_id`, `success`, `progress`, `n_cheap`, `n_refine`, `cum_cost`, `memory_size`, `drift_detected`, `coverage_t`, `precision_t`, `redundancy_t`, `freshness_t`, `mqc_t`, `ole_mean_t`, `ole_var_t`, `ole_ucb_t`, `drift_onset_flag_t`, `retries_t` |
| `predictions_*.jsonl` | 每个 task 的 `prediction` 和调度 metadata，例如 `llm_calls`, `retrieved_count`, `ole_*`, `coverage_t` |
| `budget_analysis.csv` | `task_count`, `total_cost`, `avg_cost`, `success_rate`, `avg_progress`, `efficiency`, `budget_unit`, `budget_limit`, `budget_actual`, `budget_source` |

注意：`total_cost` / `cum_cost` 是 runner 对 `n_cheap + n_refine` 的会计量；最终汇总中的 `runtime_llm_calls` 使用 `budget_actual`，即 LLM client tracker 观测到的实际调用数。

## 8. 记忆快照

当前可复核的记忆快照来自每条 trajectory 的 `memory_size` 序列，以及 `predictions_*.jsonl` 中的 retrieved/refinement metadata。

主实验每个 seed 结束时的 `memory_size`：

| 方法 | seed 42 | seed 123 | seed 456 | 最大 memory_size |
| --- | ---: | ---: | ---: | ---: |
| `adaptive_ole` | 81 | 55 | 79 | 81 |
| `cheap_only` | 134 | 134 | 134 | 134 |
| `exp3` | 77 | 73 | 73 | 77 |
| `fixed_low` | 69 | 63 | 71 | 71 |
| `oracle_high` | 81 | 81 | 53 | 81 |

Reduced prompt ablation 每个 seed 结束时的 `memory_size`：

| 方法 | seed 42 | seed 123 | seed 456 | 最大 memory_size |
| --- | ---: | ---: | ---: | ---: |
| `adaptive_ole` | 81 | 78 | 78 | 81 |
| `cheap_only` | 134 | 134 | 134 | 134 |
| `exp3` | 80 | 73 | 77 | 80 |
| `fixed_low` | 76 | 78 | 79 | 79 |
| `oracle_high` | 56 | 81 | 75 | 81 |

Refinement 触发统计：

| 实验 | 方法 | rows | refine rows | n_refine sum | OLE onsets |
| --- | --- | ---: | ---: | ---: | ---: |
| 主实验 | `adaptive_ole` | 402 | 402 | 1206 | 390 |
| 主实验 | `cheap_only` | 402 | 0 | 0 | 0 |
| 主实验 | `exp3` | 402 | 79 | 237 | 0 |
| 主实验 | `fixed_low` | 402 | 78 | 234 | 0 |
| 主实验 | `oracle_high` | 402 | 402 | 1206 | 0 |
| Reduced ablation | `adaptive_ole` | 402 | 402 | 1206 | 390 |
| Reduced ablation | `cheap_only` | 402 | 0 | 0 | 0 |
| Reduced ablation | `exp3` | 402 | 103 | 309 | 0 |
| Reduced ablation | `fixed_low` | 402 | 78 | 234 | 0 |
| Reduced ablation | `oracle_high` | 402 | 402 | 1206 | 0 |

## 9. 指标定义

任务级指标：

| 指标 | 定义 |
| --- | --- |
| `progress` | gold subgoal 序列中被预测 subgoal 匹配到的比例：`matched_expected_steps / len(expected_steps)`。 |
| `success` | `progress >= 2/3`。阈值来自 `AlfWorldDataset.success_progress_threshold = 2.0 / 3.0`。 |
| `partial_rows` | `progress < 1.0` 的任务行数。 |
| `n_cheap` | cheap channel 计数，当前每个任务为 1。 |
| `n_refine` | 调度器给出的 refinement 强度。 |
| `memory_size` | 当前 task 执行后 memory store 中的 MemoryEntry 数。 |
| `drift_detected` | adaptive/OLE 或 drift logic 是否认为当前状态触发高强度 refinement。 |
| `retries_t` | 该任务的请求重试次数。当前结果均为 0。 |

汇总指标：

| 指标 | 定义 |
| --- | --- |
| `rows` | 汇总的 trajectory 行数。 |
| `success_rate` | `success=True` 的比例。 |
| `avg_progress` | 所有 task 的 `progress` 均值。 |
| `min_progress` | 所有 task 的最小 `progress`。 |
| `runtime_llm_calls` | 各 run `budget_analysis.csv` 中 `budget_actual` 的求和。 |
| `efficiency` | 单 run 中 `success_rate / avg_cost`，其中 `avg_cost = total_cost / task_count`。 |

Memory quality / certificate 指标：

| 指标 | 定义 |
| --- | --- |
| `coverage_t` | memory entry embedding 到 coverage anchors 的覆盖比例。当前 hashing embedder 下未显式设置 `coverage_radius`。 |
| `precision_t` | retrieved memory entries 的 precision signal 均值；无 retrieval 时为 `NaN` 并设置 `no_retrieval_window_flag_t=True`。 |
| `redundancy_t` | memory entry 两两 cosine similarity 超过阈值的比例。 |
| `freshness_t` | 基于 entry timestamp 的指数衰减均值。 |
| `mqc_t` | adaptive 中记录的 MQC/OLE 风险分数；OLE 模式下等于 `ole_mean_t`。 |
| `ole_mean_t` | OLE ensemble 的预测均值。 |
| `ole_var_t` | OLE ensemble 的校准方差，低 coverage 状态会增加不确定性。 |
| `ole_ucb_t` | `ole_mean_t + kappa * sqrt(ole_var_t)`。 |
| `drift_onset_flag_t` | OLE UCB 超过阈值 `tau` 时触发。 |

## 10. 当前结果

主实验：134 tasks × 3 seeds。

| method | rows | success_rate | avg_progress | min_progress | partial_rows | runtime_llm_calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `adaptive_ole` | 402 | 0.8532 | 0.7890 | 0.3333 | 208 | 1206 |
| `fixed_low` | 402 | 0.7512 | 0.6314 | 0.3333 | 380 | 558 |
| `cheap_only` | 402 | 0.7463 | 0.6169 | 0.0000 | 394 | 402 |
| `exp3` | 402 | 0.7612 | 0.6177 | 0.0000 | 391 | 560 |
| `oracle_high` | 402 | 0.8408 | 0.7786 | 0.0000 | 213 | 1206 |

Reduced prompt ablation：134 tasks × 3 seeds。

| method | rows | success_rate | avg_progress | min_progress | partial_rows | runtime_llm_calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `adaptive_ole` | 402 | 0.9353 | 0.8789 | 0.3333 | 124 | 1206 |
| `cheap_only` | 402 | 0.7363 | 0.6061 | 0.0000 | 397 | 402 |
| `fixed_low` | 402 | 0.8706 | 0.7454 | 0.0000 | 261 | 558 |
| `exp3` | 402 | 0.8433 | 0.6907 | 0.0000 | 317 | 608 |
| `oracle_high` | 402 | 0.8060 | 0.7231 | 0.0000 | 263 | 1206 |

已生成汇总和图文件：

```text
results/selected_alfworld_summary.csv
results/selected_alfworld_summary.md
figs/Figure1-Avg Progress by Scheduler.pdf
figs/Figure2-Partial Rows by Scheduler.pdf
figs/Figure3-Prompt Ablation-With Hints vs Reduced.pdf
figs/Figure4-LLM Calls vs Avg Progress.pdf
```

## 11. 主要结论

1. 主实验中，`adaptive_ole` 的 `success_rate=0.8532`、`avg_progress=0.7890`，均高于 `oracle_high` 的 `success_rate=0.8408`、`avg_progress=0.7786`，但两者实际 LLM 调用数相同，都是 `1206`。

2. 相比非 oracle baseline，`adaptive_ole` 在主实验中明显更强：`fixed_low` 的 `avg_progress=0.6314`，`cheap_only` 为 `0.6169`，`exp3` 为 `0.6177`。这说明 OLE 驱动的 adaptive refinement 对 ALFWorld 子目标完成度有显著贡献。

3. 主实验中 `cheap_only` 调用数最低，为 `402`，但 `partial_rows=394`，说明单纯 cheap channel 在 ALFWorld 上大多只能产生部分正确的 subgoal plan。

4. `fixed_low` 和 `exp3` 的主实验调用数相近，分别为 `558` 和 `560`，但二者 `avg_progress` 仍明显低于 `adaptive_ole`。这说明仅靠固定频率或 bandit 随机探索 refinement 强度，未能有效定位 ALFWorld 中需要高强度 refinement 的任务状态。

5. Reduced prompt ablation 中，`adaptive_ole` 的表现进一步提升到 `success_rate=0.9353`、`avg_progress=0.8789`，并显著高于 `oracle_high` 的 `success_rate=0.8060`、`avg_progress=0.7231`。当前 reduced prompt 设置并非更弱结果，可能改变了 action plan 的输出分布，使 ALFWorld subgoal-token-progress 评估更容易匹配。

6. `adaptive_ole` 在主实验和 reduced ablation 中都对全部 402 行任务执行了 refinement，`n_refine sum=1206`，与 `oracle_high` 的 refinement 强度相同。它在当前落盘配置下更像是 OLE 风险持续触发的高强度策略，而不是节省调用的稀疏触发策略。

7. 当前结果支持的最稳妥结论是：在 ALFWorld 子目标计划评估中，OLE-driven adaptive scheduling 能显著提高 progress 和 success，但当前 certificate 阈值或 feature 分布会导致过度触发，成本优势尚未体现。下一步应重点检查 OLE `tau/kappa` 校准、coverage radius、training set label 分布，以及 adaptive burst 退出逻辑。
