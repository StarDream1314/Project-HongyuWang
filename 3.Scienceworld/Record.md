# ScienceWorld 实验

## 1. 实验目标

本项目评估 AOMP-mem 在 ScienceWorld 多步任务流上的慢反馈调度策略。核心问题是：在有限 LLM 调用预算下，使用 OLE certificate 驱动的 adaptive slow-feedback scheduling，是否能以低于 oracle high refinement 的调用成本维持接近的任务进度。

比较的调度器：

| 方法 | 代码调度器 | 说明 |
| --- | --- | --- |
| `adaptive_ole` | `adaptive` | 使用 OLE certificate 的 UCB 触发 refinement burst。 |
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
| `codes/experiments/aomp_mem/evaluation/runner.py` | 实验入口，负责参数解析、seed/scheduler 循环、结果落盘。 |
| `codes/experiments/aomp_mem/runtime.py` | 构建 dataset、LLM client、embedder、executor。 |
| `codes/experiments/aomp_mem/core/executor.py` | 串联 retrieval、scheduler、cheap/refine channel 和 memory store。 |
| `codes/experiments/aomp_mem/datasets/scienceworld.py` | ScienceWorld prompt、action 后处理和 progress/success 评估。 |
| `codes/experiments/aomp_mem/schedulers/` | 5 个调度策略实现。 |
| `codes/experiments/aomp_mem/certificate/` | OLE、MQC、SGLD、feature pipeline 等证书逻辑。 |
| `codes/scripts/summarize_scienceworld_results.py` | 汇总 `results/` 到 CSV/Markdown。 |
| `codes/scripts/plot_selected_scienceworld_figures.py` | 根据汇总结果生成论文图。 |

依赖文件：

```text
requirements.txt
```

核心依赖范围：

```text
numpy>=2.0,<3
scipy>=1.11,<2
scikit-learn>=1.4,<2
statsmodels>=0.14,<1
pandas>=2.0,<3
PyYAML>=6.0,<7
matplotlib>=3.8,<4
sentence-transformers>=2.5,<4
torch>=2.2
```

复现前建议设置：

```powershell
$env:PYTHONPATH="$PWD\codes"
$env:PYTHONHASHSEED="0"
```

API运行时需要通过环境变量或项目外层 `.env.local` 提供 OpenAI 兼容接口配置，例如：

```env
OPENAI_API_KEY=...
OPENAI_BASE_URL=...
OPENAI_MODEL=...
```

## 3. 固定配置与校验文件

预算配置：

```yaml
exact_match:
  tokens: 450000
  wall_clock_s: 7200
scienceworld_family_prior:
  llm_calls: 6750
```

本次 ScienceWorld track 使用 `llm_calls` 作为预算单位。主实验为 5 个方法 × 3 个 seed，因此总预算 6750 被 runner 分摊为每 run `450.0`。Reduced prompt ablation 在当前结果配置中每 run 记录为 `2250.0`。

关键 artifact 哈希：

| 文件 | 大小 bytes | SHA256 |
| --- | ---: | --- |
| `data/aomp_mem/processed/scienceworld.jsonl` | 130935 | `91412c77fb304fd6f076c6526c28a6465752552a15a7ca3b292902eb6f932e53` |
| `data/aomp_mem/raw/scienceworld/test.jsonl` | 53823 | `528b36aa84d7b097b51abe60fc26d706891fbba40b806fadcdfdec9d249117a7` |
| `data/coverage_anchors_v2.npz` | 33172 | `2315c16606bdbd7bd24445c09c4e31e0e220c24475541e06b359b401646e319c` |
| `certificate/ole_particles_v2_real_progress_k18_add050.npz` | 16688 | `af1f0a651e1d056eb59db178bf3edc2f04d37907ae9b8492f2ae7fe61c8b0556` |
| `configs/budget_equivalence.yaml` | 96 | `b64b6113cd17445b065e12ca65cdddad41d0e12c82de64f5859ea0070e252889` |
| `configs/core_hashes.json` | 340 | `daecf856592afdb07986c34fe5f596bd7b2df202aff348c7161c4d633392da8f` |

`configs/core_hashes.json` 中记录的核心代码哈希：

| 代码文件 | SHA256 |
| --- | --- |
| `experiments/aomp_mem/core/memory.py` | `20a7a5ceed786147f69c398417b5e38aa1c3cac1b2ec82645ed4e9672b9bbbe8` |
| `experiments/aomp_mem/core/refinement.py` | `0542600d5ea0e18ea79afadef02e1df31ce15ba59dc9487676c849ac948aef9a` |
| `experiments/aomp_mem/core/retrieval.py` | `b894c35d24b4157813965fb14bc0297e4f704e2c418c89028ae812c0d8b061e0` |

## 4. 数据划分

数据源：

```text
data/aomp_mem/processed/scienceworld.jsonl
```

全量处理后数据：

| 字段 | 值 |
| --- | --- |
| 样本数 | 90 |
| task_id 范围 | `scienceworld-0` 到 `scienceworld-89` |
| difficulty 分布 | `easy=34`, `hard=56` |
| official_task_name 分布 | 空值 45, `find-animal=12`, `find-non-living-thing=6`, `lifespan-longest-lived=9`, `lifespan-shortest-lived=9`, `lifespan-longest-lived-then-shortest-lived=9` |

实验划分：

| 实验 | prompt mode | task_offset | task_limit | 实际任务 | 每 seed 行数 | 总行数 |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| 主实验 | `family_prior_experimental` | 0 | 无 | `scienceworld-0` 到 `scienceworld-89` | 90 | 270 |
| Prompt ablation | `reduced` | 20 | 20 | `scienceworld-20` 到 `scienceworld-39` | 20 | 60 |

Reduced prompt ablation 子集分布：

| 字段 | 值 |
| --- | --- |
| difficulty | `easy=3`, `hard=17` |
| official_task_name | 空值 11, `find-animal=6`, `find-non-living-thing=3` |

## 5. 随机种子

所有已落盘实验使用相同 seed 集合：

```text
42, 123, 456
```

seed 在 `ExperimentConfig.seed` 中记录，并由 `SeedManager` 基于 `numpy.random.SeedSequence` 派生运行内随机数。EXP3 调度器内部构造时使用 `np.random.default_rng(0)` 作为 arm sampling RNG。

## 6. 复现实验命令

主实验命令：

```powershell
cd "\Project-HongyuWang\3.Scienceworld"
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

Reduced prompt ablation 示例命令：

```powershell
cd "\Project-HongyuWang\3.Scienceworld"
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

对 `fixed_low`、`oracle_high`、`cheap_only` 替换 `--schedulers` 和 `--output-root` 即可。对 `adaptive` ablation 还需要增加：

```powershell
--use-ole-certificate `
--ole-particles certificate\ole_particles_v2_real_progress_k18_add050.npz
```

结果汇总：

```powershell
python codes\scripts\summarize_scienceworld_results.py
```

生成图：

```powershell
python codes\scripts\plot_selected_scienceworld_figures.py
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

当前 `results/` 下已落盘 artifact 数量：

| 文件类型 | 数量 | 总大小 bytes |
| --- | ---: | ---: |
| `config.json` | 30 | 35056 |
| `budget_analysis.csv` | 30 | 7795 |
| `trajectory_*.csv` | 30 | 223090 |
| `predictions_*.jsonl` | 30 | 923321 |

日志字段说明：

| 文件 | 关键字段 |
| --- | --- |
| `config.json` | `schedule`, `seed`, `model_id`, `dataset_name`, `top_k`, `trajectory_schema`, `extra.backend`, `extra.scienceworld_prompt_mode`, `extra.embedder_version`, `extra.coverage_anchors`, `extra.use_ole_certificate`, `extra.ole_particles`, `extra.task_offset` |
| `trajectory_*.csv` | `task_id`, `success`, `progress`, `n_cheap`, `n_refine`, `cum_cost`, `memory_size`, `drift_detected`, `coverage_t`, `precision_t`, `redundancy_t`, `freshness_t`, `mqc_t`, `ole_mean_t`, `ole_var_t`, `ole_ucb_t`, `drift_onset_flag_t`, `retries_t` |
| `predictions_*.jsonl` | 每个 task 的 `prediction` 和调度 metadata，例如 `llm_calls`, `retrieved_count`, `ole_*`, `coverage_t` |
| `budget_analysis.csv` | `task_count`, `total_cost`, `avg_cost`, `success_rate`, `avg_progress`, `efficiency`, `budget_unit`, `budget_limit`, `budget_actual`, `budget_source` |

注意：`total_cost` / `cum_cost` 是 runner 对 `n_cheap + n_refine` 的会计量；最终汇总中的 `runtime_llm_calls` 使用 `budget_actual`，即 LLM client tracker 观测到的实际调用数。

## 8. 记忆快照

当前可复核的记忆快照来自每条 trajectory 的 `memory_size` 序列，以及 `predictions_*.jsonl` 中的 retrieved/refinement metadata。

主实验每个 seed 结束时的 `memory_size`：

| 方法 | seed 42 | seed 123 | seed 456 | 最大 memory_size |
| --- | ---: | ---: | ---: | ---: |
| `adaptive_ole` | 82 | 82 | 82 | 82 |
| `cheap_only` | 90 | 90 | 90 | 90 |
| `exp3` | 50 | 50 | 50 | 51 |
| `fixed_low` | 49 | 49 | 49 | 52 |
| `oracle_high` | 49 | 49 | 49 | 49 |

Reduced prompt ablation 每个 seed 结束时的 `memory_size`：

| 方法 | seed 42 | seed 123 | seed 456 | 最大 memory_size |
| --- | ---: | ---: | ---: | ---: |
| `adaptive_ole` | 19 | 19 | 18 | 19 |
| `cheap_only` | 20 | 20 | 20 | 20 |
| `exp3` | 18 | 17 | 18 | 18 |
| `fixed_low` | 17 | 15 | 15 | 17 |
| `oracle_high` | 19 | 18 | 19 | 19 |

Refinement 触发统计：

| 实验 | 方法 | rows | refine rows | n_refine sum | drift/ole onsets |
| --- | --- | ---: | ---: | ---: | ---: |
| 主实验 | `adaptive_ole` | 270 | 60 | 180 | 15 |
| 主实验 | `cheap_only` | 270 | 0 | 0 | 0 |
| 主实验 | `exp3` | 270 | 78 | 234 | 0 |
| 主实验 | `fixed_low` | 270 | 54 | 162 | 0 |
| 主实验 | `oracle_high` | 270 | 270 | 810 | 0 |
| Reduced ablation | `adaptive_ole` | 60 | 60 | 180 | 24 |
| Reduced ablation | `cheap_only` | 60 | 0 | 0 | 0 |
| Reduced ablation | `exp3` | 60 | 24 | 72 | 0 |
| Reduced ablation | `fixed_low` | 60 | 12 | 36 | 0 |
| Reduced ablation | `oracle_high` | 60 | 60 | 180 | 0 |

## 9. 指标定义

任务级指标：

| 指标 | 定义 |
| --- | --- |
| `progress` | 在候选 gold action sequence 中取最佳匹配比例：`matched_expected_steps / len(expected_steps)`。 |
| `success` | `progress >= 0.5`。阈值来自 `ScienceWorldDataset.success_progress_threshold = 0.5`。 |
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
| `coverage_t` | memory entry embedding 到 coverage anchors 的覆盖比例，半径为 `coverage_radius=0.5`。 |
| `precision_t` | retrieved memory entries 的 precision signal 均值；无 retrieval 时为 `NaN` 并设置 `no_retrieval_window_flag_t=True`。 |
| `redundancy_t` | memory entry 两两 cosine similarity 超过阈值 `0.85` 的比例。 |
| `freshness_t` | 基于 entry timestamp 的指数衰减均值，默认 `lambda_fresh=0.20`。 |
| `mqc_t` | adaptive 中记录的 MQC/OLE 风险分数；OLE 模式下等于 `ole_mean_t`。 |
| `ole_mean_t` | OLE ensemble 的预测均值。 |
| `ole_var_t` | OLE ensemble 的校准方差，低 coverage 状态会增加不确定性。 |
| `ole_ucb_t` | `ole_mean_t + kappa * sqrt(ole_var_t)`。 |
| `drift_onset_flag_t` | OLE UCB 超过阈值 `tau` 时触发。 |

## 10. 当前结果

主实验：90 tasks × 3 seeds。

| method | rows | success_rate | avg_progress | min_progress | partial_rows | runtime_llm_calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `adaptive_ole` | 270 | 1.0000 | 0.9931 | 0.6000 | 10 | 390 |
| `fixed_low` | 270 | 1.0000 | 0.9868 | 0.5000 | 17 | 378 |
| `cheap_only` | 270 | 1.0000 | 0.9928 | 0.6000 | 13 | 270 |
| `exp3` | 270 | 1.0000 | 0.9869 | 0.5000 | 14 | 426 |
| `oracle_high` | 270 | 1.0000 | 0.9932 | 0.7000 | 12 | 810 |

Reduced prompt ablation：20 tasks × 3 seeds，task offset = 20。

| method | rows | success_rate | avg_progress | min_progress | partial_rows | runtime_llm_calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `adaptive_ole` | 60 | 0.9667 | 0.8407 | 0.2500 | 34 | 180 |
| `cheap_only` | 60 | 0.8500 | 0.7418 | 0.4286 | 42 | 60 |
| `fixed_low` | 60 | 0.8833 | 0.7613 | 0.4286 | 48 | 84 |
| `exp3` | 60 | 0.9833 | 0.8191 | 0.4286 | 38 | 108 |
| `oracle_high` | 60 | 0.9833 | 0.8343 | 0.4286 | 38 | 180 |

## 11. 主要结论

1. 在带 family-prior hints 的主实验中，5 个方法的 `success_rate` 都达到 1.0000，因此主差异主要体现在 `avg_progress`、`partial_rows` 和 `runtime_llm_calls`。

2. `oracle_high` 的主实验 `avg_progress=0.9932` 最高，但需要 810 次实际 LLM 调用。`adaptive_ole` 的 `avg_progress=0.9931` 几乎相同，只使用 390 次调用，约为 oracle_high 的 48.1%。

3. `cheap_only` 在主实验中调用数最低，为 270，且 `avg_progress=0.9928`。这说明 family-prior prompt hints 对主实验贡献很强，cheap channel 已经能解决大部分任务。

4. 与 `fixed_low` 和 `exp3` 相比，`adaptive_ole` 在主实验中取得更高 `avg_progress` 和更少 `partial_rows`：`adaptive_ole` partial rows 为 10，`fixed_low` 为 17，`exp3` 为 14。成本上 `adaptive_ole=390`，低于 `exp3=426`，略高于 `fixed_low=378`。

5. Reduced prompt ablation 显示 prompt hints 对性能影响显著。去掉 family-prior hints 后，`cheap_only` 从主实验 `avg_progress=0.9928` 降到 `0.7418`，`success_rate` 降到 `0.8500`。

6. Reduced prompt ablation 中，`adaptive_ole` 的 `avg_progress=0.8407` 是最高值，但 `success_rate=0.9667`，略低于 `exp3` 和 `oracle_high` 的 `0.9833`。这说明在更难的 reduced setting 下，adaptive OLE 提高平均进度，但仍存在个别低 progress 任务。

7. Adaptive OLE 的主实验触发了 15 次 OLE drift/onset，覆盖 60 个 refinement task row；reduced ablation 中触发 24 次 onset，并且 60 行全部走 refinement。该现象与 reduced prompt 下 memory/certificate 风险更高一致。

