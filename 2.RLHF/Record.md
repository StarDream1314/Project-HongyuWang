# 2.RLHF 实验

## 1. 实验目的

本项目研究 RLHF 场景中的双通道反馈调度问题：

- cheap feedback：信息密集、成本低，但可能发生 drift 或携带 spurious bias。
- audited feedback：更可靠，但成本更高。
- 核心问题：在 cheap judge 发生漂移时，什么时候、花多少 audited feedback 用于 calibration / monitoring，才能以较低 audit cost 保持 reward 和 stationarity。

统一抽象为 checkpoint / policy pool 上的 simplex mixture 优化：

```text
correct[k, j]  # 第 k 个 checkpoint 在第 j 个样本上是否正确
r_true[k]      # checkpoint k 的真实 reward / accuracy
u[k]           # spurious feature 方向
z_t            # 第 t 轮 checkpoint mixture
s[k]           # cheap judge score
```

cheap judge 每轮产生代理分数：

```text
s[k] = r_true[k] + beta_t * u[k] + noise[k]
```

在 `drift_iter` 后，`beta_t` 从 `beta0` 变为 `beta1`。calibration audits 用最小二乘估计漂移系数：

```text
beta_hat = dot(u_cal, s[pol_cal] - y_cal) / dot(u_cal, u_cal)
r_hat = s - beta_hat * u
```

然后使用 KL-geometry A-OMP 更新 mixture：

```text
w      = kl_prox(z, eta, g_prev)
g      = F_est(w)
z_next = kl_prox(z, eta, g)
```

## 2. 数据划分

### WDBC

数据来源：`sklearn.datasets.load_breast_cancer()`。

固定划分逻辑位于 `codes/experiments/rlhf_wdbc.py`：

```text
第一步：train_test_split(test_size=0.4, random_state=0, stratify=y)
第二步：train_test_split(test_size=0.5, random_state=1, stratify=y_tmp)
```

总样本 569 条，约为：

```text
train: 341
val:   114
test:  114
```

其中 checkpoint 训练使用 train split，最终 `correct` / `r_true` 在 test split 上计算。

### SHP 风格偏好数据

本仓库自带 JSONL 风格数据，按行存储样本：

| 文件 | 行数 | 字节 |
|---|---:|---:|
| `codes/data/askphysics_train.json` | 7364 | 13017152 |
| `codes/data/askphysics_validation.json` | 409 | 809182 |
| `codes/data/askphysics_test.json` | 587 | 978994 |
| `codes/data/askhistorians_test.json` | 164 | 787444 |

SHP in-domain：

```text
train_domain = askphysics
eval_domain  = askphysics
训练数据      = askphysics_train + askphysics_validation
测试数据      = askphysics_test
```

SHP cross-domain：

```text
train_domain = askphysics
eval_domain  = askhistorians
训练数据      = askphysics_train + askphysics_validation
测试数据      = askhistorians_test
```

### TinyLLM

TinyLLM 使用同一批 SHP 风格数据：

```text
train_domain = askphysics
eval_domain  = askphysics
训练数据      = askphysics_train + askphysics_validation
测试数据      = askphysics_test 的前 eval_subset=128 条
```

TinyLLM checkpoint pool 会缓存到：

```text
results/rlhf_tinyllm/checkpoint_pool.npz
```

## 3. 配置文件与核心参数

已有配置文件：

```text
results/rlhf_wdbc/config.json
results/rlhf_shp/config.json
results/rlhf_shp_cross_domain/config.json
results/rlhf_tinyllm/config.json
results/rlhf_tinyllm_aligned/config.json
```

### WDBC

```text
m = 20
T = 1500
eta = 0.3
tau = 0.005
drift_iter = 750
beta0 = 0.0
beta1 = 0.8
sigma_judge = 0.06
n_mon = 20
n_cal_low = 5
n_cal_high = 200
warmup = 200
burst_len = 120
threshold_scale = 2.5
seeds = 0,1,2,3,4,5,6,7,8,9
seed_models = 0
```

### SHP in-domain

```text
m = 20
T = 1000
eta = 0.3
tau = 0.01
drift_iter = 600
beta0 = 0.0
beta1 = 0.9
sigma_judge = 0.08
n_mon = 25
n_cal_low = 8
n_cal_high = 250
warmup = 150
burst_len = 120
threshold_scale = 2.5
train_domain = askphysics
eval_domain = askphysics
seeds = 0,1,2,3,4,5,6,7,8,9
seed_models = 0
```

### SHP cross-domain

```text
m = 20
T = 1000
eta = 0.3
tau = 0.01
drift_iter = 600
beta0 = 0.0
beta1 = 0.9
n_mon = 25
n_cal_low = 8
n_cal_high = 250
train_domain = askphysics
eval_domain = askhistorians
seeds = 0,1,2,3,4
```

### TinyLLM spurious conciseness drift

```text
m = 12
T = 1000
eta = 0.3
tau = 0.01
drift_iter = 600
beta0 = 0.0
beta1 = -0.9
sigma_judge = 0.08
n_mon = 25
n_cal_low = 8
n_cal_high = 250
warmup = 150
burst_len = 120
threshold_scale = 2.5
train_domain = askphysics
eval_domain = askphysics
eval_subset = 128
seeds = 0,1,2,3,4
seed_model = 0
```

Tiny Transformer 结构：

```text
byte-level tokenizer
d_model = 96
n_heads = 4
n_layers = 2
d_ff = 192
dropout = 0.0
batch_size = 8
lr = 3e-4
weight_decay = 0.01
pretrain_steps = 120
dpo_steps = 240
dpo_beta = 0.1
```

## 4. 随机种子

模型 / checkpoint pool 构造：

```text
WDBC:   seed_models = 0
SHP:    seed_models = 0
TinyLLM seed_model  = 0
```

outer-loop 多 seed：

```text
WDBC:                 0,1,2,3,4,5,6,7,8,9
SHP in-domain:        0,1,2,3,4,5,6,7,8,9
SHP cross-domain:     0,1,2,3,4
TinyLLM:              0,1,2,3,4
TinyLLM aligned:      0,1,2,3,4
```

每个 seed 内对不同 schedule 使用 `SeedSequence(seed).spawn(...)` 生成独立 RNG stream，以避免某个 schedule 的 audit 数改变后影响其他 schedule 的随机数序列。

## 5. 对比的 audit schedules

| schedule | 行为 |
|---|---|
| `monitor_only` | 只使用 monitoring audits，不做 calibration audits。 |
| `fixed_low` | 每轮固定使用低预算 calibration audits。 |
| `adaptive` | warmup 后根据 `cert_hp` 阈值触发高预算 burst。 |
| `exp3` | 将 `{0, n_cal_low, n_cal_high}` 作为 bandit arms，用 Exp3 在线选择。 |
| `high` | 每轮使用高预算 calibration audits，通常最稳但成本最高。 |

Exp3 相关参数：

```text
arms = [0, n_cal_low, n_cal_high]
exp3_rho = 0.25
exp3_xi_bound = 100.0
gamma = exp3_default_gamma(K, T) unless explicitly set
```

## 6. 指标定义

| 指标 | 定义 |
|---|---|
| `reward` | 当前 mixture 的真实收益，`dot(r_true, z_t)`。 |
| `gap` | 真实 reward operator 下的 Stampacchia stationarity gap，越低越接近真实稳定方向。 |
| `beta_true` | cheap judge 当前真实漂移系数。 |
| `beta_hat` | calibration audits 估计出的漂移系数。 |
| `cert` | 基于 monitoring stream 的 prox certificate。 |
| `cert_hp` | 加入高概率 slack 后的 certificate。 |
| `n_cal` | 当前轮 calibration audits 数量。 |
| `n_mon` | 当前轮 monitoring audits 数量。 |
| `cum_aud` | 累计 audits，包括 monitoring + calibration。 |
| `xi_sq` | lookahead 与 monitoring operator 的 discrepancy proxy。 |
| `final_reward` | 最后一轮 reward。 |
| `min_post_reward` | drift 后窗口内最低 reward。 |
| `recovery_time` | drift 后从 post-drift minimum 恢复到 pre-drift reference reward 所需轮数。 |
| `total_cal_audits` | 全部 calibration audits 数量。 |

`recovery_time` 的实现位于 `codes/src/metrics.py`，核心步骤：

```text
1. 对 reward trajectory 做 centered moving average。
2. 用 drift 前 pre_window=50 轮平均值作为 pre-drift reference。
3. 在 drift 后 post_window 内找到最低点。
4. 从最低点开始，首次回到 reference - tol 的轮数即 recovery_time。
5. 若未恢复，返回 censored value。
```

## 7. 运行与复现命令

### 依赖配置

当前仓库依赖由根目录 `requirements.txt`：

```text
numpy>=2.0,<3
pandas>=2.0,<3
matplotlib>=3.8,<4
scipy>=1.11,<2
scikit-learn>=1.4,<2
torch>=2.1,<3
```

方法：

```powershell
cd "\Project-HongyuWang\2.RLHF"
$env:PYTHONPATH="$PWD\codes"
python -m pip install -r requirements.txt
```

运行 WDBC：

```powershell
$env:PYTHONPATH="$PWD\codes"
python codes\experiments\rlhf_wdbc.py
```

运行 SHP in-domain 与 cross-domain：

```powershell
$env:PYTHONPATH="$PWD\codes"
python codes\experiments\rlhf_shp.py
```

运行 TinyLLM：

```powershell
$env:PYTHONPATH="$PWD\codes"
python codes\experiments\rlhf_tinyllm.py
```

只构建 TinyLLM checkpoint pool：

```powershell
$env:PYTHONPATH="$PWD\codes"
python codes\experiments\rlhf_tinyllm.py --build_only
```

从已有 CSV 重新生成图：

```powershell
$env:PYTHONPATH="$PWD\codes"
python codes\scripts\plot_wdbc_from_csv.py --root .
python codes\scripts\plot_shp_from_csv.py --root .
python codes\scripts\plot_tinyllm_from_csv.py --root .
```

生成 LaTeX 表格：

```powershell
python codes\scripts\make_bootstrap_shp_table.py
python codes\scripts\make_shp_cross_domain_table.py
python codes\scripts\make_tinyllm_drift_table.py
```

## 8. 结果目录

结果目录统计：

| 实验目录 | CSV | JSON | NPZ |
|---|---:|---:|---:|
| `results/rlhf_wdbc` | 50 | 2 | 0 |
| `results/rlhf_shp` | 96 | 3 | 0 |
| `results/rlhf_shp_cross_domain` | 25 | 2 | 0 |
| `results/rlhf_tinyllm` | 25 | 2 | 1 |
| `results/rlhf_tinyllm_aligned` | 25 | 2 | 0 |

图表与表格：

```text
figs/:   27 PDF files
tables/: 3 LaTeX table files
```

关键 summary 文件：

```text
results/rlhf_wdbc/summary_multiseed.json
results/rlhf_shp/summary_multiseed.json
results/rlhf_shp_cross_domain/summary_multiseed.json
results/rlhf_tinyllm/summary_multiseed.json
results/rlhf_tinyllm_aligned/summary_multiseed.json
```

关键缓存：

```text
results/rlhf_tinyllm/checkpoint_pool.npz
```

## 9. 汇总结果

以下数值来自各实验的 `summary_multiseed.json`。

### WDBC

| schedule | final reward | min post reward | recovery time | median[IQR] | total cal audits |
|---|---:|---:|---:|---:|---:|
| `monitor_only` | 0.9298±0.0000 | 0.9298±0.0000 | 751±0 | 751[751,751] | 0±0 |
| `fixed_low` | 0.9534±0.0007 | 0.9510±0.0032 | 92±88 | 34[2,166] | 7500±0 |
| `adaptive` | 0.9531±0.0020 | 0.9494±0.0043 | 152±183 | 14[0,259] | 188246±19855 |
| `exp3` | 0.9509±0.0056 | 0.9428±0.0081 | 140±150 | 48[6,130] | 39383±632 |
| `high` | 0.9541±0.0002 | 0.9522±0.0005 | 28±45 | 0[0,23] | 300000±0 |

### SHP in-domain

| schedule | final reward | min post reward | recovery time | median[IQR] | total cal audits |
|---|---:|---:|---:|---:|---:|
| `monitor_only` | 0.5385±0.0000 | 0.5397±0.0002 | 401±0 | 401[401,401] | 0±0 |
| `fixed_low` | 0.5689±0.0045 | 0.5669±0.0035 | 72±92 | 22[0,52] | 8000±0 |
| `adaptive` | 0.5723±0.0016 | 0.5690±0.0014 | 112±88 | 82[28,136] | 152813±17945 |
| `exp3` | 0.5708±0.0048 | 0.5622±0.0069 | 130±70 | 138[78,175] | 38916±575 |
| `high` | 0.5717±0.0011 | 0.5692±0.0011 | 90±95 | 28[4,130] | 250000±0 |

### SHP cross-domain

| schedule | final reward | min post reward | recovery time | median[IQR] | total cal audits |
|---|---:|---:|---:|---:|---:|
| `monitor_only` | 0.6864±0.0009 | 0.6844±0.0007 | 401±0 | 401[401,401] | 0±0 |
| `fixed_low` | 0.6901±0.0005 | 0.6888±0.0005 | 128±131 | 134[72,148] | 8000±0 |
| `adaptive` | 0.6900±0.0006 | 0.6885±0.0007 | 60±104 | 20[13,63] | 148505±29557 |
| `exp3` | 0.6899±0.0005 | 0.6882±0.0012 | 151±128 | 178[107,198] | 38664±869 |
| `high` | 0.6898±0.0006 | 0.6887±0.0004 | 65±68 | 49[36,52] | 250000±0 |

### TinyLLM spurious conciseness drift

| schedule | final reward | min post reward | recovery time | median[IQR] | total cal audits |
|---|---:|---:|---:|---:|---:|
| `monitor_only` | 0.3672±0.0000 | 0.3672±0.0000 | 401±0 | 401[401,401] | 0±0 |
| `fixed_low` | 0.4201±0.0019 | 0.4122±0.0078 | 77±156 | 19[12,56] | 8000±0 |
| `adaptive` | 0.4198±0.0019 | 0.4170±0.0067 | 76±67 | 100[42,105] | 154991±59981 |
| `exp3` | 0.4089±0.0158 | 0.4078±0.0108 | 267±228 | 401[91,401] | 38996±1274 |
| `high` | 0.4214±0.0017 | 0.4170±0.0016 | 38±45 | 22[14,73] | 250000±0 |

### TinyLLM aligned verbosity drift

| schedule | final reward | min post reward | recovery time | median[IQR] | total cal audits |
|---|---:|---:|---:|---:|---:|
| `monitor_only` | 0.4240±0.0017 | 0.4179±0.0016 | 0±0 | 0[0,0] | 0±0 |
| `fixed_low` | 0.4201±0.0019 | 0.4122±0.0078 | 77±156 | 19[12,56] | 8000±0 |
| `adaptive` | 0.4198±0.0019 | 0.4170±0.0067 | 76±67 | 100[42,105] | 154991±59981 |
| `exp3` | 0.4113±0.0134 | 0.4164±0.0025 | 205±230 | 133[91,401] | 38996±1274 |
| `high` | 0.4214±0.0017 | 0.4170±0.0016 | 38±45 | 22[14,73] | 250000±0 |

## 10. 主要结论

1. `monitor_only` 在 spurious drift 下不能校准 cheap judge bias，WDBC、SHP、TinyLLM spurious 三组实验中均表现为 recovery time 达到 censored value，且 final reward 明显低于带 calibration 的 schedule。

2. calibration audits 是必要的。即使是 `fixed_low`，也能在 WDBC、SHP 和 TinyLLM spurious 中大幅提升 final reward，并显著缩短 drift 后恢复时间。

3. `high` 通常最稳定，但 audit cost 最高。它在 WDBC 和 TinyLLM spurious 中给出最高或接近最高的 final reward / min post reward，但 calibration audits 固定达到 250000 或 300000 量级。

4. `adaptive` 的价值在于用 certificate 触发高预算 burst，在多数实验中接近 `high` 的 reward，但 audit cost 明显低于 `high`。不过在 WDBC 中当前 artifacts 显示 adaptive calibration cost 仍较高，且 recovery time 方差较大。

5. `exp3` 的 audit cost 低于 `adaptive` 和 `high`，但稳定性不如 `fixed_low` / `adaptive` / `high`。在 TinyLLM spurious 中，`exp3` 的 recovery time 和 reward 方差尤其明显。

6. cross-domain setting 中，各 calibration schedule 的 final reward 差异较小，但均优于 `monitor_only`；`adaptive` 和 `high` 在 recovery time 上更有优势。

7. TinyLLM aligned verbosity drift 显示 reward 本身不总能反映 stationarity 问题：`monitor_only` 的 final reward 可以较高，但 appendix 表格显示其 final stationarity gap 明显偏大；这支持同时报告 reward 与 stationarity/certificate 指标。
