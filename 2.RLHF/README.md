# 2.RLHF实验

本仓库包含一组小规模 RLHF 实验，用来研究一个双通道反馈问题：cheap feedback 信息密集、成本低，但可能漂移或带偏；audited feedback 更可靠，但成本更高；因此，什么时候花费 audited feedback 本身就是一个值得研究的调度问题。

本实验会先构造一组 checkpoint / policy pool，再模拟一个会漂移的 cheap judge，用 audited labels 校准漂移，并在同一个 A-OMP 更新框架下比较多种 audit schedule。

## 实验组成

| 实验 | 入口脚本 | policy / checkpoint pool | 漂移类型 | 主要输出 |
|---|---|---|---|---|
| WDBC | `codes/experiments/rlhf_wdbc.py` | 20 个 logistic regression 分类器 | spurious coefficient-norm bias | `results/rlhf_wdbc/`, `figs/rlhf_real_*.pdf` |
| SHP | `codes/experiments/rlhf_shp.py` | 20 个 TF-IDF + logistic regression 偏好模型 | response-length bias | `results/rlhf_shp/`, `figs/rlhf_shp_*.pdf` |
| SHP 跨域 | `codes/experiments/rlhf_shp.py` | 在 `askphysics` 训练，在 `askhistorians` 测试 | 跨域下的 response-length bias | `results/rlhf_shp_cross_domain/`, `figs/rlhf_shp_cross_domain_*.pdf` |
| TinyLLM | `codes/experiments/rlhf_tinyllm.py` | 12 个 tiny Transformer DPO checkpoint | spurious conciseness drift | `results/rlhf_tinyllm/`, `figs/rlhf_tinyllm_*.pdf` |
| TinyLLM aligned | `codes/experiments/rlhf_tinyllm.py` | 复用 TinyLLM checkpoint pool | aligned verbosity drift 消融 | `results/rlhf_tinyllm_aligned/`, `figs/rlhf_tinyllm_aligned_*.pdf` |

## 方法概览

每个实验最终都会被整理成同一个抽象接口：

```text
correct[k, j]  # 第 k 个 checkpoint 在第 j 个测试样本上是否正确
r_true[k]      # 第 k 个 checkpoint 的真实 reward / accuracy
u[k]           # spurious feature 方向
z_t            # 第 t 轮 checkpoint 混合分布
```

每一轮外层循环中，cheap judge 生成代理分数：

```text
s[k] = r_true[k] + beta_t * u[k] + noise[k]
```

其中 `beta_t` 会在 `drift_iter` 后改变，使 cheap channel 开始偏向 spurious feature。audited feedback 用来估计这个偏置：

```text
beta_hat = dot(u_cal, s[pol_cal] - y_cal) / dot(u_cal, u_cal)
r_hat = s - beta_hat * u
```

然后 A-OMP 用去偏后的 `r_hat` 更新 checkpoint 混合分布：

```text
w      = kl_prox(z, eta, g_prev)
g      = F_est(w)
z_next = kl_prox(z, eta, g)
```

最终比较不同 schedule 的 reward、stationarity gap、drift 后恢复速度和总 audit cost。

## 目录结构

```text
2.RLHF/
  README.md
  requirements.txt
  codes/
    data/
      askphysics_train.json
      askphysics_validation.json
      askphysics_test.json
      askhistorians_test.json
    experiments/
      rlhf_wdbc.py
      rlhf_shp.py
      rlhf_tinyllm.py
    scripts/
      plot_wdbc_from_csv.py
      plot_shp_from_csv.py
      plot_tinyllm_from_csv.py
      make_bootstrap_shp_table.py
      make_shp_cross_domain_table.py
      make_tinyllm_drift_table.py
    src/
      aomp.py
      audit_schedules.py
      metrics.py
      plotting.py
      simplex.py
  results/
    rlhf_wdbc/
    rlhf_shp/
    rlhf_shp_cross_domain/
    rlhf_tinyllm/
    rlhf_tinyllm_aligned/
  figs/
  tables/
```

## 环境准备

推荐使用 Python 3.10+。以下命令以 Windows PowerShell 为例，并假设从仓库根目录运行。

```powershell
cd "\Project-HongyuWang\2.RLHF"
conda create -n rlhf python=3.10 -y
conda activate rlhf
python -m pip install -r requirements.txt
$env:PYTHONPATH="$PWD\codes"
```

TinyLLM 依赖 `torch`。如果需要特定 CPU / CUDA 版本的 PyTorch，可以先按本机环境安装合适的 PyTorch，再执行：

```powershell
python -m pip install -r requirements.txt
```

## 复现实验

### 1. 运行 WDBC

WDBC 是最轻量的实验：

```powershell
$env:PYTHONPATH="$PWD\codes"
python codes\experiments\rlhf_wdbc.py
```

预期输出：

```text
results/rlhf_wdbc/config.json
results/rlhf_wdbc/summary_multiseed.json
results/rlhf_wdbc/trajectory_<schedule>_seed<seed>.csv
figs/rlhf_real_*.pdf
```

### 2. 运行 SHP 和 SHP 跨域实验

该实验使用仓库中自带的 SHP 风格 JSON 子集：

```text
codes/data/askphysics_train.json
codes/data/askphysics_validation.json
codes/data/askphysics_test.json
codes/data/askhistorians_test.json
```

运行：

```powershell
$env:PYTHONPATH="$PWD\codes"
python codes\experiments\rlhf_shp.py
```

预期输出：

```text
results/rlhf_shp/summary_multiseed.json
results/rlhf_shp_cross_domain/summary_multiseed.json
results/rlhf_shp/trajectory_<schedule>_seed<seed>.csv
results/rlhf_shp_cross_domain/trajectory_<schedule>_seed<seed>.csv
figs/rlhf_shp_*.pdf
figs/rlhf_shp_cross_domain_*.pdf
tables/bootstrap_shp_table.tex
tables/shp_cross_domain_table.tex
```

### 3. 运行 TinyLLM

TinyLLM 会训练 tiny Transformer 和 DPO checkpoint pool：

```powershell
$env:PYTHONPATH="$PWD\codes"
python codes\experiments\rlhf_tinyllm.py
```

预期输出：

```text
results/rlhf_tinyllm/checkpoint_pool.npz
results/rlhf_tinyllm/summary_multiseed.json
results/rlhf_tinyllm_aligned/summary_multiseed.json
results/rlhf_tinyllm/trajectory_<schedule>_seed<seed>.csv
results/rlhf_tinyllm_aligned/trajectory_<schedule>_seed<seed>.csv
figs/rlhf_tinyllm_*.pdf
figs/rlhf_tinyllm_aligned_*.pdf
tables/tinyllm_drift_reward_cert_table.tex
```

如果只想构建并缓存 TinyLLM checkpoint pool：

```powershell
$env:PYTHONPATH="$PWD\codes"
python codes\experiments\rlhf_tinyllm.py --build_only
```

## 图表

绘制图表：

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

## 对比的schedule

| Schedule | 行为 |
|---|---|
| `monitor_only` | 只使用 monitoring audits，不做 calibration audits。 |
| `fixed_low` | 每轮使用固定低预算 calibration audits。 |
| `adaptive` | 默认使用低预算；当 `cert_hp` 超过 warmup 阈值时触发高预算 burst。 |
| `exp3` | 把 audit 预算 `{0, low, high}` 当作 bandit arms，在线更新选择概率。 |
| `high` | 每轮使用高预算 calibration audits；通常最稳，但成本最高。 |

## 关键指标

| 指标 | 含义 |
|---|---|
| `reward` | 当前 checkpoint 混合策略的真实收益，`dot(r_true, z_t)`。 |
| `gap` | 真实 reward operator 下的 stationarity gap；越低表示越接近真实稳定方向。 |
| `beta_hat` | 通过 audited calibration 估计出的 cheap judge drift 系数。 |
| `n_cal` | 当前轮使用的 calibration audits 数量。 |
| `cum_aud` | 累计 audit cost。 |
| `final_reward` | 最后一轮 reward。 |
| `min_post_reward` | drift 后窗口内最低 reward。 |
| `recovery_time` | drift 后恢复所需轮数。 |
| `total_audits` | monitoring + calibration 的总 audit 数量。 |





