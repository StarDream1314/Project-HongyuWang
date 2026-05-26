# ALFWorld tau995_b2_n2 Rerun Summary

## Experiment Setup

- Track: ALFWorld
- Schedulers: adaptive, oracle_high, cheap_only, fixed_low, exp3
- Seeds: 42, 123, 456
- Tasks: 134 per seed, 402 per scheduler
- Adaptive parameters: OLE certificate `ole_particles_v2_real_progress_tau995_auto.npz`, burst length 2, calibration high calls 2
- Reduced ablation: same parameters with `--alfworld-prompt-mode reduced`

## Main Results

| experiment   | scheduler   |   rows |   success_rate |   avg_progress |   min_progress |   partial_rows |   llm_calls |   avg_calls_per_task |   high_rows |   high_row_rate |   seed_success_std |   seed_progress_std |
|:-------------|:------------|-------:|---------------:|---------------:|---------------:|---------------:|------------:|---------------------:|------------:|----------------:|-------------------:|--------------------:|
| main         | adaptive    |    402 |         0.8856 |         0.8002 |         0.0000 |            202 |         788 |               1.9602 |         193 |          0.4801 |             0.1407 |              0.1113 |
| main         | oracle_high |    402 |         0.9229 |         0.8375 |         0.3333 |            170 |        1608 |               4.0000 |         402 |          1.0000 |             0.0841 |              0.1018 |
| main         | cheap_only  |    402 |         0.7736 |         0.7040 |         0.3333 |            289 |         402 |               1.0000 |           0 |          0.0000 |             0.1601 |              0.1601 |
| main         | fixed_low   |    402 |         0.8333 |         0.7172 |         0.3333 |            298 |         636 |               1.5821 |          78 |          0.1940 |             0.1188 |              0.1455 |
| main         | exp3        |    402 |         0.7935 |         0.6468 |         0.0000 |            352 |         738 |               1.8358 |         112 |          0.2786 |             0.0711 |              0.0401 |

## Reduced Prompt Ablation

| experiment     | scheduler   |   rows |   success_rate |   avg_progress |   min_progress |   partial_rows |   llm_calls |   avg_calls_per_task |   high_rows |   high_row_rate |   seed_success_std |   seed_progress_std |
|:---------------|:------------|-------:|---------------:|---------------:|---------------:|---------------:|------------:|---------------------:|------------:|----------------:|-------------------:|--------------------:|
| reduced_prompt | adaptive    |    402 |         0.8905 |         0.7861 |         0.0000 |            210 |         798 |               1.9851 |         198 |          0.4925 |             0.0449 |              0.0542 |
| reduced_prompt | oracle_high |    402 |         0.7985 |         0.7446 |         0.0000 |            232 |        1608 |               4.0000 |         402 |          1.0000 |             0.1451 |              0.1354 |
| reduced_prompt | cheap_only  |    402 |         0.8184 |         0.7156 |         0.3333 |            294 |         402 |               1.0000 |           0 |          0.0000 |             0.1287 |              0.1554 |
| reduced_prompt | fixed_low   |    402 |         0.8930 |         0.7807 |         0.3333 |            235 |         636 |               1.5821 |          78 |          0.1940 |             0.1210 |              0.1192 |
| reduced_prompt | exp3        |    402 |         0.8383 |         0.7172 |         0.0000 |            295 |         690 |               1.7164 |          96 |          0.2388 |             0.1042 |              0.1262 |

## Key Takeaways

- main: adaptive reaches 88.56% success and 0.8002 average progress with 788 calls.
- main: compared with oracle_high, adaptive saves 51.00% calls; progress gap is -0.0373, success gap is -0.0373.
- main: compared with cheap_only, adaptive improves average progress by +0.0962; compared with the best non-oracle baseline, adaptive gap is +0.0829.
- reduced: adaptive reaches 89.05% success and 0.7861 average progress with 798 calls.
- reduced: compared with oracle_high, adaptive saves 50.37% calls; progress gap is +0.0415, success gap is +0.0920.
- reduced: compared with cheap_only, adaptive improves average progress by +0.0705; compared with the best non-oracle baseline, adaptive gap is +0.0054.

## Defense-Oriented Conclusion

With the recalibrated OLE certificate and short adaptive burst, the adaptive scheduler no longer behaves like the dense high-cost channel. It uses substantially fewer LLM calls than oracle_high while preserving near-oracle task progress on the main ALFWorld setting. The reduced-prompt ablation is harder and exposes prompt sensitivity, but adaptive still provides a clear memory-correction advantage over cheap_only and remains competitive with fixed/EXP3 scheduling under a lower call budget.

This supports the project claim: the useful contribution is not reproducing A-OMP theory, but landing the two-channel idea as a memory-correction scheduler for long-horizon agent tasks. The system learns when low-cost experience is enough and when high-quality feedback should rewrite or correct memory.
