# ALFWorld Final Result Summary

| category | prompt_mode | method | rows | success_rate | avg_progress | min_progress | partial_rows | runtime_llm_calls |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| main | with_hints | adaptive | 402 | 0.8856 | 0.8002 | 0.0000 | 202 | 788 |
| main | with_hints | oracle_high | 402 | 0.9229 | 0.8375 | 0.3333 | 170 | 1608 |
| main | with_hints | cheap_only | 402 | 0.7736 | 0.7040 | 0.3333 | 289 | 402 |
| main | with_hints | fixed_low | 402 | 0.8333 | 0.7172 | 0.3333 | 298 | 636 |
| main | with_hints | exp3 | 402 | 0.7935 | 0.6468 | 0.0000 | 352 | 738 |
| prompt_ablation | reduced | adaptive | 402 | 0.8905 | 0.7861 | 0.0000 | 210 | 798 |
| prompt_ablation | reduced | oracle_high | 402 | 0.7985 | 0.7446 | 0.0000 | 232 | 1608 |
| prompt_ablation | reduced | cheap_only | 402 | 0.8184 | 0.7156 | 0.3333 | 294 | 402 |
| prompt_ablation | reduced | fixed_low | 402 | 0.8930 | 0.7807 | 0.3333 | 235 | 636 |
| prompt_ablation | reduced | exp3 | 402 | 0.8383 | 0.7172 | 0.0000 | 295 | 690 |

Note: this table reports task performance only; OLE certificate coverage is stored with the certificate artifact.
