# ALFWorld Final Result Summary

| category | prompt_mode | method | rows | success_rate | avg_progress | min_progress | partial_rows | runtime_llm_calls |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| main | with_hints | adaptive_ole | 402 | 0.8532 | 0.7890 | 0.3333 | 208 | 1206 |
| main | with_hints | fixed_low | 402 | 0.7512 | 0.6314 | 0.3333 | 380 | 558 |
| main | with_hints | cheap_only | 402 | 0.7463 | 0.6169 | 0.0000 | 394 | 402 |
| main | with_hints | exp3 | 402 | 0.7612 | 0.6177 | 0.0000 | 391 | 560 |
| main | with_hints | oracle_high | 402 | 0.8408 | 0.7786 | 0.0000 | 213 | 1206 |
| prompt_ablation | reduced | adaptive_ole | 402 | 0.9353 | 0.8789 | 0.3333 | 124 | 1206 |
| prompt_ablation | reduced | cheap_only | 402 | 0.7363 | 0.6061 | 0.0000 | 397 | 402 |
| prompt_ablation | reduced | fixed_low | 402 | 0.8706 | 0.7454 | 0.0000 | 261 | 558 |
| prompt_ablation | reduced | exp3 | 402 | 0.8433 | 0.6907 | 0.0000 | 317 | 608 |
| prompt_ablation | reduced | oracle_high | 402 | 0.8060 | 0.7231 | 0.0000 | 263 | 1206 |

Note: this table reports task performance only; OLE certificate coverage is stored with the certificate artifact.
