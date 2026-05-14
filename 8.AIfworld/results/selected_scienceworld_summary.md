# ScienceWorld Final Result Summary

| category | prompt_mode | method | rows | success_rate | avg_progress | min_progress | partial_rows | runtime_llm_calls |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| main | with_hints | adaptive_ole | 60 | 1.0000 | 0.9721 | 0.7000 | 10 | 180 |
| main | with_hints | fixed_low | 60 | 1.0000 | 0.9218 | 0.5000 | 18 | 84 |
| main | with_hints | cheap_only | 60 | 1.0000 | 0.9119 | 0.5000 | 26 | 60 |
| main | with_hints | exp3 | 60 | 0.9667 | 0.9134 | 0.0000 | 17 | 108 |
| main | with_hints | oracle_high | 60 | 1.0000 | 0.9507 | 0.7000 | 17 | 180 |
| prompt_ablation | reduced | adaptive_ole | 60 | 0.9667 | 0.8407 | 0.2500 | 34 | 180 |
| prompt_ablation | reduced | cheap_only | 60 | 0.8500 | 0.7418 | 0.4286 | 42 | 60 |
| prompt_ablation | reduced | fixed_low | 60 | 0.8833 | 0.7613 | 0.4286 | 48 | 84 |
| prompt_ablation | reduced | exp3 | 60 | 0.9833 | 0.8191 | 0.4286 | 38 | 108 |
| prompt_ablation | reduced | oracle_high | 60 | 0.9833 | 0.8343 | 0.4286 | 38 | 180 |

Note: coverage=1.000 is an auxiliary certificate coverage check, not the main performance claim.
