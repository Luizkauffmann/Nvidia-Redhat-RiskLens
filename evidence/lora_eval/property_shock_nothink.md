# Base vs LoRA on `property_shock` (4 attempts per question, thinking off)

| | base | LoRA |
|---|---:|---:|
| mean reward | 0.751 | 0.939 |
| correctness | 0.781 | 0.944 |
| grounding | 0.790 | 0.931 |
| tool_selection | 0.717 | 0.850 |
| call_validity | 0.758 | 0.982 |
| efficiency | 0.997 | 1.000 |
| tool calls / errors | 82 / 1 | 79 / 3 |
| answers with no tool call | 14 | 0 |
| output tokens / episode | 417.5 | 567.4 |

| Question | base | LoRA | Δ |
|---|---:|---:|---:|
| q01_previous_allowance | 0.504 | 0.994 | +0.490 |
| q02_current_allowance | 0.506 | 0.991 | +0.485 |
| q03_dollar_change | 0.059 | 0.983 | +0.923 |
| q04_pct_change | 0.506 | 0.996 | +0.490 |
| q05_drivers | 0.994 | 0.897 | -0.096 |
| q06_top_portfolio | 0.983 | 0.865 | -0.117 |
| q07_top_borrowers | 0.968 | 0.804 | -0.164 |
| q08_pd | 0.786 | 0.966 | +0.180 |
| q09_lgd | 0.990 | 0.991 | +0.001 |
| q10_exposure | 0.918 | 0.978 | +0.059 |
| q11_macro_variable | 0.296 | 0.896 | +0.600 |
| q12_weights | 1.000 | 1.000 | +0.000 |
| q13_model_change | 1.000 | 0.996 | -0.004 |
| q14_cre | 0.754 | 0.732 | -0.022 |
| q15_reconcile | 1.000 | 0.993 | -0.007 |
