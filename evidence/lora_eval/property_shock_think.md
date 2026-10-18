# Base vs LoRA on `property_shock` (4 attempts per question, thinking on)

| | base | LoRA |
|---|---:|---:|
| mean reward | 0.964 | 0.959 |
| correctness | 0.978 | 0.978 |
| grounding | 0.929 | 0.930 |
| tool_selection | 0.983 | 0.917 |
| call_validity | 0.919 | 0.939 |
| efficiency | 1.000 | 1.000 |
| tool calls / errors | 97 / 11 | 92 / 9 |
| answers with no tool call | 0 | 0 |
| output tokens / episode | 970.4 | 1041.7 |

| Question | base | LoRA | Δ |
|---|---:|---:|---:|
| q01_previous_allowance | 0.993 | 0.978 | -0.015 |
| q02_current_allowance | 1.000 | 0.998 | -0.002 |
| q03_dollar_change | 0.985 | 0.983 | -0.002 |
| q04_pct_change | 0.996 | 0.995 | -0.001 |
| q05_drivers | 0.983 | 0.983 | -0.000 |
| q06_top_portfolio | 0.966 | 0.984 | +0.018 |
| q07_top_borrowers | 0.886 | 0.971 | +0.085 |
| q08_pd | 0.970 | 0.975 | +0.005 |
| q09_lgd | 0.973 | 0.970 | -0.003 |
| q10_exposure | 0.968 | 0.961 | -0.007 |
| q11_macro_variable | 0.955 | 0.869 | -0.086 |
| q12_weights | 1.000 | 1.000 | +0.000 |
| q13_model_change | 0.998 | 0.983 | -0.015 |
| q14_cre | 0.792 | 0.750 | -0.043 |
| q15_reconcile | 0.993 | 0.991 | -0.003 |
