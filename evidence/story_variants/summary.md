# Story variants

Same bank, same dates (2026-07-31 → 2026-08-31), different story. Built by `python -m scripts.build_story_variants`; every figure comes from the CECL engine's official runs and sequential attribution. `design` = what the variant config says should happen; `observed` = what the attribution found.

| Variant | July $M | August $M | Change $M | Primary driver (observed) | Top portfolio | Top macro variable | Design checks |
|---|---:|---:|---:|---|---|---|---|
| `consumer_stress` | 97.32 | 120.70 | +23.38 | macroeconomic (+16.65) | C&I | unemployment_rate | 3/3 PASS |
| `credit_led` | 114.80 | 178.93 | +64.13 | credit_deterioration (+61.93) | CRE | unemployment_rate | 3/3 PASS |
| `growth_led` | 97.32 | 116.37 | +19.05 | portfolio_growth (+14.82) | CRE | unemployment_rate | 2/2 PASS |
| `property_shock` | 97.49 | 114.98 | +17.49 | macroeconomic (+15.87) | CRE | cre_price_growth | 4/4 PASS |
| `weights_shift` | 97.32 | 116.32 | +19.00 | model_methodology (+12.27) | CRE | unemployment_rate | 2/2 PASS |

## Driver attribution ($M)

| Variant | macroeconomic | credit_deterioration | portfolio_growth | model_methodology | paydowns_runoff |
|---|---:|---:|---:|---:|---:|
| `consumer_stress` | +16.65 | +3.81 | +2.50 | +0.00 | +0.42 |
| `credit_led` | +0.00 | +61.93 | +2.32 | +0.00 | -0.11 |
| `growth_led` | +0.00 | +3.81 | +14.82 | +0.00 | +0.42 |
| `property_shock` | +15.87 | +4.18 | +2.28 | +0.00 | -4.85 |
| `weights_shift` | +0.00 | +3.81 | +2.50 | +12.27 | +0.42 |

## Design checks

- **consumer_stress**: Consumer stress: unemployment shock, property prices flat
  - allowance_increases: expected `True`, observed `True` ✓
  - primary_driver: expected `macroeconomic`, observed `macroeconomic` ✓
  - primary_macro_variable: expected `unemployment_rate`, observed `unemployment_rate` ✓
- **credit_led**: Credit-led: macro flat, a large CRE cohort deteriorates sharply
  - allowance_increases: expected `True`, observed `True` ✓
  - primary_driver: expected `credit_deterioration`, observed `credit_deterioration` ✓
  - primary_portfolio: expected `CRE`, observed `CRE` ✓
- **growth_led**: Growth-led: macro flat, heavy new originations
  - allowance_increases: expected `True`, observed `True` ✓
  - primary_driver: expected `portfolio_growth`, observed `portfolio_growth` ✓
- **property_shock**: Property shock: CRE prices and rates worsen, unemployment flat
  - allowance_increases: expected `True`, observed `True` ✓
  - primary_driver: expected `macroeconomic`, observed `macroeconomic` ✓
  - primary_portfolio: expected `CRE`, observed `CRE` ✓
  - primary_macro_variable: expected `cre_price_growth`, observed `cre_price_growth` ✓
- **weights_shift**: Weights shift: scenario paths flat, weight moved to severe adverse
  - allowance_increases: expected `True`, observed `True` ✓
  - primary_driver: expected `model_methodology`, observed `model_methodology` ✓
