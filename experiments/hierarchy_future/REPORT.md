# Hierarchy demand features on the future-driver control

Exploratory, isolated. Nothing in `day1_ml/`, `shared/` or `results/day1/` was changed.

**Question:** do the hierarchy aggregate features of notebook 03c (section 6) add value once the model already has the full price and promotion plan?

**Control:** `lgbm_future_price_levels` (notebook 03). L2, LocalStandardScaler, notebook 02 LightGBM parameters, lags and rolling means, the department, class and subclass categoricals, and all ten plan columns. Each configuration adds the 11 aggregate columns per hierarchy level. Nothing else changes: same 2,000 SKUs, 14-week horizon and cutoffs.

**Acceptance:** development WAPE at least 0.2 points lower, and better at 5 or more of the 7 development cutoffs.

## Checks

- **Control reproduction:** development WAPE 0.3109 and final WAPE 0.2782, matching the saved notebook 03 forecasts. The largest absolute difference is 0.016 units over 280,000 rows. This holds only in the notebook kernel's environment, `packt_2day_cohort_2026` (lightgbm 4.7.0, mlforecast 1.1.0). The default `python` (lightgbm 4.6.0, mlforecast 1.0.2) gives 0.3122.
- **Origin safety:** the 03c aggregate code was used unchanged. After every cutoff, sales were scrambled and promotion flags flipped. All aggregate values in the 14 forecast weeks stayed the same at all 10 cutoffs. In those weeks, 97.8 to 99.9% of the values are present.
- **Training coverage:** training rows and `(unique_id, ds)` keys are identical to the control at every cutoff (945,000 rows in total). MLForecast drops rows only for null target lags, not for null exogenous values, so no matched-row control was needed. About 21% of the aggregate training values are missing, mostly in early weeks before the 52-week reference exists. LightGBM handles them natively.

## Results

These are the six specified runs, using notebook 03's `random_state`. Gain is the control's WAPE minus the model's, in points; positive is better.

| model | dev WAPE | gain (pts) | dev wins | final WAPE | gain (pts) | dev bias | final bias | dev mean RMSSE | dev APC | final APC | accepted |
|---|---|---|---|---|---|---|---|---|---|---|---|
| control | 0.3109 | — | — | 0.2782 | — | 0.069 | 0.055 | 1.658 | 0.130 | 0.090 | — |
| + department | 0.3157 | −0.48 | 1/7 | 0.2812 | −0.30 | 0.083 | 0.070 | 1.661 | 0.131 | 0.095 | no |
| + class | 0.3153 | −0.45 | 1/7 | 0.2824 | −0.41 | 0.083 | 0.075 | 1.648 | 0.129 | 0.099 | no |
| + subclass | 0.3099 | +0.09 | 4/7 | 0.2796 | −0.14 | 0.074 | 0.074 | 1.639 | 0.127 | 0.098 | no |
| + department, class | 0.3170 | −0.62 | 2/7 | 0.2802 | −0.20 | 0.088 | 0.066 | 1.660 | 0.129 | 0.096 | no |
| + all three | 0.3160 | −0.51 | 1/7 | 0.2850 | −0.67 | 0.086 | 0.077 | 1.648 | 0.128 | 0.095 | no |

**Seed check (subclass only):** each subclass refit was compared with a control refit using the same seed.

| seed | gain (pts) | dev wins |
|---|---|---|
| 0 | +0.09 | 4/7 |
| 1 | +0.02 | 3/7 |
| 2 | −0.16 | 1/7 |

The mean gain is about −0.02 points.

### Development WAPE by volume segment

The segments are recomputed at each cutoff. Change is in points against the control, and negative is better.

| model | low | mid | high | wins (low / mid / high) |
|---|---|---|---|---|
| control WAPE | 0.755 | 0.407 | 0.296 | |
| + department | +0.12 | +0.24 | +0.50 | 3 / 3 / 0 |
| + class | −0.21 | −0.07 | +0.50 | 6 / 4 / 1 |
| + subclass | −0.14 | −0.59 | −0.05 | 5 / 6 / 4 |
| + department, class | −0.05 | +0.03 | +0.68 | 3 / 4 / 2 |
| + all three | −0.09 | −0.08 | +0.57 | 4 / 4 / 1 |

## Findings

- **No configuration passes.** Department, class and the combinations are worse on development by 0.45 to 0.62 points, better at only 1 or 2 of the 7 cutoffs, and also worse on the final window.
- **Subclass is noise.** Its +0.09 points with 4/7 wins falls short of both thresholds. Across three seeds the mean gain is about −0.02 points, and the final window is worse with every seed.
- **Over-forecast bias increases** with every configuration: from 6.9% to between 7.4% and 8.8% on development, and from 5.5% to between 6.6% and 7.7% on the final window.
- **Mean RMSSE and development APC improve slightly** for class, subclass and the combinations. Final-window APC is worse for every configuration (0.095 to 0.099, against 0.090).
- **The losses are in high-volume products**, which carry most of the portfolio WAPE: +0.5 to +0.7 points for every configuration except subclass. Subclass helps mid-volume products (−0.59 points, 6/7 wins), but that does not carry through to the portfolio.
- **Interpretation:** lagged group demand mostly reflects promotions run in that group 14 or more weeks earlier. Once the model has each product's own plan, that signal is largely redundant, and it tends to raise the forecast level.

**Recommendation:** keep `lgbm_future_price_levels` as it is. Do not add the hierarchy aggregate features to the future-driver model.

## Reproduce (from the repository root)

```
PY=~/anaconda3/envs/packt_2day_cohort_2026/python.exe
$PY experiments/hierarchy_future/hierarchy_future.py check
$PY experiments/hierarchy_future/hierarchy_future.py models
$PY experiments/hierarchy_future/hierarchy_future.py models lgbm_future_price_levels fpl_subclass_agg --seed 1   # and --seed 2
$PY experiments/hierarchy_future/hierarchy_future.py evaluate
```

Run `evaluate` after adding the seed runs and APC changes in the third decimal place, because APC is scored on the blocks common to all loaded models. The table above uses the six-model evaluation.
