# Shared code

Used by the Day 1 notebooks in `day1_ml/`.

## Forecast format

One long table. Every model, cutoff and horizon is a set of rows.

| column | meaning |
|---|---|
| `unique_id` | series |
| `cutoff` | last week of data available when the forecast was made (not the first target week) |
| `ds` | target week |
| `model` | model or strategy name |
| `yhat` | point forecast, clipped at zero |
| `experiment` | notebook that produced it |
| `information_tier` | see below |

Actuals are never stored with forecasts. They come from the prepared panel and are joined on `unique_id` and `ds`.

## Information tiers

From least to most information used. Tiers are reported in separate tables and never ranked together.

| tier | meaning |
|---|---|
| `historical_only` | only data observed on or before the cutoff |
| `planned_scenario` | also a promotion plan. The dataset has no real plan, so realized promotion flags for the first four forecast weeks stand in for one. That is an assumption, not a measured capability. |
| `oracle` | realized promotions for the whole horizon |
| `leaky_diagnostic` | realized future selling prices (a realized average price reveals sales: 84% of weeks priced exactly at the regular price have zero sales, against 2% of other weeks), or candidates chosen with the final-window actuals |

A selection strategy or average takes the highest tier among its candidate pool (`combine_tiers`).

## Windows

- Final 14 weeks: cutoff 2025-04-09, targets 2025-04-16 to 2025-07-16.
- Development: seven cutoffs four weeks apart, 2024-07-17 to 2025-01-01. Their targets end on or before 2025-04-09.
- Stability support: cutoffs 2025-02-12 and 2025-03-12, used only for APC at the final cutoff. Their later targets overlap the final window, so they are never scored for accuracy.
- A decision on date *d* may use only development runs whose whole 14-week horizon was observed by *d* (`completed_runs(d)`).

**The final window was inspected while these notebooks were built.** Choices such as per-series target scaling and the 26-week rolling mean were made after looking at it, and some parameters in notebook 07 were set after seeing individual models' scores on it. Treat final-window numbers as exploratory benchmarks, not an untouched holdout. Comparisons that hold on the development runs are better evidence.

All defined in `prepare_data.py`. `teaching_sample()` returns the 2,000 series used by Day 2's `bootcamp.load_panel()` (checked: identical ids). It excludes products with under 66 weeks of history, constant series and series with any non-positive price, so short-history and new products are not represented.

## Scoring pipeline

```
forecast submissions (long)  ->  score(): utilsforecast evaluate()  ->  + APC, coverage  ->  leaderboard()
```

Selection strategies and averages are saved as forecast submissions too, so they go through the same path.

`score(fcst, actuals, by=())` in `evaluation.py` is the only scoring function. It keeps the rows every model forecasts, pivots to one column per model, and calls `evaluate()`:

| column | how |
|---|---|
| `wape` | `nd` on the whole group as one id: `sum|f - y| / sum|y|`, the portfolio (or segment) value, never an average of SKU values |
| `accuracy` | `1 - wape` |
| `bias` | `cfe` (sum of `f - y`) on the group, divided by the group's sum of actuals; positive means overforecast |
| `mae` | `mae` on the group |
| `mean_rmsse`, `median_rmsse` | `rmsse` with `seasonality=52`, one value per series and cutoff, scale from weeks up to that cutoff only, then mean and median |
| `eligible_series` | series (times cutoffs, for multi-cutoff scopes) with a defined RMSSE scale |

Undefined values differ between `utilsforecast` versions (0 up to 0.2.14; NaN or inf from 0.2.15), so `score` sets them: a group with no sales gets no WAPE, and a series with no usable seasonal scale (under 53 weeks, or never different from the same week a year earlier) gets no RMSSE. `by` adds grouping columns (segment, horizon); pass `rmsse=False` when the group splits a series' forecast window, such as the horizon.

APC is custom (`apc`), because it compares forecasts from two origins and `utilsforecast` has no stability metric yet (Nixtla/utilsforecast#255). Forecasts for the same target weeks from origins 28 days apart are compared. Targets are counted from the newer cutoff; two complete four-week blocks are kept and the last two overlapping weeks are dropped. `apc = sum|new - old| / sum|old|` per SKU block. Every model in a table is scored on the same SKU, cutoff and block: a block that is incomplete or has `sum|old| = 0` for any model is dropped for all. `apc_coverage` is the share of a model's blocks that were scored and `zero_base` counts its blocks with `sum|old| = 0`. `apc` pools the blocks; `apc_median` is the median over SKUs. A model without forecasts from both origins has no APC (blank), not zero, and does not affect the common population.

## Leaderboard (`leaderboard.py`)

`leaderboard(fcst, actuals, scope)` is the canonical leaderboard: `score` joined with forecast `coverage` (measured before the common-row restriction), APC and the model's tier, one row per model or strategy, ordered by tier and then WAPE. `scope` is `"test"` (final window, APC on the two pairs ending there), `"dev"` (development runs) or a list of cutoffs. Columns:

`model, information_tier, experiment, wape, accuracy, bias, mean_rmsse, median_rmsse, apc, apc_median, apc_coverage, coverage, eligible_series, series, rows, mae, zero_base`

`breakdown` runs `score` by horizon or by a segment column. Notebook 07's selection cost calls `evaluate()` with `mae` and `bias` directly, with the selection group as the id.

## Using it from Day 2

Day 2 forecasts are wide (`unique_id`, `ds`, one column per model) and share the final cutoff, so an adapter is a few lines:

```python
from shared.leaderboard import to_long, save_forecasts
from shared.prepare_data import TEST_CUTOFF

wide = pd.read_parquet("<day2 forecast file>.parquet")
long = to_long(wide, TEST_CUTOFF, "day2_<lab>")
long["information_tier"] = long.model.map(tiers)       # model -> tier, supplied by Day 2
save_forecasts(long, "day2_<lab>")
```

What Day 2 needs to supply is listed in `results/day1/README.md`.

## Tests

```
python -m pytest shared
```

- `test_framework.py`: WAPE, bias and RMSSE by hand; WAPE and bias as portfolio ratios, not SKU averages; grouped scores against direct `utilsforecast` calls; RMSSE against `utilsforecast.rmsse`, including the undefined-scale case; RMSSE scale from history up to the cutoff only; scoring on common rows; APC edge cases and common support; conversion of the committed baseline outputs.
- `test_tiers.py`: tier propagation, tier labels on saved forecasts, price features confined to the leaky model, evidence completeness on each decision date.
- `test_leakage.py`: reruns notebooks 03, 03b and 07 after scrambling what the decision date should not see (slow, a few minutes).
