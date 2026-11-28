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

## Information

`information_tier` records what a forecast needs to know on its cutoff date, from least to most. Notebooks display the business description.

| `information_tier` | shown as | meaning |
|---|---|---|
| `history` | Sales history and calendar only | sales up to the cutoff, calendar terms and fixed item attributes (department, class, subclass) |
| `historical_drivers` | Historical exogenous features | also past prices and promotions, each observed on or before the cutoff (lag 14 or more, or the regular price carried forward from the cutoff) |
| `partial_future_drivers` | Partial future-driver visibility | also the promotion plan for the first four forecast weeks (optional section of notebook 03) |
| `future_drivers` | Full future-driver visibility | also the price and promotion plan for all 14 weeks: promotion flag, regular price and promotional discount |
| `leakage_diagnostic` | Leakage diagnostic | uses what was not available on the cutoff or is derived from the target: the realized selling price (a week with no sales gets the regular price, so the price reveals the zeros), or candidates chosen with the final-window actuals |

The dataset has no copy of the plans as they stood on each cutoff, so recorded promotion flags, regular prices and promotional prices stand in for them. Future-driver scores are conditional on the plan being carried out as recorded. In promotion weeks the realized selling price has the same distribution with or without sales, so it stands in for the promotional price; outside promotion weeks it is never used as a plan (`add_drivers` in `features.py`).

A selection strategy or average takes the highest tier among its candidate pool (`combine_tiers`).

### Scenarios

A business scenario decides which tiers a deployable forecast may use (`SCENARIOS`, `eligible`, `scenario_pool`):

| scenario | example | may use |
|---|---|---|
| `no_future_drivers` | a manufacturer without the retailer's price and promotion plans | `history`, `historical_drivers` |
| `future_drivers` | a retailer forecasting its own sales | everything up to `future_drivers` |

Leakage diagnostics are never eligible. Model selection (notebook 07) builds each scenario's candidate pool with `usable_models` in `model_selection.py` (a wrapper around `eligible`), so a strategy is compared only with models whose information requirements the scenario meets. Rows with different information may sit side by side to show what the information is worth, never to compare architectures.

## Driver features (`features.py`)

| function | builds |
|---|---|
| `add_drivers` | `promo`, `regular_price`, `discount` (promotional discount, 0 outside promotions), `planned_price`, `price_index` (planned price over the median regular price of the 52 weeks ending 14 weeks earlier), and the realized `rel_price` |
| `future_drivers` | the plan columns plus promotion-calendar features (`promo_features`) for every week |
| `partial_plan` | the plan for the first `plan_weeks` forecast weeks, then no promotion at the cutoff's regular price |
| `historical_drivers`, `historical_future` | promotion and price history as of 14 weeks before each week, and the regular price; `historical_future` builds `X_df` with the regular price carried forward from the cutoff |
| `realized_drivers`, `day2_covariates` | leakage diagnostics: the plan's price columns from the realized selling price, and the Day 2 neural covariates |
| `item_attributes`, `in_horizon` | static attributes per series; the 14 weeks after a cutoff |

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

`breakdown` runs `score` by horizon or by a segment column.

`load_forecasts` rejects a file whose `information_tier` is not in `TIERS`, so a file saved by an older version of this code fails on load instead of being relabeled in a notebook.

## Model selection (`model_selection.py`)

The selection and averaging machinery of notebook 07. Every function takes its data as arguments.

| function | does |
|---|---|
| `usable_models(forecasts, future_drivers)` | the models a deployable forecast may use without, or with, the price and promotion plan |
| `hierarchy(panel)` | portfolio, department, class, subclass and SKU of every series |
| `selection_evidence(forecasts, actuals, models)` | development forecasts of a candidate pool with actuals, on the rows every candidate forecasts |
| `cost_table`, `choose` | selection cost per group and candidate (`evaluate()` with `mae` and `bias`, the group as the id), and the chosen model per series with the portfolio fallback |
| `strategy` | the chosen forecasts at each decision date, using only the runs completed by then (`completed_runs`) |
| `average`, `top3`, `best_individual` | equal-weight averages, the three best models on given runs, the best single model on a decision date |
| `wape_by_cutoff` | WAPE per model and cutoff, each cutoff on its own common rows |
| `ensemble_evidence`, `residuals`, `ensemble_rules`, `mean_correlation` | the ensemble experiments: leave-one-out top-three evidence, error matrices, and the diversity rules C to F |

A strategy or average takes the highest tier of its candidates (`information_of`).

## Leakage diagnostics (`diagnostics.py`)

`leakage_diagnostics(panel, cutoffs)` fits the two models that read the realized selling price, `lgbm_realized_price_leaky` and `lgbm_day2_covariates_leaky` (notebook 03, optional section 12), with the notebook 03 model setup. Their forecasts carry `leakage_diagnostic`, so `usable_models` never offers them to model selection; they exist for like-for-like comparisons with Day 2 models whose covariates come from the realized price.

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
- `test_information.py`: historical drivers and their `X_df` unchanged when everything after the cutoff is scrambled; future frames with one row per series and forecast week, the expected columns and no missing plan values; planned drivers unchanged when the realized selling price outside promotions is scrambled (the diagnostics must change); the partial plan.
- `test_tiers.py`: tier order and propagation, scenario eligibility and pools, tier labels on saved forecasts, every selection's SKU forecasts drawn from eligible candidates, leakage diagnostics absent from every selection and ensemble, evidence completeness on each decision date, the Day 2 files and leaderboard flags consistent with the saved forecasts.
- `test_leakage.py`: reruns the modeling cells of notebooks 03 (with its optional sections, including the leakage diagnostics), 03b, 03c and 03d after scrambling sales, realized prices or the plan after the cutoff, and checks each forecast against its own tier; reruns notebook 07 after scrambling sales after a decision date (slow, several minutes).
