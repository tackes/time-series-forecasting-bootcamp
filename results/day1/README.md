# Day 1 results

`forecasts/` holds one parquet file per notebook in the format described in `shared/README.md`. All forecasts are for the 2,000-series teaching sample at ten cutoffs.

| file | written by |
|---|---|
| `leaderboard.csv` | notebook 08: the canonical leaderboard, every model and strategy on the final 14 weeks, with `information_tier`. Compare within a tier; only `historical_only` rows are operational. |
| `forecasts/01_baselines.parquet` ... `07_model_selection.parquet` | notebooks 01 to 07 (`03b_promotion_spikes` from notebook 03b) |
| `forecasts/03_catboost.parquet` | notebook 03, optional section: three CatBoost models and `lgbm_exog_categorical` (historical_only) |
| `forecasts/03_categorical.parquet` | notebook 03, optional section: `lgbm_l1_std_categorical` and `lgbm_promo_calendar_categorical` (historical_only) |
| `forecasts/03c_behavior_features.parquet` | notebook 03c (optional): `lgbm_l1_cat_*` feature-group experiments on top of `lgbm_l1_std_categorical` (historical_only) |
| `forecasts/04_partition_granularity.parquet` | notebook 04, optional section: one LightGBM per department, class or subclass (groups under 10 series use the global model) |
| `forecasts/06_max_scaled.parquet` | notebook 06, optional section 8: `lgbm_l1_max` and `lgbm_tweedie_max` (target divided by each series' training maximum; historical_only). Not yet in the leaderboard or the selection pools |
| `forecasts/04_limited_history.parquet` | notebook 04: 400 products, shortened history. Partial coverage; not in the leaderboards |
| `day2_benchmarks.parquet` | notebook 08: five benchmark models for Day 2, at the final and stability cutoffs (recommended; see below) |
| `day2_handoff.parquet` | notebook 08: the larger set of 15 models, including the objective comparison and the planned, oracle and leaky references |

The final 14 weeks were inspected during development, so the leaderboards are exploratory benchmarks, not untouched holdout results.

Rerun the notebooks in order to regenerate. Most take one to three minutes on a laptop CPU; with their optional sections 03 takes about ten, 03c about three, 04 about five and 07 about 21.

## For Day 2: five benchmark models

`day2_benchmarks.parquet` is the recommended file for comparing Day 2 models with Day 1. It needs only pandas with Parquet support (`pyarrow`).

| model | what it is |
|---|---|
| `Naive` | last observed week |
| `SeasonalNaive` | same week last year |
| `lgbm_baseline` | basic global LightGBM: lags, rolling means, calendar and Fourier terms, L2 objective, per-series standardized target |
| `lgbm_l1_std_categorical` | as above with the L1 objective, plus department, class and subclass as categorical features |
| `lgbm_l1_cat_price_response` | plus historical price features; the best single Day 1 model |

- **Tier.** All five are `historical_only`: each forecast uses only data observed up to its cutoff. No realized future sales, prices or promotions. Compare Day 2 models with them only if those Day 2 models also use no future information.
- **Sample.** The 2,000 series of Day 2's `bc.load_panel()`; ids and actuals are identical.
- **Horizon and origins.** 14 weekly steps (`W-WED`). Cutoff 2025-04-09 forecasts 2025-04-16 to 2025-07-16, Day 2's test window. Cutoffs 2025-02-12 and 2025-03-12 are included only for the forecast-stability measure (APC); ignore them otherwise.
- **Rows.** 420,000: 5 models x 3 cutoffs x 2,000 series x 14 weeks. The final cutoff alone is 140,000.
- **Columns.** `unique_id`, `cutoff`, `ds`, `model`, `yhat`, `experiment` (the Day 1 notebook), `information_tier`. `cutoff` and `ds` are timestamps.

```python
import pandas as pd

day1 = pd.read_parquet("results/day1/day2_benchmarks.parquet")
day1_final = day1[day1["cutoff"] == pd.Timestamp("2025-04-09")]

# Day 2 format: unique_id, ds and one column per model
day1_wide = day1_final.pivot(index=["unique_id", "ds"], columns="model", values="yhat").reset_index()
```

`day1_wide` has the same shape as the files `bc.save_forecasts` writes. Merge it with a Day 2 forecast on `unique_id` and `ds` (convert Day 2's `ds` with `pd.to_datetime` first) and score with `bc.score` as usual. Scored that way, the Day 1 models give the same numbers as on Day 1:

| model | WAPE | mean RMSSE | median RMSSE |
|---|---|---|---|
| `lgbm_l1_cat_price_response` | 0.344 | 1.249 | 0.664 |
| `lgbm_l1_std_categorical` | 0.350 | 1.257 | 0.667 |
| `lgbm_baseline` | 0.419 | 1.302 | 0.733 |
| `SeasonalNaive` | 0.438 | 1.525 | 0.929 |
| `Naive` | 0.470 | 1.332 | 0.771 |

**WAPE and mean RMSSE rank differently.** WAPE is total absolute error over total sales, so it is dominated by the high-volume products. RMSSE is computed per series, scaled by that series' own seasonal-naive error, then averaged, so every product counts equally and the hardest products dominate the mean. That is why Naive has the worse WAPE but the better mean RMSSE of the two baselines. Day 1 ranks on WAPE and Day 2 on mean RMSSE, so report both.

## Day 1 forecasts for Day 2: the larger file

`day2_handoff.parquet` (written by notebook 08) holds the Day 1 models Day 2 compares against, so no Day 1 notebook has to be rerun.

| model | what it is | tier |
|---|---|---|
| `Naive`, `SeasonalNaive`, `HistoricAverage` | statistical baselines (01) | historical_only |
| `lgbm_baseline` | basic global LightGBM: L2 objective, per-series standardized target (02) | historical_only |
| `lgbm_raw`, `lgbm_l1_raw`, `lgbm_tweedie_raw` | the raw-target objective comparison: L2, L1, Tweedie 1.3 (02, 06) | historical_only |
| `lgbm_l1_std` | L1 objective, standardized target (06) | historical_only |
| `lgbm_l1_std_categorical` | plus department, class and subclass as categories (03) | historical_only |
| `lgbm_l1_cat_price_response` | plus historical price features; best historical-only individual model (03c) | historical_only |
| `Average_top3_ml`, `Average_top3_ml_behavior` | top-three ML averages, members chosen on completed development runs (07) | historical_only |
| `lgbm_promo_calendar_planned` | promotion plan for the first 4 weeks (03) | planned_scenario |
| `lgbm_promo_calendar_oracle` | realized promotions for all 14 weeks (03) | oracle |
| `lgbm_price_leaky` | realized future prices (03) | leaky_diagnostic |

**Format.** One row per `unique_id`, `cutoff`, `ds`, `model`, with `yhat`, `experiment` (the Day 1 notebook) and `information_tier`. `cutoff` is the last week of data the forecast used.

**Sample and window.** The 2,000 series of `shared.prepare_data.teaching_sample()`, which is the same set as Day 2's `bc.load_panel()` (checked: identical ids and identical actuals in the test window). Weekly, `W-WED`. The final cutoff is 2025-04-09, so the 14 target weeks are 2025-04-16 to 2025-07-16, which is Day 2's test window (`bc.TEST_START`). The file also has cutoffs 2025-02-12 and 2025-03-12, which APC needs. Forecasts at the seven development cutoffs are in `forecasts/`.

**Metrics** (`shared/evaluation.py`, on rows every model forecasts):
- WAPE = sum |forecast - actual| / sum actual, over all series and weeks together.
- Bias = sum (forecast - actual) / sum actual.
- RMSSE per series: RMSE over the 14 weeks divided by the in-sample RMSE of the 52-week seasonal naive, from history up to the cutoff. It is reported as a mean and a median over the series where that scale exists (`eligible_series`).
- APC: the change between this forecast and the one made four weeks earlier for the same target weeks. A model with only the final cutoff gets no APC.

On SeasonalNaive and Naive these give exactly the Day 2 numbers (WAPE 0.438 and 0.470, mean RMSSE 1.525 and 1.332).

**Loading and comparing.** Run from the repository root. `day2` is any Day 2 forecast frame with `unique_id`, `ds` and one column per model, as `bc.save_forecasts` writes it:

```python
import pandas as pd
from shared.prepare_data import load_panel, TEST_CUTOFF
from shared.leaderboard import to_long, leaderboard

day1 = pd.read_parquet("results/day1/day2_handoff.parquet")
day1 = day1[day1.model.isin(["SeasonalNaive", "lgbm_l1_std", "lgbm_l1_cat_price_response"])]

day2 = pd.read_parquet("day2_neural/results/day2/forecasts/01_gfm_default.parquet")   # unique_id, ds, NHITS, ...
day2 = to_long(day2.assign(ds=pd.to_datetime(day2.ds)), TEST_CUTOFF, "day2_01", information_tier="historical_only")

actuals = load_panel()[["unique_id", "ds", "y"]]
board = leaderboard(pd.concat([day1, day2]), actuals, scope="test")
board[["model", "information_tier", "wape", "bias", "mean_rmsse", "median_rmsse", "apc", "coverage"]]
```

`leaderboard` scores every model on the same rows and sorts within information tier. Set `information_tier` to the right label for each Day 2 model (see below). Compare models only within a tier.

## Known differences with the Day 2 code

- `bootcamp.py` looks for its data and results under `day2_neural/` (`day2_neural/data`, `day2_neural/results/day2`, `day2_neural/results/day1/leaderboard.csv`). In this repository the data is in `data/` and the Day 1 results are in `results/day1/`. Set `BOOTCAMP_DATA_DIR` to `data/`. `bc.leaderboard()` will not find the Day 1 file until its path is changed, and importing `bootcamp` creates `day2_neural/results/`.
- `bc.leaderboard()` sorts every row by mean RMSSE and ignores `information_tier`, so it would rank Day 1 oracle and leaky rows next to operational ones. Filter on `information_tier` first.
- Day 2 test dates are `datetime64[ms]`; convert with `pd.to_datetime` before joining, as above.
- `to_long` clips negative forecasts to zero, for Day 1 and Day 2 alike.

## What Day 2 needs to provide

To score Day 2 models with the shared evaluator and place them on the right leaderboard:

1. One parquet per lab with `unique_id`, `ds` and one column per model (what `bc.save_forecasts` already writes), for the 2,000-series sample at cutoff 2025-04-09. Nothing in the modeling notebooks needs to change.
2. A table `model,information_tier` using the four tiers. Proposed labels, to be confirmed by Day 2:
   - `historical_only`: NHITS, NHITS_noexog, NHITS_tuned, NHITS_MQ, NBEATS_*, DLinear, AutoETS, the ETS reconciliations, LightGBM_conf.
   - `oracle` or `leaky_diagnostic`: LightGBM from `00_recap_leaderboard`, LightGBMx, LightGBMx_mono and NHITSx all receive realized future prices (`sale_price`, `discount_depth`, `price_ratio_to_rolling_mean`, `price_index`), so they are `leaky_diagnostic` until rerun with promotion-only covariates.
   - Ensembles and selections take the highest tier of their members. `ENS_mean_top3` and `SEL_by_class` choose members on test-window errors, so they are `leaky_diagnostic` regardless.
3. For APC: the same models forecast from cutoffs 2025-02-12 and 2025-03-12. For validation-based selection: the development cutoffs (2024-07-17 to 2025-01-01, four weeks apart). Without these, APC is blank and Day 2 selections can only use the final window.
4. Ensembles and selections chosen on the validation window (14 weeks before the final window), not on final-window errors.
