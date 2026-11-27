# Day 1 results

`forecasts/` holds one parquet file per notebook in the format described in `shared/README.md`. All forecasts are for the 2,000-series teaching sample at ten cutoffs. Every row carries `information_tier`, what the forecast needs to know on its cutoff date (`shared/README.md`, "Information").

| file | written by |
|---|---|
| `leaderboard.csv` | notebook 08: every model and strategy on the final 14 weeks, with `information`, `information_tier` and the scenario flags `without_future_drivers` and `with_future_drivers` |
| `forecasts/01_baselines.parquet` ... `07_model_selection.parquet` | notebooks 01 to 07 (`03b_promotion_spikes` from notebook 03b, `03c_behavior_features` from 03c) |
| `forecasts/03_exogenous.parquet` | notebook 03: `lgbm_categorical`, the five `lgbm_future_*` plan models, `lgbm_hist_drivers`, and the leakage diagnostics `lgbm_realized_price_leaky` and `lgbm_day2_covariates_leaky` |
| `forecasts/03_partial_plan.parquet` | notebook 03, optional: `lgbm_partial_plan4`, the plan for the first four weeks only |
| `forecasts/03_catboost.parquet` | notebook 03, optional: three CatBoost models, `lgbm_hist_exog` and `lgbm_exog_categorical` |
| `forecasts/04_partition_granularity.parquet` | notebook 04, optional section: one LightGBM per department, class or subclass (groups under 10 series use the global model) |
| `forecasts/04_limited_history.parquet` | notebook 04: 400 products, shortened history. Partial coverage; not in the leaderboards |
| `forecasts/06_l1_features.parquet` | notebook 06, section 7: the main notebook 03 models refit with the L1 objective |
| `forecasts/06_raw_quantile.parquet` | notebook 06, section 4: `lgbm_quantile60_raw`, the 0.6 quantile on the raw target. Not in the leaderboard or the selection pools |
| `forecasts/06_max_scaled.parquet` | `experiments/06_max_scaling/` (formerly an optional section of notebook 06): `lgbm_l1_max` and `lgbm_tweedie_max`. Not in the leaderboard or the selection pools |
| `day2_benchmarks.parquet` | notebook 08: eight benchmark models for Day 2, at the final and stability cutoffs (recommended; see below) |
| `day2_handoff.parquet` | notebook 08: the larger set of 20 models |

The final 14 weeks were inspected during development, so the leaderboards are exploratory benchmarks, not untouched holdout results. Future-driver scores use realized promotion flags, regular prices and promotional prices as stand-ins for the plan, so they assume the plan is carried out as recorded.

Rerun the notebooks in numerical order to regenerate (run times in `day1_ml/README.md`).

## For Day 2: eight benchmark models

`day2_benchmarks.parquet` is the recommended file for comparing Day 2 models with Day 1. It needs only pandas with Parquet support (`pyarrow`). Compare a Day 2 model only with the group that has the same information.

| model | what it is | information | WAPE | mean RMSSE | median RMSSE |
|---|---|---|---|---|---|
| `Naive` | last observed week | history | 0.470 | 1.332 | 0.771 |
| `SeasonalNaive` | same week last year | history | 0.438 | 1.525 | 0.929 |
| `lgbm_baseline` | global LightGBM: lags, rolling means, calendar and Fourier terms, L2, per-series standardized target (02) | history | 0.419 | 1.302 | 0.733 |
| `lgbm_hist_drivers` | plus department, class and subclass and the historical price and promotion drivers (03) | historical drivers | 0.396 | 1.266 | 0.711 |
| `lgbm_l1_hist_drivers` | the same with the L1 objective (06); the best individual model without future drivers | historical drivers | 0.349 | 1.255 | 0.671 |
| `lgbm_future_price_levels` | `lgbm_categorical` (03) plus the price and promotion plan: promotion flag and calendar, promotional discount, price index, regular and planned price | future drivers | 0.278 | 1.148 | 0.570 |
| `lgbm_l1_future_price_levels` | the same with the L1 objective (06); the best individual model | future drivers | 0.268 | 1.155 | 0.568 |
| `lgbm_day2_covariates_leaky` | `lgbm_categorical` plus the promotion calendar and the Day 2 covariates `discount_depth` and `price_index`, computed as in `bootcamp.py` | leakage diagnostic | 0.254 | 1.116 | 0.540 |

Final 14 weeks. Scored with `bc.score`, the Day 1 models give the same numbers as on Day 1.

- **Day 2 models without covariates** (NHITS_noexog, NBEATS, DLinear, AutoETS, LightGBM_noexog, the zero-shot foundation models) compare with the first five rows.
- **Day 2 models with future covariates** (NHITSx, LightGBMx, LightGBMx_mono, the LightGBM of `00_recap_leaderboard`, Chronos2_zs_cov, Chronos2_ft, TabPFN) receive `discount_depth` and `price_index` (and in `00_recap_leaderboard` also `sale_price` and `price_ratio_to_rolling_mean`), all computed from the realized selling price. A week with no sales gets the regular price, so these covariates reveal zero-sales weeks: from them a classifier separates zero-sales weeks with an AUC of 0.90, against 0.63 from the planned drivers. These Day 2 models are therefore not comparable with `lgbm_future_price_levels`; their Day 1 equivalent is `lgbm_day2_covariates_leaky`, a leakage diagnostic. To compare a Day 2 model with the planned-driver models, rerun it with planned covariates (the promotion flag, `discount` and `price_index` from `shared.features.add_drivers`).

**Information equivalence checked.** `lgbm_day2_covariates_leaky` reproduces `bootcamp._add_price_features`: `discount_depth = max(0, 1 - sale_price / 4-week rolling mean of sale_price)` with the current week included, and `price_index = sale_price / median sale_price` of the training weeks. At the final cutoff the reference is the same as Day 2's (weeks before 2025-04-16). Day 2's static covariates (`log_mean_sales`, `share_zero_weeks`, `share_promo_weeks`) come from the training rows only and are not future information.

- **Sample.** The 2,000 series of Day 2's `bc.load_panel()`; ids and actuals are identical.
- **Horizon and origins.** 14 weekly steps (`W-WED`). Cutoff 2025-04-09 forecasts 2025-04-16 to 2025-07-16, Day 2's test window. Cutoffs 2025-02-12 and 2025-03-12 are included only for the forecast-stability measure (APC); ignore them otherwise.
- **Rows.** 672,000: 8 models x 3 cutoffs x 2,000 series x 14 weeks. The final cutoff alone is 224,000.
- **Columns.** `unique_id`, `cutoff`, `ds`, `model`, `yhat`, `experiment` (the Day 1 notebook), `information_tier`. `cutoff` and `ds` are timestamps.

```python
import pandas as pd

day1 = pd.read_parquet("results/day1/day2_benchmarks.parquet")
day1_final = day1[day1["cutoff"] == pd.Timestamp("2025-04-09")]

# Day 2 format: unique_id, ds and one column per model
day1_wide = day1_final.pivot(index=["unique_id", "ds"], columns="model", values="yhat").reset_index()
```

`day1_wide` has the same shape as the files `bc.save_forecasts` writes. Merge it with a Day 2 forecast on `unique_id` and `ds` (convert Day 2's `ds` with `pd.to_datetime` first) and score with `bc.score` as usual.

**WAPE and mean RMSSE rank differently.** WAPE is total absolute error over total sales, so it is dominated by the high-volume products. RMSSE is computed per series, scaled by that series' own seasonal-naive error, then averaged, so every product counts equally and the hardest products dominate the mean. That is why Naive has the worse WAPE but the better mean RMSSE of the two baselines. Day 1 ranks on WAPE and Day 2 on mean RMSSE, so report both.

## Day 1 forecasts for Day 2: the larger file

`day2_handoff.parquet` (written by notebook 08) holds the Day 1 models Day 2 compares against, so no Day 1 notebook has to be rerun.

| model | what it is | information |
|---|---|---|
| `Naive`, `SeasonalNaive`, `HistoricAverage` | statistical baselines (01) | history |
| `lgbm_baseline`, `lgbm_raw` | global LightGBM, L2, standardized and raw target (02) | history |
| `lgbm_l1_std`, `lgbm_l1_raw`, `lgbm_tweedie_raw` | the objective comparison (06) | history |
| `lgbm_categorical`, `lgbm_l1_std_categorical` | plus department, class and subclass, L2 (03) and L1 (06) | history |
| `lgbm_hist_drivers`, `lgbm_l1_hist_drivers` | plus historical price and promotion drivers, L2 (03) and L1 (06) | historical drivers |
| `Average_top3_ml`, `Average_top3_ml_behavior` | top-three averages without future drivers, members chosen on completed development runs (07) | historical drivers |
| `lgbm_future_promo_calendar` | promotion plan and calendar features (03) | future drivers |
| `lgbm_future_price_levels`, `lgbm_l1_future_price_levels` | the full plan, L2 (03) and L1 (06) | future drivers |
| `Average_top3_future` | top-three average with future drivers (07) | future drivers |
| `lgbm_realized_price_leaky` | the plan's price columns from the realized selling price (03) | leakage diagnostic |
| `lgbm_day2_covariates_leaky` | the Day 2 covariates (03) | leakage diagnostic |

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
day1 = day1[day1.model.isin(["SeasonalNaive", "lgbm_l1_std", "lgbm_l1_hist_drivers"])]

day2 = pd.read_parquet("day2_neural/results/day2/forecasts/01_gfm_default.parquet")   # unique_id, ds, NHITS, ...
day2 = to_long(day2.assign(ds=pd.to_datetime(day2.ds)), TEST_CUTOFF, "day2_01", information_tier="history")

actuals = load_panel()[["unique_id", "ds", "y"]]
board = leaderboard(pd.concat([day1, day2]), actuals, scope="test")
board[["model", "information", "wape", "bias", "mean_rmsse", "median_rmsse", "apc", "coverage"]]
```

`leaderboard` scores every model on the same rows and sorts by information, then WAPE. Set `information_tier` to the right label for each Day 2 model (see below).

## Known differences with the Day 2 code

- `bootcamp.py` looks for its data and results under `day2_neural/` (`day2_neural/data`, `day2_neural/results/day2`, `day2_neural/results/day1/leaderboard.csv`). In this repository the data is in `data/` and the Day 1 results are in `results/day1/`. Set `BOOTCAMP_DATA_DIR` to `data/`. `bc.leaderboard()` will not find the Day 1 file until its path is changed, and importing `bootcamp` creates `day2_neural/results/`.
- `bc.leaderboard()` sorts every row by mean RMSSE and ignores `information_tier`, so it would rank Day 1 leakage diagnostics next to deployable models. Filter on `without_future_drivers` or `with_future_drivers` first.
- Day 2 test dates are `datetime64[ms]`; convert with `pd.to_datetime` before joining, as above.
- `to_long` clips negative forecasts to zero, for Day 1 and Day 2 alike.

## What Day 2 needs to provide

To score Day 2 models with the shared evaluator and place them on the right leaderboard:

1. One parquet per lab with `unique_id`, `ds` and one column per model (what `bc.save_forecasts` already writes), for the 2,000-series sample at cutoff 2025-04-09. Nothing in the modeling notebooks needs to change.
2. A table `model,information_tier`. Proposed labels, to be confirmed by Day 2:
   - `history`: NHITS, NHITS_noexog, NHITS_tuned, NHITS_MQ, NBEATS_*, DLinear, AutoETS, the ETS reconciliations, LightGBM_conf, LightGBM_noexog, Chronos2_zs, TimesFM_zs.
   - `leakage_diagnostic`: the LightGBM of `00_recap_leaderboard`, LightGBMx, LightGBMx_mono, NHITSx, Chronos2_zs_cov, Chronos2_ft and TabPFN_TS, while their covariates come from the realized selling price. Rerun with planned covariates, they become `future_drivers`.
   - Ensembles and selections take the highest tier of their members. `ENS_mean_top3` and `SEL_by_class` choose members on test-window errors, so they are `leakage_diagnostic` regardless.
3. For APC: the same models forecast from cutoffs 2025-02-12 and 2025-03-12. For validation-based selection: the development cutoffs (2024-07-17 to 2025-01-01, four weeks apart). Without these, APC is blank and Day 2 selections can only use the final window.
4. Ensembles and selections chosen on the validation window (14 weeks before the final window), not on final-window errors.
