# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# # Day 2 · 00 (full data) — Recap: train on everything, score on the sample (10 min)
#
# The original Day 2 labs train and score on a 2,000-series sample so they run on a laptop. Here
# we **train on every eligible series** and **score on the same 2,000**, so every number below can
# be put next to the original labs.
#
# | | |
# |---|---|
# | Training panel | every series that passes the filters (≈17,000) |
# | Scored on | the same stratified 2,000-series sample as `../code` |
# | Test window | the final **14 weeks**, from **2025-04-16** |
# | Lead score | **median** RMSSE, with the mean alongside |
#
# This notebook
#
# 1. loads the full panel and shows which series were dropped and why,
# 2. picks the example series every later notebook plots,
# 3. re-creates the Day 1 reference models on the full panel, and
# 4. looks at their errors and their wins over the seasonal-naive yardstick.

# %%
import lightgbm as lgb
import pandas as pd
import polars as pl
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean
from mlforecast.target_transforms import LocalStandardScaler
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, Naive, SeasonalNaive

import gd_helpers as gd
import gd_plots as gplt

bc = gd.bc
gd.status()
gplt.use_style()

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## Intuition: why train on everything if we only score 2,000?
#
# A global model learns one shared pattern from many series. More series means more examples of
# rare things (deep discounts, top sellers, Easter weeks) to learn from. Scoring on a fixed sample
# keeps the comparison fair and the notebooks quick: the model can be trained on all ≈17,000
# series, but every lab is judged on the same 2,000.

# %% [markdown]
# ## 1. The panel and the evaluation sample

# %%
panel = gd.train_panel()
train, test = bc.split(panel, cutoff=bc.TEST_START)
eval_ids = gd.eval_ids()
info = gplt.series_table(panel)

print(f"{panel['unique_id'].n_unique():,} training series | {len(eval_ids):,} scored | "
      f"train {train['ds'].min()} → {train['ds'].max()} | "
      f"test {test['ds'].min()} → {test['ds'].max()}")

by_class = (info.assign(scored=info.index.isin(eval_ids))
            .groupby("sale_class")
            .agg(series=("scored", "size"), scored=("scored", "sum"),
                 mean_weekly_units=("mean_weekly_units", "mean"),
                 share_zero=("zero_share", "mean")))
by_class.round(2)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ### Which series are left out, and why
#
# The filters are the same as the original labs: no zero or negative prices, enough history for a
# 52-week yardstick, and not constant. The last row is **not applied**. It shows what the early
# research would also have dropped (series with a zero run of 26+ weeks). Later research found
# that dropping sparse series made the error much worse, so we keep them
# (`docs/gfm_learnings.md` §2).

# %%
gd.dropped_series()

# %% [markdown]
# ## 2. Example series: high, medium and low volume
#
# Picked once here and reused by every notebook, so you can watch the same series improve.
# Units are on the left axis; sale price (solid) and regular price (dashed) on the right.

# %%
examples = gplt.pick_examples(panel)
gplt.plot_series_price(panel, examples)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# Two axes on one chart can mislead: each axis can be stretched to suggest a link. The same
# series as two stacked panels shares only the dates:

# %%
gplt.plot_series_panels(panel, examples["high"][0])

# %% [markdown]
# ## 3. Day 1 reference models on the full panel
#
# Local models (Naive, SeasonalNaive, AutoETS) fit each series on its own, so we only need to fit
# them on the scored series. LightGBM is global: it is trained on all series.
#
# `AutoETS(season_length=1)` has no yearly season. ETS cannot carry a 52-week season, which is
# what SeasonalNaive is for.

# %%
def baseline_forecasts() -> pd.DataFrame:
    """Fit the four reference models; forecasts for the eval ids only."""
    local = StatsForecast(models=[Naive(), SeasonalNaive(season_length=bc.SEASON),
                                  AutoETS(season_length=1)], freq=bc.FREQ, n_jobs=gd.N_JOBS)
    with bc.Timer() as t_local:
        stats = local.forecast(df=bc.to_nixtla(gd.only(train, eval_ids)), h=bc.H)

    exog = ["promo", "sale_price", "discount_depth", "price_ratio_to_rolling_mean"]
    mlf = MLForecast(
        models={"LightGBM": lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05,
                                              num_leaves=63, random_state=bc.RANDOM_SEED,
                                              verbosity=-1)},
        freq=bc.FREQ, lags=[1, 2, 4, 13, 52],
        lag_transforms={1: [RollingMean(4), RollingMean(13)]},
        date_features=["month", "week"], target_transforms=[LocalStandardScaler()],
    )
    with bc.Timer() as t_lgb:
        mlf.fit(bc.to_nixtla(train, exog), static_features=[])
    future = gd.only(bc.to_nixtla(test, exog).drop(columns="y"), eval_ids)
    lgbm = mlf.predict(h=bc.H, X_df=future, ids=eval_ids)
    out = stats.reset_index().merge(lgbm, on=["unique_id", "ds"])
    return out


baselines = gd.cached("00_baselines", baseline_forecasts)
gd.save_forecasts(baselines, "00_baselines")
gd.record(baselines, test, train, lab="00 baselines (full panel)")
gd.median_board()

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ### Reading the result
#
# * The board is sorted by **median** RMSSE. The mean is pulled up by a minority of intermittent,
#   mostly-zero series, so the median tells you about the typical item.
# * LightGBM now learns from ≈17,000 series instead of 2,000. Compare its row with the original
#   lab 00 to see what more data buys a global model.
#
# *Reference run: full panel, GPU box. Exact numbers vary a little with hardware; the reading
# should not.*

# %% [markdown]
# ## 4. Where the errors are
#
# The same per-series errors, three ways: a histogram (log scale), the share of series at or
# below each error (ECDF), and the spread by sales class.

# %%
metrics = gd.score(baselines, test, train)
rmsse = gd.per_series(metrics, "rmsse")
gplt.plot_error_dist(rmsse, info)

# %%
gplt.plot_class_heatmap(rmsse, info)

# %% [markdown]
# ### Marginal wins over the yardstick
#
# How often does LightGBM beat SeasonalNaive, series by series? Counting wins weights every
# series the same; weighting by volume asks whether the wins are where the sales are.

# %%
gplt.plot_wins(rmsse, info, model="LightGBM", baseline="SeasonalNaive")

# %%
gd.save_config("00_baselines", {
    "n_train_series": panel["unique_id"].n_unique(),
    "n_eval_series": len(eval_ids),
    "eval_ids_path": str(gd.EVAL_IDS_PATH.relative_to(gd.ROOT)),
    "examples": examples,
    "baselines": ["Naive", "SeasonalNaive", "AutoETS", "LightGBM"],
    "median_rmsse": rmsse.median().round(4).to_dict(),
    "mean_rmsse": rmsse.mean().round(4).to_dict(),
})

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## Discussion
#
# * Two thirds of the series are class J (slow movers). Mean RMSSE weighs each of them the same
#   as a top seller. Which business question does the median answer, and which does WAPE answer?
# * LightGBM wins most series against the yardstick, but look at the volume-weighted bars: are
#   the wins in the classes that carry the sales?
#
# **Try it:** change `n_per_tier` in `gplt.pick_examples` and delete `example_ids.json` to pick a
# different set of examples.
