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
# # Day 2 · 00 (full data) — Lab: train on everything, score on the sample (10 min)
#
# The original Day 2 labs train and score on a 2,000-series sample, so they run on a laptop.
# Here we **train on every eligible series** and **score on the same 2,000**. Every number can
# then sit next to the original labs.
#
# | | |
# |---|---|
# | Training panel | every series that passes the filters (≈17,000) |
# | Scored on | the same stratified 2,000-series sample as `../code` |
# | Test window | the final **14 weeks**, from **2025-04-16** |
# | Lead score | **median** RMSSE, with the mean alongside |
#
# Goals
#
# 1. Load the full panel and see which series were dropped, and why.
# 2. Pick the example series that every later notebook plots.
# 3. Refit the Day 1 reference models on the full panel.
# 4. Learn the charts we use all day: price on a second axis, the error ECDF, error by class
#    and marginal wins.
#
# **In the chain:** reads no config, writes `00_baselines`.

# %%
import lightgbm as lgb
import pandas as pd
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
# Think of a new member of staff learning what a promotion does to sales. Put them on one aisle
# and they see a deep discount a few times a year. Walk them round the whole store and they see
# one every week. A global model is the same: more series means more examples of rare things,
# such as deep discounts, top sellers and holiday weeks.
#
# The exam stays the same, though. Every lab is marked on the same 2,000 series, so the scores
# are comparable and the notebooks stay quick.

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 1. The panel and the evaluation sample
#
# Load the training panel and the 2,000 scored ids, then count series by sales class. Class A
# holds the top sellers and class J the slowest movers.

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
# The filters are those of the original labs: no zero or negative prices, enough history for a
# 52-week yardstick, and not constant. The last row is **not applied**. It shows what the early
# research would also have dropped: series with a zero run of 26 weeks or more. Later research
# found that removing sparse series made the error 6–11× worse, so we keep them
# (`docs/gfm_learnings.md` §2, [100k set]).

# %%
gd.dropped_series()

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 2. Example series: high, medium and low volume
#
# Two series per volume tier (high: classes A–B, medium: D–F, low: I–J), picked near the
# middle of each tier. The first in each tier has a price cut in the test window, so price
# response shows. They are picked once here and reused by every notebook.
#
# **How to read this chart.** Units sit on the left axis, in the lower half. Sale price (solid
# step line) and regular price (dashed) sit on the right axis, near the top. Blue bands are
# promo weeks, and the dots in them are coloured by discount depth. The grey area is the test
# window.

# %%
examples = gplt.pick_examples(panel)
gplt.plot_series_price(panel, examples)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# Two axes on one chart can mislead: either scale can be stretched to suggest a link that is
# not there. Here is the same series as two stacked panels, which share only the dates.

# %%
gplt.plot_series_panels(panel, examples["high"][0])

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 3. Day 1 reference models on the full panel
#
# Local models (Naive, SeasonalNaive, AutoETS) fit each series on its own, so we fit them on
# the scored series only. LightGBM is global, so it trains on all series.
#
# `AutoETS(season_length=1)` has no yearly season: ETS cannot carry a 52-week season. That is
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
# * The board is sorted by **median** RMSSE. A minority of intermittent, mostly-zero series pulls
#   the mean up, so the median describes the typical item.
# * LightGBM now learns from ≈17,000 series instead of 2,000. Compare its row with the original
#   lab 00 to see what more data buys a global model: median ≈ __ (mean ≈ __).
#   <!-- TODO: fill after GPU run -->
# * SeasonalNaive is the yardstick. A model that cannot beat it on the median is not yet useful.
#
# *Reference run: full panel, GPU box. Exact numbers vary with seed and hardware; the reading
# should not.*

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 4. Where the errors are
#
# One score per model hides a lot. These charts show the same per-series errors three ways.
#
# * **Histogram** (log scale): the spread of errors, with each model's median dashed.
# * **ECDF**: for any error on the x axis, the share of series at or below it. A curve that
#   rises sooner (further left) is better. Where two curves cross, one model wins on the easy
#   series and the other on the hard ones. The dotted line marks RMSSE = 1.
# * **Box plot by class**: whether the errors sit in the top sellers or the slow movers.
#
# The heatmap after it gives the median error for each sales class and model; greener is better.

# %%
metrics = gd.score(baselines, test, train)
rmsse = gd.per_series(metrics, "rmsse")
gplt.plot_error_dist(rmsse, info)

# %%
gplt.plot_class_heatmap(rmsse, info)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ### Marginal wins over the yardstick
#
# How often does LightGBM beat SeasonalNaive, series by series? The chart has three panels.
#
# * **Share won, by class.** Blue bars count wins, so every series gets one vote. Orange bars
#   weight each series by its mean weekly units, so they ask whether the wins are where the
#   sales are. The dashed line is 50%.
# * **Per-series improvement**: (baseline − model) ÷ baseline, clipped to ±100%. Green is a win.
# * **Head to head**: one dot per series, log axes. Below the diagonal, LightGBM is better.

# %%
gplt.plot_wins(rmsse, info, model="LightGBM", baseline="SeasonalNaive")

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ### Reading the result
#
# * Look for a gap between the two bars in a class. Many wins by count but fewer by volume means
#   the model wins on small items and loses on big ones.
# * LightGBM beats SeasonalNaive on ≈ __% of series, ≈ __% by volume.
#   <!-- TODO: fill after GPU run -->
#
# *Exact numbers vary with seed and hardware; the reading should not.*

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
# * Where LightGBM wins on count, does it also win on volume? Are the wins in the classes that
#   carry the sales?
#
# **Try it:** change `n_per_tier` in `gplt.pick_examples` and delete `example_ids.json` to pick
# a different set of examples.
