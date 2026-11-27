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
# # Day 2 · 09 (full data) — Lab: intervals for the selected GFM (15 min)
#
# The original demo 06 builds intervals three ways on the 2,000-series sample. Here we build
# them for our selected model, trained on every series. Conformal intervals by horizon and
# group were an open question in the research (`docs/gfm_learnings.md` §9), so this is new work.
#
# | Method | How the interval is made |
# |---|---|
# | `GFM_quantile` | the selected config trained with a multi-quantile loss |
# | `GFM_conf_step` | split conformal around `GFM_selected`: one bag of errors per horizon week |
# | `GFM_conf_class` | the same, with one bag per sales class × horizon week |
# | `ETS_conf` | AutoETS with per-series conformal intervals, as in the original demo |
#
# Goals
#
# 1. Fit the quantile GFM, and fill the conformal bags on the **validation** window.
# 2. Score every method once on the test window: coverage, width, interval (Winkler) score and
#    an approximate CRPS.
# 3. Check calibration by sales class and by horizon week.
#
# **In the chain:** reads the selected config (`gd.selected_config()`, and its saved test
# forecasts) and `00_baselines`, writes `09_intervals`.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from neuralforecast.losses.pytorch import HuberMQLoss, MQLoss
from statsforecast import StatsForecast
from statsforecast.models import AutoETS
from statsforecast.utils import ConformalIntervals

import gd_helpers as gd
import gd_plots as gplt
import gd_price as gp

bc = gd.bc
gd.status()
gplt.use_style()

LEVELS = [80, 95]
QUANTILE_SEEDS = gd.SEEDS[:1]   # one seed: see section 2
MIN_GROUP = 40                  # smallest class × week bag; smaller bags use the week's bag

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## Intuition: honest about what we do not know
#
# ![Conformal intervals in two steps](../slides/img/conformal.png)
#
# A point forecast says "about 12 units". An 80% interval says "between 7 and 18, and we
# expect to be outside that range one week in five". **Calibration** checks that promise;
# **sharpness** prefers the narrowest interval that keeps it. Conformal prediction is "expect
# to be as wrong as you were before": collect the model's misses on weeks it did not train on,
# and put that much around every new forecast.

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 1. Data: the selected config
#
# `gp.prepare` rebuilds the selected config's training data exactly.

# %%
cfg = gd.selected_config()
baselines_cfg = gd.load_config("00_baselines")
raw = gd.train_panel()
synthetic_seed = cfg.get("synthetic_seed")
panel, train, test, data = gp.prepare(cfg, panel=raw, synthetic_seed=synthetic_seed)
info = gplt.series_table(panel)
eval_ids = gd.eval_ids()
y_shift = cfg.get("y_shift", gd.Y_SHIFT)
print(f"{data.fit_df['unique_id'].nunique():,} training series (with copies) | "
      f"{len(eval_ids):,} scored | {cfg['max_steps']:,} steps per fit")

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 2. The quantile GFM
#
# The same N-BEATSx, features and recipe as the selected model, with two changes. It learns
# the 2.5, 10, 50, 90 and 97.5% quantiles with a Huber-smoothed pinball loss. And it keeps the
# weights with the best multi-quantile validation loss.
#
# This fit uses **one seed**, an exception to the 3-seed rule (`docs/progress.md`). We judge an
# interval by how often it holds across 28,000 series-weeks, which we expect a second seed to
# move little. Its median point forecast is the 1-seed row on the board. Set
# `QUANTILE_SEEDS = gd.SEEDS` to check.

# %%
def nbeatsx_quantile(seed):
    """The selected N-BEATSx with a multi-quantile loss in place of the recipe's losses."""
    return gd.build_nbeatsx(cfg["params"], data, cfg["max_steps"], seed, "GFM_quantile",
                            loss=HuberMQLoss(level=LEVELS), valid_loss=MQLoss(level=LEVELS))


assert data.futr_exog, "the selected config has future covariates (the exogenous stack needs them)"
runs = [gd.fit_seed(nbeatsx_quantile, QUANTILE_SEEDS[0], data, "09_quantile",
                    save_as="09_quantile")]
for seed in QUANTILE_SEEDS[1:]:
    runs.append(gd.fit_seed(nbeatsx_quantile, seed, data, "09_quantile"))
gd.summaries(runs)[["seed", "fit_seconds", "oscillation", "best_frac", "neg_share"]].round(3)

# %%
gplt.plot_training(runs, warmup=gd.warmup_steps(cfg["max_steps"]),
                   title="Quantile GFM: training and validation loss")

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# `gd.predict` removes the `+1` target shift from the model columns only, and
# `bc.model_columns` leaves out the `-lo-` / `-hi-` columns. So we remove the shift from the
# interval columns here, then average over seeds.

# %%
quantile = pd.concat([r["forecast"] for r in runs])
quantile = quantile.groupby(["unique_id", "ds"], as_index=False).mean()
quantile = quantile.rename(columns={"GFM_quantile-median": "GFM_quantile"})
for col in quantile.columns:
    if "-lo-" in col or "-hi-" in col:
        quantile[col] = quantile[col] - y_shift
quantile.head()

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 3. Conformal intervals around `GFM_selected`
#
# Conformal needs errors on weeks the model never trained on. So we refit the selected config
# (3 seeds) on the weeks before `VALID_START` and forecast the validation window.
# `gp.prepare(cfg, cutoff=bc.VALID_START)` rebuilds every per-series statistic (statics,
# oversampling counts, reference price) from those earlier weeks only.

# %%
_, inner_train, valid, inner_data = gp.prepare(cfg, panel=raw, synthetic_seed=synthetic_seed,
                                               cutoff=bc.VALID_START)
print(f"calibration fit: train to {inner_train['ds'].max()}, "
      f"forecast {valid['ds'].min()} → {valid['ds'].max()}")


def nbeatsx_point(seed):
    """The selected config, as notebook 05 built it."""
    return gd.build_nbeatsx(cfg["params"], inner_data, cfg["max_steps"], seed, "GFM_selected")


calib_runs = gd.fit_seeds(nbeatsx_point, inner_data, "09_calibration",
                          seeds=cfg.get("seeds", gd.SEEDS))
calib = gd.seed_forecasts(calib_runs)[["unique_id", "ds", "GFM_selected_mean"]]
calib = calib.rename(columns={"GFM_selected_mean": "GFM_selected"})

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# Each error is divided by the series' typical week-on-week change, so errors from fast and
# slow sellers can share a bag. The error keeps its sign: demand is skewed, and a signed bag
# lets the interval reach further above the forecast than below it. The scale comes from the
# weeks before `VALID_START` and is used for both windows.

# %%
def naive_scale(hist: pl.DataFrame) -> pd.Series:
    """Mean absolute week-on-week change per series; 1 unit for a series that never moved."""
    s = (hist.sort("unique_id", "ds").group_by("unique_id")
         .agg(pl.col("y").diff().abs().mean().alias("scale"))
         .to_pandas().set_index("unique_id")["scale"])
    return s.where(s > 0, 1.0).fillna(1.0)


def with_keys(fcst: pd.DataFrame) -> pd.DataFrame:
    """Add the horizon week (0-13) and the sales class to a forecast frame."""
    out = fcst.sort_values(["unique_id", "ds"]).copy()
    out["step"] = out.groupby("unique_id").cumcount()
    out["sale_class"] = out["unique_id"].map(info["sale_class"])
    return out


scale = naive_scale(inner_train)
calib = with_keys(calib.merge(bc.to_nixtla(valid), on=["unique_id", "ds"]))
calib["score"] = (calib["y"] - calib["GFM_selected"].clip(lower=0)) / \
    calib["unique_id"].map(scale)
print(f"{len(calib):,} calibration errors from {calib['unique_id'].nunique():,} series")

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# Split conformal takes the error quantiles of each bag, with the small-sample correction:
# with `n` errors, the upper bound is the ⌈(n + 1)(1 − α/2)⌉-th smallest. The table shows the
# upper 80% bound per class and horizon week, in units of each series' typical weekly change.

# %%
def conformal_bounds(scores: pd.Series, level: int) -> tuple[float, float]:
    """Lower and upper split-conformal quantiles of signed scores at ``level`` %."""
    s = np.sort(scores.dropna().to_numpy())
    n = len(s)
    a = (1 - level / 100) / 2
    k_hi = min(int(np.ceil((n + 1) * (1 - a))) - 1, n - 1)
    k_lo = max(int(np.floor((n + 1) * a)) - 1, 0)
    return float(s[k_lo]), float(s[k_hi])


def conformal_table(calib: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """One row per bag (``by``): its size and the bounds at each level."""
    rows = []
    for key, g in calib.groupby(by):
        row = dict(zip(by, key))
        row["n"] = len(g)
        for lvl in LEVELS:
            row[f"lo_{lvl}"], row[f"hi_{lvl}"] = conformal_bounds(g["score"], lvl)
        rows.append(row)
    return pd.DataFrame(rows)


def apply_conformal(point: pd.DataFrame, table: pd.DataFrame, by: list[str],
                    alias: str) -> pd.DataFrame:
    """Intervals around ``point['GFM_selected']`` from a bag table, named like neuralforecast."""
    out = point.merge(table, on=by, how="left")
    s = out["unique_id"].map(scale)
    out[alias] = out["GFM_selected"]
    for lvl in LEVELS:
        out[f"{alias}-lo-{lvl}"] = (out[alias] + out[f"lo_{lvl}"] * s).clip(lower=0)
        out[f"{alias}-hi-{lvl}"] = (out[alias] + out[f"hi_{lvl}"] * s).clip(lower=0)
    keep = [c for c in out.columns if c.startswith(alias)]
    return out[["unique_id", "ds"] + keep]


by_step = conformal_table(calib, ["step"])
by_class = conformal_table(calib, ["sale_class", "step"])
small = by_class["n"] < MIN_GROUP                      # fall back to the week's bag
fallback = by_class[["step"]].merge(by_step, on="step", how="left")
for lvl in LEVELS:
    for side in ("lo", "hi"):
        col = f"{side}_{lvl}"
        by_class.loc[small, col] = fallback.loc[small, col].to_numpy()
print(f"{small.sum()} of {len(by_class)} class × week bags have fewer than {MIN_GROUP} errors "
      "and use the week's bag")
by_class.pivot(index="sale_class", columns="step", values="hi_80").round(2)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# Now the intervals for the test window, around the selected model's saved forecasts.

# %%
point = with_keys(gd.only(gd.selected_forecasts(cfg), eval_ids))
point["GFM_selected"] = point["GFM_selected"].clip(lower=0)
conf_step = apply_conformal(point, by_step, ["step"], "GFM_conf_step")
conf_class = apply_conformal(point, by_class, ["sale_class", "step"], "GFM_conf_class")

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 4. A local reference: AutoETS with per-series conformal intervals
#
# As in the original demo, each series is calibrated on its own three rolling windows. With
# three errors per week, a 95% interval is "the largest error seen so far", which caps its
# coverage near 75%.

# %%
def ets_conformal() -> pd.DataFrame:
    """AutoETS with 3-window conformal intervals, fitted on each eval series alone."""
    sf = StatsForecast(models=[AutoETS(season_length=1, alias="ETS_conf",
                                       prediction_intervals=ConformalIntervals(n_windows=3,
                                                                               h=bc.H))],
                       freq=bc.FREQ, n_jobs=gd.N_JOBS)
    with bc.Timer() as t:
        out = sf.forecast(df=bc.to_nixtla(gd.only(train, eval_ids)), h=bc.H, level=LEVELS)
    print(f"AutoETS + conformal: {t.seconds:.0f}s")
    return out.reset_index() if "unique_id" not in out.columns else out


ets = gd.cached("09_ets_conformal", ets_conformal)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 5. Scoring the intervals
#
# For an interval [l, u] at level 1 − α the **interval score** is
# `(u − l) + (2/α)(l − y)·1[y < l] + (2/α)(y − u)·1[y > u]`: the width plus a penalty for
# every miss. **CRPS** is approximated by the average pinball loss over the five quantiles we
# have. Both are scaled, like RMSSE, by each series' in-sample seasonal-naive error, and
# summarised by the median over series, with the mean alongside for CRPS.

# %%
def seasonal_scale(hist: pl.DataFrame) -> pd.Series:
    """In-sample mean absolute 52-week change per series (the RMSSE / MASE scale)."""
    lagged = (hist.sort("unique_id", "ds")
              .with_columns(pl.col("y").diff(bc.SEASON).over("unique_id").abs().alias("d")))
    s = lagged.group_by("unique_id").agg(pl.col("d").mean()).to_pandas().set_index("unique_id")
    return s["d"].replace(0, np.nan)


def pinball(y, q, tau):
    """Pinball (quantile) loss of quantile forecast ``q`` at level ``tau``."""
    return np.maximum(tau * (y - q), (tau - 1) * (y - q))


def interval_scores(fc: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Coverage, scaled width, scaled interval score and approximate CRPS per model."""
    rows = []
    for m in models:
        row = {"model": m}
        quantiles = {0.5: fc[m].clip(lower=0)}
        for lvl in LEVELS:
            lo = fc[f"{m}-lo-{lvl}"].clip(lower=0)
            hi = fc[f"{m}-hi-{lvl}"].clip(lower=0)
            a = 1 - lvl / 100
            miss = (2 / a) * ((lo - fc["y"]).clip(lower=0) + (fc["y"] - hi).clip(lower=0))
            iscore = ((hi - lo + miss) / fc["scale"]).groupby(fc["unique_id"]).mean()
            row[f"coverage_{lvl}"] = ((fc["y"] >= lo) & (fc["y"] <= hi)).mean()
            row[f"width_{lvl}"] = ((hi - lo) / fc["scale"]).median()
            row[f"interval_score_{lvl}"] = iscore.median()
            quantiles[a / 2] = lo
            quantiles[1 - a / 2] = hi
        loss = sum(pinball(fc["y"], q, tau) for tau, q in quantiles.items())
        crps = (2 * loss / len(quantiles) / fc["scale"]).groupby(fc["unique_id"]).mean()
        row["crps_median"] = crps.median()
        row["crps_mean"] = crps.mean()
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


INTERVAL_MODELS = ["GFM_quantile", "GFM_conf_step", "GFM_conf_class", "ETS_conf"]
fc = bc.to_nixtla(gd.only(test, eval_ids))[["unique_id", "ds", "y"]]
for frame in (quantile, conf_step, conf_class, ets):
    fc = fc.merge(frame, on=["unique_id", "ds"], how="left")
fc = with_keys(fc)
fc["scale"] = fc["unique_id"].map(seasonal_scale(train))
fc = fc.dropna(subset=["scale"])
scores = interval_scores(fc, INTERVAL_MODELS)
scores.round(3)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ### Reading the result
#
# * Coverage first: is each method close to its promise (0.80 and 0.95)? Under-coverage means
#   the intervals are too narrow to trust; over-coverage means they are wider than they need
#   to be. Both conformal methods reach ≈ 0.79 at 80%; the quantile GFM over-covers (≈ 0.85)
#   and ETS under-covers (≈ 0.68).
# * At 80% the quantile GFM has the best interval score (≈ 3.17) and CRPS (≈ 0.32 median),
#   with its intervals barely wider than conformal's. With 05c's stability settings (cleaned
#   prices, scaler floor, input clip) its intervals are narrower than 05's (width ≈ 2.44
#   against ≈ 2.72) and closer to their promise (0.85 against 0.86).
# * At 95% conformal's interval score is several times worse (≈ 36 and ≈ 68): a few large
#   misses, past the widest error it has seen, dominate.
# * Then the scores: among the methods that keep their promise, lower interval score and CRPS
#   are better. Width alone is not a score.
# * Expect `ETS_conf` to under-cover at 95%, for the reason given in section 4.
#
# *Exact numbers vary with seed and hardware.*

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 6. Calibration by sales class and by horizon week
#
# An average coverage of 80% can hide intervals that are too wide on fast sellers and too
# narrow on slow ones. Left: the share of actuals inside the 80% interval, one bar per method
# in each class. Right: the same share by weeks ahead. The dotted line at 0.8 is the promise;
# a calibrated method sits on it in every class and every week.

# %%
covered = fc[["unique_id", "sale_class", "step"]].copy()
for m in INTERVAL_MODELS:
    lo = fc[f"{m}-lo-80"].clip(lower=0)
    hi = fc[f"{m}-hi-80"].clip(lower=0)
    covered[m] = (fc["y"] >= lo) & (fc["y"] <= hi)
coverage_by_class = covered.groupby("sale_class")[INTERVAL_MODELS].mean()
coverage_by_step = covered.groupby("step")[INTERVAL_MODELS].mean()

fig, axes = plt.subplots(1, 2, figsize=(15, 4.3))
x = np.arange(len(coverage_by_class))
width = 0.8 / len(INTERVAL_MODELS)
for i, m in enumerate(INTERVAL_MODELS):
    axes[0].bar(x + i * width, coverage_by_class[m], width, label=m)
    axes[1].plot(coverage_by_step.index + 1, coverage_by_step[m], "o-", label=m)
axes[0].set_xticks(x + 0.4 - width / 2, coverage_by_class.index)
axes[0].set(ylim=(0, 1), xlabel="sales class", ylabel="share inside the 80% interval",
            title="Coverage of the 80% interval by class")
axes[1].set(ylim=(0, 1), xlabel="weeks ahead", ylabel="share inside the 80% interval",
            title="Coverage of the 80% interval by horizon week")
for ax in axes:
    ax.axhline(0.8, color="0.2", ls=":", lw=1)
axes[0].legend(ncol=2)
fig.tight_layout()
plt.show()

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ### Reading the result
#
# * By class: if the per-class bags earn their keep, `GFM_conf_class` sits closer to 0.8 than
#   the pooled week bags (`GFM_conf_step`). Watch the slow classes (H–J), where zeros make
#   coverage jumpy.
# * By week: coverage should not slide as the horizon grows. If it does, the bands are too
#   narrow far out.
# * The first two test weeks hold Good Friday and Anzac Day, and the validation window has no
#   event week (`docs/progress.md`). A dip in weeks 1–2 is the calibration window failing to
#   look like the test window.
#
# *Exact numbers vary with seed and hardware.*
#
# ## 7. Fan charts on the example series
#
# A fan chart shades the interval around the forecast. Here the bands are 80% intervals: the
# quantile GFM's, and the class × week conformal band around `GFM_selected`. The conformal band
# has one width per class and week ahead (scaled to the series), so it cannot widen for a
# promotion; the quantile GFM's can. Either way, about four actuals in five should fall inside.

# %%
bands = {"GFM_quantile": ("GFM_quantile-lo-80", "GFM_quantile-hi-80"),
         "GFM_conf_class": ("GFM_conf_class-lo-80", "GFM_conf_class-hi-80")}
gplt.plot_series_price(panel, baselines_cfg["examples"], fc,
                       models=["GFM_quantile", "GFM_conf_class"], bands=bands,
                       title="80% intervals: quantile GFM and class × week conformal")

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 8. On the board
#
# The quantile GFM's median is a point forecast too, so it goes on the board. The conformal
# methods reuse `GFM_selected`'s point forecast, which is already there.

# %%
point_fcst = quantile[["unique_id", "ds", "GFM_quantile"]]
quantile_seconds = gd.summaries(runs)["fit_seconds"].mean()
gd.save_forecasts(point_fcst, "09_intervals")
gd.record(point_fcst, test, train, lab="09 uncertainty",
          fit_seconds={"GFM_quantile": quantile_seconds},
          notes=f"median of the quantile GFM, {len(QUANTILE_SEEDS)} seed(s)")
gd.median_board("09")

# %%
gd.save_config("09_intervals", {
    "levels": LEVELS,
    "quantile_model": {"loss": "HuberMQLoss", "valid_loss": "MQLoss",
                       "seeds": QUANTILE_SEEDS, "saved_model": "09_quantile",
                       "y_shift": y_shift},
    "conformal": {"point_model": "GFM_selected",
                  "calibration": {"train_before": bc.VALID_START, "window": "valid",
                                  "seeds": cfg.get("seeds", gd.SEEDS),
                                  "n_errors": len(calib)},
                  "score": "signed error / mean absolute week-on-week change",
                  "groups": {"GFM_conf_step": ["step"],
                             "GFM_conf_class": ["sale_class", "step"]},
                  "min_group": MIN_GROUP,
                  "n_small_groups": int(small.sum())},
    "scores": scores.round(4).to_dict(orient="index"),
    "coverage_80_by_class": coverage_by_class.round(3).to_dict(),
    "best_interval_score_80": scores["interval_score_80"].idxmin(),
})

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## Discussion
#
# * The conformal guarantee is for the average over series and weeks, and it assumes the test
#   weeks look like the validation weeks. Which test weeks break that?
# * Bags by class trade size for fit: smaller bags follow each class's error shape, but each
#   bound rests on fewer errors. How many errors does a 95% bound need?
# * The quantile GFM learns a width per item and week, but has no guarantee. Would
#   conformalising it (CQR) be worth the extra step?
#
# **Try it:** set `MIN_GROUP = 10_000` so every class uses the pooled week bag, and compare
# coverage by class.
