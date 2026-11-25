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
# # Day 2 · 06 — Demo: uncertainty, proper scoring rules and conformal prediction (15 min)
#
# A lecture segment with a live demo. Three ways to get an 80 % / 95 % interval for the same data:
#
# 1. **Model-based quantiles**: NHITS trained with a multi-quantile loss (`MQLoss`).
# 2. **Conformal intervals** around a point model: `mlforecast` LightGBM with `PredictionIntervals`
#    (split conformal on rolling windows of the training data).
# 3. **Conformal intervals** around a local model: `statsforecast` AutoETS with `ConformalIntervals`.
#
# And two ways to judge them:
#
# * **Calibration**: does the 80 % interval contain the truth 80 % of the time?
# * **Proper scoring rules**: scaled CRPS (approximated by the average quantile loss) and the interval
#   (Winkler) score. Proper means you cannot improve your expected score by reporting something other than your
#   true belief, so narrow-but-wrong intervals are punished.

# %%
import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

import bootcamp as bc

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)
palette = ["#000000", "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#E69F00"]
plt.rcParams.update({"figure.figsize": (12, 5), "axes.prop_cycle": plt.cycler(color=palette)})

LEVELS = [80, 95]

panel = bc.load_panel()
train, test = bc.split(panel)
Y_train, Y_test = bc.to_nixtla(train), bc.to_nixtla(test)

# %% [markdown]
# ## Intuition: calibration, sharpness and honest scores
#
# ![Pinball loss and interval quality](slides/img/pinball.png)
#
# * A weather forecaster who says "30 % chance of rain" should see rain on about 30 % of those days: that is
#   **calibration**. "Between 0 and 100 mm" is always right and useless: among calibrated forecasts, prefer the
#   **sharpest**.
# * The **pinball loss** is what makes a model produce a quantile: for the 90 % quantile, being 1 unit too low
#   costs 9 times as much as being 1 unit too high, so the best answer sits high.
# * **Conformal** is "be as wrong as you were before": collect the model's misses on held-out weeks and put
#   that much around every new forecast. It works for any model, as long as the future looks like the past you
#   calibrated on.
#
# ![Conformal in two steps](slides/img/conformal.png)

# %% [markdown]
# ## 1. NHITS with a multi-quantile loss

# %%
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MQLoss
from neuralforecast.models import NHITS

nf = NeuralForecast(models=[NHITS(h=bc.H, input_size=4 * bc.H, loss=MQLoss(level=LEVELS), scaler_type="robust",
                                  max_steps=bc.MAX_STEPS, random_seed=bc.RANDOM_SEED, **bc.trainer_kwargs())],
                    freq=bc.FREQ)
nf.fit(df=Y_train)
nh = nf.predict()
nh.columns = [c.replace("NHITS", "NHITS_MQ") for c in nh.columns]
nh = nh.rename(columns={"NHITS_MQ-median": "NHITS_MQ"})

# %% [markdown]
# ## 2. LightGBM + conformal intervals

# %%
import lightgbm as lgb
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean
from mlforecast.target_transforms import LocalStandardScaler
from mlforecast.utils import PredictionIntervals

mlf = MLForecast(
    models={"LightGBM_conf": lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=63,
                                               random_state=bc.RANDOM_SEED, verbosity=-1)},
    freq=bc.FREQ, lags=[1, 2, 4, 13], lag_transforms={1: [RollingMean(4), RollingMean(13)]},
    date_features=["month", "week"], target_transforms=[LocalStandardScaler()],
)
mlf.fit(Y_train, prediction_intervals=PredictionIntervals(n_windows=3, h=bc.H))
lg = mlf.predict(h=bc.H, level=LEVELS)

# %% [markdown]
# ## 3. AutoETS + conformal intervals

# %%
from statsforecast import StatsForecast
from statsforecast.models import AutoETS
from statsforecast.utils import ConformalIntervals

sf = StatsForecast(models=[AutoETS(season_length=1, alias="ETS_conf",
                                   prediction_intervals=ConformalIntervals(n_windows=3, h=bc.H))],
                   freq=bc.FREQ, n_jobs=-1)
et = sf.forecast(df=Y_train, h=bc.H, level=LEVELS)

fc = nh.merge(lg, on=["unique_id", "ds"]).merge(et, on=["unique_id", "ds"]).merge(Y_test, on=["unique_id", "ds"])
MODELS = ["NHITS_MQ", "LightGBM_conf", "ETS_conf"]

# %% [markdown]
# ### Why 3 calibration windows per series is not enough
#
# Both conformal methods above calibrate **each series on its own**: for every series and horizon step they
# have only `n_windows = 3` past errors. The 95 % interval is then essentially "± the largest of 3 past errors",
# and a new error is larger than all 3 previous ones with probability 1/4, so coverage cannot be better than
# about 75 %, whatever the model. (Split conformal needs ⌈(n+1)(1−α)⌉ ≤ n calibration scores: 19 for 95 %.)
#
# The fix is to **pool**: divide each past error by the series' scale, put the errors of all series in one bag
# per horizon step (≈ 52,000 scores instead of 3), take the quantile, and scale it back. The assumption changes
# from "this series' past errors look like its future errors" to "scaled errors are alike across series".

# %%
cv_lgb = mlf.cross_validation(df=Y_train, n_windows=3, h=bc.H, step_size=bc.H)   # same 3 windows, refit each
sn_scale = (Y_train.sort_values(["unique_id", "ds"]).groupby("unique_id")["y"]
            .apply(lambda y: np.abs(np.diff(y.values, n=1)).mean()).replace(0, np.nan).rename("sn_scale"))
cv_lgb = cv_lgb.merge(sn_scale, left_on="unique_id", right_index=True)
cv_lgb["step"] = cv_lgb.groupby(["unique_id", "cutoff"]).cumcount()
cv_lgb["score"] = (cv_lgb["y"] - cv_lgb["LightGBM_conf"]).abs() / cv_lgb["sn_scale"]

def conformal_q(scores: pd.Series, level: int) -> float:
    s = np.sort(scores.dropna().values)
    k = int(np.ceil((len(s) + 1) * level / 100)) - 1          # finite-sample corrected rank
    return s[min(k, len(s) - 1)]

fc["step"] = fc.groupby("unique_id").cumcount()
fc = fc.merge(sn_scale, left_on="unique_id", right_index=True, how="left")
fc["LightGBM_pooled"] = fc["LightGBM_conf"]
for lvl in LEVELS:
    q = cv_lgb.groupby("step")["score"].apply(conformal_q, level=lvl)
    half = fc["step"].map(q) * fc["sn_scale"].fillna(0)
    fc[f"LightGBM_pooled-lo-{lvl}"] = (fc["LightGBM_pooled"] - half).clip(lower=0)
    fc[f"LightGBM_pooled-hi-{lvl}"] = fc["LightGBM_pooled"] + half
MODELS = ["NHITS_MQ", "LightGBM_conf", "LightGBM_pooled", "ETS_conf"]

# %% [markdown]
# ## 4. Score the intervals
#
# All scores come from `utilsforecast.losses`, the library behind `bc.score`; we only average them.
#
# * **Coverage**: share of test weeks inside the interval.
# * **Winkler (interval) score**: for an interval [l, u] at level 1 − α it is
#   `(u − l) + (2/α)(l − y)·1[y < l] + (2/α)(y − u)·1[y > u]`: width plus a penalty for every miss. We divide
#   it, like RMSSE, by each series' in-sample seasonal-naive MAE so that series are comparable.
# * **Scaled CRPS** is approximated from the quantiles we have (2.5, 10, 50, 90 and 97.5 %) by twice the average
#   quantile (pinball) loss, scaled the same way: `scaled_mqloss`, the M5-uncertainty style metric.
#   (utilsforecast also has a `scaled_crps`, but it divides by the sum of the actuals in the test window, which
#   explodes on intermittent items with a quiet 14 weeks; we avoid it.)

# %%
from utilsforecast.losses import coverage, mae, scaled_mqloss, winkler_score

for m in MODELS:                                   # no negative sales
    for lvl in LEVELS:
        fc[f"{m}-lo-{lvl}"] = fc[f"{m}-lo-{lvl}"].clip(lower=0)
train_y = Y_train[["unique_id", "ds", "y"]]
# in-sample seasonal-naive MAE per series: the RMSSE/MASE scale (0 = repeats exactly, dropped)
snaive = train_y.assign(snaive=train_y.groupby("unique_id")["y"].shift(bc.SEASON)).dropna()
scale = mae(snaive, models=["snaive"]).set_index("unique_id")["snaive"].replace(0, np.nan).rename("scale")
fc = fc.merge(scale, left_on="unique_id", right_index=True)
taus = np.array([0.025, 0.1, 0.5, 0.9, 0.975])
qcols = lambda m: [f"{m}-lo-95", f"{m}-lo-80", m, f"{m}-hi-80", f"{m}-hi-95"]   # in the order of taus

rows = []
for m in MODELS:
    r = {"model": m}
    for lvl in LEVELS:
        r[f"coverage_{lvl}"] = coverage(fc, models=[m], level=lvl)[m].mean()
        r[f"scaled_width_{lvl}"] = ((fc[f"{m}-hi-{lvl}"] - fc[f"{m}-lo-{lvl}"]) / fc.scale).median()
        w = winkler_score(fc, models=[m], level=lvl).set_index("unique_id")[m]
        r[f"scaled_interval_score_{lvl}"] = (w / scale).mean()
    r["scaled_crps"] = 2 * scaled_mqloss(fc, models={m: qcols(m)}, quantiles=taus,
                                         seasonality=bc.SEASON, train_df=train_y)[m].replace(np.inf, np.nan).mean()
    rows.append(r)
uq = pd.DataFrame(rows).set_index("model")
uq.round(3)

# %% [markdown]
# Cross-check: `bc.metric_suite` scores the same intervals with the same library calls, inside the harness every
# lab uses, plus the point metrics of each method's centre. They agree with the table above to the third
# decimal; the small differences come from the suite also clipping negative point forecasts and upper bounds at 0.

# %%
suite = bc.metric_suite(fc, Y_train, models=MODELS, baseline=None, info=bc.series_info(panel)).set_index("model")
suite[["coverage_80", "coverage_95", "scaled_winkler_80", "scaled_winkler_95", "scaled_crps",
       "mean_rmsse", "median_mase", "pooled_wape", "bias_pct"]].round(3)

# %% [markdown]
# ### Reading the result
#
# ![Coverage vs promise](slides/img/res_lab06.png)
#
# * **NHITS with a quantile loss** over-covers (≈ 87 % for a promised 80 %, ≈ 98 % for 95 %) and still has by far
#   the best scores (interval score ≈ 29 vs ≈ 37, scaled CRPS ≈ 2.7 vs ≈ 3.7): it learned **per-item, per-week**
#   widths.
# * **Per-series conformal under-covers** (≈ 66–68 % for 80 %, ≈ 73–75 % for 95 %). The main cause is arithmetic, not
#   drift: with 3 calibration windows each series has 3 scores per horizon step, and a 95 % interval from 3 scores
#   is just "the largest error seen so far". The chance a new error beats the largest of 3 is 1 in 4, so coverage
#   tops out near 75 %, which is what we got.
# * **Pooled conformal** (scale-free errors pooled across all series, then rescaled per series) fixes the
#   coverage: ≈ 79 % and ≈ 93 %. But its scores barely move: honest width is not the same as useful width. It
#   still gives every item the same *relative* width, wide for steady sellers and too narrow for spiky ones.
# * Fixes that keep the guarantee and add shape: calibrate per class, or conformalise the quantile model (CQR).
#
# *Reference run: all 17,241 eligible series, 4-core laptop CPU, 500 steps. Your numbers will differ a little; the
# reading should not.*

# %% [markdown]
# ### Calibration by velocity class
#
# Average coverage can hide a model that is too wide on fast movers and too narrow on slow movers.

# %%
info = bc.series_info(panel).select("unique_id", "sale_class").to_pandas()
cov = fc.merge(info, on="unique_id")
for m in MODELS:
    cov[m] = (cov.y >= cov[f"{m}-lo-80"].clip(lower=0)) & (cov.y <= cov[f"{m}-hi-80"])
ax = cov.groupby("sale_class")[MODELS].mean().plot(marker="o")
ax.axhline(0.8, color="black", ls=":")
ax.set_ylabel("coverage of the 80 % interval")
ax.set_title("Calibration by sales-velocity class")
plt.tight_layout()

# %%
uid = info[info.sale_class == "D"].unique_id.iloc[0]
fig, axes = plt.subplots(1, len(MODELS), figsize=(20, 4), sharey=True)
h = Y_train[(Y_train.unique_id == uid) & (Y_train.ds >= "2024-10-01")]
for ax, m in zip(axes, MODELS):
    f = fc[fc.unique_id == uid]
    ax.plot(h.ds, h.y, color="black")
    ax.plot(f.ds, f.y, color="black", ls=":")
    ax.plot(f.ds, f[m], color=palette[1])
    for lvl, alpha in zip(LEVELS, [0.35, 0.15]):
        ax.fill_between(f.ds, f[f"{m}-lo-{lvl}"].clip(lower=0), f[f"{m}-hi-{lvl}"], color=palette[1], alpha=alpha)
    ax.set_title(m)
plt.tight_layout()

# %% [markdown]
# The point forecasts go on the leaderboard too: the NHITS median and the LightGBM mean (the ETS point forecast
# is the same AutoETS that notebook 00 already recorded).

# %%
POINT = ["NHITS_MQ", "LightGBM_conf"]
bc.save_forecasts(fc[["unique_id", "ds"] + POINT], "06_uncertainty", actuals=test)
# all four interval methods with their 80 % and 95 % bounds, and the LightGBM backtest the pooled conformal used
bc.save_raw(fc[["unique_id", "ds", "y"] + [c for c in fc.columns if any(c == m or c.startswith(m + "-") for m in MODELS)]],
            "06_uncertainty", "test")
bc.save_raw(cv_lgb.drop(columns=["sn_scale", "step", "score"]), "06_uncertainty_lightgbm", "cv")
bc.record(fc[["unique_id", "ds"] + POINT], test, train, lab="06 uncertainty")
uq.to_csv(bc.RESULTS_DIR / "uncertainty_scores.csv")

# %% [markdown]
# ## Take-aways
#
# * **Coverage alone is not a score**: a 0-to-infinity interval has 100 % coverage. Use a proper score
#   (interval score, CRPS) and look at coverage as a diagnostic.
# * **Conformal** gives finite-sample marginal coverage under exchangeability, around any point model, for the
#   price of a few extra rolling fits. It does not give coverage per series or per promo week.
# * **Quantile losses** make the network learn shape (e.g. right skew on promo weeks) but have no coverage
#   guarantee; you can still conformalise them afterwards.
