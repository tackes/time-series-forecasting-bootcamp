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
# # Day 2 · 04 (full data) — Lab: does the model respond to price? (30 min)
#
# An accurate forecast that ignores price is no use for planning a promotion. The original lab
# 03 moves the planned price and asks whether the forecast moves the right way. The research
# found that **how** you move the price matters as much as the model (`docs/gfm_learnings.md`
# §5):
#
# | Section | What we do | Research finding |
# |---|---|---|
# | 2 | check what `regular_price` means here | not a reliable shelf price on this data |
# | 3 | an empirical benchmark from history | a two-way within regression [100k set] |
# | 4–8 | three arms, chosen on validation | deflation and oversampling helped [single store] |
# | 9 | two ways to sweep price | moving derived features too flipped the sign [100k set] |
# | 10 | LightGBM's price response | it shrinks as the shift grows: a tree artefact |
#
# The sweep is the standard one [method]: move **one** price channel (`price_index`) by
# ±5/10/20%, keep everything else as observed, and report the midpoint (arc) elasticity. We
# report on series selling at least 3 units a week in classes A–I.
#
# Goals
#
# 1. Build a benchmark elasticity from history, and learn why it is so strong on this data.
# 2. Test whether CPI deflation and discount oversampling improve the price response, choosing
#    on the validation window.
# 3. See how the sweep method, and the model type, can change the answer.
#
# **In the chain:** reads `03_features`, `02_shortlist` and `00_baselines`, writes `04_price`.

# %%
import functools

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean
from mlforecast.target_transforms import LocalStandardScaler

import gd_helpers as gd
import gd_plots as gplt
import gd_price as gp

bc = gd.bc
gd.status()
gplt.use_style()

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## Intuition: elasticity is a "what if", not a forecast
#
# Elasticity asks: if the price were 10% lower and nothing else changed, how many more units
# would we sell? We cannot observe that, so we ask the model. It is like a flight simulator:
# hold the promo flag, the calendar and the history fixed, move one lever (the price), and watch
# the forecast. A sensible model sells more when the price falls (negative elasticity). A
# **perverse** one sells more when the price rises.

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 1. Data and the settings carried forward
#
# Everything starts from `03_features`: model settings, calendar features and future
# covariates. `gp.prepare(cfg)` rebuilds the training data from any config.

# %%
cfg00 = gd.load_config("00_baselines")
cfg02 = gd.load_config("02_shortlist")
cfg03 = gd.load_config("03_features")
examples = cfg00["examples"]
noise_floor = cfg02["noise_floor"]

raw = gd.train_panel()
params = cfg03["params"]
futr_exog = cfg03["futr_exog"]
price_channel = "price_index"
print(f"future covariates: {futr_exog}\nnoise floor from 02: {noise_floor:.4f}")
pd.Series(cfg03["features"])

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 2. What does `regular_price` mean here?
#
# Oversampling needs a rule for "a real discount week". The obvious rule is "sale price at least
# 10% below the regular price". Before trusting it, check how often items sell *above* their
# regular price, and how well each rule lines up with the promo flag. These numbers come from
# the data alone, so they do not depend on any model run.

# %%
# raw (undeflated) prices, no oversampling
panel_a, train_a, test_a, _ = gp.prepare({"futr_exog": bc.FUTR_EXOG_COLS}, panel=raw)
info = gplt.series_table(panel_a)

price_check = train_a.select(
    (pl.col("sale_price") > pl.col("regular_price")).mean().alias("share above regular"),
    (pl.col("sale_price") < 0.9 * pl.col("regular_price")).mean()
    .alias("share < 0.9 × regular"),
    (pl.col("discount_depth") >= 0.10).mean().alias("share discount_depth ≥ 0.10"),
).to_pandas().T.rename(columns={0: "share of training weeks"})
price_check.round(3)

# %%
rule_rows = []
for rule in ("regular", "rolling"):
    flagged = gp.add_discount_flag(train_a, rule=rule).select(
        pl.col("is_discount").cast(pl.Float64), (pl.col("promo") > 0).cast(pl.Float64))
    rule_rows.append({
        "rule": rule,
        "share of weeks flagged": flagged["is_discount"].mean(),
        "share of promo weeks flagged": flagged.filter(pl.col("promo") > 0)["is_discount"].mean(),
        "correlation with promo flag": np.corrcoef(flagged["is_discount"], flagged["promo"])[0, 1],
    })
pd.DataFrame(rule_rows).set_index("rule").round(3)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ### Reading the result
#
# * About 30% of training weeks sell above the "regular" price, so it is not a ceiling. It
#   behaves more like an occasional list price that the shelf price drifts above.
# * The "below regular" rule lines up well with the promo flag, but it flags many more weeks.
#   An item that sits below a stale regular price for months counts as "on discount" the whole
#   time, though its price never moved.
# * The **rolling** rule compares each week with the item's own 4-week average price
#   (`discount_depth ≥ 0.10`, ≈ 8.5% of weeks). It needs no reference price and flags only weeks
#   where the price actually **fell**. That is the variation oversampling should show the
#   network more of.
# * This differs from the research's large set, where the promo flag barely tracked real
#   discounts (r ≈ −0.03, §3, [100k set]). We oversample with the rolling rule; the **Try it**
#   at the end swaps them.

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 3. The empirical benchmark
#
# A two-way within regression of log(units + 1) on log price, on the series we report on.
# Removing each series' average and each week's average leaves only price moves *within* an
# item, net of what every item did that week.

# %%
report_ids = gp.report_ids(train_a)
empirical = gp.empirical_elasticity(train_a, report_ids)
band = gp.empirical_band(empirical)
empirical_all = float(empirical.loc[empirical["sale_class"] == "all", "elasticity"].iloc[0])
print(f"{len(report_ids):,} report series | empirical (all) {empirical_all:.2f} | "
      f"band (IQR of classes A-I) {band[0]:.2f} to {band[1]:.2f} | "
      f"research band {gp.RESEARCH_BAND}")
empirical.round(2)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ### Reading the result
#
# On the research's large multi-store set this regression gave −0.4 to −0.7 [100k set]. Here
# it is far stronger, ≈ −3.6 (data only, no model involved). Why?
#
# * Price variation here is mostly **temporary promotional cuts**, not lasting price changes.
#   A one-week 20% cut with a display and a flyer can double sales for that week.
# * Shoppers **pull purchases forward** into the promo week and buy less afterwards. A
#   regression on weekly data counts the spike but not the dip spread over the following weeks.
# * The promo flag and the price cut arrive together, so the regression credits price with part
#   of the promo's effect.
# * The data itself is **cleaner** than the later research sets: the promo flag lines up with
#   real price cuts, and prices are recorded consistently. A clear, well-measured signal gives
#   a strong estimate rather than one shrunk towards zero by noise.
#
# So the band describes "a short promotion", not "a permanent price change". The charts shade
# it as a reference, not a target. The research constant is printed for comparison only.

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 4. Three arms
#
# Each arm adds one change to the one before, as the research did (§1, [single store]).
#
# * **A**: the 03 features, with CPI deflation switched off (so B can switch it on).
# * **B**: A + CPI deflation. Prices are in base-week money, so inflation is not a price rise.
# * **C**: B + oversampling: extra copies of series with real discount weeks (rolling rule, ×5,
#   capped at 15 copies), counted on training weeks only. Copies are dropped before scoring.
#
# All arms use the same step budget, so they differ only in their data. Each arm is built
# twice: on the weeks before `VALID_START`, to choose, and on every training week, to report.

# %%
OVERSAMPLE = {"rule": "rolling", "cap": gp.COPY_CAP, "multiplier": gp.COPY_MULTIPLIER}
base_features = {**cfg03["features"], "deflate": False}
arm_cfgs = {
    "A_features": {**cfg03, "features": base_features, "oversample": None},
    "B_cpi": {**cfg03, "features": {**base_features, "deflate": True}, "oversample": None},
    "C_cpi_over": {**cfg03, "features": {**base_features, "deflate": True},
                   "oversample": OVERSAMPLE},
}
arm_data = {}
valid_data = {}
for arm, cfg in arm_cfgs.items():
    arm_data[arm] = gp.prepare(cfg, panel=raw)[3]
    _, inner_train, valid, valid_data[arm] = gp.prepare(cfg, panel=raw, cutoff=bc.VALID_START)
    n_fit = arm_data[arm].fit_df["unique_id"].nunique()
    n_valid = valid_data[arm].fit_df["unique_id"].nunique()
    print(f"{arm:12s} {n_fit:>7,} training series (incl. copies), {n_valid:>7,} before "
          f"VALID_START")

valid_report_ids = gp.report_ids(inner_train)
max_steps = cfg03.get("max_steps") or gd.scaled_max_steps(raw["unique_id"].n_unique())
example_ids = gplt.example_ids(examples)
print(f"{max_steps:,} steps per fit | seeds {gd.SEEDS} | "
      f"{len(valid_report_ids):,} report series on validation")


# %%
def builder(arm: str, data: gd.NixtlaData):
    """Model factory for one arm on one window (the alias names the forecast column)."""
    def build(seed):
        return gd.build_nbeatsx(params, data, max_steps, seed, f"NBEATSx_{arm}")
    return build


# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 5. How we will choose an arm (fixed before looking)
#
# The rule is written down before any arm is fitted, and it reads the **validation** window
# only:
#
# 1. keep arms whose median elasticity at ±20% (seed-mean sweep) is **negative**;
# 2. and whose median validation RMSSE is within the 02 **noise floor** of arm A;
# 3. of those, choose the one with the **lowest share of perverse** series. If none qualify, A.
#
# The test window is scored once, in section 7, for the report. Notebook 05 keeps the chosen
# arm's data and tunes the model on top of it.

# %%
SELECTION_RULE = ("on the validation window: lowest share perverse among arms with median e20 "
                  "< 0 and median RMSSE <= arm A + 02 noise floor; else A")

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 6. Choosing on the validation window
#
# Each arm is fitted with 3 seeds on the weeks before `VALID_START` and scored on the 14 weeks
# after it. While each model is in memory we sweep the price on the report series of that
# window. The table shows why each arm did or did not qualify.

# %%
def valid_extras(nf, data):
    """The single-channel sweep on the validation window, kept with each tuning run."""
    return {"sweep": gp.price_sweep(nf, data, valid_report_ids, channel=price_channel)}


valid_runs = {}
for arm in arm_cfgs:
    with bc.Timer() as t:
        valid_runs[arm] = gd.fit_seeds(builder(arm, valid_data[arm]), valid_data[arm],
                                       f"04_valid_{arm}", extras=valid_extras)
    if gd.TRAIN:
        print(f"{arm} (validation): {t.seconds / 60:.1f} min for {len(gd.SEEDS)} seeds")

valid_median = {}
valid_reports = {}
for arm, runs in valid_runs.items():
    rmsse = gd.per_series(gd.score(gd.seed_forecasts(runs), valid, inner_train))
    valid_median[arm] = float(rmsse[f"NBEATSx_{arm}_mean"].median())
    valid_reports[arm] = gp.elasticity_report(gp.arc_elasticity(gp.seed_mean_sweep(runs)))

candidates = []
for arm in arm_cfgs:
    negative = valid_reports[arm]["median_e20"] < 0
    accurate = valid_median[arm] <= valid_median["A_features"] + noise_floor
    if negative and accurate:
        candidates.append(arm)
if candidates:
    selected_arm = min(candidates, key=lambda a: valid_reports[a]["share_perverse"])
else:
    selected_arm = "A_features"

choice_rows = []
for arm in arm_cfgs:
    choice_rows.append({"arm": arm, "median valid RMSSE": valid_median[arm],
                        "median valid e20": valid_reports[arm]["median_e20"],
                        "share perverse (valid)": valid_reports[arm]["share_perverse"],
                        "qualifies": arm in candidates})
choice = pd.DataFrame(choice_rows).set_index("arm")
print(f"selected arm: {selected_arm}")
choice.round(4)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 7. Refit on every training week, score the test window once
#
# The same three arms, refitted with the same seeds on all training weeks. These fits are the
# reported result: nothing below changes the choice. After each fit we sweep the price on a
# single channel. For arm A we also sweep the **compound** way of the original lab 03, which
# recomputes `discount_depth` from the moved price. Each run also probes sustained and one-week
# price shocks, and keeps forecasts at −20% / base / +20% for the example series.

# %%
def arm_extras(compound: bool):
    """What each fit computes before it is discarded: sweeps, probes and example scenarios."""
    def extras(nf, data):
        out = {"sweep": gp.price_sweep(nf, data, report_ids, channel=price_channel),
               "probe": gp.probe(nf, data, report_ids, channel=price_channel),
               "scenarios": gp.scenario_forecasts(nf, data, example_ids, channel=price_channel)}
        if compound:
            out["sweep_compound"] = gp.price_sweep(nf, data, report_ids, channel=price_channel,
                                                   compound=True)
        return out
    return extras


arms = {}
for arm in arm_cfgs:
    with bc.Timer() as t:
        arms[arm] = gd.fit_seeds(builder(arm, arm_data[arm]), arm_data[arm], f"04_{arm}",
                                 extras=arm_extras(compound=(arm == "A_features")))
    if gd.TRAIN:
        print(f"{arm}: {t.seconds / 60:.1f} min for {len(gd.SEEDS)} seeds")

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 8. Accuracy of each arm
#
# The seed-mean forecast of each arm goes on the board, then the training-health table from
# notebook 01. Arm C's `n_train_series` includes its copies.

# %%
forecasts = {arm: gd.seed_forecasts(runs) for arm, runs in arms.items()}
board_fcst = forecasts["A_features"][["unique_id", "ds"]]
seed_scores = {}
test_median = {}
for arm, runs in arms.items():
    rmsse = gd.per_series(gd.score(forecasts[arm], test_a, train_a))
    seed_scores[arm] = gd.seed_medians(runs, test_a, train_a)
    test_median[arm] = float(rmsse[f"NBEATSx_{arm}_mean"].median())
    board_fcst = board_fcst.merge(forecasts[arm][["unique_id", "ds", f"NBEATSx_{arm}_mean"]],
                                  on=["unique_id", "ds"])

fit_seconds = {}
for arm, runs in arms.items():
    fit_seconds[f"NBEATSx_{arm}_mean"] = gd.summaries(runs)["fit_seconds"].mean()
gd.save_forecasts(board_fcst, "04_price")
gd.record(board_fcst, test_a, train_a, lab=f"04 price arms ({len(gd.SEEDS)}-seed mean)",
          fit_seconds=fit_seconds, notes="fit_seconds is per seed")
gd.median_board("04")

# %%
health_cols = ["arm", "seed", "fit_seconds", "n_train_series", "oscillation", "best_frac",
               "neg_share"]
health = [gd.summaries(runs).assign(arm=arm) for arm, runs in arms.items()]
pd.concat(health)[health_cols].round(3)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 9. Price response: single channel against compound
#
# Every sweep is averaged over seeds before the elasticity is taken. Each row is one report
# (§5, [method]):
#
# * `median_e20`: median arc elasticity at ±20%, the headline;
# * `share_perverse`: share of series with e > 0; `share_flat`: share with |e| < 0.05;
# * `n_estimable`: series whose forecast moved at all;
# * `invariance`: median at ±5% ÷ median at ±20%. Near 1 means the response grows in
#   proportion to the price move.

# %%
elasticity = {"A_features (compound)": gp.arc_elasticity(
    gp.seed_mean_sweep(arms["A_features"], "sweep_compound"))}
for arm, runs in arms.items():
    elasticity[arm] = gp.arc_elasticity(gp.seed_mean_sweep(runs))
reports = {name: gp.elasticity_report(el) for name, el in elasticity.items()}
pd.DataFrame(reports).T.round(3)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# Did the choice carry over? Each arm's validation numbers, which made the choice, next to its
# test numbers, which only report it.

# %%
transfer_rows = []
for arm in arm_cfgs:
    transfer_rows.append({"arm": arm, "valid RMSSE": valid_median[arm],
                          "test RMSSE": test_median[arm],
                          "valid e20": valid_reports[arm]["median_e20"],
                          "test e20": reports[arm]["median_e20"],
                          "valid perverse": valid_reports[arm]["share_perverse"],
                          "test perverse": reports[arm]["share_perverse"],
                          "selected": arm == selected_arm})
pd.DataFrame(transfer_rows).set_index("arm").round(4)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ### Reading the result
#
# * Compare the compound row with `A_features`: the **same fitted model**, swept two ways. The
#   compound sweep also raises `discount_depth` when it cuts the price, so the model sees "a
#   deeper discount" and "a lower price" at once. The two inputs move together in training, so
#   the network may have split the effect between them in any proportion. Moving both reads off
#   a mixture that can even have the wrong sign. In the research it gave +0.45 (§5, [100k set]).
# * Median e20: compound ≈ −0.94, single channel ≈ −0.60; B ≈ −1.13, C ≈ −1.25.
#
# *Reference run: full panel, GPU box, 3 seeds. Exact numbers vary with seed and hardware; the
# reading should not.*

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 10. LightGBM's response to price
#
# The reference LightGBM from notebook 00, with `price_index` in place of the raw sale price,
# trained on the full panel. We move `price_index` alone, over a wider range of shifts, and
# check how much of the tree's split gain the price and promo features carry.

# %%
LGBM_EXOG = ["promo", "price_index", "discount_depth", "price_ratio_to_rolling_mean"]
LGBM_SHIFTS = (0.02, 0.05, 0.10, 0.20, 0.30)


@functools.cache
def fit_lgbm() -> MLForecast:
    """LightGBM as in notebook 00 on every training series, with ``price_index``.

    Cached in memory, so the sweep and the gain table share one fit when both are computed.
    """
    mlf = MLForecast(
        models={"LightGBM": lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05,
                                              num_leaves=63, random_state=bc.RANDOM_SEED,
                                              verbosity=-1)},
        freq=bc.FREQ, lags=[1, 2, 4, 13, 52],
        lag_transforms={1: [RollingMean(4), RollingMean(13)]},
        date_features=["month", "week"], target_transforms=[LocalStandardScaler()],
    )
    with bc.Timer() as t:
        mlf.fit(bc.to_nixtla(train_a, LGBM_EXOG), static_features=[])
    print(f"LightGBM fit: {t.seconds / 60:.1f} min")
    return mlf


def lgbm_sweep() -> pd.DataFrame:
    """Total forecast units per report series at base price and at ±each shift."""
    mlf = fit_lgbm()
    future = gd.only(bc.to_nixtla(test_a, LGBM_EXOG).drop(columns="y"), report_ids)
    shifts = [0.0]
    for m in LGBM_SHIFTS:
        shifts += [-m, m]
    rows = []
    for s in shifts:
        fcst = mlf.predict(h=bc.H, X_df=gp.shift_prices(future, s, price_channel),
                           ids=report_ids)
        total = fcst.groupby("unique_id")["LightGBM"].sum().rename("units").reset_index()
        rows.append(total.assign(shift=s))
    return pd.concat(rows, ignore_index=True)


def lgbm_gain() -> pd.DataFrame:
    """Share of LightGBM's total split gain per feature."""
    booster = fit_lgbm().models_["LightGBM"].booster_
    gain = pd.DataFrame({"feature": booster.feature_name(),
                         "gain": booster.feature_importance("gain")})
    gain["share"] = gain["gain"] / gain["gain"].sum()
    return gain.sort_values("share", ascending=False).reset_index(drop=True)


lgbm_el_all = gp.arc_elasticity(gd.cached("04_lgbm_sweep", lgbm_sweep))
gain = gd.cached("04_lgbm_gain", lgbm_gain)
lgbm_el = lgbm_el_all[lgbm_el_all["shift"].round(4).isin(gp.SHIFTS)]
elasticity["LightGBM"] = lgbm_el
reports["LightGBM"] = gp.elasticity_report(lgbm_el)
price_gain = gain.loc[gain["feature"].isin(LGBM_EXOG), "share"].sum()
print(f"price and promo features: {price_gain:.1%} of LightGBM's split gain")
gain.head(8).round(3)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# The next chart has one line per model: the median arc elasticity at each shift size, with the
# empirical band shaded. A smooth response gives a flat line. A line that drifts towards zero as
# the shift grows is responding less than in proportion.

# %%
shrink = pd.DataFrame({"LightGBM": lgbm_el_all.groupby("shift")["e_arc"].median()})
for arm in arms:
    shrink[f"NBEATSx {arm}"] = elasticity[arm].groupby("shift")["e_arc"].median()
fig, ax = plt.subplots(figsize=(8, 4))
for i, col in enumerate(shrink.columns):
    s = shrink[col].dropna()
    ax.plot(s.index * 100, s.values, "o-", color=gplt.MODEL_COLOURS[i], label=col)
ax.axhspan(*band, color=gplt.COLOURS["band"], alpha=0.15, label="empirical band")
ax.axhline(0, color="0.3", lw=0.8)
ax.set(xlabel="price shift, ±%", ylabel="median arc elasticity",
       title="Median elasticity against shift size (a smooth response is flat)")
ax.legend()
fig.tight_layout()
plt.show()
shrink.round(3)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ### Reading the result
#
# * A tree predicts a constant between split points, and nothing changes beyond the last split
#   it has seen. Once a price move passes that split, a bigger move adds nothing, so dividing
#   by the size of the move makes the "elasticity" shrink as the shift grows. Small moves can
#   also stay inside one leaf and read as zero. The shape comes from the splits, not shoppers.
# * In the research, price features carried ≈ 1.5% of LightGBM's split gain, against over 60%
#   for a 13-week rolling mean (§5). Here: ≈ 38%.
# * So: use LightGBM for accuracy and the network for price response.
#
# *Exact numbers vary with seed and hardware; the reading should not.*

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 11. What the selected arm does on the example series
#
# The chart from notebook 00, now with three forecasts: at the planned price (base), and with
# every test-week price 20% lower and 20% higher. The −20% line should sit above base and the
# +20% line below it. If they cross or swap, that series' response is perverse.

# %%
scenarios = pd.concat([run["scenarios"] for run in arms[selected_arm]])
scenarios = scenarios.groupby(["unique_id", "ds"], as_index=False).mean()
gplt.plot_series_price(panel_a, examples, scenarios,
                       models=["-20% price", "base", "+20% price"],
                       title=f"Arm {selected_arm}: forecasts at -20% / base / +20% price")

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 12. The elasticity charts
#
# Arm A first, then the selected arm, both from the single-channel sweep. Each chart has three
# panels, and the shaded band on each is the empirical inter-quartile range across classes from
# section 3.
#
# * **Histogram** of per-series arc elasticity at ±20%, clipped to [−5, 2]. Dashed: the median
#   of all series. Dotted: the median of responsive series only (|e| > 0.05).
# * **Box plot by sales class**, A–I. The red line is zero.
# * **Median against shift size**, for the arc, down-only and up-only elasticity. Flat lines mean
#   the response is in proportion to the move. The research found price rises to be
#   extrapolation, because few sales happened at higher prices (§5), so trust the down line more.

# %%
gplt.plot_elasticity(elasticity["A_features"], info, band=band,
                     title="Arm A, single-channel sweep")

# %%
if selected_arm != "A_features":
    gplt.plot_elasticity(elasticity[selected_arm], info, band=band,
                         title=f"Arm {selected_arm} (selected), single-channel sweep")

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# The **strength chart** sorts each series into a category by its elasticity at ±20%, and
# shows one bar per arm, split by the share of series in each category:
#
# | Category | Elasticity e |
# |---|---|
# | Highly elastic | e ≤ −3 |
# | Elastic | −3 < e ≤ −1 |
# | Inelastic | −1 < e ≤ −0.05 |
# | Flat | −0.05 < e ≤ 0 |
# | Perverse | e > 0 |
#
# The **by-class chart** after it gives each arm's median elasticity per sales class, with the
# empirical value per class dashed in black and zero dotted.

# %%
strength = {name: gp.e20(el) for name, el in elasticity.items()}
gplt.plot_elasticity_strength(strength, title="Strength of price response at ±20%")

# %%
gplt.plot_elasticity_by_class(strength, info, empirical=empirical)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ### Reading the result
#
# * Read the strength chart from the right. The orange "perverse" share is what we most want to
#   shrink. Does CPI deflation (B) or oversampling (C) shrink it? In the research both did
#   (§1, [single store]). Perverse share: A ≈ 1.0%, B ≈ 0.6%, C ≈ 0.1%.
# * Expect the empirical line to sit well below the models. The regression credits price with
#   part of the promo's lift; the models see the promo flag as a separate input, held fixed in
#   the sweep.
#
# *Exact numbers vary with seed and hardware; the reading should not.*

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 13. Probes: a sustained cut against a one-week pulse
#
# The oversampled arm C, all seeds pooled. Price shocks of −10%, −20%, −30% and +20%
# (§5, [method]). A **sustained** shock lasts the whole horizon; a **pulse** lasts one week.
#
# * Left: median lift in units by shock size, one line per mode. A cut should lift sales.
# * Right: for the pulse, the median change in the weeks that were not shocked. Ideally ≈ 0.
#
# The research's pass bar was a median elasticity of −0.3 or stronger, with the right sign. On
# this data, with an empirical value of ≈ −3.6, that bar is easy to clear.
#
# These arm probes move `price_index` alone, as the sweep does: a cut arrives with no discount
# depth. They were computed while each arm's model was in memory, and the arms were not saved,
# so they stay as they are. From 05b on, the probes rebuild every price input (next section).

# %%
probe_c = pd.concat([run["probe"] for run in arms["C_cpi_over"]], ignore_index=True)
probe_table = gp.probe_summary(probe_c)
gplt.plot_probe(probe_table, title="Arm C_cpi_over: probe responses (price_index alone)")
probe_table.round(3)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ### LightGBM under the phase 7b probes
#
# LightGBM is refitted in a few minutes, so we can probe it the way 05b, 07 and 10 probe every
# model: the price moves and every input built from it is rebuilt (`discount_depth` and
# `price_ratio_to_rolling_mean`, from the 4-week rolling mean), in fixed probes and in random
# promotion calendars (three one-week cuts per series, 10–30% deep, in ordinary weeks). The
# results go to 10's final price-response table.

# %%
lgbm_hist = gd.only(bc.to_nixtla(train_a, LGBM_EXOG), report_ids)
lgbm_futr = gd.only(bc.to_nixtla(test_a, LGBM_EXOG).drop(columns="y"), report_ids)
lgbm_calendars = gp.promo_calendars(lgbm_futr, report_ids)
print("price inputs rebuilt, largest gap from the data:",
      gp.rebuild_check(lgbm_hist, lgbm_futr))


def lgbm_predict(futr: pd.DataFrame) -> pd.DataFrame:
    """``predict(futr)`` for the LightGBM above, for ``gp.price_probes``."""
    ids = sorted(futr["unique_id"].unique())
    fcst = fit_lgbm().predict(h=bc.H, X_df=futr[["unique_id", "ds"] + LGBM_EXOG], ids=ids)
    return fcst[["unique_id", "ds", "LightGBM"]].rename(columns={"LightGBM": "yhat"})


def lgbm_probes() -> pd.DataFrame:
    return gp.price_probes(lgbm_predict, lgbm_hist, lgbm_futr, report_ids)


def lgbm_calendar() -> pd.DataFrame:
    return gp.calendar_probe(lgbm_predict, lgbm_hist, lgbm_futr, lgbm_calendars)


lgbm_probe_table = gp.probe_report(gd.cached("04_lgbm_probe", lgbm_probes))
lgbm_calendar_table = gp.calendar_report(gd.cached("04_lgbm_calendar", lgbm_calendar))
gplt.plot_probe(lgbm_probe_table, title="LightGBM: probe responses, every price input rebuilt")
lgbm_calendar_table.round(3)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ### Reading the result
#
# * A tree reads `discount_depth` and the rolling ratio through its splits, so rebuilding them
#   can move LightGBM more than moving `price_index` alone did.
# * In the calendar table, look for lift growing with depth. A tree can give the same lift for
#   15% and 25% if both land in one leaf.
#
# *Exact numbers vary with seed and hardware.*

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 14. The accuracy cost of oversampling, and seeds against arms
#
# First the marginal wins of arm C over arm A: does oversampling cost accuracy, and where? Then
# the arm comparison chart, with an extra panel: each seed's median elasticity, empirical band
# shaded.

# %%
both = forecasts["A_features"].merge(forecasts["C_cpi_over"], on=["unique_id", "ds"])
rmsse = gd.per_series(gd.score(both, test_a, train_a))
gplt.plot_wins(rmsse, info, model="NBEATSx_C_cpi_over_mean", baseline="NBEATSx_A_features_mean")

# %%
seed_e20 = {}
for arm, runs in arms.items():
    seed_e20[arm] = []
    for run in runs:
        seed_e20[arm].append(float(gp.e20(run["sweep"]).median()))
gplt.plot_arm_comparison(arms, metrics=seed_scores, elasticity=seed_e20, band=band)
pd.DataFrame(seed_e20, index=[f"seed {s}" for s in gd.SEEDS]).round(3)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ### Reading the result
#
# * In the research each run's response was all one sign, and the seeds disagreed in size
#   (§6, [100k set]). Three seeds are the minimum for any number quoted here. Look at the spread
#   before you read a difference between arms.
# * If the seed boxes for two arms overlap, the arms are not separable on elasticity.
#
# *Exact numbers vary with seed and hardware; the reading should not.*

# %%
selected_cfg = arm_cfgs[selected_arm]
gd.save_config("04_price", {
    "params": params,
    "features": selected_cfg["features"],
    "futr_exog": futr_exog,
    "y_shift": cfg03.get("y_shift", gd.Y_SHIFT),
    "oversample": selected_cfg["oversample"],
    "price_channel": price_channel,
    "arms": {arm: {"features": c["features"], "oversample": c["oversample"]}
             for arm, c in arm_cfgs.items()},
    "elasticity": reports,
    "elasticity_valid": valid_reports,
    "lgbm_probes": lgbm_probe_table.round(4).to_dict("records"),
    "lgbm_calendar": lgbm_calendar_table.round(4).to_dict("records"),
    "empirical": empirical.to_dict("records"),
    "empirical_all": empirical_all,
    "empirical_band": list(band),
    "n_report_series": len(report_ids),
    "regular_price_check": price_check["share of training weeks"].round(4).to_dict(),
    "median_rmsse": test_median,
    "median_rmsse_valid": valid_median,
    "selection_rule": SELECTION_RULE,
    "selected_arm": selected_arm,
    "max_steps": max_steps,
    "seeds": gd.SEEDS,
})

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## Discussion
#
# * The compound and single-channel rows come from the same model. Which would you show a
#   category manager, and why?
# * The empirical benchmark here is ≈ −3.6; the research's was about −0.5. What does that say
#   about using one "plausible range" across retailers?
# * LightGBM is the most accurate model on the board. Why is it still not the model to plan
#   prices with?
#
# **Try it:** set `OVERSAMPLE["rule"] = "regular"` and rerun arm C (delete its `runs/04_C_*`
# and `runs/04_valid_C_*` folders first). How many more series get copies, and does the
# perverse share move?
