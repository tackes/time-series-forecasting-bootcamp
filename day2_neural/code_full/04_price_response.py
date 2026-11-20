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
# An accurate forecast that ignores price is no use for planning a promotion. The original lab 03
# moves the planned price and asks whether the forecast moves the right way. Our research found
# that **how** you move the price matters as much as the model:
#
# | Step | What we do | Research finding (`docs/gfm_learnings.md` §5) |
# |---|---|---|
# | 0 | check what `regular_price` means here | it is not a reliable "shelf price" on this data |
# | 1 | an empirical benchmark from history | a two-way within regression, per class |
# | 2 | three training arms, 3 seeds each | CPI deflation and discount oversampling helped |
# | 2 | two ways to sweep price | moving derived features with price can flip the sign |
# | 3 | LightGBM's price response | it shrinks as the shift grows: a tree-split artefact |
#
# The sweep is the standard one: move **one** price channel (`price_index`) by ±5/10/20%, keep
# everything else as observed, and report the midpoint (arc) elasticity on series selling at
# least 3 units a week in classes A–I.

# %%
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
# would we sell? We cannot observe that, so we ask the model. Hold the promo flag, the calendar
# and the history fixed, move the price, and watch the forecast. A sensible model sells more when
# the price falls (negative elasticity). A **perverse** one sells more when the price rises.

# %% [markdown]
# ## 1. Data and the settings carried forward
#
# Everything starts from `03_features`: model settings, calendar features and the future
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
# regular price, and how well each rule lines up with the promo flag.

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
# * About 30% of training weeks sell above the "regular" price, so it is not a ceiling. It behaves
#   more like an occasional list price that the shelf price drifts above.
# * The "below regular" rule lines up well with the promo flag, but it flags a quarter of all
#   weeks: an item that sits below a stale regular price for months counts as "on discount" the
#   whole time, though its price never moved.
# * The **rolling** rule compares each week with the item's own 4-week average price
#   (`discount_depth ≥ 0.10`). It needs no reference price and flags only weeks where the price
#   actually **fell**, which is the variation oversampling is meant to show the network more of.
# * We oversample with the rolling rule. Both rules are shown so you can see the choice; the
#   **Try it** at the end swaps them.

# %% [markdown]
# ## 3. The empirical benchmark
#
# A two-way within regression of log(units + 1) on log price, on the series we report on.
# Removing each series' average and each week's average leaves only price moves *within* an item,
# net of what every item did that week.

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
# On the research's large multi-store set this regression gave −0.4 to −0.7. Here it is far
# stronger, around −3.6. Why?
#
# * Price variation here is mostly **temporary promotional cuts**, not lasting price changes.
#   A one-week 20% cut with a display and a flyer can double sales for that week.
# * Shoppers **pull purchases forward** into the promo week and buy less afterwards. A regression
#   on weekly data counts the spike but not the dip spread over the following weeks.
# * The promo flag and the price cut arrive together, so the regression credits price with part
#   of the promo's effect.
#
# So the band below describes "a short promotion", not "a permanent price change". We shade it on
# the charts as a reference, not a target.

# %% [markdown]
# ## 4. Three arms, three seeds each
#
# * **A**: the 03 features, with CPI deflation switched off (so B can switch it on).
# * **B**: A + CPI deflation: prices in base-week money, so inflation is not a price rise.
# * **C**: B + oversampling: extra copies of series with real discount weeks (rolling rule, ×5,
#   capped at 15 copies), counted on training weeks only. Copies are dropped before scoring.
#
# All arms use the same step budget, so they differ only in their data. After each fit we sweep
# the price (single channel), and for arm A also the **compound** way of lab 03, which recomputes
# `discount_depth` from the moved price. Each run also probes sustained and one-week price shocks
# and keeps forecasts at −20% / base / +20% for the example series.

# %%
OVERSAMPLE = {"rule": "rolling", "cap": 15, "multiplier": 5}
base_features = {**cfg03["features"], "deflate": False}
arm_cfgs = {
    "A_features": {**cfg03, "features": base_features, "oversample": None},
    "B_cpi": {**cfg03, "features": {**base_features, "deflate": True}, "oversample": None},
    "C_cpi_over": {**cfg03, "features": {**base_features, "deflate": True},
                   "oversample": OVERSAMPLE},
}
arm_data = {}
for arm, cfg in arm_cfgs.items():
    arm_data[arm] = gp.prepare(cfg, panel=raw)[3]
    n_fit = arm_data[arm].fit_df["unique_id"].nunique()
    print(f"{arm:12s} {n_fit:>7,} training series (including copies)")

max_steps = cfg03.get("max_steps") or gd.scaled_max_steps(raw["unique_id"].n_unique())
example_ids = []
for tier_ids in examples.values():
    example_ids += tier_ids
print(f"{max_steps:,} steps per fit | seeds {gd.SEEDS}")


# %%
def subset(data: gd.NixtlaData, ids: list[str]) -> gd.NixtlaData:
    """The same data restricted to ``ids`` (for quick what-if predictions)."""
    return gd.NixtlaData(gd.only(data.fit_df, ids), gd.only(data.predict_df, ids),
                         gd.only(data.futr_df, ids), data.static_df, data.futr_exog,
                         data.stat_exog, data.y_shift)


def scenario_forecasts(nf, data: gd.NixtlaData, ids: list[str], shift: float = 0.20):
    """Forecasts for ``ids`` at base price and with the price ``shift`` lower and higher."""
    sub = subset(data, ids)
    out = None
    for label, s in (("-20% price", -shift), ("base", 0.0), ("+20% price", shift)):
        fcst = gd.predict(nf, sub, futr_df=gp.shift_prices(sub.futr_df, s, price_channel))
        col = bc.model_columns(fcst)[0]
        fcst = fcst[["unique_id", "ds", col]].rename(columns={col: label})
        out = fcst if out is None else out.merge(fcst, on=["unique_id", "ds"])
    return out


def arm_extras(compound: bool):
    """What each fit computes before it is discarded: sweeps, probes and example scenarios."""
    def extras(nf, data):
        out = {"sweep": gp.price_sweep(nf, data, report_ids, channel=price_channel),
               "probe": gp.probe(nf, data, report_ids, channel=price_channel),
               "scenarios": scenario_forecasts(nf, data, example_ids)}
        if compound:
            out["sweep_compound"] = gp.price_sweep(nf, data, report_ids, channel=price_channel,
                                                   compound=True)
        return out
    return extras


def builder(arm: str):
    """Model factory for one arm (the alias names the forecast column)."""
    def build(seed):
        return gd.build_nbeatsx(params, arm_data[arm], max_steps, seed, f"NBEATSx_{arm}")
    return build


arms = {}
for arm in arm_cfgs:
    with bc.Timer() as t:
        arms[arm] = gd.fit_seeds(builder(arm), arm_data[arm], f"04_{arm}",
                                 extras=arm_extras(compound=(arm == "A_features")))
    if gd.TRAIN:
        print(f"{arm}: {t.seconds / 60:.1f} min for {len(gd.SEEDS)} seeds")

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 5. How we will choose an arm (fixed before looking)
#
# The rule is written down before the results below:
#
# 1. keep arms whose median elasticity at ±20% (seed-mean sweep) is **negative**;
# 2. and whose median test RMSSE is within the 02 **noise floor** of arm A;
# 3. of those, choose the one with the **lowest share of perverse** series. If none qualify, A.
#
# This uses the test window as a diagnostic, which is only acceptable because notebook 05
# repeats the choice properly: it tunes on the validation window and scores the test window once.

# %%
SELECTION_RULE = ("lowest share perverse among arms with median e20 < 0 and median test RMSSE "
                  "<= arm A + 02 noise floor; else A")

# %% [markdown]
# ## 6. Accuracy of each arm

# %%
forecasts = {arm: gd.seed_forecasts(runs) for arm, runs in arms.items()}
board_fcst = forecasts["A_features"][["unique_id", "ds"]]
seed_scores = {}
test_median = {}
for arm in arms:
    rmsse = gd.per_series(gd.score(forecasts[arm], test_a, train_a))
    seed_cols = [c for c in rmsse.columns if not c.endswith("_mean")]
    seed_scores[arm] = rmsse[seed_cols].median()
    test_median[arm] = float(rmsse[f"NBEATSx_{arm}_mean"].median())
    board_fcst = board_fcst.merge(forecasts[arm][["unique_id", "ds", f"NBEATSx_{arm}_mean"]],
                                  on=["unique_id", "ds"])

fit_seconds = {}
for arm, runs in arms.items():
    fit_seconds[f"NBEATSx_{arm}_mean"] = gd.summaries(runs)["fit_seconds"].mean()
gd.save_forecasts(board_fcst, "04_price")
gd.record(board_fcst, test_a, train_a, lab=f"04 price arms ({len(gd.SEEDS)}-seed mean)",
          fit_seconds=fit_seconds, notes="fit_seconds is per seed")
gd.median_board()

# %%
health_cols = ["arm", "seed", "fit_seconds", "n_train_series", "oscillation", "best_frac",
               "neg_share"]
health = [gd.summaries(runs).assign(arm=arm) for arm, runs in arms.items()]
pd.concat(health)[health_cols].round(3)

# %% [markdown]
# ## 7. Price response: single channel against compound
#
# Every sweep is averaged over seeds (the seed-mean model) before the elasticity is taken.

# %%
def seed_mean_sweep(runs: list[dict], key: str = "sweep") -> pd.DataFrame:
    """Average the total units of each (series, shift) over seeds."""
    allruns = pd.concat([run[key] for run in runs], ignore_index=True)
    return allruns.groupby(["unique_id", "shift"], as_index=False)["units"].mean()


def e20(el: pd.DataFrame) -> pd.Series:
    """Per-series arc elasticity at the largest shift."""
    top = el[el["shift"] == el["shift"].max()]
    return top.set_index("unique_id")["e_arc"]


elasticity = {"A_features (compound)": gp.arc_elasticity(
    seed_mean_sweep(arms["A_features"], "sweep_compound"))}
for arm, runs in arms.items():
    elasticity[arm] = gp.arc_elasticity(seed_mean_sweep(runs))
reports = {name: gp.elasticity_report(el) for name, el in elasticity.items()}
pd.DataFrame(reports).T.round(3)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ### Reading the result
#
# * Compare the first two rows: the **same fitted model**, swept two ways. The compound sweep
#   also raises `discount_depth` when it cuts the price, so the model sees "a deeper discount" and
#   "a lower price" at once. Because these two inputs are strongly correlated in training, the
#   network may have split the effect between them in any proportion; moving both together
#   reads off a mixture that can even have the wrong sign. In the research it gave +0.45.
# * `n_estimable` counts series whose forecast moved at all; `invariance` near 1 means the
#   response grows in proportion to the price move.

# %% [markdown]
# ## 8. LightGBM's response to price
#
# The reference LightGBM from notebook 00, with `price_index` in place of the raw sale price,
# trained on the full panel. We move `price_index` alone, over a wider range of shifts.

# %%
LGBM_EXOG = ["promo", "price_index", "discount_depth", "price_ratio_to_rolling_mean"]
LGBM_SHIFTS = (0.02, 0.05, 0.10, 0.20, 0.30)
lgbm_store = {}


def fit_lgbm() -> MLForecast:
    """Lei-style LightGBM on every training series (as in 00, with ``price_index``)."""
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
    lgbm_store["mlf"] = mlf
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
    booster = lgbm_store["mlf"].models_["LightGBM"].booster_
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
# * A tree predicts a constant between split points. A small price move often stays inside one
#   leaf and changes nothing; a big one crosses a split and jumps. Divide by the size of the move
#   and the "elasticity" changes with the shift size: that is the split, not the shopper.
# * Price features carry a small share of LightGBM's gain; the lags and rolling means do the
#   work. Use LightGBM for accuracy and the network for price response.

# %% [markdown]
# ## 9. Choosing the arm

# %%
candidates = []
for arm in arms:
    negative = reports[arm]["median_e20"] < 0
    accurate = test_median[arm] <= test_median["A_features"] + noise_floor
    if negative and accurate:
        candidates.append(arm)
if candidates:
    selected_arm = min(candidates, key=lambda a: reports[a]["share_perverse"])
else:
    selected_arm = "A_features"

choice_rows = []
for arm in arms:
    choice_rows.append({"arm": arm, "median test RMSSE": test_median[arm],
                        "median e20": reports[arm]["median_e20"],
                        "share perverse": reports[arm]["share_perverse"],
                        "qualifies": arm in candidates})
choice = pd.DataFrame(choice_rows).set_index("arm")
print(f"selected arm: {selected_arm}")
choice

# %% [markdown]
# ## 10. What the selected arm does on the example series (F12)
#
# Forecasts at the planned price (base), and with every test-week price 20% lower and higher.

# %%
scenarios = pd.concat([run["scenarios"] for run in arms[selected_arm]])
scenarios = scenarios.groupby(["unique_id", "ds"], as_index=False).mean()
gplt.plot_series_price(panel_a, examples, scenarios,
                       models=["-20% price", "base", "+20% price"],
                       title=f"Arm {selected_arm}: forecasts at -20% / base / +20% price")

# %% [markdown]
# ## 11. The elasticity charts
#
# Arm A with the single-channel sweep first, then the selected arm. The shaded band is the
# empirical inter-quartile range across classes from section 3.

# %%
gplt.plot_elasticity(elasticity["A_features"], info, band=band,
                     title="Arm A, single-channel sweep")

# %%
if selected_arm != "A_features":
    gplt.plot_elasticity(elasticity[selected_arm], info, band=band,
                         title=f"Arm {selected_arm} (selected), single-channel sweep")

# %%
strength = {name: e20(el) for name, el in elasticity.items()}
gplt.plot_elasticity_strength(strength, title="Strength of price response at ±20%")

# %%
gplt.plot_elasticity_by_class(strength, info, empirical=empirical)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ### Reading the result
#
# * Read the strength chart from the right: the orange "perverse" share is what we most want to
#   shrink. Does CPI deflation (B) or oversampling (C) shrink it?
# * The empirical line is much steeper than any model. It describes short promotions with the
#   promo flag held fixed; the models see the flag as a separate input and hold it fixed too.

# %% [markdown]
# ## 12. Probes: a sustained cut against a one-week pulse
#
# The oversampled arm C, all seeds pooled. A sustained cut lasts the whole horizon; a pulse lasts
# one week. A good model lifts sales in the pulse week and leaves the other weeks nearly alone.

# %%
probe_c = pd.concat([run["probe"] for run in arms["C_cpi_over"]], ignore_index=True)
probe_table = gp.probe_summary(probe_c)
gplt.plot_probe(probe_table, title="Arm C_cpi_over: probe responses")
probe_table.round(3)

# %% [markdown]
# ## 13. The accuracy cost of oversampling, and seeds against arms

# %%
both = forecasts["A_features"].merge(forecasts["C_cpi_over"], on=["unique_id", "ds"])
rmsse = gd.per_series(gd.score(both, test_a, train_a))
gplt.plot_wins(rmsse, info, model="NBEATSx_C_cpi_over_mean", baseline="NBEATSx_A_features_mean")

# %%
seed_e20 = {}
for arm, runs in arms.items():
    seed_e20[arm] = [float(e20(gp.arc_elasticity(run["sweep"])).median()) for run in runs]
gplt.plot_arm_comparison(arms, metrics=seed_scores, elasticity=seed_e20, band=band)
pd.DataFrame(seed_e20, index=[f"seed {s}" for s in gd.SEEDS]).round(3)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ### Reading the result
#
# * In the research each seed's response was all one sign, and the seeds disagreed in size.
#   Three seeds are the minimum for any number quoted here; look at the spread before you read
#   a difference between arms.

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
    "empirical": empirical.to_dict("records"),
    "empirical_all": empirical_all,
    "empirical_band": list(band),
    "n_report_series": len(report_ids),
    "regular_price_check": price_check["share of training weeks"].round(4).to_dict(),
    "median_rmsse": test_median,
    "selection_rule": SELECTION_RULE,
    "selected_arm": selected_arm,
    "max_steps": max_steps,
    "seeds": gd.SEEDS,
})

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## Discussion
#
# * The compound and single-channel rows come from the same model. Which one would you show a
#   category manager, and why?
# * The empirical benchmark here is about −3.6, the research's about −0.5. What does that say
#   about using one "plausible range" across retailers?
# * LightGBM is the most accurate model on the board. Why is it still not the model to plan
#   prices with?
#
# **Try it:** set `OVERSAMPLE["rule"] = "regular"` and rerun arm C (delete its `runs/04_C_*`
# folders first). How many more series get copies, and does the perverse share move?
