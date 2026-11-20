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
# # Day 2 · 03 (full data) — Lab: calendar, events and the hierarchy (25 min)
#
# The original lab 03 adds price and promotion as future covariates and the hierarchy as
# statics. Price response gets its own notebook (04). Here we take the tuned N-BEATSx from
# notebook 02 and ask whether it forecasts better when it also knows **when** a week is:
#
# | Arm | Adds | Future covariates |
# |---|---|---|
# | `base` | nothing (the 02 model) | promo, discount depth, price index |
# | `calendar` | week-of-year sine/cosine (2 pairs) and event weeks | + 4 Fourier + 4 events |
# | `calendar_hierarchy` | the above plus the item's department as a static | as `calendar` |
#
# The events are Easter (Good Friday), Anzac Day, Mother's Day and Christmas.
#
# This notebook
#
# 1. builds the three arms from configs, so any of them can be rebuilt with `gp.prepare(cfg)`,
# 2. fits each with 3 seeds and **chooses on the validation window**,
# 3. refits the best feature arm on the full training window, scores the test window once and
#    compares it with the base arm (which is notebook 02's tuned model), and
# 4. looks at the event weeks themselves, which sit at the very start of the test window.
#
# See `docs/gfm_learnings.md` §7.

# %%
import numpy as np
import pandas as pd

import gd_helpers as gd
import gd_plots as gplt
import gd_price as gp

bc = gd.bc
gd.status()
gplt.use_style()

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## Intuition: smooth seasons and sharp events
#
# Fourier terms are smooth waves over the year. They can say "sales rise towards December",
# but not "this one week is Easter", because Easter moves by up to five weeks from year to year.
# An event dummy is a switch that is on in that one week, whenever it falls. A department
# static is a name tag: it lets the model share patterns among items that sit together on the
# shelf, without one model per department.

# %% [markdown]
# ## 1. The starting point and the three arms
#
# The model settings, step budget and covariates come from `02_shortlist`. Each arm is a small
# config; `gp.prepare(cfg)` turns it into training data. Feature switches that are off are
# written out, so the config says exactly what was used.

# %%
shortlist_cfg = gd.load_config("02_shortlist")
examples = gd.load_config("00_baselines")["examples"]
params = shortlist_cfg["best_params"]
max_steps = shortlist_cfg["max_steps"]
noise_floor = shortlist_cfg["noise_floor"]
base_exog = shortlist_cfg["futr_exog"]
y_shift = shortlist_cfg["y_shift"]

panel = gd.train_panel()
train, test = bc.split(panel, cutoff=bc.TEST_START)
info = gplt.series_table(panel)

NO_FEATURES = {"deflate": False, "events": False, "fourier_k": 0, "hierarchy_level": None}
CALENDAR = {**NO_FEATURES, "events": True, "fourier_k": 2}
CALENDAR_HIERARCHY = {**CALENDAR, "hierarchy_level": "department_id"}
calendar_exog = base_exog + gp.fourier_cols(2) + gp.EVENT_COLS

arm_cfgs = {
    "base": {"features": NO_FEATURES, "futr_exog": base_exog, "y_shift": y_shift},
    "calendar": {"features": CALENDAR, "futr_exog": calendar_exog, "y_shift": y_shift},
    "calendar_hierarchy": {"features": CALENDAR_HIERARCHY, "futr_exog": calendar_exog,
                           "y_shift": y_shift},
}
ALIASES = {"base": "NBEATSx_base", "calendar": "NBEATSx_calendar",
           "calendar_hierarchy": "NBEATSx_cal_hier"}

arm_rows = []
for arm, cfg in arm_cfgs.items():
    arm_rows.append({"arm": arm, **cfg["features"], "future covariates": len(cfg["futr_exog"])})
pd.DataFrame(arm_rows).set_index("arm")

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ### Where the events fall
#
# The data has fewer than two years of training weeks, so the model sees **one** Easter, one
# Anzac Day and one Mother's Day to learn from. The validation window (January to mid April
# 2025) holds **no** event at all, so it can judge the Fourier terms and the department static,
# but not the event dummies.
#
# The test window is different. Good Friday 2025 (18 April) falls in the **first test week**,
# the week starting 16 April, and Anzac Day in the second. Easter in a held-back window is
# exactly what broke the price response in the research's confirm run (§6), and how to
# represent it is still an open question (§9).

# %%
def window_of(ds: pd.Series) -> pd.Series:
    """Which window each week belongs to: inner train, validation, test or later."""
    ds = pd.to_datetime(ds)
    test_end = pd.Timestamp(bc.TEST_START) + pd.Timedelta(weeks=bc.H)
    conditions = [ds < pd.Timestamp(bc.VALID_START), ds < pd.Timestamp(bc.TEST_START),
                  ds < test_end]
    return pd.Series(np.select(conditions, ["inner train", "validation", "test"], "after test"),
                     index=ds.index)


event_panel = gp.add_events(panel)
event_weeks = (event_panel.select(["ds"] + gp.EVENT_COLS).unique("ds").sort("ds")
               .to_pandas().melt(id_vars="ds", var_name="event", value_name="on"))
event_weeks = event_weeks[event_weeks["on"] > 0].drop(columns="on").sort_values("ds")
event_weeks["window"] = window_of(event_weeks["ds"])
event_weeks.reset_index(drop=True)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ### Hierarchy ids are labels, not numbers
#
# `department_id` is a number in the data, but department 41 is not "more" than department 3.
# Fed in as a number, the network would read a distance between them that means nothing, and
# the scaler would treat it as a quantity (§7). So `gp.hierarchy_statics` one-hot encodes the
# 12 largest departments; the rest share an "other" column.

# %%
hierarchy = gp.hierarchy_statics(panel, "department_id")
print(f"{panel['department_id'].n_unique()} departments → {hierarchy.shape[1] - 1} one-hot "
      f"columns")
hierarchy.head()

# %% [markdown]
# ## 2. Fit the three arms on the tuning split
#
# `gp.prepare(cfg, cutoff=bc.VALID_START)` builds each arm on the inner training weeks and
# forecasts the validation window; the reference price and statics use the inner weeks only.
# The calendar columns are known for any week and the department is a membership, so neither
# needs any statistic from later weeks.

# %%
def fit_arm(arm: str, data: gd.NixtlaData, stage: str) -> list[dict]:
    """The 02 model on one arm's data, 3 seeds; ``stage`` is ``valid`` or ``test``."""
    def build(seed):
        return gd.build_nbeatsx(params, data, max_steps, seed, ALIASES[arm])

    return gd.fit_seeds(build, data, f"03_{stage}_{arm}")


valid_runs = {}
for arm, cfg in arm_cfgs.items():
    _, inner, valid, data = gp.prepare(cfg, panel=panel, cutoff=bc.VALID_START)
    print(f"{arm}: {len(data.futr_exog)} future covariates, {len(data.stat_exog)} statics")
    valid_runs[arm] = fit_arm(arm, data, "valid")

# %% [markdown]
# ## 3. Choose on the validation window
#
# Each seed is scored on its own (median RMSSE over the evaluation series). The rule is fixed
# before looking: start from `base`, and move to a richer arm only if it beats the current
# choice by more than the **noise floor** measured in notebook 02.

# %%
def seed_medians(seed_runs: list[dict], actual, history) -> pd.Series:
    """Median RMSSE of each seed's forecast on ``actual``, scaled by ``history``."""
    fcst = gd.seed_forecasts(seed_runs)
    rmsse = gd.per_series(gd.score(fcst, actual, history))
    seed_cols = [c for c in rmsse.columns if not c.endswith("_mean")]
    return rmsse[seed_cols].median()


valid_scores = {}
rows = []
for arm, arm_runs in valid_runs.items():
    scores = seed_medians(arm_runs, valid, inner)
    health = gd.summaries(arm_runs)
    valid_scores[arm] = scores
    rows.append({"arm": arm, "median_rmsse_valid": scores.mean(), "seed_sd": scores.std(),
                 "oscillation": health["oscillation"].median(),
                 "best_frac": health["best_frac"].median(),
                 "minutes_per_fit": health["fit_seconds"].mean() / 60})
valid_table = pd.DataFrame(rows).set_index("arm")

valid_median = valid_table["median_rmsse_valid"]
chosen = "base"
for arm in list(arm_cfgs)[1:]:                  # simplest first
    if valid_median[chosen] - valid_median[arm] > noise_floor:
        chosen = arm
feature_arms = valid_table.drop(index="base")
challenger = chosen if chosen != "base" else feature_arms["median_rmsse_valid"].idxmin()
print(f"noise floor {noise_floor:.4f} | chosen: {chosen} | refitted for test: {challenger}")
valid_table.round(4)

# %%
gplt.plot_arm_comparison(valid_runs, metrics=valid_scores)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ### Reading the result
#
# * If no arm clears the noise floor, the choice stays `base`: the extra columns cost nothing to
#   compute, but a feature that cannot be shown to help is one more thing to explain.
# * Remember what the validation window could not test: it has no event week. A calendar arm
#   that ties here may still differ in the Easter and Anzac weeks of the test window.
#
# *Reference run: full panel, GPU box. Exact numbers vary a little with hardware; the reading
# should not.*

# %% [markdown]
# ## 4. Refit on the full training window and score the test window once
#
# The best feature arm (the chosen one, or the runner-up if `base` was kept) is refitted with
# three seeds and its seed-mean forecast goes on the board. The `base` arm is not refitted: it
# is notebook 02's `NBEATSx_tuned` (same settings, same data), so we reuse its saved forecasts.

# %%
base_col = ALIASES["base"]
challenger_col = ALIASES[challenger]
_, _, _, full_data = gp.prepare(arm_cfgs[challenger], panel=panel)
test_runs = fit_arm(challenger, full_data, "test")
challenger_fcst = gd.seed_forecasts(test_runs)[["unique_id", "ds", f"{challenger_col}_mean"]]
challenger_fcst = challenger_fcst.rename(columns={f"{challenger_col}_mean": challenger_col})

gd.save_forecasts(challenger_fcst, "03_features")
gd.record(challenger_fcst, test, train,
          lab=f"03 calendar and hierarchy ({len(gd.SEEDS)}-seed mean)",
          fit_seconds={challenger_col: gd.summaries(test_runs)["fit_seconds"].mean()},
          notes=f"chosen on validation: {chosen}; fit_seconds is per seed")

base_fcst = pd.read_parquet(gd.FORECAST_DIR / "02_tuned.parquet")
base_fcst = base_fcst[["unique_id", "ds", "NBEATSx_tuned"]].rename(
    columns={"NBEATSx_tuned": base_col})
test_fcst = base_fcst.merge(challenger_fcst, on=["unique_id", "ds"])
gd.median_board()

# %%
rmsse = gd.per_series(gd.score(test_fcst, test, train))
gplt.plot_error_dist(rmsse, info, models=[challenger_col, base_col])

# %%
gplt.plot_wins(rmsse, info, model=challenger_col, baseline=base_col)

# %% [markdown]
# ## 5. The event weeks
#
# RMSSE pools the 14 test weeks. To see Easter and Anzac Day, look week by week: the total
# forecast over the evaluation series against the total sold (bias), with the event marked.

# %%
def weekly_bias(fcst: pd.DataFrame, actual, models: list[str]) -> pd.DataFrame:
    """Per test week: total forecast ÷ total actual − 1 over the eval ids (forecasts ≥ 0)."""
    act = gd.only(actual, gd.eval_ids()).select("unique_id", "ds", "y").to_pandas()
    act["ds"] = pd.to_datetime(act["ds"])
    d = act.merge(fcst, on=["unique_id", "ds"])
    d[models] = d[models].clip(lower=0)
    totals = d.groupby("ds")[["y"] + models].sum()
    out = pd.DataFrame({m: totals[m] / totals["y"] - 1 for m in models})
    out.insert(0, "units sold", totals["y"])
    return out


week_events = event_weeks.assign(ds=pd.to_datetime(event_weeks["ds"])).set_index("ds")["event"]
bias = weekly_bias(test_fcst, test, [base_col, challenger_col])
bias.insert(0, "event", week_events.reindex(bias.index).fillna(""))
bias.round(3)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ### Reading the result
#
# * Look at the first two rows. A model without events has no way to know that the week of
#   16 April is Easter; its bias there is what the event dummy has to fix.
# * One good Easter week does not make a feature: the dummy was learned from a single Easter,
#   and a seed-mean forecast can still be lucky. Compare with the other weeks' bias before
#   reading much into it.

# %% [markdown]
# The example series from notebook 00, with event weeks marked (dotted) and both forecasts:

# %%
gplt.plot_series_price(event_panel, examples, test_fcst, models=[base_col, challenger_col],
                       events=True, title="Example series with event weeks (dotted)")

# %%
gd.save_config("03_features", {
    "params": params,
    "features": arm_cfgs[chosen]["features"],
    "futr_exog": arm_cfgs[chosen]["futr_exog"],
    "y_shift": y_shift,
    "max_steps": max_steps,
    "seeds": gd.SEEDS,
    "median_rmsse": {"valid": valid_table["median_rmsse_valid"].round(4).to_dict(),
                     "test": rmsse.median().round(4).to_dict()},
    "arm": chosen,
})

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## Discussion
#
# * The validation window had no event week; the test window starts with two. Did the choice
#   made on validation hold up on test? What would a fairer tuning window look like when the
#   events you care about come once a year?
# * One Easter in training is one example. Can a network learn "Easter" from a single week, or
#   is the dummy mostly noise until there are more years of data?
# * Look at the weekly bias in the Easter week. A model that misses Easter will also misread a
#   promotion that falls in it: the lift is credited to the price, not to the holiday (§6).
# * Notebook 04 builds on the chosen features and decides whether to deflate prices.
#
# **Try it:** set `"fourier_k": 4` in `CALENDAR` (and use `gp.fourier_cols(4)`) to give the
# calendar sharper seasonal shapes, and see whether the validation score moves past the noise
# floor.
