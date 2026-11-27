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
# # Day 2 · 03 (full data) — Lab: calendar, events and the hierarchy (25 min)
#
# The original lab 03 adds price and promotion as future covariates and the hierarchy as statics.
# Here price response gets a notebook of its own (04). Instead, we take the tuned N-BEATSx from
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
# Goals
#
# 1. Build the three arms from configs, so any of them can be rebuilt with `gp.prepare(cfg)`.
# 2. Fit each with 3 seeds and **choose on the validation window**.
# 3. Refit the best feature arm, score the test window once, and compare it with `base`
#    (notebook 02's tuned model).
# 4. Look at the event weeks themselves, which sit at the start of the test window.
#
# See `docs/gfm_learnings.md` §7.
#
# **In the chain:** reads `02_shortlist` and `00_baselines`, writes `03_features`.

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
# ## Intuition: smooth seasons and sharp events
#
# Fourier terms describe the year as a smooth curve. They can tell the model that sales rise
# towards December, but not that one particular week is Easter, because Easter moves by up to
# five weeks from year to year. An event dummy does the opposite: it switches on in that one
# week, whenever it falls, and is off otherwise. A department static is a label that lets the
# model share patterns among items that sit together on the shelf, without needing one model per
# department.

# %% [markdown]
# ## 1. The starting point and the three arms
#
# The model settings, step budget and covariates come from `02_shortlist`. Each arm is a small
# config, and `gp.prepare(cfg)` turns it into training data. We write out the switches that are
# off too, so the config records exactly what was used.

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
# ### Where the events fall
#
# Our data has fewer than two years of training weeks, so the model sees **one** Easter, one
# Anzac Day and one Mother's Day to learn from. The validation window (January to early April
# 2025) holds **no** event at all, which means it can judge the Fourier terms and the department
# static, but not the event dummies.
#
# The test window is different. Good Friday 2025 (18 April) falls in the **first test week**, the
# week starting 16 April, and Anzac Day in the second. Easter also fell in our research's
# held-back window, which is one reason its price response did not transfer (§6, [100k set]).
# How best to represent Easter is still an open question (§9). The table below lists every event
# week.

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
# ### Hierarchy ids are labels, not numbers
#
# `department_id` is a number in the data, but department 41 is not "more" than department 3. If
# we fed it in as a number, the network would read a distance between departments that means
# nothing, and the scaler would treat it as a quantity (§7, [method]). So `gp.hierarchy_statics`
# one-hot encodes the 12 largest departments, and the rest share an "other" column.

# %%
hierarchy = gp.hierarchy_statics(panel, "department_id")
print(f"{panel['department_id'].n_unique()} departments → {hierarchy.shape[1] - 1} one-hot "
      f"columns")
hierarchy.head()

# %% [markdown]
# ## 2. Fit the three arms on the tuning split
#
# `gp.prepare(cfg, cutoff=bc.VALID_START)` builds each arm on the inner training weeks and
# forecasts the validation window, with the reference price and statics taken from the inner
# weeks only. Calendar columns are known for any week, and a department is a membership, so
# neither needs a statistic from later weeks.

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
# We score each seed on its own (median RMSSE over the evaluation series). The rule is fixed
# before we look: start from `base`, and only move to a richer arm if it beats the current
# choice by more than the **noise floor** from notebook 02.

# %%
valid_scores = {}
rows = []
for arm, arm_runs in valid_runs.items():
    scores = gd.seed_medians(arm_runs, valid, inner)
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
# ### Reading the result
#
# * If no arm clears the noise floor, we stay with `base`. The extra columns cost nothing to
#   compute, but a feature we can't show to help is one more thing to explain.
# * Validation median RMSSE: base ≈ 0.613, calendar ≈ 0.717, calendar + hierarchy ≈ 0.703. The
#   Fourier and event inputs made the model worse, most likely because it overfits them and
#   responds to noise.
# * Remember what validation could not test: it has no event week. A calendar arm that ties here
#   may still differ in the Easter and Anzac weeks of the test window.
#
# *Reference run: full panel, GPU box, 3 seeds. Exact numbers vary with seed and hardware; the
# reading should not.*

# %% [markdown]
# ## 4. Refit on the full training window and score the test window once
#
# We refit the best feature arm (the chosen one, or the runner-up if `base` was kept) with three
# seeds, and its seed-mean forecast goes on the board. `base` is not refitted: it is notebook
# 02's `NBEATSx_tuned` (same settings, same data), so we reuse its saved forecasts.

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
gd.median_board("03")

# %%
rmsse = gd.per_series(gd.score(test_fcst, test, train))
gplt.plot_error_dist(rmsse, info, models=[challenger_col, base_col])

# %%
gplt.plot_wins(rmsse, info, model=challenger_col, baseline=base_col)

# %% [markdown]
# ## 5. The event weeks
#
# RMSSE pools the 14 test weeks, so it hides what happens in any single week. To see Easter and
# Anzac Day we go week by week, computing the total forecast over the evaluation series ÷ the
# total sold − 1 (the bias), with each event marked. Positive means over-forecast.

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
# ### Reading the result
#
# * Look at the first two rows. A model without events cannot know that the week of 16 April is
#   Easter, so its bias there is what the event dummy has to fix.
# * Easter-week bias: base ≈ −3.6%, feature arm ≈ −2.4%.
# * One good Easter week does not make a feature. The dummy was learnt from a single Easter, and
#   a seed-mean forecast can still be lucky, so compare it with the other weeks' bias first.
#
# *Exact numbers vary with seed and hardware; the reading should not.*

# %% [markdown]
# Here are the example series from notebook 00, with both forecasts and the event weeks marked
# (dotted).

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
# ## Discussion
#
# * The validation window had no event week, while the test window starts with two. Did the
#   choice we made on validation hold up on test? What would a fairer tuning window look like
#   when the events you care about come once a year?
# * One Easter in training is one example. Can a network learn "Easter" from a single week, or
#   is the dummy mostly noise until we have more years of data?
# * A model that misses Easter may also misread a promotion that falls in it, and credit the
#   holiday's lift to the price. Why does that matter for notebook 04?
#
# **Try it:** set `"fourier_k": 4` in `CALENDAR` (and use `gp.fourier_cols(4)`) to give the
# calendar sharper seasonal shapes, and see whether the validation score moves past the noise
# floor.
