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
# # Day 2 · 05b (full data) — Lab: teaching price response with synthetic discounts (15 min)
#
# Notebook 05 picked a model that is accurate and responds to price. Its price response is still
# learnt from the discounts the shop actually ran, and those are few, short and bunched in the
# same series. Here we try a research idea (phase 7): **plant extra one-week price cuts** in the
# oversampled training copies, each with a known lift, and see whether the model learns from them.
#
# | | |
# |---|---|
# | Starts from | the selected 05 model (`05_selected`) |
# | Adds | two one-week cuts per oversampled copy, depth 10–30%, with a planted lift |
# | Originals | untouched: only the copies get synthetic weeks |
# | Measured | the same sweep and probes as 04/05, plus an **echo check** |
# | Decides | whether later notebooks use 05 or 05b (rule written before the results) |
#
# This notebook
#
# 1. builds the 05 training data with and without synthetic cuts and shows what was planted,
# 2. fits the 05 model on the synthetic version (3 seeds),
# 3. checks whether the model gives back the lift we planted,
# 4. compares price response and accuracy with 05, and applies the rule.

# %%
import matplotlib.pyplot as plt
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
# ## Intuition: a lesson with the answer written in
#
# A model learns price response from weeks where the price moved. If the shop rarely cuts prices
# outside a few promotions, there is little to learn from. Synthetic cuts are worked examples: we
# lower the price for one week in a copy of a series and raise its sales by a lift we choose. If
# the model has learnt anything, a probe with the same cut should **echo** back roughly that lift.
# The risk is that the model learns our made-up rule instead of the shop's real behaviour, so we
# check accuracy on real test weeks too.

# %% [markdown]
# ## 1. From 05's config to 05b's
#
# 05b is 05's config plus one key, `synthetic`. The cuts are planted with **one fixed random
# seed** for every training seed, so the three fits see the same training data and their spread
# is training noise only, as in 05.

# %%
baselines_cfg = gd.load_config("00_baselines")
noise_floor = gd.load_config("02_shortlist")["noise_floor"]
price_cfg = gd.load_config("04_price")
cfg_a = gd.load_config("05_selected")
price_channel = price_cfg.get("price_channel", "price_index")
examples = baselines_cfg["examples"]
example_ids = []
for tier_ids in examples.values():
    example_ids += tier_ids

cfg_b = dict(cfg_a)
oversample_added = not cfg_a.get("oversample")
if oversample_added:      # synthetic cuts live in the copies, so they need oversampling
    cfg_b["oversample"] = price_cfg.get("oversample") or {
        "rule": "rolling", "cap": gp.COPY_CAP, "multiplier": gp.COPY_MULTIPLIER}
    print("05 has no oversampling: 05b adds 04's default, so it differs from 05 in two ways.")
cfg_b["synthetic"] = {"cuts_per_copy": 2}
synthetic_seed = bc.RANDOM_SEED

raw = gd.train_panel()
panel, train, test, data_a = gp.prepare(cfg_a, panel=raw)
_, _, _, data_b = gp.prepare(cfg_b, panel=raw, synthetic_seed=synthetic_seed)
info = gplt.series_table(panel)
report_ids = gp.report_ids(train)

params = cfg_a["params"]
max_steps = cfg_a["max_steps"]
if oversample_added:
    max_steps = gd.scaled_max_steps(data_b.fit_df["unique_id"].nunique(),
                                    batch_size=params.get("batch_size", 256))
print(f"{data_b.fit_df['unique_id'].nunique():,} training series incl. copies | "
      f"{max_steps:,} steps | {len(report_ids):,} report series | seeds {gd.SEEDS}")
pd.Series({"oversample": cfg_b["oversample"], "synthetic": cfg_b["synthetic"],
           "synthetic_seed": synthetic_seed, "price_channel": price_channel})

# %% [markdown]
# ### What was planted
#
# Each copy is its original series with a few weeks changed, so comparing them row by row shows
# the planted weeks: the price is lower and the units higher. The table counts them by depth,
# with the lift we asked for and the lift that landed (units are whole numbers, so small sellers
# round).

# %%
def planted_weeks(fit_df: pd.DataFrame, y_shift: float) -> pd.DataFrame:
    """Weeks where a copy's price differs from its original series': depth and lift."""
    is_copy = fit_df["unique_id"].str.contains(gp.COPY_TAG, regex=False)
    copies = fit_df[is_copy].copy()
    copies["original"] = copies["unique_id"].str.split(gp.COPY_TAG).str[0]
    originals = fit_df[~is_copy].rename(columns={"unique_id": "original"})
    both = copies.merge(originals, on=["original", "ds"], suffixes=("_b", "_a"))
    both = both[~np.isclose(both["price_index_a"], both["price_index_b"])].copy()
    both["depth"] = (1 - both["price_index_b"] / both["price_index_a"]).round(2)
    units_a = both["y_a"] - y_shift
    both["lift"] = (both["y_b"] - y_shift) / units_a.where(units_a > 0) - 1
    return both[["unique_id", "ds", "depth", "lift"]]


planted = planted_weeks(data_b.fit_df, data_b.y_shift)
planted_table = planted.groupby("depth").agg(weeks=("lift", "size"),
                                             median_lift_landed=("lift", "median"))
planted_table["lift_asked"] = planted_table.index.map(gp.SYNTHETIC_LIFT)
print(f"{len(planted):,} planted weeks in {planted['unique_id'].nunique():,} copies")
planted_table.round(2)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 2. The rule, written before the results
#
# Later notebooks use 05b **only if all three hold**; otherwise they keep 05:
#
# 1. its median arc elasticity at ±20% (`median_e20`, report series) is **more negative** than
#    05's;
# 2. its share of right-signed series under the **sustained** probes (averaged over the four
#    shocks) is **higher** than 05's;
# 3. its test median RMSSE is no more than the 02 noise floor above 05's.
#
# The first two ask for a stronger and more consistent price response; the third says it must
# not cost real accuracy. 05's numbers come from its saved model.

# %% [markdown]
# ## 3. Fit the 05 model on the synthetic data (3 seeds)
#
# Each fit also runs the single-channel sweep and the probes on the report series, and the
# price scenarios on the example series, before the model is discarded. Only the first seed is
# saved, for later notebooks.

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


def price_extras(nf, data):
    """Sweep, probes and example scenarios, computed while the model is in memory."""
    return {"sweep": gp.price_sweep(nf, data, report_ids, channel=price_channel),
            "probe": gp.probe(nf, data, report_ids, channel=price_channel),
            "scenarios": scenario_forecasts(nf, data, example_ids)}


def synthetic_model(seed):
    return gd.build_nbeatsx(params, data_b, max_steps, seed, "GFM_synthetic")


run_name = "05b_synthetic"
runs = [gd.fit_seed(synthetic_model, gd.SEEDS[0], data_b, run_name, extras=price_extras,
                    save_as="05b_synthetic")]
runs += gd.fit_seeds(synthetic_model, data_b, run_name, seeds=gd.SEEDS[1:], extras=price_extras)
health_cols = ["seed", "fit_seconds", "oscillation", "best_frac", "final_over_best", "neg_share"]
gd.summaries(runs)[health_cols].round(3)

# %%
gplt.plot_training(runs, warmup=gd.warmup_steps(max_steps), title="05b: every seed")

# %% [markdown]
# ### 05's reference numbers
#
# The same sweep and probes on 05's saved model, computed once here so this notebook does not
# depend on how 05 stored its runs.

# %%
def reference_sweep() -> pd.DataFrame:
    return gp.price_sweep(gd.load_model(cfg_a["saved_model"]), data_a, report_ids,
                          channel=price_channel)


def reference_probe() -> pd.DataFrame:
    return gp.probe(gd.load_model(cfg_a["saved_model"]), data_a, report_ids,
                    channel=price_channel)


sweep_a = gd.cached("05b_reference_sweep", reference_sweep)
probe_a = gd.cached("05b_reference_probe", reference_probe)

# %% [markdown]
# For 05b, each series' sweep and probe are averaged over the three seeds first (the response of
# the seed-mean model), then summarised. The per-seed medians show the spread.

# %%
def seed_mean(frames: list[pd.DataFrame], keys: list[str]) -> pd.DataFrame:
    """Average the numeric columns of per-seed frames that share ``keys``."""
    return pd.concat(frames).groupby(keys, as_index=False).mean(numeric_only=True)


def e20(el: pd.DataFrame) -> pd.Series:
    """Per-series arc elasticity at the largest shift (±20%)."""
    top = el[el["shift"] == el["shift"].max()]
    return top.set_index("unique_id")["e_arc"]


sweep_b = seed_mean([r["sweep"] for r in runs], ["unique_id", "shift"])
probe_b = seed_mean([r["probe"] for r in runs], ["unique_id", "shock", "mode"])
el_a = gp.arc_elasticity(sweep_a)
el_b = gp.arc_elasticity(sweep_b)

per_seed = []
for r in runs:
    report = gp.elasticity_report(gp.arc_elasticity(r["sweep"]))
    per_seed.append({"seed": r["summary"]["seed"], **report})
per_seed = pd.DataFrame(per_seed)
reports = pd.DataFrame({"05 selected": gp.elasticity_report(el_a),
                        "05b synthetic": gp.elasticity_report(el_b)}).T
per_seed[["seed", "median_e20", "share_perverse", "share_flat"]].round(3)

# %%
reports.round(3)

# %% [markdown]
# ## 4. The echo check: does the model give back the planted lift?
#
# A one-week **pulse** probe is the same experiment as a planted cut: the price falls for one
# week and nothing else changes. We compare the lift we planted at 10%, 20% and 30% with the
# median lift each model predicts in the pulse week.

# %%
summary_a = gp.probe_summary(probe_a)
summary_b = gp.probe_summary(probe_b)

echo_rows = []
for depth in (0.10, 0.20, 0.30):
    row = {"depth": depth, "planted": gp.SYNTHETIC_LIFT[depth]}
    for name, summ in (("05", summary_a), ("05b", summary_b)):
        for mode in ("pulse", "sustained"):
            hit = summ[(summ["mode"] == mode) & np.isclose(summ["shock"], -depth)]
            row[f"{name} {mode}"] = float(hit["median_lift"].iloc[0])
    echo_rows.append(row)
echo = pd.DataFrame(echo_rows).set_index("depth")
echo.round(3)

# %%
fig, ax = plt.subplots(figsize=(8, 4))
x = echo.index * 100
ax.plot(x, echo["planted"] * 100, "s--", color="black", lw=2, label="planted")
ax.plot(x, echo["05 pulse"] * 100, "o-", color=gplt.MODEL_COLOURS[0], label="05, pulse")
ax.plot(x, echo["05b pulse"] * 100, "o-", color=gplt.MODEL_COLOURS[1], label="05b, pulse")
ax.set(xlabel="one-week price cut, %", ylabel="median lift in that week, %",
       title="Echo check: planted lift against the lift the model predicts")
ax.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# ### Reading the result
#
# <!-- DRAFT: Graeme to edit -->
# * The planted line is the answer we wrote in. 05 never saw it; 05b saw it in ~2 weeks per copy.
# * If 05b's line sits near the planted one, the model learnt the rule we planted. That shows it
#   *can* learn a price response from data like this, not that the shop's customers behave so.
# * The probe moves only the price channel. A planted cut also raised the discount depth, so a
#   partial echo is expected.

# %% [markdown]
# ## 5. Probes: sustained against pulse (F11)

# %%
gplt.plot_probe(summary_a, title="05 selected: probe responses")

# %%
gplt.plot_probe(summary_b, title="05b synthetic (seed mean): probe responses")

# %% [markdown]
# ## 6. Price response by strength and class (F6, F7)

# %%
empirical = gp.empirical_elasticity(train, report_ids)
strength = {"05 selected": e20(el_a), "05b synthetic": e20(el_b)}
gplt.plot_elasticity_strength(strength, title="Strength of price response at ±20%")

# %%
gplt.plot_elasticity_by_class(strength, info, empirical=empirical)

# %% [markdown]
# ## 7. Accuracy on the test window
#
# 05b's seed-mean forecast against 05's, series by series (F4).

# %%
forecasts_b = gd.seed_forecasts(runs).rename(columns={"GFM_synthetic_mean": "GFM_synthetic"})
seed_cols = [c for c in forecasts_b.columns if c.startswith("GFM_synthetic_s")]
forecasts_a = gd.selected_forecasts(cfg_a)          # 05's forecasts, column GFM_selected
both = forecasts_a.merge(forecasts_b, on=["unique_id", "ds"])
rmsse = gd.per_series(gd.score(both, test, train))

seed_medians = rmsse[seed_cols].median()
accuracy = pd.Series({"05 median RMSSE": rmsse["GFM_selected"].median(),
                      "05b median RMSSE": rmsse["GFM_synthetic"].median(),
                      "05b seed spread (max - min)": seed_medians.max() - seed_medians.min(),
                      "noise floor (02)": noise_floor})
accuracy.round(4)

# %%
gplt.plot_wins(rmsse, info, model="GFM_synthetic", baseline="GFM_selected")

# %%
fit_seconds = gd.summaries(runs)["fit_seconds"].mean()
gd.save_forecasts(forecasts_b[["unique_id", "ds", "GFM_synthetic"]], "05b_synthetic")
gd.record(forecasts_b[["unique_id", "ds", "GFM_synthetic"]], test, train,
          lab=f"05b synthetic ({len(gd.SEEDS)}-seed mean)", fit_seconds=fit_seconds,
          notes="fit_seconds is per seed")
gd.median_board()

# %% [markdown]
# ## 8. What the synthetic model does on the example series (F12)
#
# Forecasts at the planned price (base), and with every test-week price 20% lower and higher,
# averaged over the three seeds.

# %%
scenarios = seed_mean([r["scenarios"] for r in runs], ["unique_id", "ds"])
gplt.plot_series_price(panel, examples, scenarios, models=["-20% price", "base", "+20% price"],
                       title="05b: forecasts at -20% / base / +20% price")

# %% [markdown]
# ## 9. Apply the rule

# %%
def right_sign_share(summary: pd.DataFrame) -> float:
    """Mean share of right-signed series over the sustained probes."""
    return float(summary[summary["mode"] == "sustained"]["share_right_sign"].mean())


checks = {
    "stronger response (median_e20 more negative)":
        reports.loc["05b synthetic", "median_e20"] < reports.loc["05 selected", "median_e20"],
    "more right-signed series (sustained probes)":
        right_sign_share(summary_b) > right_sign_share(summary_a),
    "accuracy within the noise floor":
        accuracy["05b median RMSSE"] - accuracy["05 median RMSSE"] <= noise_floor,
}
use = "05b" if all(checks.values()) else "05"
print(f"later notebooks use: {use}")
pd.Series(checks)

# %%
gd.save_config("05b_synthetic", {
    **cfg_a,
    "oversample": cfg_b["oversample"],
    "oversample_added": oversample_added,
    "synthetic": cfg_b["synthetic"],
    "synthetic_seed": synthetic_seed,
    "max_steps": max_steps,
    "seeds": gd.SEEDS,
    "use": use,
    "rule": list(checks),
    "checks": {k: bool(v) for k, v in checks.items()},
    "elasticity": reports.round(4).to_dict("index"),
    "share_right_sign": {"05": right_sign_share(summary_a), "05b": right_sign_share(summary_b)},
    "echo": echo.round(4).reset_index().to_dict("records"),
    "median_rmsse": {"05": accuracy["05 median RMSSE"], "05b": accuracy["05b median RMSSE"]},
    "saved_model": "05b_synthetic",
    "forecasts": "05b_synthetic",
    "forecast_column": "GFM_synthetic",
})

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## Discussion
#
# * The echo check shows whether the model *can* learn a response it is shown. Is a planted lift a
#   fair teacher when the shop's own discounts suggest an elasticity near −3.6?
# * Synthetic cuts sit in copies only. Do they change the forecasts for the original series in
#   weeks with no price change? Look at the accuracy table and the wins chart.
# * The rule needed all three checks. Which one is the hardest to pass, and why?
#
# **Try it:** set `cuts_per_copy` to 4, or scale `gp.SYNTHETIC_LIFT` down by half, and rerun the
# echo check. Does the measured lift follow the planted one?
