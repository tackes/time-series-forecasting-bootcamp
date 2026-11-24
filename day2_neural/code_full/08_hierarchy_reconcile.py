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
# # Day 2 · 08 (full data) — Lab: a global model at the bottom of the hierarchy (20 min)
#
# The original lab 05 forecasts every level of total → department → class → subclass → item
# with AutoETS and reconciles them. Here the **item level comes from the selected global
# model**, and simple ETS models forecast the levels above. Because the model forecasts every
# item, the upper levels are true totals, not totals of a sample.
#
# This follows the research's phase 8, which is **experimental** (`docs/gfm_learnings.md` §8,
# [100k set]). Two differences: the upper levels use ETS, not the research's dynamic harmonic
# regression, and the items come from one saved seed, not a 3-seed mean.
#
# | Level | Base forecast |
# |---|---|
# | total, department, class, subclass | `AutoETS(season_length=1)`, as in the original lab |
# | item | the selected GFM, target shift removed, clipped at 0 and at 3× its largest week |
#
# | Method | Idea |
# |---|---|
# | `BottomUp` | trust the items and sum them |
# | `TopDown` (forecast proportions) | trust the total and split it down |
# | `MiddleOut` (class) | trust the classes: sum up from them and split down |
# | `MinTrace` `ols` / `wls_struct` | least-squares adjustment of every level at once |
#
# Goals
#
# 1. Forecast every item with the selected model, and every upper node with ETS.
# 2. Reconcile them and check that the results add up.
# 3. Score each method at each level, and put the item level on the board.
#
# **In the chain:** reads the selected config (`gd.selected_config()`, for its saved model)
# and `00_baselines`, writes `08_reconcile`.

# %%
from dataclasses import replace
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import (BottomUpSparse, MiddleOutSparse, MinTraceSparse,
                                          TopDownSparse)
from hierarchicalforecast.utils import aggregate
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, Naive
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import rmsse

import gd_helpers as gd
import gd_plots as gplt
import gd_price as gp

bc = gd.bc
gd.status()
gplt.use_style()

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## Intuition: the numbers must add up, so whom do you believe?
#
# ![Incoherent forecasts](../slides/img/hierarchy_tree.png)
#
# Think of a budget whose line items disagree with its total. A global model is good at items:
# it has seen thousands of them. It has never seen a department total, which is a different,
# much smoother kind of series. ETS on the total sees exactly that series. Reconciliation
# decides how much to believe each, and makes the numbers add up.

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 1. Item forecasts from the selected model
#
# The saved model forecasts **every** training series, not just the scored ones, so the upper
# levels are sums of real items. This uses the one saved seed, not the 3-seed mean on the
# board, because the other seeds were not saved (`docs/progress.md`, single-seed exceptions).

# %%
cfg = gd.selected_config()
baselines_cfg = gd.load_config("00_baselines")
panel, train, test, data = gp.prepare(cfg, synthetic_seed=cfg.get("synthetic_seed"))
info = gplt.series_table(panel)
all_ids = sorted(panel["unique_id"].unique().to_list())


def bottom_forecasts() -> pd.DataFrame:
    """The saved selected model's forecasts for every training series, clipped at 0."""
    nf = gd.load_model(cfg["saved_model"])
    every = replace(data, predict_df=gd.only(data.fit_df, all_ids),
                    futr_df=bc.to_nixtla(test, data.futr_exog).drop(columns="y"))
    with bc.Timer() as t:
        fcst = gd.predict(nf, every)          # removes the target shift
    print(f"{len(all_ids):,} series forecast in {t.seconds:.0f}s")
    col = bc.model_columns(fcst)[0]
    fcst = fcst[["unique_id", "ds", col]].rename(columns={col: "Base"})
    fcst["Base"] = fcst["Base"].clip(lower=0).astype("float32")
    return fcst


bottom = gd.cached("08_bottom_forecasts", bottom_forecasts)
bottom["ds"] = pd.to_datetime(bottom["ds"])
print(f"{bottom['unique_id'].nunique():,} items forecast by {cfg['saved_model']}")

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ### A guard: cap each item at a multiple of its largest training week
#
# The eval sample hides a problem that sums over every item expose. On the full run a handful
# of the 17,000 item forecasts exploded (one item that sells ≈130 units a week was forecast at
# millions), and on its own that put the bottom-up store total ≈ 47% too high. We cap each
# item's forecast at `CAP` times its largest training week. Real sales above 3× that happened in
# ≈ 0.5% of test weeks, so the cap seldom cuts a real spike. The cell prints how many it touches.

# %%
CAP = 3                     # item forecast ceiling, as a multiple of its largest training week
train_max = train.group_by("unique_id").agg(pl.col("y").max().alias("train_max")).to_pandas()
bottom = bottom.merge(train_max, on="unique_id", how="left")
ceiling = CAP * bottom["train_max"]
over = bottom["Base"] > ceiling
n_capped = int(bottom.loc[over, "unique_id"].nunique())
print(f"{over.sum():,} item-weeks in {n_capped} items capped "
      f"at {CAP}x their largest training week")
bottom.loc[over, ["unique_id", "ds", "Base", "train_max"]].nlargest(5, "Base")

# %%
bottom["Base"] = bottom["Base"].clip(upper=ceiling).astype("float32")
bottom = bottom.drop(columns="train_max")

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 2. Build the hierarchy
#
# As in the original lab. Items without a subclass cannot be placed and are left out (the cell
# prints how many, and how many of them are scored). The summing matrix is kept **sparse**:
# with ≈19,000 nodes and ≈17,000 items, a dense one would take several GB.

# %%
placed = panel.filter(pl.col("subclass_id") > 0)
unplaced = set(panel["unique_id"].unique()) - set(placed["unique_id"].unique())
n_series = panel["unique_id"].n_unique()
n_scored = len(unplaced & set(gd.eval_ids()))
print(f"{len(unplaced):,} of {n_series:,} items have no subclass and are left out "
      f"({n_scored} of them scored)")
df = placed.select(
    pl.lit("Total").alias("total"),
    pl.format("D{}", pl.col("department_id")).alias("department"),
    pl.format("C{}", pl.col("class_id")).alias("class"),
    pl.format("S{}", pl.col("subclass_id")).alias("subclass"),
    pl.col("unique_id").alias("item"),
    "ds", "y",
).to_pandas()
df["ds"] = pd.to_datetime(df["ds"])

spec = [
    ["total"],
    ["total", "department"],
    ["total", "department", "class"],
    ["total", "department", "class", "subclass"],
    ["total", "department", "class", "subclass", "item"],
]
Y_df, S, tags = aggregate(df, spec, sparse_s=True)
LEVELS = [key.split("/")[-1] for key in tags]
print({level: len(ids) for level, ids in zip(LEVELS, tags.values())})

test_start = pd.Timestamp(bc.TEST_START)
Y_train = Y_df[Y_df["ds"] < test_start]
Y_test = Y_df[Y_df["ds"] >= test_start]
items = tags["total/department/class/subclass/item"]
item_path = pd.Series(items, index=[p.split("/")[-1] for p in items])
level_of = {}
for key, uids in tags.items():
    for uid in uids:
        level_of[uid] = key.split("/")[-1]

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 3. Base forecasts for the upper levels
#
# AutoETS on every node above the items, with Naive as the fallback if a fit fails. We use
# `season_length=1` because the aggregates have under two years of history, too short for a
# yearly season.

# %%
upper_levels = [uids for key, uids in tags.items() if not key.endswith("/item")]
upper_ids = np.concatenate(upper_levels).tolist()


def upper_forecasts() -> pd.DataFrame:
    """AutoETS forecasts for every node above the item level."""
    sf = StatsForecast(models=[AutoETS(season_length=1)], freq=bc.FREQ, n_jobs=gd.N_JOBS,
                       fallback_model=Naive())
    with bc.Timer() as t:
        fcst = sf.forecast(df=Y_train[Y_train["unique_id"].isin(set(upper_ids))], h=bc.H)
    print(f"{len(upper_ids):,} upper nodes forecast in {t.seconds:.0f}s")
    fcst = fcst.reset_index() if "unique_id" not in fcst.columns else fcst
    return fcst.rename(columns={"AutoETS": "Base"})


upper = gd.cached("08_upper_base", upper_forecasts)
upper["ds"] = pd.to_datetime(upper["ds"])

item_base = bottom[bottom["unique_id"].isin(item_path.index)].copy()
item_base["unique_id"] = item_base["unique_id"].map(item_path)
Y_hat = pd.concat([upper[["unique_id", "ds", "Base"]], item_base], ignore_index=True)
order = {uid: i for i, uid in enumerate(S.row_labels)}       # same order as the summing matrix
Y_hat = (Y_hat.assign(_o=Y_hat["unique_id"].map(order)).sort_values(["_o", "ds"])
         .drop(columns="_o").reset_index(drop=True))
print(f"{Y_hat['unique_id'].nunique():,} nodes with a base forecast")

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 4. Reconcile
#
# The sparse versions of the original lab's methods, with two differences:
#
# * `MinTrace(mint_shrink)` is left out. It needs in-sample errors for every node, and the GFM
#   has none for the items without a costly in-sample pass. `ols` and `wls_struct` need none.
# * `MiddleOut` uses the class level. The research preferred MiddleOut (§8).

# %%
reconcilers = [
    BottomUpSparse(),
    TopDownSparse(method="forecast_proportions"),
    MiddleOutSparse(middle_level="total/department/class", top_down_method="forecast_proportions"),
    MinTraceSparse(method="ols"),
    MinTraceSparse(method="wls_struct"),
]
SHORT = {"Base": "Base (incoherent)", "Base/BottomUpSparse": "BottomUp",
         "Base/TopDownSparse_method-forecast_proportions": "TopDown",
         "Base/MiddleOutSparse_middle_level-total/department/class_top_down_method-"
         "forecast_proportions": "MiddleOut",
         "Base/MinTraceSparse_method-ols": "MinT-ols",
         "Base/MinTraceSparse_method-wls_struct": "MinT-wls_struct"}

hrec = HierarchicalReconciliation(reconcilers=reconcilers)
with bc.Timer() as t_rec:
    Y_rec = hrec.reconcile(Y_hat_df=Y_hat, S_df=S, tags=tags)
print(f"reconciled in {t_rec.seconds:.1f}s")
Y_rec = Y_rec.rename(columns=SHORT)
methods = [c for c in Y_rec.columns if c in SHORT.values()]
methods

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ### Coherence check: does everything add up?
#
# Two checks per method: the original lab's (items sum to the total) and a stricter one (every
# node equals the sum of its items, via the summing matrix). A method that fails is dropped.

# %%
def coherence(Y_rec: pd.DataFrame, method: str) -> dict:
    """Largest gap between the total and its items, and between any node and its items."""
    wide = Y_rec.pivot(index="unique_id", columns="ds", values=method)
    all_nodes = wide.loc[S.row_labels].to_numpy()
    bottom_nodes = wide.loc[S.col_labels].to_numpy()
    total = wide.loc["Total"].to_numpy()
    return {"max |total - sum of items|": float(np.abs(total - bottom_nodes.sum(axis=0)).max()),
            "max |node - sum of its items|": float(np.abs(all_nodes - S.to_csr() @ bottom_nodes)
                                                   .max())}


COHERENCE_TOL = 1e-6        # allowed gap, as a share of the average weekly total
checks = pd.DataFrame({m: coherence(Y_rec, m) for m in methods}).T
total_size = Y_train[Y_train["unique_id"] == "Total"]["y"].mean()
tolerance = COHERENCE_TOL * total_size
incoherent = [m for m in methods if m != "Base (incoherent)"
              and checks.loc[m].max() > tolerance]
print("fails the coherence check, dropped:", incoherent)
methods = [m for m in methods if m not in incoherent]
checks.round(4)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# Every reconciled method should show 0, up to rounding; only the base forecasts may disagree.
# In the original lab the dense `MiddleOut` failed this check, so run it whenever the library
# or the hierarchy changes.

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 5. The rule, written before the results
#
# Planning uses every level, so a method should not be bad at any of them. We pick the
# coherent method whose **worst level** is best: the smallest maximum, over the five levels, of
# its median RMSSE divided by the base forecast's. Ties within 1% go to `MiddleOut`, the
# research's preference, then to the simpler method.
#
# The rule reads the test window, so the chosen method is a finding to report, not a setting
# tuned for later notebooks.

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 6. Accuracy per level
#
# RMSSE here is scaled by the one-step naive error, as in the original lab, because the
# aggregates are too short for a 52-week scale. So the scores compare across levels but **not**
# with the board. Forecasts are clipped at 0. Every node at each level is scored, all 17,000
# items included. Median first, then the mean.

# %%
scored = Y_rec.merge(Y_test, on=["unique_id", "ds"])
scored[methods] = scored[methods].clip(lower=0)
ev = evaluate(scored, metrics=[partial(rmsse, seasonality=1)], models=methods,
              train_df=Y_train)
ev["level"] = ev["unique_id"].map(level_of)
by_level = ev.groupby("level")[methods].median().loc[LEVELS]
by_level_mean = ev.groupby("level")[methods].mean().loc[LEVELS]
by_level.round(3)

# %%
by_level_mean.round(3)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# Bias per level: total forecast over total actual, minus one (negative = under-forecast).

# %%
sums = scored.assign(level=scored["unique_id"].map(level_of)).groupby("level")[methods + ["y"]]
bias = sums.sum()[methods].div(sums.sum()["y"], axis=0).sub(1).loc[LEVELS]
bias.round(3)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# The chart below puts the per-level table on one scale. Each group of bars is a method; each
# bar is one level, its median RMSSE divided by the base forecast's at that level. Left of the
# dotted line at 1, reconciliation helped that level; right of it, it hurt.

# %%
rel = by_level.div(by_level["Base (incoherent)"], axis=0).drop(columns="Base (incoherent)")
ax = rel.T.plot.barh(figsize=(12, 5.5), width=0.8)
ax.axvline(1, color="black", ls=":")
ax.set_xlabel("median RMSSE relative to the base forecast (<1 = better)")
ax.set_title("Reconciliation: gain or loss at each level")
plt.tight_layout()
plt.show()

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ### Reading the result
#
# * Each method's longest bar is its worst level, which the rule uses. Expect BottomUp to leave
#   the items unchanged (ratio 1) and TopDown to leave the total unchanged.
# * In the original lab, with ETS items, BottomUp made the total ≈ 29% worse. With GFM items,
#   is BottomUp's total ratio now ≈ __? <!-- TODO: fill after GPU run -->
# * Check the bias table too: a method can win on RMSSE and still under-forecast a level.
#
# *Exact numbers vary with seed and hardware.*
#
# ### One subclass and its department, before and after
#
# The subclass and department of the first high-volume example series: actuals, the ETS base
# forecast (dotted), and each coherent method (dashed).

# %%
example = baselines_cfg["examples"]["high"][0]
path = item_path.get(example, items[0])
nodes = {"subclass": path.rsplit("/", 1)[0], "department": "/".join(path.split("/")[:2])}
start = test_start - pd.Timedelta(weeks=52)
fig, axes = plt.subplots(1, 2, figsize=(15, 4.2))
for ax, (level, node) in zip(axes, nodes.items()):
    hist = Y_df[(Y_df["unique_id"] == node) & (Y_df["ds"] >= start)]
    ax.plot(hist["ds"], hist["y"], color=gplt.COLOURS["actual"], lw=1.4, label="actual")
    fc = Y_rec[Y_rec["unique_id"] == node]
    for i, m in enumerate(methods):
        ls = ":" if m == "Base (incoherent)" else "--"
        ax.plot(fc["ds"], fc[m], ls=ls, lw=1.4, color=gplt._colour(i), label=m)
    ax.axvspan(test_start, hist["ds"].max(), color=gplt.COLOURS["test"], zorder=0)
    ax.set(title=f"{level} {node.split('/')[-1]}", ylabel="units / week")
axes[0].legend(fontsize=7)
fig.tight_layout()
plt.show()

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 7. The item level on the board
#
# The reconciled item forecasts for the scored series, on the board's scale (52-week RMSSE).
# `GFM_BottomUp` is the base GFM itself (one seed), so it shows how far the saved seed is from
# the 3-seed mean on the board.

# %%
item_rows = Y_rec[Y_rec["unique_id"].isin(set(items))]
item_fc = item_rows[["unique_id", "ds"] + methods[1:]].copy()
item_fc["unique_id"] = item_fc["unique_id"].str.split("/").str[-1]
item_fc = item_fc.rename(columns={m: f"GFM_{m}" for m in methods[1:]})
item_fc = gd.only(item_fc, gd.eval_ids())
gd.save_forecasts(item_fc, "08_reconcile")
gd.record(item_fc, test, train, lab="08 hierarchy", fit_seconds=t_rec.seconds,
          notes="item level of the reconciled hierarchy; one saved seed")
gd.median_board()

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# The same items next to `GFM_selected`, the 3-seed mean on the board (F2: histogram,
# cumulative curve and box plot by class).

# %%
item_vs_board = item_fc.merge(gd.selected_forecasts(cfg), on=["unique_id", "ds"])
item_rmsse = gd.per_series(gd.score(item_vs_board, test, train))
gplt.plot_error_dist(item_rmsse, info)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 8. Apply the rule
#
# Each method's worst-level ratio, the tie band and the preference order from section 5.

# %%
worst = rel.max()
best = worst.min()
TIE_BAND = 1.01             # within 1% of the best worst-level ratio counts as a tie
tied = worst[worst <= best * TIE_BAND].index.tolist()
preference = ["MiddleOut", "BottomUp", "TopDown", "MinT-wls_struct", "MinT-ols"]
chosen = next(m for m in preference if m in tied)
print(f"worst-level ratio: {worst.round(3).to_dict()} | chosen: {chosen}")

# %%
gd.save_config("08_reconcile", {
    "selected_from": cfg.get("forecasts", "05_selected"),
    "bottom_model": cfg["saved_model"],
    "upper_model": "AutoETS(season_length=1), Naive fallback",
    "item_cap": {"multiple_of_train_max": CAP, "item_weeks": int(over.sum()),
                 "items": n_capped},
    "levels": {level: len(ids) for level, ids in zip(LEVELS, tags.values())},
    "methods": methods,
    "dropped_incoherent": incoherent,
    "left_out": {"MinTrace(mint_shrink)": "needs in-sample item errors from the GFM"},
    "median_rmsse_by_level": by_level.round(4).to_dict("index"),
    "mean_rmsse_by_level": by_level_mean.round(4).to_dict("index"),
    "bias_by_level": bias.round(4).to_dict("index"),
    "worst_level_ratio": worst.round(4).to_dict(),
    "rule": "smallest worst-level ratio to base (median RMSSE); ties within 1% -> MiddleOut",
    "chosen": chosen,
})

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## Discussion
#
# * With a strong item model, does BottomUp now do well at the top too? If not, look at its
#   total-level bias: small item biases add up across 17,000 items.
# * TopDown trusts ETS on the total and splits it by forecast proportions. At which levels does
#   that pay?
# * MinT-ols treats every node as equally reliable. With ≈17,000 items and one total, what does
#   that do to the items?
#
# **Try it:** swap the upper-level model for `AutoARIMA` with Fourier terms (a dynamic harmonic
# regression, as in the research), delete `08_upper_base.parquet`, and rerun with `GD_TRAIN=1`.
