"""Scoring for the Day 1 forecast table, on Nixtla's utilsforecast.

Forecasts are long: unique_id, cutoff, ds, model, yhat. The cutoff is the last week of
data available when the forecast was made. Actuals (unique_id, ds, y) are joined on
unique_id and ds. evaluate() wants one column per model, so `to_wide` pivots first.

`score` is the one scoring function. Every model in the table is scored on the same rows.

- WAPE is `nd` (sum|forecast - actual| / sum|actual|) and MAE is `mae`, computed on a whole
  group of rows at once: the group is passed to evaluate() as its id, so the result is the
  portfolio (or segment) value, not an average of SKU values.
- Bias is `cfe` (sum of forecast - actual) divided by the sum of actuals in the same group.
- RMSSE is `rmsse` with a 52-week seasonal scale built from the weeks up to each cutoff,
  one value per series and cutoff, then averaged.

Undefined values differ between utilsforecast versions (0 up to 0.2.14; NaN or inf from 0.2.15),
so they are set here: a group with no sales gets no WAPE, and a series with no usable seasonal
scale (under 53 weeks, or never different from the same week a year earlier) gets no RMSSE.

APC is custom because it compares forecasts from two origins; see `apc`.
"""

from functools import partial

import pandas as pd
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import cfe, mae, nd, rmsse

SEASON = 52


def to_wide(rows, extra=()):
    """Long rows (model, yhat) to one column per model, the layout evaluate() expects.

    The cutoff column is dropped after the pivot: from utilsforecast 0.2.15, evaluate() scores
    each cutoff separately when a "cutoff" column is present, and here the rows are pooled.
    """
    wide = rows.pivot(index=["unique_id", "cutoff", "ds", "y", *extra], columns="model", values="yhat").reset_index()
    return wide.drop(columns="cutoff")


def common_rows(fcst):
    """Keep only unique_id/cutoff/ds rows that every model in the table forecasts."""
    n_models = fcst.groupby(["unique_id", "cutoff", "ds"])["model"].transform("nunique")
    return fcst[n_models == fcst["model"].nunique()]


def coverage(fcst):
    """Share of all forecast rows (any model) that each model provides."""
    universe = fcst[["unique_id", "cutoff", "ds"]].drop_duplicates().shape[0]
    return (fcst.groupby("model").size() / universe).rename("coverage")


def series_rmsse(rows, actuals):
    """utilsforecast RMSSE per series, cutoff and model, scaled on the history up to each cutoff."""
    out = []
    for cutoff, group in rows.groupby("cutoff"):
        train = actuals[actuals["ds"] <= cutoff].sort_values(["unique_id", "ds"])
        wide = to_wide(group)
        models = sorted(group["model"].unique())
        result = evaluate(wide, [partial(rmsse, seasonality=SEASON)], models=models, train_df=train).set_index("unique_id")

        yearly_change = (train["y"] - train.groupby("unique_id")["y"].shift(SEASON)).abs()
        eligible = (yearly_change > 0).groupby(train["unique_id"]).any().reindex(result.index)
        result.loc[~eligible.to_numpy(), models] = float("nan")
        out.append(result[models].reset_index().melt(id_vars="unique_id", var_name="model", value_name="rmsse").assign(cutoff=cutoff))
    return pd.concat(out)


def score(fcst, actuals, by=(), rmsse=True):
    """WAPE, accuracy, bias, MAE and RMSSE per model (and per group in `by`), on rows every model forecasts.

    `by` lists extra grouping columns already in fcst. Pass rmsse=False when a group splits a
    series' forecast window (such as horizon), because RMSSE belongs to the whole window.
    """
    by = list(by)
    rows = common_rows(fcst).merge(actuals[["unique_id", "ds", "y"]], on=["unique_id", "ds"])
    rows["group"] = rows.groupby(by, observed=True).ngroup() if by else 0
    models = sorted(rows["model"].unique())

    pooled = evaluate(to_wide(rows, extra=["group"]), [mae, nd, cfe], models=models, id_col="group")
    table = pooled.melt(id_vars=["group", "metric"], var_name="model").pivot(index=["group", "model"], columns="metric", values="value")
    total_actual = rows[rows["model"] == models[0]].groupby("group")["y"].sum()       # same rows for every model
    actual = table.index.get_level_values("group").map(total_actual).to_numpy()
    table["wape"] = table["nd"].where(actual > 0)
    table["accuracy"] = 1 - table["wape"]
    table["bias"] = table["cfe"] / actual
    table = table.join(rows.groupby(["group", "model"]).agg(rows=("y", "size"), series=("unique_id", "nunique")))

    if rmsse:
        per_series = series_rmsse(rows, actuals).merge(rows[["unique_id", "cutoff", "group"]].drop_duplicates(), on=["unique_id", "cutoff"])
        table = table.join(per_series.groupby(["group", "model"])["rmsse"].agg(mean_rmsse="mean", median_rmsse="median", eligible_series="count"))

    labels = rows.drop_duplicates("group").set_index("group")[by]
    columns = ["rows", "series", "mae", "wape", "accuracy", "bias"] + (["mean_rmsse", "median_rmsse", "eligible_series"] if rmsse else [])
    return table[columns].reset_index().merge(labels, left_on="group", right_index=True)[by + ["model"] + columns]


def apc_blocks(fcst, new_cutoffs, gap_days=28, block_weeks=4, n_blocks=2):
    """Forecast revisions per model, SKU, new cutoff and four-week block.

    The new forecast (cutoff in new_cutoffs) is compared with the forecast for the same target
    weeks made gap_days earlier. Targets are counted in weeks from the new cutoff. With 14-week
    forecasts four weeks apart, 10 weeks overlap: two complete blocks of block_weeks weeks are
    kept and the last two overlapping weeks are dropped.
    Columns: change = sum|new - old|, base = sum|old|.
    """
    new = fcst[fcst["cutoff"].isin(new_cutoffs)]
    old = fcst.assign(cutoff=fcst["cutoff"] + pd.Timedelta(days=gap_days)).rename(columns={"yhat": "yhat_old"})
    pairs = new.merge(old, on=["unique_id", "cutoff", "ds", "model"])
    pairs["week"] = (pairs["ds"] - pairs["cutoff"]).dt.days // 7
    pairs = pairs[pairs["week"].between(1, block_weeks * n_blocks)]
    pairs["block"] = (pairs["week"] - 1) // block_weeks
    pairs["change"] = (pairs["yhat"] - pairs["yhat_old"]).abs()
    pairs["base"] = pairs["yhat_old"].abs()

    blocks = pairs.groupby(["model", "unique_id", "cutoff", "block"]).agg(
        change=("change", "sum"), base=("base", "sum"), weeks=("week", "size")
    ).reset_index()
    return blocks[blocks["weeks"] == block_weeks].drop(columns="weeks")


def apc(fcst, new_cutoffs, **kwargs):
    """Absolute percentage change between forecasts of the same target weeks.

        apc = sum|new - old| / sum|old|   within one SKU block

    Every model is scored on the same SKU, new cutoff and block: blocks that are incomplete or
    have sum|old| = 0 for any model are dropped for all models. Models without forecasts from
    both origins are left out of that population and get no row, so their APC is unavailable
    rather than zero. Portfolio APC pools the blocks; median APC takes each SKU's pooled blocks
    and the median over SKUs. apc_coverage is the share of a model's complete blocks that were
    scored; zero_base counts its blocks with sum|old| = 0.
    """
    blocks = apc_blocks(fcst, new_cutoffs, **kwargs)
    keys = ["unique_id", "cutoff", "block"]
    defined = blocks[blocks["base"] > 0]
    shared = defined.groupby(keys).size()
    defined = defined.merge(shared[shared == blocks["model"].nunique()].reset_index()[keys], on=keys)

    portfolio = defined.groupby("model")[["change", "base"]].sum()
    sku = defined.groupby(["model", "unique_id"])[["change", "base"]].sum()
    sku["apc"] = sku["change"] / sku["base"]
    out = pd.DataFrame({
        "apc": portfolio["change"] / portfolio["base"],
        "apc_median": sku.groupby("model")["apc"].median(),
        "apc_coverage": defined.groupby("model").size() / blocks.groupby("model").size(),
        "zero_base": blocks[blocks["base"] == 0].groupby("model").size(),
    })
    out["apc_coverage"] = out["apc_coverage"].fillna(0)
    out["zero_base"] = out["zero_base"].fillna(0).astype(int)
    return out.reset_index()
