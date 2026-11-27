"""Exploratory experiment: do hierarchy demand features help the best future-driver model of notebook 03?

Control: lgbm_future_price_levels (notebook 03): L2, LocalStandardScaler, notebook 02 LightGBM parameters,
lags and rolling means, the three categorical attributes, the full plan (promotion calendar, discount,
price index, regular and planned price). Each configuration adds the aggregate features of notebook 03c
(section 6) for one or more hierarchy levels; nothing else changes.

Nothing in day1_ml/, shared/ or results/day1/ is modified. Output goes to experiments/hierarchy_future/results/.

    python experiments/hierarchy_future/hierarchy_future.py check        # origin-safety check of the features
    python experiments/hierarchy_future/hierarchy_future.py models [names] [--seed N]
    python experiments/hierarchy_future/hierarchy_future.py evaluate
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from shared.prepare_data import load_panel, series_segments, H, FREQ, ALL_CUTOFFS, DEV_CUTOFFS, TEST_CUTOFF
from shared.features import (LAGS, LAG_TRANSFORMS, DATE_FEATURES, LGB_PARAMS, CATEGORICAL, PROMO_CALENDAR,
                             add_drivers, item_attributes, future_drivers, in_horizon)

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
CONTROL_COLS = PROMO_CALENDAR + ["discount", "price_index", "regular_price", "planned_price"]
CONTROL_MODEL = "lgbm_future_price_levels"
CONFIGS = {
    "lgbm_future_price_levels": [],
    "fpl_dept_agg": ["department"],
    "fpl_class_agg": ["class"],
    "fpl_subclass_agg": ["subclass"],
    "fpl_dept_class_agg": ["department", "class"],
    "fpl_all_agg": ["department", "class", "subclass"],
}


# ----------------------------------------------------------------------------- features
def aggregate_features(panel, attributes):
    """Notebook 03c, section 6, unchanged: hierarchy demand features per level, joined to every product-week."""
    membership = panel[["unique_id", "ds"]].merge(attributes, on="unique_id", how="left")
    out = panel[["unique_id", "ds"]].copy()
    aggregate_cols = {}
    for level in CATEGORICAL:
        weekly = panel.assign(group=membership[level]).groupby(["group", "ds"]).agg(total=("y", "sum"), promo_share=("promo", "mean")).reset_index()
        by_group = weekly.groupby("group")
        lagged = by_group.total.shift(H)
        rolled = lambda values, window, how: values.groupby(weekly.group).transform(lambda s: s.rolling(window, min_periods=1).agg(how))
        reference = lagged.groupby(weekly.group).transform(lambda s: s.rolling(52, min_periods=26).mean())

        group_features = pd.DataFrame({
            "group": weekly.group, "ds": weekly.ds,
            f"{level}_lag14": lagged / reference,
            f"{level}_lag18": by_group.total.shift(18) / reference,
            f"{level}_lag26": by_group.total.shift(26) / reference,
            f"{level}_lag52": by_group.total.shift(52) / reference,
            f"{level}_mean4": rolled(lagged, 4, "mean") / reference,
            f"{level}_mean13": rolled(lagged, 13, "mean") / reference,
            f"{level}_mean26": rolled(lagged, 26, "mean") / reference,
            f"{level}_cv13": rolled(lagged, 13, "std") / rolled(lagged, 13, "mean"),
            f"{level}_momentum": rolled(lagged, 4, "mean") / rolled(lagged, 26, "mean"),
            f"{level}_change": lagged / by_group.total.shift(18) - 1,
            f"{level}_promo_share13": rolled(by_group.promo_share.shift(H), 13, "mean"),
        }).replace([np.inf, -np.inf], np.nan)
        aggregate_cols[level] = list(group_features.columns[2:])
        out = out.assign(group=membership[level].values).merge(group_features, on=["group", "ds"], how="left").drop(columns="group")
    return out, aggregate_cols


def prepared():
    panel = add_drivers(load_panel())
    attributes = item_attributes(panel)
    plan = future_drivers(panel)[["unique_id", "ds", *CONTROL_COLS]]
    agg, agg_cols = aggregate_features(panel, attributes)
    frame = plan.merge(agg, on=["unique_id", "ds"], how="left")
    assert len(frame) == len(plan) == len(panel)
    return panel, attributes, frame, agg_cols


def config_cols(name, agg_cols):
    return CONTROL_COLS + [c for level in CONFIGS[name] for c in agg_cols[level]]


# ----------------------------------------------------------------------------- origin-safety check
def check():
    """Scramble sales and promotion flags after each cutoff: the aggregate features of its 14 forecast weeks must not change."""
    panel = add_drivers(load_panel())
    attributes = item_attributes(panel)
    base, cols = aggregate_features(panel, attributes)
    cols = [c for level in CATEGORICAL for c in cols[level]]
    rng = np.random.default_rng(0)
    rows = []
    for cutoff in ALL_CUTOFFS:
        scrambled = panel.copy()
        after = scrambled.ds > cutoff
        scrambled.loc[after, "y"] = rng.permutation(scrambled.loc[after, "y"].to_numpy()) * 3 + 7
        scrambled.loc[after, "promo"] = 1 - scrambled.loc[after, "promo"]
        changed, _ = aggregate_features(scrambled, attributes)
        a = in_horizon(base, cutoff).sort_values(["unique_id", "ds"])
        b = in_horizon(changed, cutoff).sort_values(["unique_id", "ds"])
        pd.testing.assert_frame_equal(a[cols].reset_index(drop=True), b[cols].reset_index(drop=True))
        # the features must also be available: share of non-missing values in the forecast weeks
        rows.append({"cutoff": cutoff.date(), "horizon_rows": len(a), **{f"nonmissing_{lvl}": a[[c for c in cols if c.startswith(lvl)]].notna().mean().mean()
                                                                       for lvl in CATEGORICAL}})
        print(f"{cutoff.date()}: horizon features unchanged after scrambling", flush=True)
    pd.DataFrame(rows).round(4).to_csv(OUT / "origin_safety.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string())


# ----------------------------------------------------------------------------- models
def run_models(names, seed=None):
    import lightgbm as lgb
    from mlforecast import MLForecast
    from mlforecast.target_transforms import LocalStandardScaler
    from shared.leaderboard import to_long

    panel, attributes, frame, agg_cols = prepared()
    actuals = panel[["unique_id", "ds", "y"]]
    as_category = {col: "category" for col in CATEGORICAL}
    params = LGB_PARAMS if seed is None else {**LGB_PARAMS, "random_state": seed}
    suffix = "" if seed is None else f"_seed{seed}"
    coverage = []
    for name in names:
        cols = config_cols(name, agg_cols)
        model = name + suffix
        out, start = [], time.perf_counter()
        for cutoff in ALL_CUTOFFS:
            history = actuals[actuals.ds <= cutoff]
            train = (history.merge(frame[["unique_id", "ds", *cols]], on=["unique_id", "ds"])
                     .merge(attributes, on="unique_id").astype(as_category))
            X_df = in_horizon(frame, cutoff)[["unique_id", "ds", *cols]]
            mlf = MLForecast(
                models={model: lgb.LGBMRegressor(**params)},
                freq=FREQ, lags=LAGS, lag_transforms=LAG_TRANSFORMS, date_features=DATE_FEATURES,
                target_transforms=[LocalStandardScaler()], num_threads=1,
            )
            # preprocess + fit_models is what fit() does; done in two steps to record the training rows
            prep = mlf.preprocess(train, static_features=CATEGORICAL)
            mlf.fit_models(prep[mlf.ts.features_order_], prep["y"].to_numpy())
            keys = pd.util.hash_pandas_object(prep[["unique_id", "ds"]], index=False).sum()
            coverage.append({"model": model, "cutoff": cutoff.date(), "train_rows": len(prep), "key_hash": int(keys) & 0xFFFFFFFF,
                             "n_features": len(mlf.ts.features_order_),
                             "agg_missing_share": float(prep[[c for c in cols if c not in CONTROL_COLS]].isna().mean().mean()) if len(cols) > len(CONTROL_COLS) else 0.0})
            out.append(to_long(mlf.predict(H, X_df=X_df), cutoff, "hierarchy_future", "future_drivers"))
        out = pd.concat(out)
        out.to_parquet(OUT / f"forecasts_{model}.parquet", index=False)
        print(f"{model}: {len(cols)} dynamic columns, {time.perf_counter() - start:.0f}s", flush=True)
        if name == CONTROL_MODEL and seed is None:
            check_control(out)
    cov = pd.DataFrame(coverage)
    path = OUT / "training_coverage.csv"
    if path.exists():
        old = pd.read_csv(path)
        cov = pd.concat([old[~old.model.isin(cov.model)], cov.astype({"cutoff": str})])
    cov.to_csv(path, index=False)


def check_control(control):
    saved = pd.read_parquet(ROOT / "results" / "day1" / "forecasts" / "03_exogenous.parquet").query("model == @CONTROL_MODEL")
    key = ["unique_id", "cutoff", "ds"]
    both = control.merge(saved, on=key, suffixes=("", "_saved"))
    assert len(both) == len(saved) == len(control)
    diff = (both.yhat - both.yhat_saved).abs().max()
    print(f"control vs saved {CONTROL_MODEL}: {len(both):,} rows, max |difference| = {diff}")
    pd.Series({"rows": len(both), "max_abs_difference": float(diff)}).to_csv(OUT / "control_reproduction.csv")


# ----------------------------------------------------------------------------- evaluation
def evaluate():
    from shared.evaluation import score, common_rows
    from shared.leaderboard import leaderboard

    panel = add_drivers(load_panel())
    actuals = panel[["unique_id", "ds", "y"]]
    files = sorted(OUT.glob("forecasts_*.parquet"))
    fcst = common_rows(pd.concat([pd.read_parquet(f) for f in files], ignore_index=True))
    seeded = lambda m: m.rpartition("_seed")[2] if "_seed" in m else "0"
    control_of = lambda m: CONTROL_MODEL + ("" if seeded(m) == "0" else f"_seed{seeded(m)}")
    models = [m for m in fcst.model.unique()]

    dev = leaderboard(fcst, actuals, scope="dev").set_index("model")
    final = leaderboard(fcst, actuals, scope="test").set_index("model")
    by_origin = pd.concat([score(fcst[fcst.cutoff == c], actuals, rmsse=False).assign(cutoff=c.date()) for c in list(DEV_CUTOFFS) + [TEST_CUTOFF]]
                          ).pivot(index="model", columns="cutoff", values="wape")
    dev_cols = [c.date() for c in DEV_CUTOFFS]
    change = 100 * (by_origin - by_origin.loc[[control_of(m) for m in by_origin.index]].to_numpy())

    table = pd.DataFrame({
        "dev_wape": dev.wape, "final_wape": final.wape,
        "dev_gain_pts": -100 * (dev.wape - dev.wape.loc[[control_of(m) for m in dev.index]].to_numpy()),
        "final_gain_pts": -100 * (final.wape - final.wape.loc[[control_of(m) for m in final.index]].to_numpy()),
        "dev_wins": change[dev_cols].lt(0).sum(axis=1),
        "dev_bias": dev.bias, "final_bias": final.bias,
        "dev_mean_rmsse": dev.mean_rmsse, "final_mean_rmsse": final.mean_rmsse,
        "dev_apc": dev.apc, "final_apc": final.apc, "dev_rows": dev.rows,
    })
    table["accepted"] = (table.dev_gain_pts >= 0.2) & (table.dev_wins >= 5)
    is_control = table.index.map(lambda m: m == control_of(m))
    table.loc[is_control, ["dev_wins", "accepted"]] = np.nan
    order = sorted(table.index, key=lambda m: (seeded(m) != "0", seeded(m), m != control_of(m), list(CONFIGS).index(m.rpartition("_seed")[0] if "_seed" in m else m)))
    table = table.loc[order]
    table.round(4).to_csv(OUT / "summary.csv")
    print(table.round(4).to_string())
    print()
    change.loc[order].round(2).to_csv(OUT / "wape_change_by_origin_pts.csv")
    print("WAPE change vs control by origin, points (negative = better):")
    print(change.loc[order].round(2).to_string())

    # volume segments, recomputed at each development origin from history up to it (as notebook 03c)
    segs = pd.concat([series_segments(panel, c).assign(cutoff=c).rename_axis("unique_id").reset_index() for c in DEV_CUTOFFS])
    rows = fcst[fcst.cutoff.isin(DEV_CUTOFFS)].merge(segs[["unique_id", "cutoff", "volume"]], on=["unique_id", "cutoff"])
    seg = score(rows, actuals, by=["volume"], rmsse=False)
    wape = seg.pivot(index="model", columns="volume", values="wape")[["low", "mid", "high"]].loc[order]
    bias = seg.pivot(index="model", columns="volume", values="bias")[["low", "mid", "high"]].loc[order]
    seg_change = 100 * (wape - wape.loc[[control_of(m) for m in wape.index]].to_numpy())
    seg_wins = {}
    for v in ["low", "mid", "high"]:
        part = rows[rows.volume == v]
        o = pd.concat([score(part[part.cutoff == c], actuals, rmsse=False).assign(cutoff=c) for c in DEV_CUTOFFS]).pivot(index="model", columns="cutoff", values="wape")
        seg_wins[v] = o.lt(o.loc[[control_of(m) for m in o.index]].to_numpy()).sum(axis=1)
    segments = pd.concat({"dev_wape": wape, "change_pts": seg_change, "wins_of_7": pd.DataFrame(seg_wins).loc[order], "bias": bias}, axis=1)
    segments.round(4).to_csv(OUT / "dev_volume_segments.csv")
    print()
    print("Development, by volume segment:")
    print(segments.round(4).to_string())


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "check":
        check()
    elif stage == "models":
        args = sys.argv[2:]
        seed = None
        if "--seed" in args:
            i = args.index("--seed")
            seed = int(args[i + 1])
            args = args[:i] + args[i + 2:]
        run_models(args or list(CONFIGS), seed)
    elif stage == "evaluate":
        evaluate()
    else:
        raise SystemExit(__doc__)
