"""Research experiments removed from notebook 04 (global vs specialized models). Not part of the teaching path.

    scaling    target scaling and the earlier department models: lgbm_global_minmax, lgbm_dept_raw, lgbm_dept_std
               (departments under 30 products merged into "Other", no global fallback)
    own        partitions against the global model on only the products that got their own model
    segments   WAPE and bias by volume and intermittency (segments recomputed at each development cutoff)
    cutoffs    WAPE at every development cutoff and the final cutoff
    limited    400 products with history shortened to 13 or 26 weeks: global LightGBM against HistoricAverage

Run from the repository root with the notebook environment (packt_2day_cohort_2026):

    python experiments/04_research/research_04.py scaling limited [--save]
    python experiments/04_research/research_04.py own segments cutoffs

`scaling` and `limited` refit their models. With --save they write them to results/day1/forecasts/:
`scaling` replaces its three models in 04_global_vs_specialized (notebook 07 uses them as candidates; notebook 04
writes lgbm_velocity_std to the same file), and `limited` writes 04_limited_history. Without --save nothing is written.
`own`, `segments` and `cutoffs` read the saved forecasts of notebooks 01 to 04 and fit nothing.

Compute, last measured in the notebook (fit and 14-week forecast summed over ten cutoffs, one thread):
department 96 s (65 models), class 76 s (63), subclass 25 s (23), volume tiers 12 s (3), global about 10 s (1).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import lightgbm as lgb
from mlforecast import MLForecast
from mlforecast.target_transforms import LocalStandardScaler, LocalMinMaxScaler
from statsforecast import StatsForecast
from statsforecast.models import HistoricAverage

from shared.prepare_data import load_panel, series_segments, H, FREQ, ALL_CUTOFFS, DEV_CUTOFFS, TEST_CUTOFF
from shared.features import LAGS, LAG_TRANSFORMS, DATE_FEATURES, LGB_PARAMS
from shared.evaluation import score
from shared.leaderboard import to_long, save_forecasts, load_forecasts, leaderboard

MIN_SERIES = 10
SCALING_MODELS = ["lgbm_global_minmax", "lgbm_dept_raw", "lgbm_dept_std"]
PARTITIONS = ["lgbm_department_part", "lgbm_class_part", "lgbm_subclass_part", "lgbm_velocity_std"]

panel = load_panel()
actuals = panel[["unique_id", "ds", "y"]]
items = panel.drop_duplicates("unique_id").set_index("unique_id")
hierarchy = {level: items[f"{level}_id"].astype("string").fillna("Unknown") for level in ["department", "class", "subclass"]}
pd.set_option("display.width", 200)


def fit_groups(history, membership, name, target_transforms):
    """One MLForecast model per group of series, no fallback; the forecasts stacked."""
    forecasts = []
    for _, ids in membership.groupby(membership).groups.items():
        mlf = MLForecast(
            models={name: lgb.LGBMRegressor(**LGB_PARAMS)},
            freq=FREQ, lags=LAGS, lag_transforms=LAG_TRANSFORMS, date_features=DATE_FEATURES,
            target_transforms=target_transforms, num_threads=1,
        )
        mlf.fit(history[history.unique_id.isin(ids)])
        forecasts.append(mlf.predict(H))
    return pd.concat(forecasts)


def scaling(save):
    department = items.department_name.fillna("Unknown")
    department = department.where(department.map(department.value_counts()) >= 30, "Other")
    out = []
    for cutoff in ALL_CUTOFFS:
        history = actuals[actuals.ds <= cutoff]
        everyone = pd.Series("all", index=history.unique_id.unique())
        runs = {
            "lgbm_global_minmax": (everyone, [LocalMinMaxScaler()]),
            "lgbm_dept_raw": (department, []),
            "lgbm_dept_std": (department, [LocalStandardScaler()]),
        }
        for name, (membership, transforms) in runs.items():
            out.append(to_long(fit_groups(history, membership, name, transforms), cutoff, "04_global_vs_specialized"))
    out = pd.concat(out)
    rows = pd.concat([load_forecasts(["02_ml_baseline"]).query("model in ['lgbm_raw', 'lgbm_baseline']"), out])
    print(pd.concat({"development": leaderboard(rows, actuals, scope="dev").set_index("model")[["wape", "bias", "mean_rmsse", "apc"]],
                     "final 14 weeks": leaderboard(rows, actuals, scope="test").set_index("model")[["wape", "bias"]]}, axis=1).round(3))
    if save:
        saved = load_forecasts(["04_global_vs_specialized"])
        save_forecasts(pd.concat([saved[~saved.model.isin(SCALING_MODELS)], out]), "04_global_vs_specialized")


def saved_candidates():
    reference = load_forecasts(["01_baselines", "02_ml_baseline", "03_exogenous"])
    reference = reference[reference.model.isin(["SeasonalNaive", "HistoricAverage", "lgbm_baseline", "lgbm_categorical"])]
    return pd.concat([reference, load_forecasts(["04_global_vs_specialized", "04_partition_granularity"])])


def own(candidates):
    rows = []
    for level, membership in hierarchy.items():
        for cutoff in DEV_CUTOFFS:
            series = actuals[actuals.ds <= cutoff].unique_id.unique()
            sizes = membership[membership.index.isin(series)].value_counts()
            trained = membership[membership.isin(sizes[sizes >= MIN_SERIES].index) & membership.index.isin(series)].index
            pair = candidates[candidates.model.isin([f"lgbm_{level}_part", "lgbm_baseline"]) & (candidates.cutoff == cutoff) & candidates.unique_id.isin(trained)]
            rows.append(pair.assign(level=level, model=np.where(pair.model == "lgbm_baseline", "global", "partitioned")))
    rows = pd.concat(rows)
    table = pd.concat([score(g, actuals).assign(level=level) for level, g in rows.groupby("level")])
    print(table.pivot(index="level", columns="model", values=["series", "wape", "bias", "mean_rmsse"]).loc[list(hierarchy)].round(3))


def segments(candidates):
    origin = pd.concat([series_segments(panel, c).assign(cutoff=c) for c in DEV_CUTOFFS]).rename_axis("unique_id").reset_index()
    rows = candidates[candidates.cutoff.isin(DEV_CUTOFFS)].merge(origin[["unique_id", "cutoff", "volume", "intermittency"]], on=["unique_id", "cutoff"])
    by_volume = score(rows, actuals, by=["volume"], rmsse=False)
    by_zeros = score(rows, actuals, by=["intermittency"], rmsse=False)
    print(pd.concat({
        "wape by volume": by_volume.pivot(index="model", columns="volume", values="wape")[["low", "mid", "high"]],
        "bias by volume": by_volume.pivot(index="model", columns="volume", values="bias")[["low", "mid", "high"]],
        "wape by intermittency": by_zeros.pivot(index="model", columns="intermittency", values="wape")[["regular", "some_zeros", "zero_heavy"]],
        "bias by intermittency": by_zeros.pivot(index="model", columns="intermittency", values="bias")[["regular", "some_zeros", "zero_heavy"]],
    }, axis=1).round(3).to_string())


def cutoffs(candidates):
    lgbm = candidates[candidates.model.str.startswith("lgbm")]
    per = pd.concat([score(lgbm[lgbm.cutoff == c], actuals, rmsse=False).assign(cutoff=c.date()) for c in list(DEV_CUTOFFS) + [TEST_CUTOFF]])
    print(per.pivot(index="model", columns="cutoff", values="wape").round(3).to_string())


def limited(save):
    focal = pd.Series(sorted(panel.unique_id.unique())).sample(400, random_state=7)
    runs = []
    for weeks in [13, 26]:
        for cutoff in list(DEV_CUTOFFS) + [TEST_CUTOFF]:
            history = actuals[actuals.ds <= cutoff]
            truncated = history[~history.unique_id.isin(focal) | (history.ds > cutoff - pd.Timedelta(weeks=weeks))]
            mlf = MLForecast(
                models={f"lgbm_global_last{weeks}": lgb.LGBMRegressor(**LGB_PARAMS)},
                freq=FREQ, lags=LAGS, lag_transforms=LAG_TRANSFORMS, date_features=DATE_FEATURES,
                target_transforms=[LocalStandardScaler()], num_threads=1,
            )
            mlf.fit(truncated)
            preds = mlf.predict(H)
            preds = preds[preds.unique_id.isin(focal)]
            sf = StatsForecast(models=[HistoricAverage(alias=f"HistoricAverage_last{weeks}")], freq=FREQ, n_jobs=1)
            local = sf.forecast(df=truncated[truncated.unique_id.isin(focal)], h=H)
            runs.append(to_long(preds.merge(local, on=["unique_id", "ds"]), cutoff, "04_limited_history"))
    out = pd.concat(runs).drop_duplicates(["unique_id", "cutoff", "ds", "model"])
    full = load_forecasts(["01_baselines", "02_ml_baseline"])
    full = full[full.model.isin(["lgbm_baseline", "HistoricAverage"]) & full.unique_id.isin(focal)]
    rows = pd.concat([out, full]).query("cutoff in @DEV_CUTOFFS.tolist()")
    print(score(rows, actuals).sort_values("wape")[["model", "series", "wape", "bias", "mean_rmsse", "median_rmsse"]].round(3))
    if save:
        save_forecasts(out, "04_limited_history")


if __name__ == "__main__":
    args = sys.argv[1:]
    save = "--save" in args
    stages = [a for a in args if a != "--save"] or ["own", "segments", "cutoffs"]
    candidates = saved_candidates() if {"own", "segments", "cutoffs"} & set(stages) else None
    for stage in stages:
        print(f"\n=== {stage}")
        {"scaling": lambda: scaling(save), "limited": lambda: limited(save), "own": lambda: own(candidates),
         "segments": lambda: segments(candidates), "cutoffs": lambda: cutoffs(candidates)}[stage]()
