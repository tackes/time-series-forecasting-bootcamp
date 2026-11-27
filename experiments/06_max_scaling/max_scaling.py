"""Research experiment, moved out of notebook 06 (formerly its optional section 8): a non-negative scale for Tweedie.

The z-score scaler of notebook 02 makes the target negative, so Tweedie was compared only on raw units.
Here every series is divided by its own training maximum, y / max(y), which keeps zeros at zero, and
L1 and Tweedie are fit on that scale. The findings are in REPORT.md.

The saved forecasts in results/day1/forecasts/06_max_scaled.parquet came from this code. This script does
not overwrite them: it refits the two models, checks that they reproduce the saved file, and writes its
tables to experiments/06_max_scaling/results/.

Run from the repository root with the packt_2day_cohort_2026 environment (about a minute):

    python experiments/06_max_scaling/max_scaling.py
"""

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from mlforecast import MLForecast
from mlforecast.target_transforms import BaseTargetTransform

from shared.prepare_data import load_panel, series_segments, H, FREQ, ALL_CUTOFFS, DEV_CUTOFFS, TEST_CUTOFF
from shared.features import LAGS, LAG_TRANSFORMS, DATE_FEATURES, LGB_PARAMS
from shared.evaluation import score
from shared.leaderboard import to_long, load_forecasts, leaderboard, FOLDER

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)


class LocalMaxScaler(BaseTargetTransform):
    """y / max(y) per series, from the training rows only. A series that never sold is divided by 1."""

    def fit_transform(self, df):
        self.scale_ = df.groupby(self.id_col)[self.target_col].max().where(lambda m: m > 0, 1.0)
        return df.assign(**{self.target_col: df[self.target_col] / df[self.id_col].map(self.scale_)})

    def inverse_transform(self, df):
        scale = df[self.id_col].map(self.scale_)
        models = df.columns.drop([self.id_col, self.time_col])
        return df.assign(**{m: df[m] * scale for m in models})


def fit_max_scaled(actuals):
    models = [
        MLForecast(models={"lgbm_l1_max": lgb.LGBMRegressor(**LGB_PARAMS, objective="l1")}, freq=FREQ, lags=LAGS,
                   lag_transforms=LAG_TRANSFORMS, date_features=DATE_FEATURES, target_transforms=[LocalMaxScaler()], num_threads=1),
        MLForecast(models={"lgbm_tweedie_max": lgb.LGBMRegressor(**LGB_PARAMS, objective="tweedie", tweedie_variance_power=1.3)}, freq=FREQ,
                   lags=LAGS, lag_transforms=LAG_TRANSFORMS, date_features=DATE_FEATURES, target_transforms=[LocalMaxScaler()], num_threads=1),
    ]
    forecasts, checks = [], []
    for cutoff in ALL_CUTOFFS:
        train = actuals[actuals.ds <= cutoff]
        for mlf in models:
            mlf.fit(train)
            forecasts.append(to_long(mlf.predict(H), cutoff, "06_max_scaled"))
        scale = mlf.ts.target_transforms[0].scale_
        scaled = train.y / train.unique_id.map(scale)
        checks.append(dict(cutoff=cutoff.date(), series=len(scale), never_sold=int((train.groupby("unique_id").y.max() == 0).sum()),
                           smallest_scale=scale.min(), scaled_min=scaled.min(), scaled_max=scaled.max()))
    return pd.concat(forecasts), pd.DataFrame(checks)


def origin_segments(panel, cutoff):
    segments = series_segments(panel, cutoff)
    share = panel[panel.ds <= cutoff].groupby("unique_id").is_promotional.mean()
    segments["promotion"] = pd.cut(share, [-1, 0.05, 0.25, 1], labels=["rarely", "occasionally", "heavily"]).astype(str)
    return segments.assign(cutoff=cutoff).rename_axis("unique_id").reset_index()


def main():
    panel = load_panel()
    actuals = panel[["unique_id", "ds", "y"]]
    max_forecasts, checks = fit_max_scaled(actuals)
    checks.to_csv(OUT / "scale_checks.csv", index=False)

    saved = pd.read_parquet(FOLDER / "forecasts" / "06_max_scaled.parquet")
    key = ["unique_id", "cutoff", "ds", "model"]
    both = saved.merge(max_forecasts, on=key, suffixes=("_saved", "_new"))
    print(f"reproduction: {len(both)} of {len(saved)} rows matched, max abs difference {(both.yhat_saved - both.yhat_new).abs().max():.4f}")

    design = pd.DataFrame({"scale": ["z-score", "max", "max", "raw", "raw"], "objective": ["L1", "L1", "Tweedie 1.3", "L1", "Tweedie 1.3"]},
                          index=["lgbm_l1_std", "lgbm_l1_max", "lgbm_tweedie_max", "lgbm_l1_raw", "lgbm_tweedie_raw"])
    candidates = pd.concat([load_forecasts(["06_objectives"]).query("model in @design.index"), max_forecasts])

    dev = leaderboard(candidates, actuals, scope="dev").set_index("model")
    final = leaderboard(candidates, actuals, scope="test").set_index("model")
    table = design.join(dev[["wape", "mae", "bias", "mean_rmsse", "median_rmsse", "apc", "coverage"]])
    table = table.join(final[["wape", "mae", "bias", "mean_rmsse"]].add_prefix("final_"))
    table.round(4).to_csv(OUT / "summary.csv")
    print(table.round(3))

    origins = list(DEV_CUTOFFS) + [TEST_CUTOFF]
    by_origin = pd.concat([score(candidates[candidates.cutoff == c], actuals, rmsse=False).assign(cutoff=c.date()) for c in origins])
    wape = by_origin.pivot(index="model", columns="cutoff", values="wape").loc[design.index]
    contrasts = {"Tweedie max - L1 max": ("lgbm_tweedie_max", "lgbm_l1_max"), "L1 max - L1 z-score": ("lgbm_l1_max", "lgbm_l1_std"),
                 "Tweedie max - L1 z-score": ("lgbm_tweedie_max", "lgbm_l1_std"), "Tweedie max - Tweedie raw": ("lgbm_tweedie_max", "lgbm_tweedie_raw")}
    change = pd.DataFrame({name: (wape.loc[a] - wape.loc[b]) * 100 for name, (a, b) in contrasts.items()}).T
    dev_cols = [c.date() for c in DEV_CUTOFFS]
    change = change.assign(dev_mean=change[dev_cols].mean(axis=1), dev_sd=change[dev_cols].std(axis=1), better_at=(change[dev_cols] < 0).sum(axis=1))
    change.round(3).to_csv(OUT / "wape_change_by_origin.csv")
    print(change.round(2))

    rows = candidates[candidates.cutoff.isin(DEV_CUTOFFS)].merge(pd.concat([origin_segments(panel, c) for c in DEV_CUTOFFS]), on=["unique_id", "cutoff"])
    tables = {}
    for segment, order in [("volume", ["low", "mid", "high"]), ("intermittency", ["regular", "some_zeros", "zero_heavy"]), ("promotion", ["rarely", "occasionally", "heavily"])]:
        scores = score(rows, actuals, by=[segment], rmsse=False)
        tables[(segment, "wape")] = scores.pivot(index="model", columns=segment, values="wape")[order]
        tables[(segment, "bias")] = scores.pivot(index="model", columns=segment, values="bias")[order]
    pd.concat(tables, axis=1).loc[design.index].round(4).to_csv(OUT / "dev_segments.csv")

    # retrospective: weeks whose actual sales exceeded three times the product's median up to the cutoff
    medians = pd.concat([actuals[actuals.ds <= c].groupby("unique_id").y.median().rename("median").reset_index().assign(cutoff=c) for c in DEV_CUTOFFS])
    spike_rows = candidates[candidates.cutoff.isin(DEV_CUTOFFS)].merge(medians, on=["unique_id", "cutoff"]).merge(actuals, on=["unique_id", "ds"])
    spike_rows["week"] = np.where(spike_rows.y > 3 * spike_rows["median"], "high", "other")
    spikes = score(spike_rows.drop(columns=["y", "median"]), actuals, by=["week"], rmsse=False)
    pd.concat({m: spikes.pivot(index="model", columns="week", values=m) for m in ["wape", "bias", "rows"]}, axis=1).loc[design.index].round(4).to_csv(OUT / "dev_high_weeks.csv")


if __name__ == "__main__":
    main()
