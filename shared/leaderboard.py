"""Save, load and rank Day 1 forecasts in the common long format.

Columns: unique_id, cutoff, ds, model, yhat, experiment, information_tier.

information_tier, from least to most information used:
    historical_only    only data observed on or before the cutoff
    planned_scenario   also a promotion plan, here stood in for by realized promotion flags in the
                       first PLAN_WEEKS forecast weeks (an assumption, not real plan data)
    oracle             realized future promotions for the whole horizon
    leaky_diagnostic   realized future selling prices (which reveal sales) or test-chosen models
A selection strategy or average takes the highest tier among its candidates.
"""

from pathlib import Path

import pandas as pd

from shared.evaluation import apc, coverage, score
from shared.prepare_data import DEV_CUTOFFS, TEST_CUTOFF

FOLDER = Path(__file__).resolve().parents[1] / "results" / "day1"
KEY = ["unique_id", "cutoff", "ds", "model"]
TIERS = ["historical_only", "planned_scenario", "oracle", "leaky_diagnostic"]


def combine_tiers(tiers):
    """Tier of a strategy that can use any of these candidates."""
    return max(tiers, key=TIERS.index)


def to_long(wide, cutoff, experiment, information_tier="historical_only"):
    """Convert a wide prediction frame (unique_id, ds, one column per model) to the shared format."""
    long = wide.melt(id_vars=["unique_id", "ds"], var_name="model", value_name="yhat")
    long["yhat"] = long["yhat"].clip(lower=0)
    return long.assign(cutoff=cutoff, experiment=experiment, information_tier=information_tier)


def save_forecasts(fcst, name):
    """Write forecasts to results/day1/forecasts/<name>.parquet."""
    (FOLDER / "forecasts").mkdir(parents=True, exist_ok=True)
    assert fcst["yhat"].notna().all(), "drop rows a model cannot forecast instead of saving NaN"
    cols = KEY + ["yhat", "experiment", "information_tier"]
    fcst[cols].astype({"yhat": "float32"}).to_parquet(FOLDER / "forecasts" / f"{name}.parquet", index=False)


def load_forecasts(names=None):
    """Load saved forecasts. names is a list of file stems; None loads everything."""
    files = sorted((FOLDER / "forecasts").glob("*.parquet"))
    files = [f for f in files if names is None or f.stem in names]
    fcst = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    assert not fcst.duplicated(KEY).any(), "same model saved twice"
    return fcst


COLUMNS = ["model", "information_tier", "experiment", "wape", "accuracy", "bias", "mean_rmsse", "median_rmsse",
           "apc", "apc_median", "apc_coverage", "coverage", "eligible_series", "series", "rows", "mae", "zero_base"]


def leaderboard(fcst, actuals, scope="test"):
    """The canonical leaderboard: one row per model or strategy, ordered by tier, then WAPE.

    scope="test" scores the final cutoff and measures APC on the two 28-day pairs ending there.
    scope="dev" scores the development cutoffs; a list of cutoffs scores those. Accuracy is on
    rows every model forecasts, APC on blocks every model has; coverage is measured beforehand.
    """
    if scope == "test":
        cutoffs, new_cutoffs = [TEST_CUTOFF], [TEST_CUTOFF, TEST_CUTOFF - pd.Timedelta(days=28)]
    else:
        cutoffs = list(DEV_CUTOFFS) if scope == "dev" else list(scope)
        new_cutoffs = cutoffs[1:]

    scored = fcst[fcst["cutoff"].isin(cutoffs)]
    meta = fcst.drop_duplicates("model")[["model", "experiment", "information_tier"]]
    table = (score(scored, actuals)
             .merge(coverage(scored).reset_index(), on="model")
             .merge(apc(fcst, new_cutoffs), on="model", how="left")
             .merge(meta, on="model"))
    table["zero_base"] = table["zero_base"].astype("Int64")
    table = table.sort_values(["information_tier", "wape"], key=lambda c: c.map(TIERS.index) if c.name == "information_tier" else c)
    return table[COLUMNS].reset_index(drop=True)


def breakdown(fcst, actuals, by, scope="test", segments=None):
    """Scores by horizon or by a segment column (volume, intermittency, department)."""
    rows = fcst[fcst["cutoff"].isin([TEST_CUTOFF] if scope == "test" else list(DEV_CUTOFFS))]
    rows = rows.assign(horizon=(rows["ds"] - rows["cutoff"]).dt.days // 7)
    if segments is not None:
        rows = rows.merge(segments, left_on="unique_id", right_index=True)
    return score(rows, actuals, by=[by], rmsse=by != "horizon")   # RMSSE belongs to a whole forecast window
