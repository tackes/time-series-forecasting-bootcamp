"""Save, load and rank Day 1 forecasts in the common long format.

Columns: unique_id, cutoff, ds, model, yhat, experiment, information_tier.

information_tier records what a forecast needs to know on its cutoff date, from least to most:
    history                  sales history, calendar and fixed item attributes
    historical_drivers       also past prices and promotions, observed by the cutoff
    partial_future_drivers   also a promotion plan for the first few forecast weeks
    future_drivers           also planned prices and promotions for the whole horizon (realized values
                             stand in for the plan, which this dataset does not contain)
    leakage_diagnostic       uses information that was not available on the cutoff, or that is derived from
                             the target: realized selling prices, or choices made with the final-window actuals
A selection strategy or average takes the highest tier among its candidates.

A business scenario decides which tiers a deployable forecast may use (`SCENARIOS`). Leakage diagnostics
are never deployable.
"""

from pathlib import Path

import pandas as pd

from shared.evaluation import apc, coverage, score
from shared.prepare_data import DEV_CUTOFFS, TEST_CUTOFF

FOLDER = Path(__file__).resolve().parents[1] / "results" / "day1"
KEY = ["unique_id", "cutoff", "ds", "model"]
TIERS = ["history", "historical_drivers", "partial_future_drivers", "future_drivers", "leakage_diagnostic"]
INFORMATION = {
    "history": "Sales history and calendar only",
    "historical_drivers": "Historical exogenous features",
    "partial_future_drivers": "Partial future-driver visibility",
    "future_drivers": "Full future-driver visibility",
    "leakage_diagnostic": "Leakage diagnostic",
}
# Most information a deployable forecast may use in each business scenario.
SCENARIOS = {
    "no_future_drivers": "historical_drivers",   # e.g. a manufacturer without the retailer's price and promotion plans
    "future_drivers": "future_drivers",          # a retailer forecasting its own sales, with its own plans
}


def combine_tiers(tiers):
    """Tier of a strategy that can use any of these candidates."""
    return max(tiers, key=TIERS.index)


def eligible(tier, scenario):
    """Whether a forecast with this tier may be used in the scenario. Leakage diagnostics never are."""
    return tier != "leakage_diagnostic" and TIERS.index(tier) <= TIERS.index(SCENARIOS[scenario])


def scenario_pool(tier_of, scenario, models=None):
    """Models (from a model -> tier mapping) a selection may choose from in the scenario, in the given order."""
    models = list(tier_of.index if models is None else models)
    return [m for m in models if eligible(tier_of[m], scenario)]


def describe(tiers):
    """Business description of each tier."""
    return pd.Series(tiers).map(INFORMATION).to_numpy()


def to_long(wide, cutoff, experiment, information_tier="history"):
    """Convert a wide prediction frame (unique_id, ds, one column per model) to the shared format."""
    long = wide.melt(id_vars=["unique_id", "ds"], var_name="model", value_name="yhat")
    long["yhat"] = long["yhat"].clip(lower=0)
    return long.assign(cutoff=cutoff, experiment=experiment, information_tier=information_tier)


def save_forecasts(fcst, name):
    """Write forecasts to results/day1/forecasts/<name>.parquet."""
    (FOLDER / "forecasts").mkdir(parents=True, exist_ok=True)
    assert fcst["yhat"].notna().all(), "drop rows a model cannot forecast instead of saving NaN"
    assert fcst["information_tier"].isin(TIERS).all(), "unknown information tier"
    cols = KEY + ["yhat", "experiment", "information_tier"]
    fcst[cols].astype({"yhat": "float32"}).to_parquet(FOLDER / "forecasts" / f"{name}.parquet", index=False)


def load_forecasts(names=None):
    """Load saved forecasts. names is a list of file stems; None loads everything."""
    files = sorted((FOLDER / "forecasts").glob("*.parquet"))
    files = [f for f in files if names is None or f.stem in names]
    fcst = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    assert not fcst.duplicated(KEY).any(), "same model saved twice"
    return fcst


COLUMNS = ["model", "information", "information_tier", "experiment", "wape", "accuracy", "bias", "mean_rmsse", "median_rmsse",
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
    table["information"] = describe(table["information_tier"])
    table = table.sort_values(["information_tier", "wape"], key=lambda c: c.map(TIERS.index) if c.name == "information_tier" else c)
    return table[COLUMNS].reset_index(drop=True)


def breakdown(fcst, actuals, by, scope="test", segments=None):
    """Scores by horizon or by a segment column (volume, intermittency, department)."""
    rows = fcst[fcst["cutoff"].isin([TEST_CUTOFF] if scope == "test" else list(DEV_CUTOFFS))]
    rows = rows.assign(horizon=(rows["ds"] - rows["cutoff"]).dt.days // 7)
    if segments is not None:
        rows = rows.merge(segments, left_on="unique_id", right_index=True)
    return score(rows, actuals, by=[by], rmsse=by != "horizon")   # RMSSE belongs to a whole forecast window
