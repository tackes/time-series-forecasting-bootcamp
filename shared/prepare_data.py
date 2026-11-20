"""Prepare a common weekly retail dataset for both workshop days."""

from pathlib import Path

import pandas as pd

import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data"

H = 14
FREQ = "W-WED"
SEASON = 52

# A cutoff is the last week of data available when the forecast is made.
TEST_CUTOFF = pd.Timestamp("2025-04-09")
DEV_CUTOFFS = pd.date_range(end="2025-01-01", periods=7, freq="28D")   # targets end by 2025-04-09
STABILITY_CUTOFFS = TEST_CUTOFF - pd.to_timedelta([28, 56], unit="D")  # extra origins, used only for APC
ALL_CUTOFFS = DEV_CUTOFFS.append(STABILITY_CUTOFFS).append(pd.DatetimeIndex([TEST_CUTOFF])).sort_values()


def prepare_data():
    sales = pd.read_parquet(DATA / "sales_weekly.parquet")
    items = pd.read_csv(DATA / "items.csv")
    hierarchy = pd.read_csv(DATA / "hierarchy.csv")

    sales = sales.rename(columns={"week_start": "ds", "sales_units": "y"})
    sales["ds"] = pd.to_datetime(sales["ds"])
    sales["item_id"] = sales["unique_id"].str.split("_").str[0].astype(int)

    # Fill gaps only between each series' first and last recorded weeks.
    bounds = sales.groupby("unique_id")["ds"].agg(["min", "max"])
    weeks = pd.date_range(sales["ds"].min(), sales["ds"].max(), freq="W-WED")
    panel = pd.MultiIndex.from_product(
        [bounds.index, weeks], names=["unique_id", "ds"]
    ).to_frame(index=False)
    panel = panel.join(bounds, on="unique_id")
    panel = panel.loc[panel["ds"].between(panel["min"], panel["max"])]
    panel = panel.drop(columns=["min", "max"])

    sales["is_observed"] = True
    data = panel.merge(sales, on=["unique_id", "ds"], how="left", validate="one_to_one")
    data["is_observed"] = data["is_observed"].fillna(False).astype(bool)
    data["item_id"] = data["unique_id"].str.split("_").str[0].astype(int)

    names = hierarchy[["subclass_id", "subclass_name", "class_name", "department_name"]]
    data = data.merge(items, on="item_id", how="left", validate="many_to_one")
    data = data.merge(names, on="subclass_id", how="left", validate="many_to_one")

    # Keep missing values as missing; sales and future driver assumptions belong in the labs.
    data = data.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    data.to_parquet(DATA / "retail_prepared.parquet", index=False)
    return data


def completed_runs(decision_cutoff):
    """Development cutoffs whose whole 14-week horizon was observed on the decision date."""
    return [c for c in DEV_CUTOFFS if c + pd.Timedelta(weeks=H) <= decision_cutoff]


def teaching_sample(n_series=2000, seed=46335):
    """Same 2,000 series as day2_neural/code/bootcamp.py: stratified by sale_class."""
    df = pd.read_parquet(DATA / "retail_prepared.parquet", columns=["unique_id", "ds", "y", "sale_price", "sale_class"])
    bad_price = df.loc[df["sale_price"] <= 0, "unique_id"].unique()
    train = df[df["ds"] <= TEST_CUTOFF].groupby("unique_id")["y"].agg(["size", "std"])
    keep = train[(train["size"] >= SEASON + H) & (train["std"] > 0)].index
    keep = keep[~keep.isin(bad_price)]

    classes = df[df["unique_id"].isin(keep)].drop_duplicates("unique_id")[["unique_id", "sale_class"]]
    classes = classes.sort_values("unique_id")
    share = min(1.0, n_series / len(classes))
    rng = np.random.default_rng(seed)
    picked = []
    for _, group in classes.groupby("sale_class", sort=False):
        ids = group["unique_id"].to_numpy()
        k = min(len(ids), max(1, round(len(ids) * share)))
        picked += rng.choice(ids, size=k, replace=False).tolist()
    return sorted(picked)


def load_panel(sample=True):
    """Prepared panel, optionally restricted to the common teaching sample."""
    df = pd.read_parquet(DATA / "retail_prepared.parquet")
    if sample:
        df = df[df["unique_id"].isin(teaching_sample())]
    return df.sort_values(["unique_id", "ds"]).reset_index(drop=True)


def series_segments(df, cutoff=TEST_CUTOFF):
    """Volume, intermittency and department, from history up to the final test cutoff."""
    hist = df[df["ds"] <= cutoff].groupby("unique_id")["y"].agg(mean="mean", zero_share=lambda y: (y == 0).mean())
    hist["volume"] = pd.qcut(hist["mean"].rank(method="first"), 3, labels=["low", "mid", "high"])
    hist["intermittency"] = pd.cut(hist["zero_share"], [-1, 0.1, 0.5, 1], labels=["regular", "some_zeros", "zero_heavy"])
    dept = df.drop_duplicates("unique_id").set_index("unique_id")["department_name"].fillna("Unknown")
    hist[["volume", "intermittency"]] = hist[["volume", "intermittency"]].astype(str)
    return hist[["volume", "intermittency"]].join(dept.rename("department"))


if __name__ == "__main__":
    prepared = prepare_data()
    print(f"Prepared {len(prepared):,} rows across {prepared['unique_id'].nunique():,} series")
    print(f"Added {(~prepared['is_observed']).sum():,} missing-week rows")

