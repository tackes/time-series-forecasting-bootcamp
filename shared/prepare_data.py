"""Prepare a common weekly retail dataset for both workshop days."""

from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"


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


if __name__ == "__main__":
    prepared = prepare_data()
    print(f"Prepared {len(prepared):,} rows across {prepared['unique_id'].nunique():,} series")
    print(f"Added {(~prepared['is_observed']).sum():,} missing-week rows")

