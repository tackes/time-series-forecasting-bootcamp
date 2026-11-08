from pathlib import Path
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
df = pd.read_parquet(DATA / "retail_prepared.parquet")

print(f"Rows: {len(df):,} | Series: {df.unique_id.nunique():,}")
print(f"Dates: {df.ds.min()} to {df.ds.max()}")

columns = ["y", "sale_price", "regular_price", "is_promotional"]
print("\nMissing values:")
print(df[columns].isna().agg(["sum", "mean"]).T.rename(columns={"sum": "count", "mean": "fraction"}))

print("\nPrice checks:")
print(pd.Series({
    "sale_price <= 0": df.sale_price.le(0).sum(),
    "regular_price <= 0": df.regular_price.le(0).sum(),
    "sale_price > regular_price": df.sale_price.gt(df.regular_price).sum(),
    "promotional rows": df.is_promotional.eq(True).sum(),
}))

print("\nWeeks per series:")
print(df.groupby("unique_id").size().describe())
print("\nInserted weeks:")
print((~df.is_observed).sum() if "is_observed" in df else "is_observed column unavailable")
