"""Combine forecast submissions and calculate a shared leaderboard."""

from pathlib import Path

import pandas as pd

COLUMNS = ["split", "unique_id", "ds", "y", "model", "yhat"]


def baseline_submission(path):
    """Convert the existing baseline notebook's wide output to shared format."""
    df = pd.read_parquet(path)
    models = [c for c in ["Naive", "HistoricAverage", "SeasonalNaive", "Selected"] if c in df]
    return df.melt(
        id_vars=["split", "unique_id", "ds", "y"],
        value_vars=models,
        var_name="model",
        value_name="yhat",
    ).dropna(subset=["yhat"])[COLUMNS]


def combine_results(files, output="results/all_predictions.parquet"):
    """Combine long-format CSV/Parquet forecasts, checking common actuals."""
    frames = [pd.read_csv(f) if str(f).endswith(".csv") else pd.read_parquet(f) for f in files]
    df = pd.concat([f[COLUMNS] for f in frames], ignore_index=True)
    df["ds"] = pd.to_datetime(df["ds"])

    key = ["split", "unique_id", "ds", "model"]
    if df.duplicated(key).any():
        raise ValueError("Duplicate predictions for the same split, series, date, and model")
    if df.groupby(["split", "unique_id", "ds"])["y"].nunique().gt(1).any():
        raise ValueError("Actuals differ between submissions")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    return df


def leaderboard(predictions, split="test"):
    """Score models on the same test observations (WAPE, accuracy, bias)."""
    df = predictions[predictions["split"].eq(split)].dropna(subset=["y", "yhat"])
    counts = df.groupby(["unique_id", "ds"])["model"].nunique()
    n_models = df["model"].nunique()
    common = counts[counts.eq(n_models)].index
    df = df.set_index(["unique_id", "ds"]).loc[common].reset_index()

    df["absolute_error"] = (df["yhat"] - df["y"]).abs()
    df["error"] = df["yhat"] - df["y"]
    totals = df.groupby("model", as_index=False).agg(
        actual=("y", "sum"),
        absolute_error=("absolute_error", "sum"),
        error=("error", "sum"),
        rows=("y", "size"),
    )
    totals["WAPE"] = totals["absolute_error"] / totals["actual"]
    totals["accuracy"] = 1 - totals["WAPE"]
    totals["bias"] = totals["error"] / totals["actual"]
    return totals[["model", "rows", "WAPE", "accuracy", "bias"]].sort_values("WAPE")
