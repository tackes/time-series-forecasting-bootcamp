"""Shared evaluation helpers for the forecasting bootcamp."""

import numpy as np
import pandas as pd


def scores(df, models):
    """Overall WAPE, forecast bias, and accuracy for model columns."""
    rows = []
    for model in models:
        valid = df[["y", model]].dropna()
        actual = valid["y"].sum()
        error = valid[model] - valid["y"]
        rows.append({
            "model": model,
            "WAPE": error.abs().sum() / actual if actual else np.nan,
            "bias": error.sum() / actual if actual else np.nan,
            "accuracy": 1 - error.abs().sum() / actual if actual else np.nan,
        })
    return pd.DataFrame(rows).sort_values("WAPE")


def select_models(validation, models):
    """Select the lowest-MAE model for each series, breaking ties by model order."""
    errors = validation.groupby("unique_id").apply(
        lambda g: pd.Series({m: (g["y"] - g[m]).abs().mean() for m in models}),
        include_groups=False,
    )
    return errors.idxmin(axis=1).rename("selected_model")


def selected_forecasts(forecasts, selections, models):
    """Add the prediction from each series' validation-selected baseline."""
    result = forecasts.join(selections, on="unique_id")
    result["Selected"] = np.nan
    for model in models:
        mask = result["selected_model"].eq(model)
        result.loc[mask, "Selected"] = result.loc[mask, model]
    return result
