from pathlib import Path
import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import Naive, SeasonalNaive

DATA = Path(__file__).resolve().parents[1] / "data"
RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

h = 14
df = pd.read_parquet(DATA / "retail_prepared.parquet", columns=["unique_id", "ds", "y"])
df["ds"] = pd.to_datetime(df["ds"])
df = df.sort_values(["unique_id", "ds"])


def forecast(start):
    start = pd.Timestamp(start)
    train = df[df.ds < start]
    actual = df[(df.ds >= start) & (df.ds < start + pd.Timedelta(weeks=h))]
    sf = StatsForecast(models=[Naive(), SeasonalNaive(season_length=52)], freq="W-WED", n_jobs=-1)
    predictions = sf.forecast(df=train, h=h).reset_index()
    return actual.merge(predictions, on=["unique_id", "ds"], how="inner")


def metrics(frame, column):
    actual, predicted = frame.y.to_numpy(), frame[column].to_numpy()
    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted = actual[valid], predicted[valid]
    denominator = np.abs(actual).sum()
    return pd.Series({
        "WAPE": np.abs(actual - predicted).sum() / denominator if denominator else np.nan,
        "Bias": (predicted - actual).sum() / denominator if denominator else np.nan,
    })


validation = forecast("2025-01-08")
test = forecast("2025-04-16")
models = ["Naive", "SeasonalNaive"]

# Pick each series' best baseline using validation only.
errors = validation.groupby("unique_id").apply(
    lambda g: pd.Series({model: (g.y - g[model]).abs().sum() for model in models}),
    include_groups=False,
)
choices = errors.idxmin(axis=1).rename("selected_model")
test = test.join(choices, on="unique_id")
test["Selected"] = np.where(test.selected_model.eq("Naive"), test.Naive, test.SeasonalNaive)

scores = pd.DataFrame({model: metrics(test, model) for model in models + ["Selected"]}).T
print("\nFinal 14-week holdout:")
print(scores.round(4))
print("\nSelected models:")
print(choices.value_counts())

test.to_parquet(RESULTS / "baseline_predictions.parquet", index=False)
scores.to_csv(RESULTS / "baseline_leaderboard.csv")
