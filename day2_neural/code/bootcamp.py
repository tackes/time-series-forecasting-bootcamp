"""Shared harness for the Day 2 labs.

Every lab notebook imports this module so that all models are trained on the
same panel, split at the same date and scored the same way. Forecasts are saved
to ``results/day2/forecasts`` and every scored model gets one row in
``results/day2/leaderboard.csv``; the model-selection lab reads both.

Environment variables (set them before importing to change the workshop size):

- ``BOOTCAMP_N_SERIES``: number of series in the stratified sample (default 2000,
  ``0`` = all eligible series). Every lab must use the same value.
- ``BOOTCAMP_MAX_STEPS``: training steps for the neural models (default 500).
- ``BOOTCAMP_ACCELERATOR``: ``auto`` (default), ``cpu`` or ``gpu``.
- ``BOOTCAMP_DATA_DIR``: folder with ``sales_weekly.parquet`` etc. (default ``../data``).
"""

from __future__ import annotations

import logging
import os
import time
from datetime import date, timedelta
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae, mase, rmsse, smape

# ============================================================================
# CONFIG — shared with Day 1 and the example script nbeats_weekly_phase3.py
# ============================================================================

PUBLIC_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("BOOTCAMP_DATA_DIR", PUBLIC_DIR / "data"))
RESULTS_DIR = PUBLIC_DIR / "results" / "day2"
FORECAST_DIR = RESULTS_DIR / "forecasts"
LEADERBOARD_PATH = RESULTS_DIR / "leaderboard.csv"
DAY1_LEADERBOARD_PATH = PUBLIC_DIR / "results" / "day1" / "leaderboard.csv"

H = 14                                   # forecast horizon, weeks
FREQ = "W-WED"                           # weeks start on Wednesday
SEASON = 52                              # yearly seasonality in weeks
TEST_START = date(2025, 4, 16)           # final 14 weeks are the test set
VALID_START = TEST_START - timedelta(weeks=H)   # last 14 train weeks, for tuning
MIN_TRAIN_WEEKS = SEASON + H             # needed for a seasonal-naive scale
RANDOM_SEED = 46335

N_SERIES = int(os.environ.get("BOOTCAMP_N_SERIES", 2000))
MAX_STEPS = int(os.environ.get("BOOTCAMP_MAX_STEPS", 500))
ACCELERATOR = os.environ.get("BOOTCAMP_ACCELERATOR", "auto")

PRIMARY_METRIC = "rmsse"                 # leaderboard is ranked on mean RMSSE

# Scale-free future covariates (see the price-feature notes in Lab 03)
FUTR_EXOG_COLS = ["promo", "discount_depth", "price_index"]
HIERARCHY_COLS = ["department_id", "class_id", "subclass_id"]
# Static covariates for the neural models: numeric series descriptors computed on
# the training period. Raw hierarchy ids are labels, not quantities, so they are
# not fed to a network as numbers.
STATIC_COLS = ["log_mean_sales", "share_zero_weeks", "share_promo_weeks"]

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Keep notebook output readable: Lightning prints several lines per fit. Import it
# first, because importing it resets its loggers to INFO.
import pytorch_lightning  # noqa: E402,F401

try:
    import lightning.pytorch  # noqa: F401  (pulled in by some libraries, with its own loggers)
except ImportError:
    pass

for _name in ("pytorch_lightning", "lightning", "lightning.pytorch", "lightning_fabric", "lightning.fabric"):
    logging.getLogger(_name).setLevel(logging.ERROR)
FORECAST_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# DATA
# ============================================================================

def _add_price_features(df: pl.DataFrame) -> pl.DataFrame:
    """Relative price features, as in nbeats_weekly_phase3.py.

    Prices and promotions are treated as known in advance (a planned price
    calendar), so they are valid future covariates.
    """
    df = df.sort(["unique_id", "ds"]).with_columns(
        pl.col("sale_price").rolling_mean(window_size=4, min_samples=1)
        .over("unique_id").alias("_rolling_price_mean")
    )
    df = df.with_columns(
        (pl.col("sale_price") / pl.col("_rolling_price_mean")).alias("price_ratio_to_rolling_mean"),
        pl.col("sale_price").pct_change().over("unique_id").alias("price_pct_change_1w"),
    ).with_columns(
        pl.max_horizontal(pl.lit(0.0), 1.0 - pl.col("price_ratio_to_rolling_mean")).alias("discount_depth"),
    ).drop("_rolling_price_mean")
    cols = ["price_ratio_to_rolling_mean", "price_pct_change_1w", "discount_depth"]
    return df.with_columns(
        [pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(0.0).fill_null(0.0).alias(c) for c in cols]
    )


def load_panel(n_series: int | None = None, seed: int = RANDOM_SEED) -> pl.DataFrame:
    """Load the weekly panel in Nixtla long format (``unique_id``, ``ds``, ``y``).

    Keeps series with a clean price history and at least ``MIN_TRAIN_WEEKS``
    weeks before the test window that are not constant in training, then draws a
    sample stratified by ``sale_class`` so that every lab sees the same series.
    """
    n_series = N_SERIES if n_series is None else n_series

    sales = pl.read_parquet(DATA_DIR / "sales_weekly.parquet").rename(
        {"week_start": "ds", "sales_units": "y", "is_promotional": "promo"}
    )
    items = pl.read_csv(DATA_DIR / "items.csv").select(
        "item_id", "department_id", "class_id", "subclass_id", "item_group_id"
    )
    df = (
        sales.with_columns(pl.col("unique_id").str.split("_").list.first().cast(pl.Int64).alias("item_id"))
        .join(items, on="item_id", how="left")
        .with_columns(pl.col("promo").cast(pl.Float32))
    )

    bad_price = df.filter(pl.col("sale_price") <= 0)["unique_id"].unique()
    train = df.filter(pl.col("ds") < TEST_START)
    stats = train.group_by("unique_id").agg(pl.len().alias("n"), pl.col("y").std().alias("sd"))
    keep = stats.filter((pl.col("n") >= MIN_TRAIN_WEEKS) & (pl.col("sd") > 0))["unique_id"]
    df = df.filter(pl.col("unique_id").is_in(keep) & ~pl.col("unique_id").is_in(bad_price.implode()))

    if n_series:
        classes = df.select("unique_id", "sale_class").unique().sort("unique_id")
        frac = min(1.0, n_series / classes.height)
        rng = np.random.default_rng(seed)
        picked = []
        for (sc,), grp in classes.group_by(["sale_class"], maintain_order=True):
            ids = grp["unique_id"].to_numpy()
            k = max(1, int(round(len(ids) * frac)))
            picked.extend(rng.choice(ids, size=min(k, len(ids)), replace=False).tolist())
        df = df.filter(pl.col("unique_id").is_in(picked))

    df = _add_price_features(df)
    # Scale-free price level: price relative to the series' median training price
    # (raw prices differ by orders of magnitude between items and confuse a global net).
    ref = df.filter(pl.col("ds") < TEST_START).group_by("unique_id").agg(pl.col("sale_price").median().alias("_ref"))
    df = df.join(ref, on="unique_id").with_columns(
        (pl.col("sale_price") / pl.col("_ref")).alias("price_index")
    ).drop("_ref")
    for c in HIERARCHY_COLS:
        df = df.with_columns(pl.col(c).fill_null(0))
    return df.sort(["unique_id", "ds"])


def split(df: pl.DataFrame, cutoff: date = TEST_START) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Temporal split: everything before ``cutoff`` trains, the next ``H`` weeks test."""
    end = cutoff + timedelta(weeks=H)
    return df.filter(pl.col("ds") < cutoff), df.filter((pl.col("ds") >= cutoff) & (pl.col("ds") < end))


def to_nixtla(df: pl.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """pandas frame with a datetime ``ds`` (neuralforecast and hierarchicalforecast want pandas)."""
    cols = ["unique_id", "ds", "y"] + (cols or [])
    out = df.select(cols).to_pandas()
    out["ds"] = pd.to_datetime(out["ds"])
    return out


def static_frame(train: pl.DataFrame) -> pd.DataFrame:
    """Static covariates per series, computed on the training rows only (no test leakage)."""
    return (
        train.group_by("unique_id")
        .agg(
            pl.col("y").mean().log1p().alias("log_mean_sales"),
            (pl.col("y") == 0).mean().alias("share_zero_weeks"),
            (pl.col("promo") > 0).mean().alias("share_promo_weeks"),
        )
        .sort("unique_id")
        .to_pandas()
        .astype({c: "float32" for c in STATIC_COLS})
    )


def series_info(df: pl.DataFrame) -> pl.DataFrame:
    """Per-series attributes used to slice results: volume class and shape features."""
    feats = pl.read_parquet(DATA_DIR / "series_features.parquet").select(
        "unique_id", "cluster_label", "zero_proportion", "cv_squared", "intermittent_ratio", "mean"
    )
    info = df.select("unique_id", "sale_class", "department_id").unique("unique_id").join(feats, on="unique_id", how="left")
    # Syntetos-Boylan demand classes: ADI = 1 / share of non-zero weeks, CV² of non-zero demand
    adi = 1.0 / (1.0 - pl.col("zero_proportion")).clip(lower_bound=1e-6)
    return info.with_columns(
        pl.when((adi < 1.32) & (pl.col("cv_squared") < 0.49)).then(pl.lit("smooth"))
        .when(adi < 1.32).then(pl.lit("erratic"))
        .when(pl.col("cv_squared") < 0.49).then(pl.lit("intermittent"))
        .otherwise(pl.lit("lumpy")).alias("demand_type")
    )


# ============================================================================
# SCORING
# ============================================================================

def _as_pandas(df) -> pd.DataFrame:
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()
    df = df.reset_index() if "unique_id" not in df.columns else df.copy()
    df["ds"] = pd.to_datetime(df["ds"])
    return df


def model_columns(fcst: pd.DataFrame) -> list[str]:
    return [c for c in fcst.columns if c not in ("unique_id", "ds", "y", "cutoff", "index")
            and "-lo-" not in c and "-hi-" not in c and "_ql" not in c]


def score(fcst, test, train, models: list[str] | None = None) -> pd.DataFrame:
    """Per-series metrics in long format: ``unique_id``, ``metric``, one column per model.

    RMSSE and MASE are scaled by the in-sample seasonal-naive (52-week) error,
    the same choice as the Day 1 baseline script.
    """
    fcst, test, train = _as_pandas(fcst), _as_pandas(test), _as_pandas(train)
    models = models or model_columns(fcst)
    df = test[["unique_id", "ds", "y"]].merge(fcst[["unique_id", "ds"] + models], on=["unique_id", "ds"], how="inner")
    df[models] = df[models].clip(lower=0)          # sales cannot be negative
    out = evaluate(
        df,
        metrics=[partial(rmsse, seasonality=SEASON), partial(mase, seasonality=SEASON), smape, mae],
        models=models,
        train_df=train[["unique_id", "ds", "y"]],
    )
    out["metric"] = out["metric"].str.replace(r"_seasonality_\d+", "", regex=True)
    return out.replace([np.inf, -np.inf], np.nan)


def summarise(metrics: pd.DataFrame, fcst=None, test=None) -> pd.DataFrame:
    """One row per model: mean/median RMSSE, MASE, sMAPE and (if given) WAPE."""
    models = [c for c in metrics.columns if c not in ("unique_id", "metric")]
    piv = {m: metrics[metrics["metric"] == m].set_index("unique_id")[models] for m in metrics["metric"].unique()}
    rows = []
    for model in models:
        r = {
            "model": model,
            "mean_rmsse": piv["rmsse"][model].mean(),
            "median_rmsse": piv["rmsse"][model].median(),
            "mean_mase": piv["mase"][model].mean(),
            "mean_smape": piv["smape"][model].mean(),
            "n_series": int(piv["rmsse"][model].notna().sum()),
        }
        if fcst is not None and test is not None:
            f = _as_pandas(fcst).merge(_as_pandas(test)[["unique_id", "ds", "y"]], on=["unique_id", "ds"])
            r["wape"] = (f[model].clip(lower=0) - f["y"]).abs().sum() / f["y"].abs().sum()
        rows.append(r)
    return pd.DataFrame(rows).sort_values("mean_rmsse").reset_index(drop=True)


# ============================================================================
# FORECAST STORE + LEADERBOARD
# ============================================================================

def save_forecasts(fcst, name: str) -> Path:
    """Save point forecasts (``unique_id``, ``ds``, model columns) for the selection lab."""
    fcst = _as_pandas(fcst)
    cols = ["unique_id", "ds"] + model_columns(fcst)
    path = FORECAST_DIR / f"{name}.parquet"
    fcst[cols].to_parquet(path, index=False)
    return path


def load_forecasts() -> pd.DataFrame:
    """All saved forecasts joined into one wide frame (later files win on name clashes)."""
    out = None
    for path in sorted(FORECAST_DIR.glob("*.parquet")):
        f = pd.read_parquet(path)
        f["ds"] = pd.to_datetime(f["ds"])
        if out is None:
            out = f
        else:
            out = out.drop(columns=[c for c in f.columns if c in out.columns and c not in ("unique_id", "ds")])
            out = out.merge(f, on=["unique_id", "ds"], how="outer")
    return out


def record(fcst, test, train, lab: str, fit_seconds: float | dict | None = None,
           models: list[str] | None = None, notes: str = "") -> pd.DataFrame:
    """Score forecasts, upsert them into the leaderboard, and return their summary rows."""
    fcst = _as_pandas(fcst)
    models = models or model_columns(fcst)
    summary = summarise(score(fcst, test, train, models), fcst, test)
    summary["lab"] = lab
    if isinstance(fit_seconds, dict):
        summary["fit_seconds"] = summary["model"].map(fit_seconds)
    else:
        summary["fit_seconds"] = fit_seconds
    summary["notes"] = notes
    summary["recorded_at"] = pd.Timestamp.now().floor("s")

    board = leaderboard(include_day1=False)
    if len(board):   # re-running a lab replaces all of that lab's rows
        board = board[~board["model"].isin(summary["model"]) & (board["lab"] != lab)]
    board = pd.concat([board, summary], ignore_index=True) if len(board) else summary
    board.sort_values("mean_rmsse").to_csv(LEADERBOARD_PATH, index=False)
    return summary


def leaderboard(include_day1: bool = True) -> pd.DataFrame:
    """The Day 2 leaderboard, with Day 1 entries appended when that file exists."""
    frames = []
    if LEADERBOARD_PATH.exists():
        frames.append(pd.read_csv(LEADERBOARD_PATH))
    if include_day1 and DAY1_LEADERBOARD_PATH.exists():
        d1 = pd.read_csv(DAY1_LEADERBOARD_PATH)
        d1["lab"] = "day1:" + d1.get("lab", pd.Series(["day1"] * len(d1))).astype(str)
        frames.append(d1)
    if not frames:
        return pd.DataFrame(columns=["model", "mean_rmsse", "median_rmsse", "mean_mase", "mean_smape",
                                     "wape", "n_series", "lab", "fit_seconds", "notes", "recorded_at"])
    board = pd.concat(frames, ignore_index=True).sort_values("mean_rmsse").reset_index(drop=True)
    board.index = board.index + 1
    return board


class Timer:
    """``with Timer() as t: ...`` then ``t.seconds``."""

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self._t0


def trainer_kwargs() -> dict:
    """Lightning settings shared by every neural model in the labs."""
    return dict(
        accelerator=ACCELERATOR,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        logger=False,
    )
