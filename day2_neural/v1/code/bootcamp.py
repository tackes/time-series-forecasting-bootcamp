"""Shared harness for the Day 2 labs.

Every lab notebook imports this module so that all models are trained on the
same panel, split at the same date and scored the same way. Forecasts are saved
to ``results/day2/forecasts`` and every scored model gets one row in
``results/day2/leaderboard.csv``; the model-selection lab reads both.

Environment variables (set them before importing to change the workshop size):

- ``BOOTCAMP_N_SERIES``: ``0`` (default) uses every eligible series (17,241 of the
  18,692); a positive number draws a sample of that size stratified by
  ``sale_class``, for slow machines. Every lab must use the same value.
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
from utilsforecast.losses import bias, mae, mase, rmsse, smape, wape

# ============================================================================
# CONFIG — shared with Day 1 and the example script nbeats_weekly_phase3.py
# ============================================================================

PUBLIC_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("BOOTCAMP_DATA_DIR", PUBLIC_DIR / "data"))
RESULTS_DIR = PUBLIC_DIR / "results" / "day2"
FORECAST_DIR = RESULTS_DIR / "forecasts"   # test-window point forecasts, the pool for the selection lab
RAW_DIR = RESULTS_DIR / "raw"              # everything each lab predicted, with actuals: raw/<window>/<name>.parquet
LEADERBOARD_PATH = RESULTS_DIR / "leaderboard.csv"
DAY1_LEADERBOARD_PATH = PUBLIC_DIR / "results" / "day1" / "leaderboard.csv"

H = 14                                   # forecast horizon, weeks
FREQ = "W-WED"                           # weeks start on Wednesday
SEASON = 52                              # yearly seasonality in weeks
TEST_START = date(2025, 4, 16)           # final 14 weeks are the test set
VALID_START = TEST_START - timedelta(weeks=H)   # last 14 train weeks, for tuning
MIN_TRAIN_WEEKS = SEASON + H             # needed for a seasonal-naive scale
RANDOM_SEED = 46335

N_SERIES = int(os.environ.get("BOOTCAMP_N_SERIES", 0))
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
RAW_DIR.mkdir(parents=True, exist_ok=True)


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
    weeks before the test window that are not constant in training. With
    ``n_series > 0`` it draws a sample stratified by ``sale_class`` (the same
    sample in every lab).
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
    """Per-series attributes used to slice results: volume class, demand type and shape cluster.

    Demand statistics are computed here from the training weeks. The numeric columns
    in ``series_features.parquet`` are standardised (z-scores) and its ``cv_squared``
    is constant, so only its ``cluster_label`` is used (computed by the data provider
    on the full history; fine for slicing results, not as a model input).
    """
    feats = pl.read_parquet(DATA_DIR / "series_features.parquet").select("unique_id", "cluster_label")
    demand = (
        df.filter(pl.col("ds") < TEST_START)
        .group_by("unique_id")
        .agg(
            (pl.col("y") == 0).mean().alias("zero_share"),
            pl.col("y").mean().alias("mean_weekly_units"),
            (pl.col("y").filter(pl.col("y") > 0).std() / pl.col("y").filter(pl.col("y") > 0).mean())
            .pow(2).fill_null(0.0).alias("cv2_nonzero"),
        )
    )
    info = (df.select("unique_id", "sale_class", "department_id").unique("unique_id")
            .join(demand, on="unique_id", how="left").join(feats, on="unique_id", how="left"))
    # Syntetos-Boylan demand classes: ADI = average weeks between non-zero sales
    # (= 1 / share of non-zero weeks), CV² = squared coefficient of variation of the non-zero sales
    adi = 1.0 / (1.0 - pl.col("zero_share")).clip(lower_bound=1e-6)
    return info.with_columns(
        adi.alias("adi"),
        pl.when((adi < 1.32) & (pl.col("cv2_nonzero") < 0.49)).then(pl.lit("smooth"))
        .when(adi < 1.32).then(pl.lit("erratic"))
        .when(pl.col("cv2_nonzero") < 0.49).then(pl.lit("intermittent"))
        .otherwise(pl.lit("lumpy")).alias("demand_type"),
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

    All metrics come from ``utilsforecast.losses`` (no hand-written formulas); RMSSE, MASE and MAE were
    checked against ``sktime`` on 1,000 series (identical to 1e-13).

    * RMSSE and MASE are scaled by the in-sample seasonal-naive (52-week) error, the same choice as the
      Day 1 baseline script. A series whose sales repeat exactly 52 weeks apart has a zero scale; its
      score is infinite and is dropped (3 of the 17,241 series), which is why ``n_series`` is 17,238.
    * sMAPE follows the utilsforecast convention, mean of ``|y - ŷ| / (|y| + |ŷ|)``, which runs from 0 to 1:
      half of the "0–200 %" convention used by sktime and the M3/M4 papers. Weeks where both are 0 count as 0.
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


def _pooled(metric, f: pd.DataFrame, model: str) -> float:
    """Apply a utilsforecast metric to all series and weeks pooled into one."""
    return float(metric(f[["y", model]].assign(unique_id="all"), models=[model])[model].iloc[0])


def summarise(metrics: pd.DataFrame, fcst=None, test=None) -> pd.DataFrame:
    """One row per model.

    From the per-series metrics: mean and median RMSSE, mean and median MASE, mean sMAPE.
    With ``fcst`` and ``test``, two volume-weighted numbers over all series and weeks pooled
    (``utilsforecast`` WAPE and bias):

    * ``wape``: sum |y - ŷ| / sum |y|, so best sellers count most;
    * ``bias_pct``: sum (ŷ - y) / sum y. Positive = over-forecasting (excess stock), negative =
      under-forecasting (lost sales). A model can have a good RMSSE and still be biased, which on
      intermittent items usually means forecasting a smooth mean above the frequent zeros.
    """
    models = [c for c in metrics.columns if c not in ("unique_id", "metric")]
    piv = {m: metrics[metrics["metric"] == m].set_index("unique_id")[models] for m in metrics["metric"].unique()}
    rows = []
    for model in models:
        r = {
            "model": model,
            "mean_rmsse": piv["rmsse"][model].mean(),
            "median_rmsse": piv["rmsse"][model].median(),
            "mean_mase": piv["mase"][model].mean(),
            "median_mase": piv["mase"][model].median(),
            "mean_smape": piv["smape"][model].mean(),
            "n_series": int(piv["rmsse"][model].notna().sum()),
        }
        if fcst is not None and test is not None:
            f = _as_pandas(fcst).merge(_as_pandas(test)[["unique_id", "ds", "y"]], on=["unique_id", "ds"])
            f = f[f[model].notna()].copy()           # same rows in numerator and denominator
            f[model] = f[model].clip(lower=0)        # sales cannot be negative (as in score)
            r["wape"] = _pooled(wape, f, model)
            r["bias_pct"] = _pooled(bias, f, model) / f["y"].mean()
        rows.append(r)
    return pd.DataFrame(rows).sort_values("mean_rmsse").reset_index(drop=True)


# ============================================================================
# FORECAST STORE + LEADERBOARD
# ============================================================================

def _with_actuals(fcst: pd.DataFrame, actuals) -> pd.DataFrame:
    if actuals is None or "y" in fcst.columns:
        return fcst
    return fcst.merge(_as_pandas(actuals)[["unique_id", "ds", "y"]], on=["unique_id", "ds"], how="left")


def save_forecasts(fcst, name: str, actuals=None) -> Path:
    """Save test-window point forecasts (``unique_id``, ``ds``, model columns, and ``y`` if given)
    for the selection lab. Use :func:`save_raw` as well to keep intervals and backtests."""
    fcst = _with_actuals(_as_pandas(fcst), actuals)
    cols = ["unique_id", "ds"] + model_columns(fcst) + (["y"] if "y" in fcst.columns else [])
    path = FORECAST_DIR / f"{name}.parquet"
    fcst[cols].to_parquet(path, index=False)
    return path


def save_raw(fcst, name: str, window: str, actuals=None) -> Path:
    """Keep everything a lab predicted so any metric can be computed later without refitting.

    One tidy-wide parquet per lab and window in ``results/day2/raw/<window>/<name>.parquet``:
    ``unique_id``, ``ds``, ``cutoff`` (backtests), ``y`` (the actuals) and every forecast column,
    including prediction intervals (``<model>-lo-80`` ...) and quantiles. ``window`` is ``"test"``
    (the leaderboard window), ``"valid"`` (the 14 weeks before it) or ``"cv"`` (rolling-origin
    backtests on the training data). :func:`load_raw` reads them back; :func:`metric_suite` scores them.
    """
    fcst = _with_actuals(_as_pandas(fcst).drop(columns=["index"], errors="ignore"), actuals)
    front = [c for c in ("unique_id", "ds", "cutoff", "y") if c in fcst.columns]
    path = RAW_DIR / window / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    fcst[front + [c for c in fcst.columns if c not in front]].to_parquet(path, index=False)
    return path


def load_raw(window: str = "test", names: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """The raw forecast files of one window, as {lab name: frame}."""
    out = {}
    for path in sorted((RAW_DIR / window).glob("*.parquet")):
        if names is None or path.stem in names:
            out[path.stem] = pd.read_parquet(path)
    return out


# ============================================================================
# METRIC SUITE
# ============================================================================

def metric_suite(df, train, models: list[str] | None = None, baseline: str | None = "SeasonalNaive",
                 levels: tuple[int, ...] = (80, 95), info=None) -> pd.DataFrame:
    """Many metrics, one row per model, all computed by ``utilsforecast`` or ``sktime``.

    ``df`` holds ``unique_id``, ``ds``, ``y``, optionally ``cutoff`` (backtests), the model columns and,
    where a model has them, interval columns ``<model>-lo-<level>`` / ``<model>-hi-<level>``.
    ``train`` is the training history the scaled metrics are scaled by (for backtests, utilsforecast
    uses only the weeks before each cutoff). Forecasts are clipped at 0 (sales cannot be negative).

    Columns, by family (see the README for definitions and why some metrics are left out):

    * scaled, per series, then mean and median: RMSSE, MASE (seasonal-naive 52-week scale);
    * relative to ``baseline`` (default SeasonalNaive): ``avgrelmae`` / ``avgrelrmse``, the geometric
      mean over series of each series' MAE / RMSE ratio to the baseline (Davydenko & Fildes 2013), and
      ``share_beats_baseline``, the share of series with a lower MAE than the baseline;
    * percentage: mean sMAPE (0–1 scale), mean MAAPE (arctangent APE, defined at y = 0), mean RMSLE;
    * pooled over all series and weeks: WAPE, MAE and RMSE (units per week), bias % (positive = over-forecast);
    * share of series over-forecast in total (negative cumulative forecast error);
    * mean RMSSE by demand type (smooth, erratic, intermittent, lumpy);
    * intervals, where present: coverage, scaled Winkler score and median scaled width per level, and
      scaled CRPS approximated by twice the scaled multi-quantile loss over the available quantiles.
    """
    from utilsforecast import losses as L
    # sktime's functional API: the MeanSquaredLogError *class* in sktime 1.3 ignores square_root=True
    # and returns MSLE, and the classes are ~2,000x slower on 17,000 series.
    from sktime.performance_metrics.forecasting import (mean_arctangent_absolute_percentage_error,
                                                         mean_squared_log_error)

    df = _as_pandas(df).copy()
    df["ds"] = pd.to_datetime(df["ds"])
    train = _as_pandas(train)[["unique_id", "ds", "y"]].copy()
    train["ds"] = pd.to_datetime(train["ds"])
    models = models or model_columns(df)
    has_cut = "cutoff" in df.columns
    for c in df.columns:
        if c in models or "-lo-" in c or "-hi-" in c:
            df[c] = df[c].clip(lower=0)
    per = lambda fn, ms, **kw: fn(df, models=ms, **kw).replace([np.inf, -np.inf], np.nan)
    rows = {m: {"model": m} for m in models}

    def put(frame, key, agg=("mean",)):
        for m in models:
            col = frame[m]
            for a in agg:
                rows[m][f"{a}_{key}" if a else key] = getattr(col, a)()

    put(per(L.rmsse, models, seasonality=SEASON, train_df=train), "rmsse", ("mean", "median"))
    put(per(L.mase, models, seasonality=SEASON, train_df=train), "mase", ("mean", "median"))
    put(per(L.smape, models), "smape")

    # relative to the baseline: geometric mean of per-series ratios (series where either is 0 are skipped)
    if baseline and baseline in df.columns:
        both = list(dict.fromkeys(models + [baseline]))
        mae_s, rmse_s = per(L.mae, both), per(L.rmse, both)
        for m in models:
            for key, fr in (("avgrelmae", mae_s), ("avgrelrmse", rmse_s)):
                r = (fr[m] / fr[baseline]).replace([0, np.inf], np.nan).dropna()
                rows[m][key] = float(np.exp(np.log(r).mean()))
            rows[m]["share_beats_baseline"] = float((mae_s[m] < mae_s[baseline]).mean())

    # sktime metrics, per series (one instance = one series and cutoff): arrays of shape (weeks, instances)
    key = ["unique_id", "cutoff"] if has_cut else ["unique_id"]
    step = df.groupby(key).cumcount()
    wide = lambda col: df.assign(_k=step).pivot_table(index="_k", columns=key, values=col, aggfunc="first")
    y_w = wide("y")
    for m in models:
        p_w = wide(m).reindex(columns=y_w.columns)
        ok = ~(y_w.isna() | p_w.isna()).any().to_numpy()          # instances with a full horizon
        yt, yp = y_w.to_numpy()[:, ok], p_w.to_numpy()[:, ok]
        rows[m]["mean_maape"] = float(np.mean(mean_arctangent_absolute_percentage_error(yt, yp, multioutput="raw_values")))
        rows[m]["mean_rmsle"] = float(np.mean(mean_squared_log_error(yt, yp, multioutput="raw_values", square_root=True)))

    # pooled over all series and weeks (volume-weighted), on the rows where the model has a forecast
    pooled = df.assign(unique_id="all").drop(columns="cutoff", errors="ignore")
    for m in models:
        pm = pooled[pooled[m].notna()]
        for name, fn in (("wape", L.wape), ("mae", L.mae), ("rmse", L.rmse)):
            rows[m][f"pooled_{name}"] = float(fn(pm, models=[m])[m].iloc[0])
        rows[m]["bias_pct"] = float(L.bias(pm, models=[m])[m].iloc[0]) / pm["y"].mean()
    cfe = per(L.cfe, models)
    for m in models:
        rows[m]["share_series_over"] = float((cfe[m] < 0).mean())

    # mean RMSSE by demand type
    rm = per(L.rmsse, models, seasonality=SEASON, train_df=train)
    info = series_info(load_panel()) if info is None else info
    info = info.to_pandas() if isinstance(info, pl.DataFrame) else info
    types = info[["unique_id", "demand_type"]]
    by = rm.merge(types, on="unique_id").groupby("demand_type")[models].mean()
    for t in by.index:
        for m in models:
            rows[m][f"rmsse_{t}"] = by.loc[t, m]

    # intervals and quantiles, for the models that have them
    for m in models:
        lv = [l for l in sorted(levels) if f"{m}-lo-{l}" in df.columns and f"{m}-hi-{l}" in df.columns]
        # A wider level identical to a narrower one is mislabelled, not scored: TimesFM 2.5 only has the
        # 10 %, ..., 90 % quantiles, and timecopilot returns those for any wider interval without a warning.
        for wide, narrow in [(b, a) for i, a in enumerate(lv) for b in lv[i + 1:]]:
            if wide in lv and np.allclose(df[f"{m}-lo-{wide}"], df[f"{m}-lo-{narrow}"], equal_nan=True) \
                    and np.allclose(df[f"{m}-hi-{wide}"], df[f"{m}-hi-{narrow}"], equal_nan=True):
                print(f"metric_suite: {m}'s {wide} % interval equals its {narrow} % interval; {wide} % not scored")
                lv.remove(wide)
        if not lv:
            continue
        # the seasonal-naive scale of each series (and cutoff), read off the library: MASE = MAE / scale
        g = key
        scale = (per(L.mae, [m]).set_index(g)[m] / per(L.mase, [m], seasonality=SEASON, train_df=train)
                 .set_index(g)[m]).replace([0, np.inf], np.nan).rename("_ms_scale")
        d = df.drop(columns="_ms_scale", errors="ignore").join(scale, on=g)
        for l in lv:
            rows[m][f"coverage_{l}"] = per(L.coverage, [m], level=l)[m].mean()
            w = per(L.winkler_score, [m], level=l).set_index(g)[m]
            rows[m][f"scaled_winkler_{l}"] = (w / scale).mean()
            rows[m][f"scaled_width_{l}"] = ((d[f"{m}-hi-{l}"] - d[f"{m}-lo-{l}"]) / d["_ms_scale"]).median()
        taus, cols = [0.5], [m]
        for l in lv:
            a = (1 - l / 100) / 2
            taus += [a, 1 - a]; cols += [f"{m}-lo-{l}", f"{m}-hi-{l}"]
        sq = per(L.scaled_mqloss, {m: cols}, quantiles=np.array(taus), seasonality=SEASON, train_df=train)
        rows[m]["scaled_crps"] = 2 * sq[m].mean()
    return pd.DataFrame(rows.values()).sort_values("mean_rmsse").reset_index(drop=True)


# ============================================================================
# HIERARCHY (Lab 05)
# ============================================================================

def mint_wls_var(Y_hat, Y_fitted, S, model="AutoETS"):
    """MinT with W = diag(in-sample error variance), solved with preconditioned conjugate gradients.

    Same estimator as hierarchicalforecast's MinTrace(method="wls_var"): reconciled = S (S'W⁻¹S)⁻¹ S'W⁻¹ ŷ.
    The library's sparse version uses an unpreconditioned solver and stalls here, because node variances range from
    about 1 (slow items) to about 10¹³ (the store total).
    """
    from scipy import sparse
    from scipy.sparse.linalg import LinearOperator, cg

    rows = list(S.row_labels)
    Sm = sparse.csr_matrix(S.to_csr())
    # W: mean squared in-sample one-step error per node, as in hierarchicalforecast's dense MinTrace("wls_var")
    err2 = (Y_fitted.y - Y_fitted[model]) ** 2
    res = (err2.groupby(Y_fitted.unique_id).sum() / Y_fitted.groupby("unique_id").size()).reindex(rows).to_numpy() + 2e-8
    winv = 1.0 / res
    St = Sm.T.tocsr()
    n_b = Sm.shape[1]
    # A = S'W⁻¹S is dense (the total node touches every item), so never form it: apply it as an operator and
    # solve with conjugate gradients, preconditioned by its diagonal (which fixes the huge scale differences).
    A = LinearOperator((n_b, n_b), matvec=lambda v: St @ (winv * (Sm @ v)))
    diagA = np.asarray(Sm.multiply(Sm).T @ winv).ravel()
    M = LinearOperator((n_b, n_b), matvec=lambda v: v / diagA)
    yhat = Y_hat.pivot(index="unique_id", columns="ds", values=model).reindex(rows)
    B = St @ (winv[:, None] * yhat.to_numpy())                                 # (n_bottom, h)
    cols_out = []
    for j in range(B.shape[1]):
        x, info = cg(A, B[:, j], M=M, rtol=1e-12, atol=0.0, maxiter=5000)
        if info != 0:
            raise RuntimeError(f"CG did not converge (info={info}) for horizon step {j}")
        cols_out.append(x)
    Xb = np.column_stack(cols_out)
    rec = pd.DataFrame(Sm @ Xb, index=rows, columns=yhat.columns)
    return rec.stack().rename("MinT-wls_var").reset_index().rename(columns={"level_0": "unique_id"})


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
        return pd.DataFrame(columns=["model", "mean_rmsse", "median_rmsse", "mean_mase", "median_mase", "mean_smape",
                                     "wape", "bias_pct", "n_series", "lab", "fit_seconds", "notes", "recorded_at"])
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
