"""Plumbing for the full-data Day 2 notebooks (``day2_neural/code_full``).

The notebooks train on every eligible series and score on the original labs' 2,000-series
sample, so their numbers line up with the original labs. This module sits on top of
``bootcamp.py`` (imported, never edited):

- it points ``bootcamp``'s data and output paths at ``results/day2_full`` (the original labs'
  ``results/day2`` is never touched), then ``bc.split``, ``bc.score`` and ``bc.record`` work
  unchanged;
- it loads the full training panel and the fixed evaluation ids;
- it saves and loads configs, forecasts, tables and models;
- it holds the GFM training recipe from the research (see ``docs/gfm_learnings.md`` §3):
  scaled step budget, warm-up then step decay, best-weights retention, gradient clipping,
  Huber training loss with MAE validation, and the target shifted by one.

Environment variables:

- ``GD_TRAIN=1``: fit models and write results (the GPU run). Default: load saved results.
- ``GD_REFRESH=1``: with ``GD_TRAIN=1``, recompute cached tables (``cached``) even if they
  exist. Without it a TRAIN run reuses any table already on disk, so a crashed run resumes.
- ``GD_SMOKE=1``: tiny panel and a few training steps, written to ``results/day2_full_smoke``.
- ``GD_N_JOBS``: StatsForecast worker processes (default all cores; 1 in a smoke run).
- ``BOOTCAMP_ACCELERATOR``: ``auto`` (default), ``cpu`` or ``gpu``, read by ``bootcamp``.
- ``CUDA_VISIBLE_DEVICES``: which GPU to use. Every fit uses one device (``trainer_kwargs``);
  ``run_gpu.sh`` defaults this to ``0``.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import sys
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytorch_lightning as pl_lightning
import torch
from hierarchicalforecast.methods import MinTraceSparse
from scipy import sparse
from scipy.sparse.linalg import LinearOperator

CODE_FULL = Path(__file__).resolve().parent
ROOT = CODE_FULL.parents[1]
sys.path.insert(0, str(ROOT / "day2_neural" / "v1" / "code"))

import bootcamp as bc  # noqa: E402  (the shared Day 2 harness; needs the path above)



def _quiet_known_warnings() -> None:
    """Silence the known noisy library warnings only, so real ones still show on the GPU box.

    Lightning's advice (``PossibleUserWarning``: few workers, small logging interval) and the
    deprecation and user warnings that lightning, neuralforecast and the Nixtla libraries raise
    on every fit are hidden. Anything else, e.g. from torch or our own code, is shown.
    """
    noisy = (r"(pytorch_lightning|lightning|lightning_fabric|neuralforecast|statsforecast|"
             r"mlforecast|utilsforecast|coreforecast|hierarchicalforecast)(\.|$)")
    for category in (FutureWarning, UserWarning, DeprecationWarning):
        warnings.filterwarnings("ignore", category=category, module=noisy)
    try:
        from lightning_fabric.utilities.warnings import PossibleUserWarning
    except ImportError:
        from pytorch_lightning.utilities.warnings import PossibleUserWarning
    warnings.filterwarnings("ignore", category=PossibleUserWarning)
    # bootcamp (read-only here) uses polars and pandas idioms those libraries now deprecate
    for category in (FutureWarning, DeprecationWarning):
        warnings.filterwarnings("ignore", category=category, module="bootcamp")


_quiet_known_warnings()

# ============================================================================
# SETTINGS AND PATHS
# ============================================================================

TRAIN = os.environ.get("GD_TRAIN") == "1"
REFRESH = os.environ.get("GD_REFRESH") == "1"
SMOKE = os.environ.get("GD_SMOKE") == "1"

SEEDS = [46335, 11207, 90218] if not SMOKE else [46335, 11207]
N_JOBS = int(os.environ.get("GD_N_JOBS", 1 if SMOKE else -1))   # StatsForecast workers
SMOKE_N_SERIES = 150          # training series in a smoke run
SMOKE_N_EVAL = 60             # of which scored
EVAL_N_SERIES = 2000         # the original labs' sample (bc.N_SERIES now defaults to 0)
SMOKE_MAX_STEPS = 20

OUT_DIR = ROOT / "results" / ("day2_full_smoke" if SMOKE else "day2_full")
CONFIG_DIR = OUT_DIR / "configs"
FORECAST_DIR = OUT_DIR / "forecasts"
TABLE_DIR = OUT_DIR / "tables"
RUNS_DIR = OUT_DIR / "runs"
MODEL_DIR = ROOT / "models" / ("day2_full_smoke" if SMOKE else "day2_full")
EVAL_IDS_PATH = (OUT_DIR if SMOKE else ROOT / "results") / "eval_ids.csv"
EXAMPLES_PATH = OUT_DIR / "example_ids.json"
CPI_PATH = CODE_FULL / "data" / "cpi_weekly_deflator.csv"

for _d in (CONFIG_DIR, FORECAST_DIR, TABLE_DIR, RUNS_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Point the shared harness at the full-data locations. Its functions read these at call time.
bc.DATA_DIR = ROOT / "data"
bc.RESULTS_DIR = OUT_DIR
bc.FORECAST_DIR = FORECAST_DIR
bc.RAW_DIR = OUT_DIR / "raw"
bc.LEADERBOARD_PATH = OUT_DIR / "leaderboard.csv"
bc.DAY1_LEADERBOARD_PATH = OUT_DIR / "no_day1_leaderboard.csv"   # Day 1 has no RMSSE board

Y_SHIFT = 1.0                 # the recipe trains on y + 1 (safe robust scaling, ratios)


def status() -> None:
    """Print the run mode, so every notebook says where it reads and writes."""
    mode = "TRAIN (fit and save)" if TRAIN else "LOAD (read saved results)"
    smoke = " | SMOKE run" if SMOKE else ""
    print(f"{mode}{smoke} | results: {OUT_DIR.relative_to(ROOT)} | "
          f"accelerator: {bc.ACCELERATOR}")


# ============================================================================
# DATA: train on everything eligible, score on the original labs' sample
# ============================================================================

def eval_ids() -> list[str]:
    """The fixed evaluation series: the original labs' stratified sample (written once to csv)."""
    if not EVAL_IDS_PATH.exists():
        if SMOKE:   # every non-J series of the tiny panel, topped up with class J
            classes = train_panel().select("unique_id", "sale_class").unique().sort("unique_id")
            ids = classes.filter(pl.col("sale_class") != "J")["unique_id"].to_list()
            rest = classes.filter(pl.col("sale_class") == "J")["unique_id"].to_numpy()
            rng = np.random.default_rng(bc.RANDOM_SEED)
            n_more = max(0, min(len(rest), SMOKE_N_EVAL - len(ids)))
            ids += rng.choice(rest, size=n_more, replace=False).tolist()
        else:
            ids = bc.load_panel(n_series=EVAL_N_SERIES, seed=bc.RANDOM_SEED)["unique_id"].unique()
            ids = ids.to_list()
        pd.DataFrame({"unique_id": sorted(ids)}).to_csv(EVAL_IDS_PATH, index=False)
    return pd.read_csv(EVAL_IDS_PATH)["unique_id"].astype(str).tolist()


def train_panel() -> pl.DataFrame:
    """Every series that passes the original labs' filters (``load_panel(n_series=0)``).

    A smoke run uses a small sample instead (``SMOKE_N_SERIES``).
    """
    n = SMOKE_N_SERIES if SMOKE else 0
    return bc.load_panel(n_series=n, seed=bc.RANDOM_SEED)


def dropped_series() -> pd.DataFrame:
    """Why series are missing from the training panel: the original labs' filters, in order.

    Also shows what the early research filters (a zero run of 26+ weeks, fewer than 52 weeks
    of history) would have removed on top. They are reported, not applied.
    """
    raw = pl.read_parquet(bc.DATA_DIR / "sales_weekly.parquet").rename(
        {"week_start": "ds", "sales_units": "y"})
    train = raw.filter(pl.col("ds") < bc.TEST_START)
    bad_price = set(raw.filter(pl.col("sale_price") <= 0)["unique_id"].unique().to_list())
    stats = train.group_by("unique_id").agg(pl.len().alias("n"), pl.col("y").std().alias("sd"))
    short = set(stats.filter(pl.col("n") < bc.MIN_TRAIN_WEEKS)["unique_id"].to_list())
    flat = set(stats.filter(pl.col("sd").fill_null(0) == 0)["unique_id"].to_list())
    zero_run = _longest_zero_run(train)
    long_zero = set(zero_run.filter(pl.col("run") >= 26)["unique_id"].to_list())

    all_ids = set(raw["unique_id"].unique().to_list())
    rows, left = [], set(all_ids)
    for reason, ids in [("price <= 0 in any week", bad_price),
                        (f"fewer than {bc.MIN_TRAIN_WEEKS} training weeks", short),
                        ("constant in training (incl. all zero)", flat)]:
        hit = left & ids
        rows.append({"filter": reason, "applied": True, "series_dropped": len(hit)})
        left -= hit
    rows.append({"filter": "early research: zero run of 26+ weeks", "applied": False,
                 "series_dropped": len(left & long_zero)})
    out = pd.DataFrame(rows)
    print(f"{len(all_ids):,} series in the data, {len(left):,} kept for training")
    return out


def _longest_zero_run(df: pl.DataFrame) -> pl.DataFrame:
    """Longest run of consecutive zero-sales weeks per series."""
    runs = (df.sort("unique_id", "ds")
            .with_columns((pl.col("y") == 0).alias("zero"))
            .with_columns((pl.col("zero") != pl.col("zero").shift()).fill_null(True)
                          .cum_sum().over("unique_id").alias("block")))
    return (runs.filter(pl.col("zero")).group_by("unique_id", "block").len()
            .group_by("unique_id").agg(pl.col("len").max().alias("run")))


def only(df, ids):
    """Rows of ``df`` (pandas or polars) whose ``unique_id`` is in ``ids``."""
    if isinstance(df, pl.DataFrame):
        return df.filter(pl.col("unique_id").is_in(list(ids)))
    df = df.reset_index() if "unique_id" not in df.columns else df
    return df[df["unique_id"].isin(set(ids))]


@dataclass
class NixtlaData:
    """Everything a neuralforecast fit and predict needs, in pandas.

    ``fit_df`` is the training frame (possibly with oversampled copies and ``y`` shifted),
    ``predict_df`` the history used to forecast the eval series, ``futr_df`` their future
    covariates and ``static_df`` the statics for every id in either frame.
    """

    fit_df: pd.DataFrame
    predict_df: pd.DataFrame
    futr_df: pd.DataFrame
    static_df: pd.DataFrame | None
    futr_exog: list[str] = field(default_factory=list)
    stat_exog: list[str] = field(default_factory=list)
    y_shift: float = 0.0


def nixtla_data(train: pl.DataFrame, test: pl.DataFrame, futr_exog: list[str] | None = None,
                statics: bool = True, y_shift: float = Y_SHIFT,
                predict_ids: list[str] | None = None,
                extra_statics: pd.DataFrame | None = None) -> NixtlaData:
    """Build the pandas frames for a fit on ``train`` and a forecast over ``test``.

    Statics come from ``bc.static_frame`` (training weeks only), plus any ``extra_statics``
    (one row per ``unique_id``). Forecasts are made for ``predict_ids`` (default: the eval ids)
    to keep prediction fast.
    """
    futr_exog = list(futr_exog or [])
    ids = predict_ids if predict_ids is not None else eval_ids()
    fit_df = bc.to_nixtla(train, futr_exog)
    fit_df["y"] = fit_df["y"] + y_shift
    predict_df = only(fit_df, ids)
    futr_df = only(bc.to_nixtla(test, futr_exog).drop(columns="y"), ids)
    static_df = bc.static_frame(train) if statics else None
    stat_exog = list(bc.STATIC_COLS) if statics else []
    if statics and extra_statics is not None:
        static_df = static_df.merge(extra_statics, on="unique_id", how="left").fillna(0.0)
        stat_exog += [c for c in extra_statics.columns if c != "unique_id"]
    return NixtlaData(fit_df, predict_df, futr_df, static_df, futr_exog, stat_exog, y_shift)


def subset_data(data: NixtlaData, ids) -> NixtlaData:
    """The same data restricted to ``ids``, for quick what-if predictions (sweeps, probes).

    Statics are kept whole: ``predict`` picks the rows it needs.
    """
    return NixtlaData(only(data.fit_df, ids), only(data.predict_df, ids),
                      only(data.futr_df, ids), data.static_df, data.futr_exog,
                      data.stat_exog, data.y_shift)


# ============================================================================
# SCORING AND THE BOARD
# ============================================================================

def score(fcst, test, train) -> pd.DataFrame:
    """``bc.score`` on the eval ids only: per-series metrics in long format."""
    ids = eval_ids()
    return bc.score(only(fcst, ids), only(test, ids), only(train, ids))


def record(fcst, test, train, lab: str, **kwargs) -> pd.DataFrame:
    """``bc.record`` on the eval ids only, so every score covers the original labs' 2,000
    series."""
    ids = eval_ids()
    return bc.record(only(fcst, ids), only(test, ids), only(train, ids), lab=lab, **kwargs)


def median_board(upto: str | None = None) -> pd.DataFrame:
    """The full-data leaderboard, sorted by median RMSSE (mean alongside).

    ``upto`` is the calling notebook's number ("07", "07b"): only rows from that notebook and
    earlier ones are kept, so a notebook's board never shows models that come after it. Every
    lab label starts with its notebook number, and plain string order is notebook order
    ("05" < "05b" < "06").
    """
    board = bc.leaderboard(include_day1=False)
    if upto is not None:
        number = board["lab"].astype(str).str.split().str[0]
        board = board[number <= upto]
    cols = ["model", "median_rmsse", "mean_rmsse", "wape", "n_series", "lab", "fit_seconds"]
    board = board[[c for c in cols if c in board.columns]]
    return board.sort_values("median_rmsse").reset_index(drop=True).round(3)


def per_series(metrics: pd.DataFrame, metric: str = "rmsse") -> pd.DataFrame:
    """Wide frame of one metric: index ``unique_id``, one column per model."""
    return metrics[metrics["metric"] == metric].drop(columns="metric").set_index("unique_id")


# ============================================================================
# CONFIGS, TABLES, FORECASTS, MODELS
# ============================================================================

@contextmanager
def _atomic(path: Path):
    """Write to a temporary name next to ``path``, then rename it into place.

    A crash mid-write leaves the old file (or none), never a half-written one that a LOAD run
    would fail to read. Works for files and folders.
    """
    tmp = path.with_name(f".{path.name}.tmp")
    if tmp.is_dir():
        shutil.rmtree(tmp)
    yield tmp
    if path.is_dir():
        shutil.rmtree(path)
    os.replace(tmp, path)


def save_config(name: str, config: dict) -> Path:
    """Write ``configs/<name>.json``; the next notebook reads it instead of hardcoding."""
    path = CONFIG_DIR / f"{name}.json"
    with _atomic(path) as tmp:
        tmp.write_text(json.dumps(_no_nan(config), indent=2, default=_jsonable,
                                  allow_nan=False))
    return path


def _no_nan(x):
    """Replace NaN and infinity with ``None`` (null), so configs are strict JSON."""
    if isinstance(x, dict):
        return {k: _no_nan(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_no_nan(v) for v in x]
    if isinstance(x, (float, np.floating)) and not np.isfinite(x):
        return None
    return x


def load_config(name: str) -> dict:
    """Read a config written by an earlier notebook."""
    path = CONFIG_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found: run the notebook that writes it first.")
    return json.loads(path.read_text())


def _jsonable(x):
    """``json.dumps`` fallback: numpy numbers become Python numbers; dates and paths strings."""
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, (date, pd.Timestamp, Path)):
        return str(x)
    raise TypeError(f"not JSON serialisable: {type(x)}")


def cached(name: str, compute) -> pd.DataFrame:
    """Load ``tables/<name>.parquet``, or (in a TRAIN run) compute it, save it and return it.

    This is how notebooks stay readable on the day: heavy work sits inside ``compute`` and
    only runs on the GPU box. A TRAIN run reuses a table that already exists, so a crashed run
    resumes; set ``GD_REFRESH=1`` to recompute it. A missing table in a LOAD run is an error,
    not a silent refit. ``compute`` must not rely on another ``cached`` call having run.
    """
    path = TABLE_DIR / f"{name}.parquet"
    if path.exists() and not (TRAIN and REFRESH):
        return pd.read_parquet(path)
    if not TRAIN:
        raise FileNotFoundError(f"{path.name} missing: run this notebook once with GD_TRAIN=1.")
    out = compute()
    with _atomic(path) as tmp:
        out.to_parquet(tmp, index=False)
    return out


STABILITY_CONFIG = "05c_input_clip"     # 05c's choice of price cleaning, floor and clip


def selected_config() -> dict:
    """The model every notebook after 05 builds on.

    05c's chosen arm if it has one (05's settings with cleaned prices, the scaler floor and,
    for ``clean_floor_clip``, the input clip: ``docs/stability_choices.md``); otherwise 05b's
    if it won; otherwise 05's. ``selected_from`` says which. The floor and clip travel in
    ``params`` (``scale_floor``, ``input_clip``), so ``build_nbeatsx`` applies them to every
    refit of the selected model.
    """
    if (CONFIG_DIR / f"{STABILITY_CONFIG}.json").exists():
        stable = load_config(STABILITY_CONFIG)
        arm = stable.get("use")
        if arm in ("clean_floor", "clean_floor_clip"):
            base = load_config("05_selected")              # 05c's arms are built on 05
            clip = stable["input_clip"] if arm == "clean_floor_clip" else None
            alias = "GFM_" + arm
            return {**base, "price_clean": stable["price_clean"],
                    "params": {**base["params"], "scale_floor": stable["scale_floor"],
                               "input_clip": clip},
                    "selected_from": "05c", "use": arm, "forecasts": f"05c_{arm}",
                    "forecast_column": alias, "saved_model": "05c_selected",
                    "selected_runs": f"05c_{arm}"}
    synthetic = CONFIG_DIR / "05b_synthetic.json"
    if synthetic.exists():
        cfg = load_config("05b_synthetic")
        if cfg.get("use") == "05b":
            return {**cfg, "selected_from": "05b"}
    return {**load_config("05_selected"), "selected_from": "05"}


def selected_forecasts(cfg: dict | None = None) -> pd.DataFrame:
    """Forecasts of the selected model, with the column renamed to ``GFM_selected``.

    05 saves ``05_selected.parquet`` (column ``GFM_selected``); if 05b won, its config names its
    own file and column (``forecasts``, ``forecast_column``).
    """
    cfg = cfg or selected_config()
    name = cfg.get("forecasts", "05_selected")
    col = cfg.get("forecast_column", "GFM_selected")
    fcst = pd.read_parquet(FORECAST_DIR / f"{name}.parquet")
    return fcst[["unique_id", "ds", col]].rename(columns={col: "GFM_selected"})


def save_forecasts(fcst, name: str) -> Path:
    """Point forecasts for the eval ids, for the selection notebook.

    The same file ``bc.save_forecasts`` writes (``forecasts/<name>.parquet``: ``unique_id``,
    ``ds`` and the model columns), written atomically.
    """
    fcst = only(fcst, eval_ids())
    fcst = fcst.to_pandas() if isinstance(fcst, pl.DataFrame) else fcst
    path = FORECAST_DIR / f"{name}.parquet"
    with _atomic(path) as tmp:
        fcst[["unique_id", "ds"] + bc.model_columns(fcst)].to_parquet(tmp, index=False)
    return path


def save_model(nf, name: str) -> Path:
    """Save a fitted ``NeuralForecast`` without its training data (tens of MB, not GB)."""
    path = MODEL_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with _atomic(path) as tmp:
        nf.save(str(tmp), save_dataset=False, overwrite=True)
        stability = _stability_of(nf.models[0])
        if stability:                 # the library does not save a patched scaler: note it
            (Path(tmp) / STABILITY_FILE).write_text(json.dumps(stability))
    return path


def copy_model(source: str, name: str) -> Path:
    """Save an already-saved model under a second name (e.g. the winner as ``05_selected``)."""
    path = MODEL_DIR / name
    with _atomic(path) as tmp:
        shutil.copytree(MODEL_DIR / source, tmp)
    return path


def load_model(name: str):
    """Load a model saved by ``save_model`` (predict with an explicit ``df=``)."""
    from neuralforecast import NeuralForecast

    path = MODEL_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: download the models or run with GD_TRAIN=1.")
    nf = NeuralForecast.load(str(path))
    if (path / STABILITY_FILE).exists():  # reapply the scaler floor and clip it was fitted with
        stability = json.loads((path / STABILITY_FILE).read_text())
        for model in nf.models:
            floor_robust_scaler(model, stability["scale_floor"], clip=stability["input_clip"])
    return nf


# ============================================================================
# THE TRAINING RECIPE (docs/gfm_learnings.md §3)
# ============================================================================

def scaled_max_steps(n_series: int, batch_size: int = 256, visits: int = 174,
                     floor: int = 2_000, ceiling: int = 250_000) -> int:
    """Training steps for each series to be drawn about ``visits`` times.

    A fixed budget under-trains a bigger panel. The research settled on ~174 visits per series.
    The floor is lower than the research's 10k because this panel is smaller.
    """
    if SMOKE:
        return SMOKE_MAX_STEPS
    steps = visits * n_series / batch_size
    return int(min(max(steps, floor), ceiling))


def warmup_then_step(step: int, warmup_steps: int, decay_every: int,
                     gamma: float = 0.5) -> float:
    """Learning-rate multiplier: linear warm-up, then ×``gamma`` every ``decay_every`` steps."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    return gamma ** ((step - warmup_steps) // decay_every)


class BestWeights(pl_lightning.Callback):
    """Keep the weights with the lowest validation loss (in memory) and restore them at the end.

    The research ran the full step budget instead of stopping early, because most runs finished
    above their own best validation loss. Restoring the best weights made 6 of 6 seeds usable.
    """

    def __init__(self):
        self.best_loss, self.best_step, self._state = float("inf"), None, None

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking or not pl_module.valid_trajectories:
            return
        step, loss = pl_module.valid_trajectories[-1]
        if loss < self.best_loss:
            self.best_loss, self.best_step = loss, step
            state = pl_module.state_dict()
            self._state = {k: v.detach().cpu().clone() for k, v in state.items()}

    def on_fit_end(self, trainer, pl_module):
        if self._state is not None:
            pl_module.load_state_dict(self._state)
        self._state = None                            # free the copy; not saved with the model
        pl_module.best_step = self.best_step          # read back by ``health``


def trainer_kwargs() -> dict:
    """``bc.trainer_kwargs`` pinned to one device.

    Lightning's default ``devices="auto"`` trains on every visible GPU. On a multi-GPU box that
    starts distributed training inside the notebook kernel, which hangs, and it would also
    split each batch across cards. One device keeps every fit the same as a single-GPU run.
    """
    return dict(bc.trainer_kwargs(), devices=1)


def recipe_kwargs(max_steps: int, recipe: bool = True, val_check_steps: int | None = None,
                  warmup_frac: float = 0.05, n_decays: int = 3) -> dict:
    """Training settings for a neuralforecast model, with the research recipe on or off.

    Off is the library default (MAE loss, StepLR halving, no clipping). On adds Huber loss with
    MAE validation, 5% warm-up then ``n_decays`` halvings, gradient clip 1.0 and best weights.
    """
    from neuralforecast.losses.pytorch import MAE, HuberLoss

    val_check_steps = val_check_steps or max(5, min(500, max_steps // 40))
    kwargs = dict(max_steps=max_steps, val_check_steps=val_check_steps,
                  early_stop_patience_steps=-1, **trainer_kwargs())
    if not recipe:
        return dict(kwargs, loss=MAE(), valid_loss=MAE())
    warmup = max(1, int(warmup_frac * max_steps))
    decay_every = max(1, (max_steps - warmup) // (n_decays + 1))
    schedule = partial(warmup_then_step, warmup_steps=warmup, decay_every=decay_every)
    return dict(kwargs, loss=HuberLoss(), valid_loss=MAE(),
                lr_scheduler=torch.optim.lr_scheduler.LambdaLR,
                lr_scheduler_kwargs={"lr_lambda": schedule},
                gradient_clip_val=1.0, callbacks=[BestWeights()])


def warmup_steps(max_steps: int, warmup_frac: float = 0.05) -> int:
    """The warm-up length ``recipe_kwargs`` uses (for marking it on plots)."""
    return max(1, int(warmup_frac * max_steps))


# The research's phase 4 defaults (single store). Chosen on a test window in the research,
# so treat them as a starting point, not a tuned answer (docs/gfm_learnings.md §1).
NBEATSX_DEFAULTS = {
    "input_size": 14,
    "stack_types": ["identity", "trend", "seasonality", "exogenous"],
    "n_blocks": [2, 2, 2, 2],
    "mlp_size": 512,
    "mlp_depth": 2,
    "n_harmonics": 3,
    "n_polynomials": 3,
    "dropout_prob_theta": 0.257,
    "learning_rate": 4.83e-4,
    "batch_size": 256,
    "scaler_type": "robust",
}


def _floored_robust_statistics(x, mask, dim=-1, eps=1e-6, floor: float = 0.0):
    """The library's robust statistics, with the spread never below ``floor``."""
    from neuralforecast.common._scalers import robust_statistics

    median, spread = robust_statistics(x, mask, dim=dim, eps=eps)
    return median, spread.clamp(min=floor)


def _clipped_robust_scaler(x, x_median, x_mad, clip: float = float("inf")):
    """The library's robust scaling, with every channel but the first (the target) clipped
    to ``[-clip, clip]``. Windows are ``[batch, time, channel]``, the target first."""
    z = (x - x_median) / x_mad
    return torch.cat([z[:, :, :1], z[:, :, 1:].clamp(-clip, clip)], dim=2)


def floor_robust_scaler(model, floor: float, clip: float | None = None):
    """Floor the robust window scaler's spread at ``floor`` in ``model``, in place.

    ``clip`` also bounds the scaled covariates (not the target) at ``±clip``, so a large real
    price move cannot reach the exogenous stack at any size.

    The library scales each input over the input weeks by its median and MAD. A covariate
    that barely moved there (a price drifting by 0.001 with CPI) gets a tiny spread, so a
    discount in the forecast weeks becomes hundreds of scaled units. The floor bounds that.
    The patch is on the model instance, so it survives the copy ``NeuralForecast.fit`` makes,
    but a model loaded from disk has the library's scaler again.
    """
    from neuralforecast.common._scalers import TemporalNorm

    norms = [m for m in model.modules()
             if isinstance(m, TemporalNorm) and m.scaler_type == "robust"]
    if not norms:
        raise ValueError("floor_robust_scaler: the model has no robust scaler")
    for norm in norms:
        norm.compute_statistics = partial(_floored_robust_statistics, floor=floor)
        if clip is not None:
            norm.scaler = partial(_clipped_robust_scaler, clip=clip)
    return model


STABILITY_FILE = "stability.json"     # next to a saved model: the floor and clip to reapply


def _stability_of(model) -> dict | None:
    """``{"scale_floor", "input_clip"}`` if ``floor_robust_scaler`` patched ``model``."""
    from neuralforecast.common._scalers import TemporalNorm

    for m in model.modules():
        stats = getattr(m, "compute_statistics", None) if isinstance(m, TemporalNorm) else None
        if isinstance(stats, partial) and stats.func is _floored_robust_statistics:
            scaler = m.scaler
            clip = scaler.keywords["clip"] if isinstance(scaler, partial) else None
            return {"scale_floor": stats.keywords["floor"], "input_clip": clip}
    return None


def build_nbeatsx(params: dict, data: NixtlaData, max_steps: int, seed: int, alias: str,
                  recipe: bool = True, loss=None, valid_loss=None):
    """An NBEATSx from a params dict (as stored in configs) plus the training recipe.

    ``mlp_size`` and ``mlp_depth`` give every stack the same MLP: ``[[size] * depth] * stacks``.
    ``scale_floor`` and ``input_clip`` (05c) floor the robust scaler's spread and clip the
    scaled covariates (``floor_robust_scaler``).
    ``loss`` and ``valid_loss`` replace the recipe's losses when given (e.g. a multi-quantile
    loss for intervals); everything else in the recipe stays.
    """
    from neuralforecast.models import NBEATSx

    p = {**NBEATSX_DEFAULTS, **params}
    floor, clip = p.pop("scale_floor", None), p.pop("input_clip", None)
    mlp_units = [[p.pop("mlp_size")] * p.pop("mlp_depth")] * len(p["stack_types"])
    if not data.futr_exog:            # the exogenous stack needs future covariates
        p["stack_types"] = [s for s in p["stack_types"] if s != "exogenous"]
        p["n_blocks"] = p["n_blocks"][:len(p["stack_types"])]
        mlp_units = mlp_units[:len(p["stack_types"])]
    train_kwargs = recipe_kwargs(max_steps, recipe=recipe)
    if loss is not None:
        train_kwargs["loss"] = loss
    if valid_loss is not None:
        train_kwargs["valid_loss"] = valid_loss
    model = NBEATSx(h=bc.H, futr_exog_list=data.futr_exog or None,
                    stat_exog_list=data.stat_exog or None, mlp_units=mlp_units,
                    random_seed=seed, alias=alias, **p, **train_kwargs)
    if floor is not None:             # 05c's scaler floor (and clip): docs/stability_choices.md
        floor_robust_scaler(model, floor, clip=clip)
    return model


# ============================================================================
# FITTING: one seed at a time, cached so a crash resumes and LOAD runs are instant
# ============================================================================

def trajectory(model, every: int = 50) -> pd.DataFrame:
    """Training and validation loss by step (training loss averaged into ``every``-step bins)."""
    tr = pd.DataFrame(model.train_trajectories, columns=["step", "train_loss"])
    tr["step"] = (tr["step"] // every) * every
    tr = tr.groupby("step", as_index=False)["train_loss"].mean()
    va = pd.DataFrame(model.valid_trajectories, columns=["step", "valid_loss"])
    return tr.merge(va, on="step", how="outer").sort_values("step").reset_index(drop=True)


def health(traj: pd.DataFrame, max_steps: int, best_step: int | None) -> dict:
    """Training health from the validation curve (never from elasticity).

    - ``oscillation``: median relative jump between validation checks in the second half;
    - ``best_frac``: where the best weights came from (above ~0.65 suggests under-training);
    - ``final_over_best``: final validation loss relative to the best one.
    """
    va = traj.dropna(subset=["valid_loss"])
    if va.empty:
        return {}
    tail = va[va["step"] >= va["step"].max() / 2]["valid_loss"].to_numpy()
    jumps = np.abs(np.diff(tail)) / tail[:-1] if len(tail) > 1 else np.array([np.nan])
    best = va["valid_loss"].min()
    return {"oscillation": float(np.median(jumps)),
            "best_step": best_step,
            "best_frac": (best_step / max_steps) if best_step is not None else None,
            "best_valid": float(best),
            "final_over_best": float(va["valid_loss"].iloc[-1] / best)}


def _run_dir(name: str, seed: int) -> Path:
    """Folder that caches one seed of one named run: ``runs/<name>/seed<seed>``."""
    return RUNS_DIR / name / f"seed{seed}"


def clear_gpu() -> None:
    """Free Python garbage and the GPU memory torch has cached, between fits."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def fit_seed(build_model, seed: int, data: NixtlaData, name: str, val_size: int = bc.H,
             extras=None, save_as: str | None = None) -> dict:
    """Fit one model for one seed, forecast the eval ids, and cache the results.

    ``build_model(seed)`` returns an unfitted neuralforecast model whose alias is the column
    name. ``extras(nf, data)`` may return a dict of extra frames (e.g. a price sweep), cached
    with the run. ``save_as`` also saves the fitted model to ``models/`` (only for models a
    later notebook or a demo loads; read it back with ``load_model``). Returns ``forecast``,
    ``trajectory``, ``summary`` and the extras: the same as ``load_runs`` reads from disk.

    The fitted model is not kept in memory: each one holds its own copy of the training data,
    so an HPO notebook would otherwise grow by one panel per fit. Everything is on disk before
    ``summary.json``, which is written last, so a crash at any point loses only this fit.
    """
    from neuralforecast import NeuralForecast

    run_dir = _run_dir(name, seed)
    done = run_dir / "summary.json"
    if done.exists():                     # finished earlier (resume or LOAD); delete to refit
        if TRAIN and save_as and not (MODEL_DIR / save_as).exists():
            print(f"{save_as} is not saved, but {run_dir} is cached: delete that folder to "
                  "refit and save it.")
        return _load_run(run_dir)
    if not TRAIN:
        raise FileNotFoundError(f"{run_dir} missing: run this notebook once with GD_TRAIN=1.")

    clear_gpu()
    model = build_model(seed)
    nf = NeuralForecast(models=[model], freq=bc.FREQ)
    with bc.Timer() as t:
        nf.fit(df=data.fit_df, static_df=data.static_df, val_size=val_size)
    fcst = predict(nf, data)
    fitted = nf.models[0]
    traj = trajectory(fitted)
    summary = {"name": name, "seed": seed, "alias": model.alias or type(model).__name__,
               "fit_seconds": round(t.seconds, 1), "max_steps": fitted.max_steps,
               "n_train_series": int(data.fit_df["unique_id"].nunique()),
               "neg_share": float((fcst.select_dtypes("number") < 0).mean().mean()),
               **health(traj, fitted.max_steps, getattr(fitted, "best_step", None))}

    run_dir.mkdir(parents=True, exist_ok=True)
    out = {"forecast": fcst, "trajectory": traj, "summary": summary}
    if extras:
        out.update(extras(nf, data))
    for key, frame in out.items():
        if key == "summary":
            continue
        suffix = ".csv" if key == "trajectory" else ".parquet"
        with _atomic(run_dir / f"{key}{suffix}") as tmp:
            if suffix == ".csv":
                frame.to_csv(tmp, index=False)
            else:
                frame.to_parquet(tmp, index=False)
    if save_as:
        save_model(nf, save_as)
    with _atomic(done) as tmp:                                           # written last
        tmp.write_text(json.dumps(summary, indent=2, default=_jsonable))
    del nf, model, fitted
    clear_gpu()
    return out


def _load_run(run_dir: Path) -> dict:
    """A finished run read back from disk, in the same shape ``fit_seed`` returns."""
    out = {"summary": json.loads((run_dir / "summary.json").read_text()),
           "trajectory": pd.read_csv(run_dir / "trajectory.csv")}
    for path in sorted(run_dir.glob("*.parquet")):
        out[path.stem] = pd.read_parquet(path)
    return out


def load_runs(name: str, seeds: list[int] | None = None) -> list[dict]:
    """Every finished seed of a named run, read from ``runs/<name>/``.

    For pulling any run into a notebook, e.g. a 05 trial or a 02 grid cell named in a table:
    ``gd.load_runs("05_trial_<key>")``. Same shape as ``fit_seeds`` returns, so ``seed_forecasts``,
    ``summaries`` and the plots take it directly. ``list_runs`` shows the names.
    """
    folder = RUNS_DIR / name
    found = {}
    for run_dir in folder.glob("seed*"):
        if (run_dir / "summary.json").exists():
            found[int(run_dir.name.removeprefix("seed"))] = run_dir
    if not found:
        raise FileNotFoundError(f"no finished runs in runs/{name}: see list_runs().")
    if seeds is None:                 # the order fit_seeds uses, then any other seeds
        seeds = [s for s in SEEDS if s in found]
        seeds += sorted(set(found) - set(seeds))
    missing = [s for s in seeds if s not in found]
    if missing:
        raise FileNotFoundError(f"{name}: no finished run for seed(s) {missing}.")
    return [_load_run(found[s]) for s in seeds]


def list_runs(prefix: str = "") -> pd.DataFrame:
    """One row per finished fit on disk (name, seed, model, minutes, training health).

    ``prefix`` narrows it to one notebook, e.g. ``list_runs("05_trial")``.
    """
    rows = []
    for path in sorted(RUNS_DIR.glob(f"{prefix}*/seed*/summary.json")):
        s = json.loads(path.read_text())
        rows.append({"name": path.parent.parent.name, "seed": s["seed"], "alias": s["alias"],
                     "fit_minutes": round(s["fit_seconds"] / 60, 1),
                     "best_frac": s.get("best_frac"), "best_valid": s.get("best_valid"),
                     "max_steps": s["max_steps"], "n_train_series": s["n_train_series"]})
    return pd.DataFrame(rows)


def predict(nf, data: NixtlaData, futr_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Forecast the eval ids, undoing the target shift. Returns unclipped forecasts."""
    futr = data.futr_df if futr_df is None else futr_df
    static = data.static_df
    if static is not None:
        static = static[static["unique_id"].isin(set(data.predict_df["unique_id"]))]
    fcst = nf.predict(df=data.predict_df, static_df=static, futr_df=futr)
    fcst = fcst.reset_index() if "unique_id" not in fcst.columns else fcst
    for col in bc.model_columns(fcst):
        fcst[col] = fcst[col] - data.y_shift
    return fcst


def fit_seeds(build_model, data: NixtlaData, name: str, seeds: list[int] | None = None,
              save_first_as: str | None = None, **kwargs) -> list[dict]:
    """``fit_seed`` for each seed (3 by default). Each run caches separately.

    ``save_first_as`` saves the first seed's model under that name as soon as it is fitted.
    """
    runs = []
    for i, seed in enumerate(seeds or SEEDS):
        save_as = save_first_as if i == 0 else None
        runs.append(fit_seed(build_model, seed, data, name, save_as=save_as, **kwargs))
    return runs


def seed_forecasts(runs: list[dict]) -> pd.DataFrame:
    """One wide frame with a column per seed (``<alias>_s<seed>``) and the seed mean."""
    out, cols = None, []
    for run in runs:
        alias, seed = run["summary"]["alias"], run["summary"]["seed"]
        f = run["forecast"][["unique_id", "ds", alias]]
        f = f.rename(columns={alias: f"{alias}_s{seed}"})
        cols.append(f"{alias}_s{seed}")
        out = f if out is None else out.merge(f, on=["unique_id", "ds"])
    out[f"{alias}_mean"] = out[cols].mean(axis=1)
    return out


def seed_medians(runs: list[dict], actual, history) -> pd.Series:
    """Median RMSSE of each seed's own forecast on ``actual`` (scaled by ``history``).

    One value per seed, indexed ``<alias>_s<seed>``; the seed-mean forecast is left out.
    """
    rmsse = per_series(score(seed_forecasts(runs), actual, history))
    seed_cols = [c for c in rmsse.columns if not c.endswith("_mean")]
    return rmsse[seed_cols].median()


def summaries(runs: list[dict]) -> pd.DataFrame:
    """The per-seed summaries as a table."""
    return pd.DataFrame([r["summary"] for r in runs])


def params_key(params: dict) -> str:
    """Short stable hash of a parameter dict, for caching HPO trials."""
    blob = json.dumps(params, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:10]


# ============================================================================
# HPO STUDIES
# ============================================================================

def _trial_settings(trial) -> dict:
    """The settings a trial ran (or was queued to run) with."""
    return trial.system_attrs.get("fixed_params") or dict(trial.params)


def run_study(study, objective, n_trials: int, first_trials: list[dict] | None = None,
              max_failures: int = 3) -> None:
    """Run an Optuna study until it has ``n_trials`` finished trials, resuming after a crash.

    The study lives in SQLite and every fit is cached on disk, so a stopped GPU run picks up
    where it left off:

    - on the first run, ``first_trials`` (e.g. coverage cells) are queued ahead of the sampler;
    - a trial the crash interrupted is marked failed and its settings queued again. Its
      finished seeds load from the cache, so only the interrupted fit is redone;
    - a trial that runs out of GPU memory is marked failed, the GPU cleared and its settings
      queued again. After ``max_failures`` of these the study stops with an error (rerun the
      notebook: a fresh process starts with empty memory). Any other error stops it at once.
    """
    from optuna.trial import TrialState

    if not study.trials and first_trials:
        for settings in first_trials:
            study.enqueue_trial(settings)
    for trial in study.get_trials(deepcopy=False, states=[TrialState.RUNNING]):
        study._storage.set_trial_state_values(trial._trial_id, state=TrialState.FAIL)
        study.enqueue_trial(_trial_settings(trial))
        print(f"trial {trial.number} was interrupted: queued again")

    failures = 0
    while len(study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])) < n_trials:
        failed_before = {t.number for t in study.get_trials(states=[TrialState.FAIL])}
        study.optimize(objective, n_trials=1, catch=(torch.cuda.OutOfMemoryError,))
        for trial in study.get_trials(states=[TrialState.FAIL]):
            if trial.number in failed_before:
                continue
            failures += 1
            clear_gpu()
            if failures > max_failures:
                raise RuntimeError(f"{failures} trials ran out of GPU memory: rerun the notebook "
                                   "(it resumes), or lower batch_size.")
            study.enqueue_trial(_trial_settings(trial))
            print(f"trial {trial.number} ran out of GPU memory: queued again")


# ============================================================================
# PARETO FRONT AND KNEE (docs/gfm_learnings.md §6)
# ============================================================================

def pareto_front(df: pd.DataFrame, x: str = "accuracy", y: str = "elasticity") -> pd.DataFrame:
    """Rows not dominated on (``x``, ``y``), both lower-is-better, sorted by ``x``."""
    pts = df[[x, y]].to_numpy()
    keep = []
    for i, (a, b) in enumerate(pts):
        dominated = ((pts[:, 0] <= a) & (pts[:, 1] <= b) & ((pts[:, 0] < a) | (pts[:, 1] < b)))
        keep.append(not dominated.any())
    return df[keep].sort_values(x)


def knee_by_gain(front: pd.DataFrame, x: str = "accuracy", y: str = "elasticity"):
    """Index of the front point with the most ``y`` gained per unit of ``x`` given up."""
    base = front.loc[front[x].idxmin()]
    lost = (front[x] - base[x]).replace(0, np.nan)
    gain = (base[y] - front[y]) / lost
    return gain.idxmax()


def knee_by_distance(front: pd.DataFrame, x: str = "accuracy", y: str = "elasticity"):
    """Index of the front point furthest from the chord joining its two ends (axes scaled
    0-1)."""
    pts = front[[x, y]].to_numpy(dtype=float)
    span = pts.max(axis=0) - pts.min(axis=0)
    span[span == 0] = 1.0
    p = (pts - pts.min(axis=0)) / span
    a, b = p[0], p[-1]
    chord = (b - a) / (np.linalg.norm(b - a) or 1.0)
    dist = np.abs((p[:, 0] - a[0]) * chord[1] - (p[:, 1] - a[1]) * chord[0])
    return front.index[int(np.argmax(dist))]



# ============================================================================
# CHRONOS-2 (07, 08)
# ============================================================================

CHRONOS_REPO = "autogluon/chronos-2-small" if SMOKE else "amazon/chronos-2"
CHRONOS_BATCH = 256           # series (target + covariates) per predict batch
CHRONOS_QUANTILES = {0.025: "-lo-95", 0.1: "-lo-80", 0.9: "-hi-80", 0.975: "-hi-95"}


def chronos_pipeline(repo: str = CHRONOS_REPO):
    """The Chronos-2 pipeline, on the GPU when there is one (weights cache outside the repo)."""
    from chronos import Chronos2Pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return Chronos2Pipeline.from_pretrained(repo, device_map=device)


def chronos_forecast(pipe, df: pd.DataFrame, future_df: pd.DataFrame | None = None,
                     alias: str = "Chronos2", h: int = bc.H,
                     quantiles: dict | None = None) -> pd.DataFrame:
    """Chronos-2 forecasts of the ``h`` weeks after each series in ``df``.

    ``df`` is long (``unique_id``, ``ds``, ``y`` and any covariates); covariates also in
    ``future_df`` are known in advance. Returns the point forecast as ``alias`` and, for each
    level in ``quantiles`` ({level: suffix}), a column ``alias + suffix``.
    """
    quantiles = quantiles or {}
    df = df.sort_values(["unique_id", "ds"])
    if future_df is not None:
        future_df = future_df.sort_values(["unique_id", "ds"])
    out = pipe.predict_df(df, future_df=future_df, id_column="unique_id", timestamp_column="ds",
                          target="y", prediction_length=h,
                          quantile_levels=list(quantiles) or [0.5],
                          batch_size=CHRONOS_BATCH, freq=bc.FREQ)
    names = {"predictions": alias}
    for q, suffix in quantiles.items():
        names[str(q)] = f"{alias}{suffix}"
    return out.rename(columns=names)[["unique_id", "ds"] + list(names.values())]


# ============================================================================
# TIMESTRAL AND T0-GAMMA (07b)
# ============================================================================

FOUNDATION_REPOS = {"timestral": "mistralai/Timestral-209M", "t0": "theforecastingcompany/t0-gamma"}
FOUNDATION_BATCH = 256        # series per forward pass


def foundation_model(kind: str, device: str | None = None):
    """Timestral (``kind="timestral"``) or t0-gamma (``"t0"``), weights from the HF cache."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if kind == "timestral":
        from timestral import TimestralForecaster
        return TimestralForecaster.from_pretrained(FOUNDATION_REPOS[kind], device=device)
    from t0 import T0Forecaster
    return T0Forecaster.from_pretrained(FOUNDATION_REPOS[kind]).to(device).eval()


def panel_arrays(hist: pd.DataFrame, future: pd.DataFrame | None, covariates: list[str],
                 h: int) -> tuple[list, np.ndarray, np.ndarray | None, list]:
    """Long frames to the arrays Timestral and t0 read.

    Returns ``(ids, context, known, future_weeks)``: ``context`` is ``(B, T)`` sales, NaN
    before a series starts; ``known`` is ``(B, C, T + h)`` covariates over the context and the
    ``h`` future weeks (``None`` without covariates), filled across a series' missing start.
    Every series must share the same ``h`` future weeks.
    """
    weeks = np.sort(pd.to_datetime(hist["ds"]).unique())
    hist = hist.assign(ds=pd.to_datetime(hist["ds"]))
    ids = sorted(hist["unique_id"].unique())
    context = (hist.pivot(index="unique_id", columns="ds", values="y")
               .reindex(index=ids, columns=weeks).to_numpy(dtype=np.float32))
    future_weeks = list(pd.date_range(pd.Timestamp(weeks[-1]) + pd.Timedelta(weeks=1),
                                      periods=h, freq=bc.FREQ))
    if not covariates:
        return ids, context, None, future_weeks
    both = pd.concat([hist[["unique_id", "ds"] + covariates],
                      future.assign(ds=pd.to_datetime(future["ds"]))[["unique_id", "ds"]
                                                                      + covariates]])
    known = []
    for col in covariates:
        wide = (both.pivot(index="unique_id", columns="ds", values=col)
                .reindex(index=ids, columns=list(weeks) + future_weeks))
        known.append(wide.ffill(axis=1).bfill(axis=1).fillna(0.0).to_numpy(dtype=np.float32))
    return ids, context, np.stack(known, axis=1), future_weeks


def foundation_forecast(kind: str, model, hist: pd.DataFrame, future: pd.DataFrame | None = None,
                        alias: str = "FM", h: int = bc.H,
                        covariates: list[str] | None = None) -> pd.DataFrame:
    """Zero-shot forecasts from Timestral or t0-gamma, in long form.

    Uses the developers' recommended settings: symmetric averaging and clipping at 0 for a
    series that never went negative (both models), plus, for t0, integer snapping and the
    missing start read as padding. Returns ``alias`` (median) and the 80% band
    (``alias-lo-80``, ``alias-hi-80``).
    """
    covariates = list(covariates or [])
    ids, context, known, future_weeks = panel_arrays(hist, future, covariates, h)
    out = []
    for start in range(0, len(ids), FOUNDATION_BATCH):
        part = slice(start, start + FOUNDATION_BATCH)
        cov = None if known is None else known[part]
        if kind == "timestral":
            with torch.inference_mode():
                fc = model.forecast(context[part], prediction_length=h, known_covariates=cov)
            q = fc.quantiles[:, 0].float().cpu().numpy()                 # (B, H, Q)
            levels = np.array(fc.quantile_levels)
            cols = [int(np.abs(levels - lvl).argmin()) for lvl in (0.1, 0.5, 0.9)]
            q = q[..., cols]
        else:
            fc = model.predict(context[part], h, quantile_levels=(0.1, 0.5, 0.9),
                               future_covariates=cov, trim_leading_missing=True,
                               symmetric_averaging=True, clip_nonnegative=True,
                               snap_integers=True)
            q = fc.quantiles.float().cpu().numpy()                         # (B, H, 3)
        out.append(q)
    q = np.concatenate(out)
    frame = pd.DataFrame({"unique_id": np.repeat(ids, h), "ds": np.tile(future_weeks, len(ids)),
                          f"{alias}-lo-80": q[..., 0].ravel(), alias: q[..., 1].ravel(),
                          f"{alias}-hi-80": q[..., 2].ravel()})
    return frame[["unique_id", "ds", alias, f"{alias}-lo-80", f"{alias}-hi-80"]]


# ============================================================================
# HIERARCHY (08)
# ============================================================================

def rolling_one_step(predict, frame: pd.DataFrame, origins) -> pd.DataFrame:
    """One-step-ahead forecasts at each week in ``origins``, for MinT's residuals.

    ``frame`` is long (``unique_id``, ``ds``, ``y`` and covariates) and covers the origins and
    the ``bc.H`` weeks after the last one. For each origin, ``predict(hist, futr)`` sees only
    the rows before it (``hist``) and the covariates of the ``bc.H`` weeks from it (``futr``,
    no ``y``) and returns ``unique_id``, ``ds``, ``yhat``; the forecast for the origin week is
    kept. Returns ``unique_id``, ``ds`` (= origin), ``yhat``.
    """
    frame = frame.assign(ds=pd.to_datetime(frame["ds"]))
    weeks = np.sort(frame["ds"].unique())
    out = []
    for origin in pd.to_datetime(list(origins)):
        hist = frame[frame["ds"] < origin]
        ahead = weeks[(weeks >= origin)][:bc.H]
        futr = frame[frame["ds"].isin(ahead) & frame["unique_id"].isin(set(hist["unique_id"]))]
        fc = predict(hist, futr.drop(columns="y"))
        fc = fc.assign(ds=pd.to_datetime(fc["ds"]))
        out.append(fc.loc[fc["ds"] == origin, ["unique_id", "ds", "yhat"]])
    return pd.concat(out, ignore_index=True)


class MinTraceDirect(MinTraceSparse):
    """``MinTraceSparse`` with an exact, direct solve in place of the library's iterative one.

    The library solves ``S' W^-1 S x = S' W^-1 y`` with plain ``bicgstab`` and ignores its exit
    code. Under ``wls_var`` the error variances in this panel span many orders of
    magnitude, so that solve stops early and returns forecasts in the millions without a
    warning. Here the bottom rows of ``S`` are the identity, so ``S' W^-1 S`` is a diagonal
    (the items) plus one low-rank term per upper node, and the Woodbury identity solves it
    with one Cholesky factor of an (upper nodes x upper nodes) matrix: 2,118 x 2,118 in 08.
    ``W`` is the library's own, so the method is unchanged; only the solve is.
    """

    def _get_PW_matrices(self, S, y_hat, y_insample=None, y_hat_insample=None):
        from scipy.linalg import cho_factor, cho_solve
        _, W = super()._get_PW_matrices(S, y_hat, y_insample, y_hat_insample)
        S = sparse.csr_matrix(S)
        n_nodes, n_bottom = S.shape
        upper = S[: n_nodes - n_bottom]
        if (S[n_nodes - n_bottom:] != sparse.identity(n_bottom)).nnz:
            raise ValueError("MinTraceDirect needs the bottom rows of S to be the identity")
        var = W.diagonal()
        var_up, var_bottom = var[: n_nodes - n_bottom], var[n_nodes - n_bottom:]
        # (D + U C U')^-1 = D^-1 - D^-1 U (C^-1 + U' D^-1 U)^-1 U' D^-1, with D = 1/var_bottom,
        # U = upper', C = 1/var_up.
        inner = (upper @ sparse.diags(var_bottom) @ upper.T).toarray() + np.diag(var_up)
        factor = cho_factor(inner)

        def solve(y):
            v = (S.T @ (np.ravel(y) / var)) * var_bottom       # D^-1 S' W^-1 y
            return v - var_bottom * (upper.T @ cho_solve(factor, upper @ v))

        return LinearOperator((n_bottom, n_nodes), matvec=solve), W


def floor_residual_sd(insample: pd.DataFrame, col: str = "Base",
                      min_sd: float = 1e-3) -> pd.DataFrame:
    """Give every node's in-sample residual at least ``min_sd`` spread.

    MinT's ``wls_var`` and ``mint_shrink`` divide by each node's residual variance, so a node
    whose forecast always equals its sales (an item that never sold, forecast at 0) breaks
    them. For such nodes only, ``±min_sd`` is added to the in-sample forecast in alternate
    weeks: far below a unit, so their weight in the reconciliation is simply very high.
    """
    out = insample.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    sd = (out["y"] - out[col]).groupby(out["unique_id"]).transform("std").fillna(0)
    flat = sd < min_sd
    sign = np.where(out.groupby("unique_id").cumcount() % 2 == 0, 1.0, -1.0)
    out.loc[flat, col] = out.loc[flat, col] + min_sd * sign[flat.to_numpy()]
    return out


def split_by_share(parent_fc: pd.DataFrame, child_base: pd.DataFrame, parent_of: pd.Series,
                   col: str) -> pd.DataFrame:
    """Split each parent's forecast ``col`` to its children by their base-forecast shares.

    ``parent_fc``: ``unique_id`` (parent), ``ds``, ``col``. ``child_base``: ``unique_id``
    (child), ``ds``, ``Base``. ``parent_of`` maps child id -> parent id. A parent whose
    children's base forecasts sum to 0 is split equally. The children sum to their parent.
    """
    child = child_base.assign(parent=child_base["unique_id"].map(parent_of))
    grp = child.groupby(["parent", "ds"])["Base"]
    total = grp.transform("sum")
    count = grp.transform("size")
    share = np.where(total > 0, child["Base"] / total.where(total > 0, 1.0), 1.0 / count)
    parent = parent_fc.rename(columns={"unique_id": "parent", col: "_parent"})
    child = child.assign(share=share).merge(parent[["parent", "ds", "_parent"]],
                                            on=["parent", "ds"], how="left")
    child[col] = child["share"] * child["_parent"]
    return child[["unique_id", "ds", col]]
