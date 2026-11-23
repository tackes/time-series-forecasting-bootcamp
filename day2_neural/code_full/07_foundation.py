# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# # Day 2 · 07 (full data) — Lab: a foundation model against the selected GFM (20 min)
#
# The original lab 04 runs pretrained foundation models on a CPU subset. Here we put the
# strongest of them, **Chronos-2**, next to our selected global model (`GFM_selected`), on the
# same 2,000 evaluation series. The research listed this comparison as an open question
# (`docs/gfm_learnings.md` §9), so this is new work.
#
# | Variant | What it uses |
# |---|---|
# | `Chronos2_zs` | zero-shot, sales history only |
# | `Chronos2_zs_cov` | zero-shot, with the selected config's future covariates in context |
# | `Chronos2_ft` | fine-tuned on every training series, with covariates (GPU, optional) |
#
# The references are `GFM_selected`, LightGBM and SeasonalNaive, read from their saved
# forecasts.
#
# Goals
#
# 1. Run the Chronos-2 variants on the validation and test windows.
# 2. Choose a variant on **validation**, then score the test window once.
# 3. See where Chronos-2 beats the GFM, and where it does not.
#
# **In the chain:** reads the selected config (`gd.selected_config()`) and `00_baselines`,
# writes `07_foundation`.

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## Intuition: a well-travelled forecaster against a local expert
#
# Chronos-2 has seen millions of series from many industries, but never this shop. Shown a
# series' history, it forecasts the next 14 weeks with no training at all. Our GFM has seen
# only this shop, but all of it, with prices and promotions. It is the travelling consultant
# against the shop manager: who wins, and on which items?

# %%
import functools
import os
import tempfile

import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline
from chronos.chronos2.preprocess import from_data_frame

import gd_helpers as gd
import gd_plots as gplt
import gd_price as gp

bc = gd.bc
gd.status()
gplt.use_style()

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 1. Settings
#
# These set the model size, whether to fine-tune, and the interval levels.
#
# * The full Chronos-2 (≈120M parameters, ≈0.5 GB) on the GPU run; the small one (28M) in a
#   CPU smoke run. Weights download once to the Hugging Face cache, outside the repo.
# * Fine-tuning is off unless `GD_FM_FINETUNE=1`. It needs a GPU on the full panel.
# * Zero-shot forecasts have no random seed: the same input always gives the same forecast.

# %%
GPU = torch.cuda.is_available()
DEVICE = "cuda" if GPU else "cpu"
CHRONOS_REPO = "autogluon/chronos-2-small" if gd.SMOKE else "amazon/chronos-2"
FM_N_SERIES = 20 if gd.SMOKE else None        # None: every eval series
FINETUNE = os.environ.get("GD_FM_FINETUNE") == "1"
FT_STEPS = 5 if gd.SMOKE else 1_000
FT_LR = 1e-6                                  # the library's recommendation for full fine-tuning
FT_BATCH = 32 if gd.SMOKE else 256
BATCH = 256                                   # series (target + covariates) per predict batch
LEVELS = [80, 95]
QUANTILES = {0.025: "-lo-95", 0.1: "-lo-80", 0.9: "-hi-80", 0.975: "-hi-95"}
print(f"GPU: {GPU} | model: {CHRONOS_REPO} | fine-tune: {FINETUNE} ({FT_STEPS} steps)")

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 2. Data: the selected config's panel and covariates
#
# `gp.prepare` rebuilds the selected config's training data exactly, so Chronos-2 sees the same
# deflated prices, events and calendar terms as the GFM. The context is each series' full
# training history. For the validation window the data is rebuilt with
# `cutoff=bc.VALID_START`, so the reference price uses only the weeks before it.

# %%
cfg = gd.selected_config()
baselines_cfg = gd.load_config("00_baselines")
raw = gd.train_panel()
synthetic_seed = cfg.get("synthetic_seed")
panel, train, test, data = gp.prepare(cfg, panel=raw, synthetic_seed=synthetic_seed)
_, fit_part, valid, _ = gp.prepare(cfg, panel=raw, synthetic_seed=synthetic_seed,
                                   cutoff=bc.VALID_START)
info = gplt.series_table(panel)
covariates = cfg["futr_exog"]

examples = baselines_cfg["examples"]
example_ids = gplt.example_ids(examples)
fm_ids = gd.eval_ids()
if FM_N_SERIES:                      # smoke run: the examples first, then fill up
    rest = [u for u in fm_ids if u not in example_ids]
    fm_ids = (example_ids + rest)[:FM_N_SERIES]
print(f"{len(fm_ids):,} series forecast | covariates: {covariates}")

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 3. Chronos-2 in a few lines
#
# Chronos-2 reads a long data frame: one row per series and week, the target, and any
# covariates. Covariates that also appear in `future_df` are treated as **known in advance**
# (planned price and promotions, events, calendar). We keep its median and the 80% and 95%
# bands, named like neuralforecast's columns.

# %%
def chronos_forecast(pipe, hist, future, ids, alias, covariates=None) -> pd.DataFrame:
    """Chronos-2 forecasts of the ``H`` weeks after ``hist`` for ``ids``: point and bands."""
    covariates = list(covariates or [])
    df = gd.only(bc.to_nixtla(hist, covariates), ids).sort_values(["unique_id", "ds"])
    future_df = None
    if covariates:
        future_df = gd.only(bc.to_nixtla(future, covariates).drop(columns="y"), ids)
        future_df = future_df.sort_values(["unique_id", "ds"])
    out = pipe.predict_df(df, future_df=future_df, id_column="unique_id", timestamp_column="ds",
                          target="y", prediction_length=bc.H, quantile_levels=list(QUANTILES),
                          batch_size=BATCH, freq=bc.FREQ)
    names = {"predictions": alias}
    for q, suffix in QUANTILES.items():
        names[str(q)] = f"{alias}{suffix}"
    return out.rename(columns=names)[["unique_id", "ds"] + list(names.values())]


def finetune(pipe, hist, covariates, seed):
    """A copy of Chronos-2 fine-tuned on every series in ``hist`` (covariates known ahead).

    The checkpoint the library writes goes to a temporary folder outside the repo.
    """
    df = bc.to_nixtla(hist, covariates).sort_values(["unique_id", "ds"])
    inputs = from_data_frame(df, target_columns=["y"], prediction_length=bc.H,
                             known_covariates_names=covariates, id_column="unique_id",
                             timestamp_column="ds")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        return pipe.fit(inputs, prediction_length=bc.H, finetune_mode="full",
                        learning_rate=FT_LR, num_steps=FT_STEPS, batch_size=FT_BATCH,
                        output_dir=tmp, remove_printer_callback=True, seed=seed)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# Each variant runs once per window: validation (history before `VALID_START`) to compare the
# variants, and test to score them. Fine-tuning uses the weeks before `VALID_START` of **all**
# eligible series, so one fine-tuned model is honest on both windows. It is the only variant
# with a seed, so it runs 3 seeds and the forecast is the seed mean.

# %%
@functools.cache
def run_chronos() -> tuple[pd.DataFrame, pd.DataFrame]:
    """All variants on both windows: forecasts (long by ``window``) and the time each step took.

    Cached in memory, so the two tables below come from one run when both are computed.
    """
    seconds = {}
    pipe = Chronos2Pipeline.from_pretrained(CHRONOS_REPO, device_map=DEVICE)
    variants = {"Chronos2_zs": (pipe, None), "Chronos2_zs_cov": (pipe, covariates)}
    if FINETUNE:
        for seed in gd.SEEDS:
            with bc.Timer() as t:
                tuned = finetune(pipe, fit_part, covariates, seed)
            seconds[f"finetune_s{seed}"] = t.seconds
            variants[f"Chronos2_ft_s{seed}"] = (tuned, covariates)
    frames = []
    for window, (hist, future) in {"valid": (fit_part, valid), "test": (train, test)}.items():
        out = None
        for alias, (model, covs) in variants.items():
            with bc.Timer() as t:
                f = chronos_forecast(model, hist, future, fm_ids, alias, covs)
            seconds[f"{alias}_{window}"] = t.seconds
            print(f"{window:5s} {alias:22s} {t.seconds:7.1f}s")
            out = f if out is None else out.merge(f, on=["unique_id", "ds"])
        frames.append(out.assign(window=window))
    timing = pd.DataFrame({"step": list(seconds), "seconds": list(seconds.values())})
    return pd.concat(frames, ignore_index=True), timing


def chronos_forecasts() -> pd.DataFrame:
    """The forecasts from ``run_chronos``."""
    return run_chronos()[0]


def chronos_timing() -> pd.DataFrame:
    """The timings from ``run_chronos``, one row per step."""
    return run_chronos()[1]


fm_all = gd.cached("07_chronos", chronos_forecasts)
timing = gd.cached("07_chronos_seconds", chronos_timing)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# The fine-tuned forecast is the mean of its seeds, point and bands alike. The table shows how
# long each step took.

# %%
ft_cols = [c for c in fm_all.columns if c.startswith("Chronos2_ft_s")]
ft_points = [c for c in ft_cols if c in bc.model_columns(fm_all)]
if ft_points:
    fm_all["Chronos2_ft"] = fm_all[ft_points].mean(axis=1)
    for suffix in QUANTILES.values():
        fm_all[f"Chronos2_ft{suffix}"] = fm_all[[f"{c}{suffix}" for c in ft_points]].mean(axis=1)
variants = [c for c in ["Chronos2_zs", "Chronos2_zs_cov", "Chronos2_ft"] if c in fm_all]
fm_valid = fm_all[fm_all["window"] == "valid"].drop(columns="window")
fm_test = fm_all[fm_all["window"] == "test"].drop(columns="window")
timing

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 4. Which variant? Decided on the validation window
#
# The variant with the lowest validation median RMSSE goes forward, with the mean and the
# cost per 1,000 series alongside. The test window plays no part in the choice.

# %%
valid_metrics = gd.score(fm_valid[["unique_id", "ds"] + variants + ft_points], valid, fit_part)
valid_summary = bc.summarise(valid_metrics, fm_valid, valid)
valid_summary = valid_summary.sort_values("median_rmsse").reset_index(drop=True)
n_fm = len(fm_ids)
step_seconds = timing.set_index("step")["seconds"]
valid_seconds = valid_summary["model"].map(lambda m: step_seconds.get(f"{m}_valid", np.nan))
valid_summary["seconds_per_1k_series"] = 1000 * valid_seconds / n_fm
best_variant = valid_summary.loc[valid_summary["model"].isin(variants), "model"].iloc[0]
print(f"best Chronos-2 variant on validation (median RMSSE): {best_variant}")
valid_summary[["model", "median_rmsse", "mean_rmsse", "wape", "seconds_per_1k_series"]].round(3)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ### Reading the result
#
# * Do the covariates help (`zs_cov` against `zs`)? Chronos-2 reads them in context, with no
#   training, so a gain here is the price and promotion calendar paying off for free.
# * If fine-tuning ran: is its gain bigger than the 02 noise floor? Fine-tuning costs GPU time
#   on every refresh.
# * Note the cost column: zero-shot takes ≈ __ s per 1,000 series.
#   <!-- TODO: fill after GPU run -->
#
# *Exact numbers vary with seed and hardware.*

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 5. On the test window, against the selected GFM
#
# The Chronos-2 rows go on the board. If they cover only a subset of series, their `n_series`
# says so, and notebook 10 leaves them out of the standings.

# %%
fm_seconds = {m: step_seconds.get(f"{m}_test") for m in variants}
gd.save_forecasts(fm_test[["unique_id", "ds"] + variants], "07_foundation")
gd.record(fm_test[["unique_id", "ds"] + variants], test, train, lab="07 foundation",
          fit_seconds=fm_seconds,
          notes=f"{n_fm} series; {CHRONOS_REPO}; fine-tuned: {bool(ft_points)}")
gd.median_board()

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# For a like-for-like comparison, every model is scored on the series Chronos-2 forecast.

# %%
baselines = pd.read_parquet(gd.FORECAST_DIR / "00_baselines.parquet")
baselines["ds"] = pd.to_datetime(baselines["ds"])
references = ["GFM_selected", "LightGBM", "SeasonalNaive"]
both = fm_test.merge(gd.selected_forecasts(cfg), on=["unique_id", "ds"], how="left")
both = both.merge(baselines[["unique_id", "ds", "LightGBM", "SeasonalNaive"]],
                  on=["unique_id", "ds"], how="left")
test_metrics = gd.score(both, test, train)
rmsse = gd.per_series(test_metrics)
test_summary = bc.summarise(test_metrics, both, test).sort_values("median_rmsse")
test_summary[["model", "median_rmsse", "mean_rmsse", "wape", "n_series"]].round(3)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ### Reading the result
#
# * Median RMSSE first: where does the chosen variant sit against `GFM_selected` (≈ __ against
#   ≈ __)? <!-- TODO: fill after GPU run --> The mean sits alongside.
# * The research found LightGBM the most accurate model but untrustworthy on price (§5,
#   [100k set]). If Chronos-2 is close on accuracy, the same question applies to it.
#
# *Exact numbers vary with seed and hardware.*
#
# ## 6. Where the errors are
#
# Per-series RMSSE for every model (F2): a histogram, a cumulative curve and a box plot by
# class.

# %%
gplt.plot_error_dist(rmsse, info, models=variants + references)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# By demand type (as in the original lab): smooth, erratic, intermittent and lumpy series.

# %%
by_type = rmsse.join(info["demand_type"]).groupby("demand_type")
by_type.median().round(3).assign(n=by_type.size())

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ### Marginal wins: the best Chronos-2 variant against the selected GFM
#
# Bars above 0.5 are classes where Chronos-2 wins more series than it loses.

# %%
gplt.plot_wins(rmsse, info, model=best_variant, baseline="GFM_selected")

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 7. The example series
#
# Chronos-2's 80% band is shaded around its forecast. Watch the promo weeks: does it follow the
# planned price, or smooth over it?

# %%
shown = [best_variant, "GFM_selected"]
bands = {best_variant: (f"{best_variant}-lo-80", f"{best_variant}-hi-80")}
gplt.plot_series_price(panel, examples, both, models=shown, bands=bands)

# %%
test_median = test_summary.set_index("model")["median_rmsse"]
gd.save_config("07_foundation", {
    "repo": CHRONOS_REPO,
    "n_series": n_fm,
    "covariates": covariates,
    "variants": variants,
    "finetune": {"ran": bool(ft_points), "steps": FT_STEPS, "learning_rate": FT_LR,
                 "mode": "full", "seed_columns": ft_points},
    "selected_variant": best_variant,
    "selection_rule": "lowest median RMSSE on the validation window",
    "median_rmsse_valid": valid_summary.set_index("model")["median_rmsse"].round(4).to_dict(),
    "median_rmsse_test": test_median.round(4).to_dict(),
    "seconds": step_seconds.round(1).to_dict(),
    "gpu": GPU,
})

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## Discussion
#
# * Chronos-2 sees at most ≈ 100 weekly points per series here. What does it bring that a
#   model trained on 17,000 of this shop's series does not?
# * In the wins by class, is Chronos-2 strongest where history is sparse, and weakest where
#   promotions drive the spikes? What would that tell you about using both?
# * This notebook does not sweep Chronos-2's price response. Its forecasts could be swept like
#   the GFM's (`gp.shift_prices` on the future covariates). Would you trust the answer?
#
# **Try it:** set `cross_learning=True` in `chronos_forecast` so Chronos-2 shares information
# across the series in a batch, and compare the validation table.
