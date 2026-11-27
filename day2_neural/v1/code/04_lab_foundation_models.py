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
# # Day 2 · 04 — Lab: foundation models, fine-tune or not? (30 min)
#
# We run pretrained time-series foundation models (all of them through `timecopilot`) and decide, **on the validation
# window**, whether fine-tuning is worth it:
#
# | Variant | What it uses |
# |---|---|
# | `Chronos2_zs` | Chronos-2, zero-shot, target history only |
# | `Chronos2_zs_cov` | Chronos-2, zero-shot, with planned price and promotion as future covariates (in-context) |
# | `Chronos2_ft` | Chronos-2 fine-tuned on our training windows (LoRA), then forecast with covariates |
# | `TimesFM_zs` | Google's TimesFM 2.5 (200M), zero-shot, target history only: a second opinion from a different pretrained model |
# | `TabPFN_TS` (optional) | TabPFN-TS: a tabular foundation model on calendar + running-index features |
#
# Reference points on the same series: SeasonalNaive and a default NHITS.
#
# **Hardware.** The lab uses every eligible series, like the other labs. On a CPU the foundation models are slow
# at this size (Chronos-2 and TimesFM forecast every series; fine-tuning trains on all of them), so if you are short
# of time set `FM_N_SERIES` (below, or as an environment variable) to a number to run on a random subset. On a 4-core
# laptop CPU all 17,241 series take about 2 hours and 1,000 series about 12 minutes. Weights download from Hugging Face on first use
# (Chronos-2 ≈ 0.5 GB, TimesFM 2.5 ≈ 1 GB): **run the first two cells before the break.**

# %%
import logging
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import torch

import bootcamp as bc

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)
palette = ["#000000", "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#E69F00"]
plt.rcParams.update({"figure.figsize": (12, 5), "axes.prop_cycle": plt.cycler(color=palette)})

GPU = torch.cuda.is_available()
FM_N_SERIES = int(os.environ.get("FM_N_SERIES", 0)) or None  # 0/unset = every eligible series; a number = random subset if short of time
FT_STEPS = 1000 if GPU else 150        # fine-tuning steps
RUN_TABPFN = GPU                       # TabPFN-TS needs a GPU (or a Prior Labs API token for cloud mode)
CHRONOS_REPO = "amazon/chronos-2"      # "autogluon/chronos-2-small" (28M) is ~4x faster on CPU
TIMESFM_REPO = "google/timesfm-2.5-200m-pytorch"
LEVELS = [80, 95]                      # prediction intervals from the models' own quantiles
print(f"GPU: {GPU} | series: {FM_N_SERIES or 'all'} | fine-tune steps: {FT_STEPS}")

# %%
from huggingface_hub import snapshot_download

for repo in (CHRONOS_REPO, TIMESFM_REPO):
    snapshot_download(repo)            # pre-fetch the weights

# %% [markdown]
# ## Intuition: the well-travelled forecaster
#
# * **Zero-shot**: a consultant who has seen millions of series from every industry, but never your shop. You
#   show them the last 93 weeks and ask for the next 14. No training, minutes to the first forecast.
# * **Fine-tuned**: the same consultant after a week of induction on your data. Better on your quirks
#   (intermittency, promo spikes), but now you pay for training and can overfit.
# * **A second consultant**: TimesFM was pretrained by Google on a different corpus with a different
#   architecture (a decoder-only transformer on patches). If two zero-shot models disagree, neither is a free win.
# * **Covariates in context**: Chronos-2 reads the planned price and promotion calendar like notes in the
#   margin, without retraining.
#
# **How to read your results.** Compare against the seed noise floor (Lab 02) and the leaderboard: if zero-shot is
# within a few percent of NHITSx / LightGBM, it is a strong baseline for almost no effort. Fine-tune only if the
# validation gain is clearly bigger than the seed-to-seed noise and worth the GPU time.
#
# *Reference run: all 17,241 eligible series on a 4-core laptop CPU, no GPU, 150 LoRA steps; the whole notebook
# took 1 h 55 min, more than half of it TimesFM. The "Reading the result" cells give the reference numbers; fill in the
# decision table with your own before you read them.*

# %% [markdown]
# ## Data: validation split first, test split last

# %%
panel = bc.load_panel()
if FM_N_SERIES:
    ids = (panel.select("unique_id", "sale_class").unique().sort("unique_id")
           .sample(n=FM_N_SERIES, seed=bc.RANDOM_SEED)["unique_id"])
    panel = panel.filter(pl.col("unique_id").is_in(ids.implode()))
X = bc.FUTR_EXOG_COLS          # promo, discount_depth, price_index

train, test = bc.split(panel)
fit_part, valid = bc.split(train, cutoff=bc.VALID_START)
n = panel["unique_id"].n_unique()
print(f"{n} series")


def frames(hist: pl.DataFrame, future: pl.DataFrame):
    """History with covariates, and the known-future covariates for the next H weeks."""
    return bc.to_nixtla(hist, X), bc.to_nixtla(future, X).drop(columns="y")

# %% [markdown]
# ## 1. Zero-shot vs fine-tuned, scored on the validation window

# %%
from timecopilot.models.foundation.chronos import Chronos, ChronosFinetuningConfig
from timecopilot.models.foundation.timesfm import TimesFM

zs = Chronos(repo_id=CHRONOS_REPO, alias="Chronos2_zs", batch_size=64)
zs_cov = Chronos(repo_id=CHRONOS_REPO, alias="Chronos2_zs_cov", batch_size=64)
ft = Chronos(repo_id=CHRONOS_REPO, alias="Chronos2_ft", batch_size=64,
             finetuning_config=ChronosFinetuningConfig(finetune_steps=FT_STEPS, finetune_mode="lora",
                                                       learning_rate=1e-5, batch_size=64))
tfm = TimesFM(repo_id=TIMESFM_REPO, alias="TimesFM_zs", batch_size=64,
              context_length=128)  # our histories are <= 93 weeks; the 2048 default only adds padding (2x slower, same forecasts)


def run_fms(hist: pl.DataFrame, future: pl.DataFrame) -> tuple[pd.DataFrame, dict]:
    df_x, X_df = frames(hist, future)
    secs, out = {}, []
    for model, kwargs in [(zs, dict(df=df_x[["unique_id", "ds", "y"]])),
                          (zs_cov, dict(df=df_x, X_df=X_df)),
                          (ft, dict(df=df_x, X_df=X_df)),
                          (tfm, dict(df=df_x[["unique_id", "ds", "y"]]))]:
        with bc.Timer() as t:
            out.append(model.forecast(h=bc.H, freq=bc.FREQ, level=LEVELS, **kwargs).set_index(["unique_id", "ds"]))
        secs[model.alias] = t.seconds
        print(f"{model.alias:16s} {t.seconds:7.1f}s  ({1000 * t.seconds / n:6.1f}s per 1k series)")
    if RUN_TABPFN:
        from timecopilot.models.foundation.tabpfn import TabPFN
        tab = TabPFN(alias="TabPFN_TS")
        with bc.Timer() as t:
            out.append(tab.forecast(df=df_x, X_df=X_df, h=bc.H, freq=bc.FREQ).set_index(["unique_id", "ds"]))
        secs["TabPFN_TS"] = t.seconds
    return pd.concat(out, axis=1).reset_index(), secs

# %%
_ = zs.forecast(df=bc.to_nixtla(fit_part.filter(pl.col("unique_id") == panel["unique_id"][0])), h=bc.H, freq=bc.FREQ)  # warm-up / load
fm_valid, secs_valid = run_fms(fit_part, valid)

# %% [markdown]
# References on the **same** series and window: SeasonalNaive and a default NHITS (as in Lab 01, but with a
# 56-week lookback).

# %%
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive


def references(hist: pl.DataFrame) -> tuple[pd.DataFrame, dict]:
    Y = bc.to_nixtla(hist)
    with bc.Timer() as t_sn:
        sn = StatsForecast(models=[SeasonalNaive(season_length=bc.SEASON)], freq=bc.FREQ).forecast(df=Y, h=bc.H)
    nf = NeuralForecast(models=[NHITS(h=bc.H, input_size=4 * bc.H, scaler_type="robust", max_steps=bc.MAX_STEPS,
                                      start_padding_enabled=True, random_seed=bc.RANDOM_SEED, **bc.trainer_kwargs())], freq=bc.FREQ)
    with bc.Timer() as t_nh:
        nf.fit(df=Y)
        nh = nf.predict()
    return sn.merge(nh, on=["unique_id", "ds"]), {"SeasonalNaive": t_sn.seconds, "NHITS": t_nh.seconds}


ref_valid, secs_ref = references(fit_part)
valid_fc = fm_valid.merge(ref_valid, on=["unique_id", "ds"])
valid_summary = bc.summarise(bc.score(valid_fc, valid, fit_part), valid_fc, valid)
valid_summary["seconds"] = valid_summary["model"].map({**secs_valid, **secs_ref})
valid_summary["seconds_per_1k_series"] = 1000 * valid_summary["seconds"] / n
bc.save_raw(valid_fc, "04_foundation", "valid", actuals=valid)   # every forecast and interval, with actuals
valid_summary[["model", "mean_rmsse", "median_rmsse", "wape", "seconds", "seconds_per_1k_series"]].round(3)

# %% [markdown]
# The same validation forecasts through the full metric suite (`bc.metric_suite`: scaled, relative to
# SeasonalNaive, percentage, pooled, bias, by demand type, and interval scores for the foundation models'
# 80 % and 95 % intervals; definitions in the README).

# %%
info = bc.series_info(panel)
bc.metric_suite(valid_fc.merge(bc.to_nixtla(valid), on=["unique_id", "ds"]), bc.to_nixtla(fit_part),
                info=info).set_index("model").T.round(3)

# %% [markdown]
# ### Where does fine-tuning change things?

# %%
m = bc.score(valid_fc, valid, fit_part)
m = m[m.metric == "rmsse"].merge(bc.series_info(panel).to_pandas(), on="unique_id")
cols = ["Chronos2_zs", "Chronos2_zs_cov", "Chronos2_ft", "TimesFM_zs", "NHITS", "SeasonalNaive"]
m.groupby("demand_type")[cols].mean().round(3).assign(n=m.groupby("demand_type").size())

# %% [markdown]
# ### Reading the result
#
# * **Zero-shot Chronos-2 is already competitive.** With target history only it scores ≈ 1.59 on validation,
#   level with the default NHITS (≈ 1.54) and well ahead of SeasonalNaive (≈ 2.23), with no training at all and
#   about 4 s per 1,000 series.
# * **The two zero-shot models agree.** TimesFM 2.5 lands at ≈ 1.60, next to Chronos-2 (≈ 1.59), so neither is
#   a free win over the other; TimesFM is also ≈ 25× slower on a CPU (≈ 120 s per 1,000 series).
# * **The covariates are the big lever.** Reading the planned price and promotion calendar in context takes
#   Chronos-2 from ≈ 1.59 to ≈ 1.44 mean RMSSE, its median from ≈ 0.70 to ≈ 0.58 and its WAPE from ≈ 0.32 to
#   ≈ 0.24. Most of it comes from **erratic** series (≈ 1.02 → 0.68), the promo-driven ones; intermittent and lumpy
#   series barely move.
# * **Fine-tuning does not pay on validation.** 150 LoRA steps make Chronos-2 *worse* on the mean (≈ 1.47 vs
#   ≈ 1.44) with the same median, for ≈ 10 extra minutes of CPU. The validation verdict is: zero-shot with
#   covariates.
# * **The intervals come free, and they are honest.** Chronos-2's own quantiles cover 76–79 % of validation weeks
#   for a promised 80 % and 95 % for 95 %, with no calibration step: compare the 66–68 % of per-series conformal in
#   Lab 06. Its covariates make the intervals *narrower* (median scaled width 1.9 vs 2.5) at almost the same
#   coverage.
# * **Bias**: with covariates Chronos-2 is unbiased in total volume (+0.1 %); without them it under-forecasts by
#   6 %, TimesFM by 7 %.
# * **A library pitfall**: TimesFM's 95 % row in the suite repeats its 80 % row. TimesFM 2.5 only predicts the
#   10 %, ..., 90 % quantiles, and timecopilot returns those when asked for 2.5 % and 97.5 %, without a warning.
#   `bc.metric_suite` now detects identical intervals and skips the wider one (as in Lab 07); ignore TimesFM's
#   95 % numbers in the table above.

# %% [markdown]
# ## 2. Decision
#
# Fill in from the validation table above (not the test window):
#
# | Question | Answer |
# |---|---|
# | Does zero-shot Chronos-2 beat SeasonalNaive? NHITS? | |
# | Do the two zero-shot models agree (`Chronos2_zs` vs `TimesFM_zs`)? | |
# | Do the covariates help (`zs_cov` vs `zs`)? On which demand types? | |
# | Does fine-tuning beat zero-shot by more than the seed noise (Lab 02)? | |
# | Extra cost of fine-tuning, in seconds per 1k series, and on what hardware? | |
# | **Decision**: zero-shot, fine-tune, or stay with the tuned global model? | |
#
# Things that push towards **fine-tuning**: many series, a stable domain, a GPU, a gain on validation that
# survives a second fold. Towards **zero-shot**: few or new series, no GPU, frequent re-training, and a small or
# noisy validation gain.

# %%
decision = valid_summary.set_index("model")[["mean_rmsse", "seconds_per_1k_series"]]
gain_ft = decision.loc["Chronos2_zs_cov", "mean_rmsse"] - decision.loc["Chronos2_ft", "mean_rmsse"]
extra_cost = decision.loc["Chronos2_ft", "seconds_per_1k_series"] - decision.loc["Chronos2_zs_cov", "seconds_per_1k_series"]
print(f"fine-tuning: RMSSE gain {gain_ft:+.4f} for {extra_cost:+.0f}s per 1k series")

# %% [markdown]
# ## 3. Final run on the test window
#
# Refit (fine-tune) on the full training period and forecast the test window. If you ran a subset, the
# leaderboard row is computed on that subset only (see `n_series`), so compare it with the reference rows below
# rather than with the full leaderboard.

# %%
fm_test, secs_test = run_fms(train, test)
ref_test, secs_ref_test = references(train)
bc.save_forecasts(fm_test, "04_foundation", actuals=test)
bc.save_raw(fm_test.merge(ref_test, on=["unique_id", "ds"]), "04_foundation", "test", actuals=test)
bc.record(fm_test, test, train, lab="04 foundation", fit_seconds=secs_test,
          notes=f"{n} series; {CHRONOS_REPO}, {TIMESFM_REPO}; ft steps {FT_STEPS}; GPU={GPU}")
test_summary = bc.summarise(bc.score(fm_test.merge(ref_test, on=["unique_id", "ds"]), test, train))
test_summary[["model", "mean_rmsse", "median_rmsse", "n_series"]].round(3)

# %% [markdown]
# ### Reading the result
#
# * **The validation decision holds.** On the test window fine-tuning gains ≈ 0.01 mean RMSSE (≈ 1.28 vs ≈ 1.29),
#   half the seed noise of Lab 02 (≈ 0.02), with the same median (≈ 0.52 vs ≈ 0.51). Zero-shot with covariates is
#   the defensible choice.
# * **Chronos-2 with covariates is the best single model of the day.** The fine-tuned model scores ≈ 1.28 on the
#   test mean and the zero-shot one ≈ 1.29, a gap inside the seed noise. Even the zero-shot model, the one
#   validation picked, beats NHITSx (≈ 1.30), the tuned NHITS (≈ 1.33) and LightGBM (≈ 1.35) on the mean, and by
#   more on the median (≈ 0.51 vs ≈ 0.57) and WAPE (≈ 0.23 vs ≈ 0.28), without training a thing.
# * **The catch is cost.** On a CPU it needs ≈ 20 s per 1,000 series just to forecast, while NHITS trains *and*
#   forecasts in ≈ 2 s per 1,000 series; TimesFM needs ≈ 120 s. A GPU removes most of that gap.

# %% [markdown]
# ## Discussion
#
# * Did the validation decision hold on the test window?
# * Chronos-2 sees at most ~93 weekly points of context here; its 8k-token context is irrelevant. What *is*
#   it bringing?
# * Licences differ: Chronos-2 and TimesFM 2.5 are Apache-2.0; recent TabPFN weights are non-commercial. Check before production.
