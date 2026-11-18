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
# # Day 2 · 02 — Lab: hyperparameter optimisation, what is worth tuning (30 min)
#
# Two experiments on NHITS, both scored on the **validation window** (the 14 weeks before the test window), so
# the test set stays untouched until the final refit:
#
# 1. **One-at-a-time sensitivity.** Move one hyperparameter at a time away from the default and measure how
#    much validation RMSSE moves. Big spread → worth tuning; flat → leave the default.
# 2. **A small Optuna search** (`AutoNHITS`, TPE sampler) over the parameters that mattered, then fANOVA
#    importances from the study.
#
# Rule of thumb for N-BEATS / N-HiTS on retail data (check it against your own results below):
#
# | Usually worth tuning | Usually not |
# |---|---|
# | `input_size` (lookback) | `mlp_units` width / depth |
# | `learning_rate`, `max_steps` | `n_blocks`, number of stacks |
# | `scaler_type` | activation function |
# | `loss` (matches the metric?) | `n_pool_kernel_size` details |

# %%
import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

import bootcamp as bc

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)
optuna.logging.set_verbosity(optuna.logging.WARNING)
palette = ["#000000", "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#E69F00"]
plt.rcParams.update({"figure.figsize": (12, 5), "axes.prop_cycle": plt.cycler(color=palette)})

N_TRIALS = 8               # Optuna trials; 30+ is better if you have a GPU
# Each sensitivity fit takes ~30 s on a laptop CPU at the default size. Run these four in class (~5 min) and
# hand the others to volunteers, or set RUN_ALL = True (~12 min).
RUN_PARAMS = ["input_size", "learning_rate", "loss", "mlp_units"]
RUN_ALL = False

# %%
from neuralforecast import NeuralForecast
from neuralforecast.auto import AutoNHITS
from neuralforecast.losses.pytorch import MAE, MSE, HuberLoss
from neuralforecast.models import NHITS

panel = bc.load_panel()
train, test = bc.split(panel)
fit_part, valid = bc.split(train, cutoff=bc.VALID_START)     # tune on this split only
Y_fit, Y_train = bc.to_nixtla(fit_part), bc.to_nixtla(train)
print(f"tuning fit: → {fit_part['ds'].max()} | validation: {valid['ds'].min()} → {valid['ds'].max()}")

# %% [markdown]
# ## Intuition: tuning is a budget
#
# Imagine a mixing desk with twenty knobs. Most change the sound very little; one or two change it a lot. The
# sensitivity sweep finds out which knobs matter **before** you spend money searching over all of them. And every
# gain must be compared with the noise floor, measured below by refitting the same model with different random
# seeds: a knob whose effect is smaller than that is not a finding.

# %% [markdown]
# ## 1. One-at-a-time sensitivity

# %%
DEFAULT = dict(input_size=2 * bc.H, learning_rate=1e-3, scaler_type="robust", loss=MAE(),
               mlp_units=[[512, 512]] * 3, n_blocks=[1, 1, 1], max_steps=bc.MAX_STEPS)

GRID = {
    "input_size": [bc.H, 2 * bc.H, 4 * bc.H],
    "learning_rate": [1e-4, 1e-3, 5e-3],
    "scaler_type": ["identity", "standard", "robust"],
    "loss": [MAE(), MSE(), HuberLoss()],
    "max_steps": [bc.MAX_STEPS // 2, bc.MAX_STEPS, bc.MAX_STEPS * 2],
    "mlp_units": [[[128, 128]] * 3, [[512, 512]] * 3, [[1024, 1024]] * 3],
    "n_blocks": [[1, 1, 1], [2, 2, 2], [3, 3, 3]],
}


def label(v):
    return type(v).__name__ if hasattr(v, "is_distribution_output") else str(v[0][0]) + "w" if isinstance(v, list) and isinstance(v[0], list) else str(v)


_cache: dict[str, tuple[float, float]] = {}


def valid_rmsse(seed: int = bc.RANDOM_SEED, **overrides) -> tuple[float, float]:
    params = {**DEFAULT, **overrides}
    key = repr(sorted((k, label(v)) for k, v in params.items())) + f"|seed={seed}"
    if key in _cache:                      # the default setting appears in every row of the grid
        return _cache[key]
    model = NHITS(h=bc.H, random_seed=seed, **params, **bc.trainer_kwargs())
    nf = NeuralForecast(models=[model], freq=bc.FREQ)
    with bc.Timer() as t:
        nf.fit(df=Y_fit)
    s = bc.summarise(bc.score(nf.predict(), valid, fit_part, ["NHITS"]))
    _cache[key] = (s.loc[0, "mean_rmsse"], t.seconds)
    return _cache[key]


rows = []
for param, values in GRID.items():
    if not RUN_ALL and param not in RUN_PARAMS:
        continue
    for v in values:
        score, secs = valid_rmsse(**{param: v})
        rows.append({"param": param, "value": label(v), "valid_rmsse": score, "fit_seconds": secs})
        print(f"{param:14s} {label(v):>10s}  RMSSE {score:.4f}  ({secs:.0f}s)")
sens = pd.DataFrame(rows)

# %%
spread = (sens.groupby("param")["valid_rmsse"].agg(lambda s: s.max() - s.min())
          .sort_values(ascending=True).rename("spread"))
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
spread.plot.barh(ax=axes[0], color=palette[1])
axes[0].set_title("Validation RMSSE range when varying one parameter")
axes[0].set_xlabel("max − min mean RMSSE")
for i, (param, g) in enumerate(sens.groupby("param")):
    axes[1].plot(range(len(g)), g["valid_rmsse"], marker="o", label=param)
axes[1].set_xticks([0, 1, 2], ["low", "middle", "high"])
axes[1].set_title("Validation RMSSE per setting")
axes[1].legend(fontsize=9)
plt.tight_layout()
spread.sort_values(ascending=False)

# %% [markdown]
# ### The noise floor: same settings, different random seeds
#
# A neural net starts from random weights and sees mini-batches in random order. Refit the **default** setting
# with two more seeds: the spread you get is what "changing nothing" does to the score. A parameter whose spread
# is not clearly bigger than this is not a finding.

# %%
seed_scores = [valid_rmsse()[0]] + [valid_rmsse(seed=s)[0] for s in (1, 2)]
seed_spread = max(seed_scores) - min(seed_scores)
print("default setting, 3 seeds:", [round(x, 4) for x in seed_scores], f"| seed spread {seed_spread:.4f}")
print("parameters whose spread beats the seed spread:", list(spread[spread > seed_spread].sort_values(ascending=False).index))

# %% [markdown]
# **Read the chart before the search.** Parameters at the bottom of the bar chart are not worth search budget:
# fix them at the default. A parameter whose effect is smaller than the seed spread above is not a finding.

# %% [markdown]
# ### Reading the result
#
# * In the reference run the **loss** was by far the biggest knob: validation RMSSE ≈ 1.26 with MAE or Huber,
#   ≈ 1.51 with MSE. MSE squares the error, so it chases the rare promo spikes and overshoots every normal week.
# * Lookback, width and learning rate each moved the score by less than 0.06.
#
# *Reference run: 2,000 series, laptop CPU, 500 steps. Your numbers will differ a little; the reading should not.*

# %% [markdown]
# ## 2. A small Optuna search over what mattered
#
# `AutoNHITS` with the Optuna backend uses its own internal validation split (`val_size`), here the last
# 14 weeks of the training data. The config function is the search space.

# %%
def config(trial: optuna.Trial) -> dict:
    return {
        "input_size": trial.suggest_categorical("input_size", [bc.H, 2 * bc.H, 3 * bc.H, 4 * bc.H]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "scaler_type": trial.suggest_categorical("scaler_type", ["standard", "robust"]),
        "max_steps": trial.suggest_categorical("max_steps", [bc.MAX_STEPS, 2 * bc.MAX_STEPS]),
        "mlp_units": [[512, 512]] * 3,                 # fixed: not worth tuning
        "random_seed": bc.RANDOM_SEED,
        **bc.trainer_kwargs(),
    }


auto = AutoNHITS(h=bc.H, loss=MAE(), valid_loss=MAE(), config=config, backend="optuna",
                 search_alg=optuna.samplers.TPESampler(seed=bc.RANDOM_SEED), num_samples=N_TRIALS)
nf = NeuralForecast(models=[auto], freq=bc.FREQ)
with bc.Timer() as t_hpo:
    nf.fit(df=Y_train, val_size=bc.H)
study = nf.models[0].results
print(f"{N_TRIALS} trials in {t_hpo.seconds:.0f}s | best valid MAE {study.best_value:.3f}")
study.best_params

# %%
trials = study.trials_dataframe()[["number", "value", "params_input_size", "params_learning_rate",
                                   "params_scaler_type", "params_max_steps"]].sort_values("value")
try:
    importance = optuna.importance.get_param_importances(study)
    print("fANOVA importance:", {k: round(v, 3) for k, v in importance.items()})
except Exception as e:                       # needs enough completed trials
    print("importance not available:", e)
trials.head(8)

# %% [markdown]
# ## 3. Refit the winner and score it on the test window

# %%
fcst = nf.predict().rename(columns={"AutoNHITS": "NHITS_tuned"})
bc.save_forecasts(fcst, "02_hpo")
bc.record(fcst, test, train, lab="02 HPO", fit_seconds=t_hpo.seconds,
          notes=f"{N_TRIALS} TPE trials; best {study.best_params}")
bc.leaderboard()[["model", "mean_rmsse", "median_rmsse", "wape", "fit_seconds", "lab"]]

# %% [markdown]
# ### Reading the result
#
# ![Lab 02 results](slides/img/res_lab02.png)
#
# * **The noise floor.** Refitting the default with three seeds moved validation RMSSE by ≈ 0.009. All four swept
#   parameters beat that, but only the **loss** beats it by a mile (0.25: MSE chases the few huge weeks and is
#   ≈ 20 % worse). Lookback, width and learning rate move the score by 0.03–0.06, a few times the seed noise.
# * **Optuna**: 8 trials took ≈ 540 s and moved test RMSSE from ≈ 1.193 to ≈ 1.174. That is **15× the compute for
#   ≈ 1.6 %**, a gain only about twice the seed noise, and LightGBM (≈ 2 s, ≈ 1.169) is still ahead.
# * fANOVA credits the learning rate with almost all the importance (0.97), but that is importance *within the
#   ranges we searched*, from 8 points. The one-at-a-time sweep is the more honest picture of what is worth tuning.
# * Rule of thumb: get the **loss** and the **scaling** right, then stop tuning.
#
# *Reference run: 2,000 series, laptop CPU, 500 steps. Your numbers will differ a little; the reading should not.*

# %% [markdown]
# ## Discussion
#
# * How much of the gain came from **one** parameter (usually the lookback)? Could you have got it with a
#   three-point manual sweep instead of a search?
# * The search cost N× the default fit. Compare the RMSSE gain per CPU-minute with Day 1's LightGBM.
# * Tuning on a single validation window overfits that window. With more budget, score each trial with
#   `cross_validation(n_windows=3)` instead.
