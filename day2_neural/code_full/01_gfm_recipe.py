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
# # Day 2 · 01 (full data) — Lab: a training recipe for global models (20 min)
#
# The original lab 01 trains NHITS and N-BEATS with library defaults for 500 steps. On ≈17,000
# series that under-trains, and the result moves a lot from one random seed to the next. Our own
# research ran into exactly this, and settled on a short **training recipe**:
#
# | Ingredient | Why |
# |---|---|
# | Steps scaled to the panel (≈174 visits per series) | a fixed budget under-trains big panels |
# | 5% learning-rate warm-up, then 3 halvings | fewer early blow-ups, smaller oscillations |
# | Run the full budget, **keep the best validation weights** | most runs ended above their best |
# | Gradient clipping at 1.0 | stops single bad batches from throwing training off |
# | Huber loss to train, MAE to validate | robust to outliers; validate on what we report |
# | Target shifted by one (`y + 1`) | robust scaling divides by zero on all-zero windows |
#
# This notebook fits N-BEATSx with the recipe **off** and **on**, plus NHITS with it on,
# with 3 seeds each. It compares accuracy, how much the seeds disagree, and training health.
# See `docs/gfm_learnings.md` §3.

# %%
import numpy as np
import pandas as pd
from neuralforecast.models import NHITS

import gd_helpers as gd
import gd_plots as gplt

bc = gd.bc
gd.status()
gplt.use_style()

# %% [markdown]
# ## 1. Data and the starting configuration
#
# The future covariates are the original labs' scale-free set (`promo`, `discount_depth`,
# `price_index`), with the three numeric statics. The starting settings are the research's
# **phase 4 defaults**. One honest caveat: the research chose them by scoring trials on a test
# window, a mistake it fixed later. Here every tuning decision uses the validation window, and
# the test window is scored once.

# %%
baselines_cfg = gd.load_config("00_baselines")
panel = gd.train_panel()
train, test = bc.split(panel, cutoff=bc.TEST_START)
info = gplt.series_table(panel)

recipe_data = gd.nixtla_data(train, test, futr_exog=bc.FUTR_EXOG_COLS)            # y + 1
plain_data = gd.nixtla_data(train, test, futr_exog=bc.FUTR_EXOG_COLS, y_shift=0.0)

params = dict(gd.NBEATSX_DEFAULTS)
max_steps = gd.scaled_max_steps(recipe_data.fit_df["unique_id"].nunique(),
                                batch_size=params["batch_size"])
print(f"{recipe_data.fit_df['unique_id'].nunique():,} training series -> {max_steps:,} steps "
      f"per fit | seeds {gd.SEEDS}")
pd.Series(params)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## Intuition: why a recipe and not just more steps?
#
# Training a network is a walk downhill in the fog. A **warm-up** takes small steps until you
# have your footing. **Clipping** stops one steep patch from sending you over a cliff. **Keeping
# the best weights** is remembering the lowest point you passed, rather than wherever you were
# standing when the time ran out.

# %% [markdown]
# ## 2. Three arms, three seeds each
#
# * `NBEATSx_off`: same model and budget, library defaults (MAE loss, step decay, no clipping,
#   final weights, no shift).
# * `NBEATSx_recipe`: the full recipe.
# * `NHITS_recipe`: NHITS with the recipe, for comparison with the original lab.
#
# Each fit validates on the last 14 training weeks (`val_size = H`). The recipe uses that window
# to pick its best weights. Every run is cached, so this cell only trains on the GPU box.

# %%
def nbeatsx_off(seed):
    return gd.build_nbeatsx(params, plain_data, max_steps, seed, "NBEATSx_off", recipe=False)


def nbeatsx_recipe(seed):
    return gd.build_nbeatsx(params, recipe_data, max_steps, seed, "NBEATSx_recipe")


def nhits_recipe(seed):
    return NHITS(h=bc.H, input_size=params["input_size"], futr_exog_list=recipe_data.futr_exog,
                 stat_exog_list=recipe_data.stat_exog, scaler_type="robust",
                 batch_size=params["batch_size"], learning_rate=params["learning_rate"],
                 random_seed=seed, alias="NHITS_recipe", **gd.recipe_kwargs(max_steps))


arms = {
    "NBEATSx_off": gd.fit_seeds(nbeatsx_off, plain_data, "01_nbeatsx_off"),
    "NBEATSx_recipe": gd.fit_seeds(nbeatsx_recipe, recipe_data, "01_nbeatsx_recipe"),
    "NHITS_recipe": gd.fit_seeds(nhits_recipe, recipe_data, "01_nhits_recipe"),
}
# Keep one model for later notebooks to predict live: the first recipe seed.
if gd.TRAIN and arms["NBEATSx_recipe"][0]["nf"] is not None:
    gd.save_model(arms["NBEATSx_recipe"][0]["nf"], "01_nbeatsx_recipe")

# %% [markdown]
# ## 3. Training curves: recipe off vs on
#
# Loss against step on a log scale. The dotted line is where the warm-up ends; the dashed green
# line is the step whose weights the recipe kept.

# %%
gplt.plot_training(arms["NBEATSx_off"][:1] + arms["NBEATSx_recipe"][:1] + arms["NHITS_recipe"][:1],
                   warmup=gd.warmup_steps(max_steps), title="First seed of each arm")

# %% [markdown]
# ## 4. Accuracy and seed spread
#
# Each seed is scored on its own, then the three seeds are averaged into one forecast.

# %%
forecasts = {name: gd.seed_forecasts(runs) for name, runs in arms.items()}
seed_scores = {}
health = []
for name, runs in arms.items():
    rmsse = gd.per_series(gd.score(forecasts[name], test, train))
    seed_cols = [c for c in rmsse.columns if not c.endswith("_mean")]
    seed_scores[name] = rmsse[seed_cols].median()
    health.append(gd.summaries(runs).assign(arm=name))

spread = pd.DataFrame({name: {"median of seed medians": s.median(),
                              "seed spread (max - min)": s.max() - s.min()}
                       for name, s in seed_scores.items()}).T
spread.round(3)

# %%
gplt.plot_arm_comparison(arms, metrics=seed_scores)

# %% [markdown]
# ### Training health
#
# These are screened on the validation curve only, never on price response:
# * `oscillation`: how much validation loss jumps between checks late in training;
# * `best_frac`: where the kept weights came from (above ≈0.65 suggests the budget is too short);
# * `neg_share`: the share of negative forecasts.

# %%
health_cols = ["arm", "seed", "fit_seconds", "oscillation", "best_frac", "final_over_best",
               "neg_share"]
pd.concat(health)[health_cols].round(3)

# %% [markdown]
# ## 5. On the board, and against the Day 1 models

# %%
board_fcst = forecasts["NBEATSx_off"][["unique_id", "ds"]]
for name in arms:
    board_fcst = board_fcst.merge(forecasts[name][["unique_id", "ds", f"{name}_mean"]],
                                  on=["unique_id", "ds"])
fit_seconds = {}
for name, runs in arms.items():
    fit_seconds[f"{name}_mean"] = gd.summaries(runs)["fit_seconds"].mean()
gd.save_forecasts(board_fcst, "01_recipe")
gd.record(board_fcst, test, train, lab="01 recipe (3-seed mean)", fit_seconds=fit_seconds,
          notes="fit_seconds is per seed")
gd.median_board()

# %%
baselines = pd.read_parquet(gd.FORECAST_DIR / "00_baselines.parquet")
both = board_fcst.merge(baselines, on=["unique_id", "ds"])
rmsse = gd.per_series(gd.score(both, test, train))
gplt.plot_error_dist(rmsse, info, models=["NBEATSx_recipe_mean", "NBEATSx_off_mean",
                                          "LightGBM", "SeasonalNaive"])

# %%
gplt.plot_wins(rmsse, info, model="NBEATSx_recipe_mean", baseline="SeasonalNaive")

# %%
gplt.plot_wins(rmsse, info, model="NBEATSx_recipe_mean", baseline="LightGBM")

# %% [markdown]
# ## 6. The example series, with forecasts

# %%
gplt.plot_series_price(panel, baselines_cfg["examples"], both,
                       models=["NBEATSx_recipe_mean", "LightGBM"])

# %%
best = "NBEATSx_recipe"
gd.save_config("01_recipe", {
    "model": "NBEATSx",
    "params": params,
    "recipe": True,
    "y_shift": gd.Y_SHIFT,
    "futr_exog": recipe_data.futr_exog,
    "stat_exog": recipe_data.stat_exog,
    "max_steps": max_steps,
    "seeds": gd.SEEDS,
    "minutes_per_fit": {n: round(sec / 60, 2) for n, sec in fit_seconds.items()},
    "median_rmsse": {n: float(rmsse[f"{n}_mean"].median()) for n in arms},
    "seed_spread": spread["seed spread (max - min)"].round(4).to_dict(),
    "saved_model": "01_nbeatsx_recipe",
    "selected": best,
})

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## Discussion
#
# * Does the recipe win on accuracy, on seed spread, or both? In the research the bigger prize
#   was **stability**: with best-weights retention every seed was usable.
# * Look at `best_frac`. If the kept weights come from the very end, the budget is too short; if
#   from the middle, the run kept going after its best.
# * The minutes per fit (in the config) size the search in notebook 02.
#
# **Try it:** set `warmup_frac=0` in `gd.recipe_kwargs` and look at the first few hundred steps
# of the training curve.
