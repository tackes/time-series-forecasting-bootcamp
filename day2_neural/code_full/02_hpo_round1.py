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
# # Day 2 · 02 (full data) — Lab: hyperparameter search, round 1 (25 min)
#
# The original lab 02 moves one NHITS setting at a time and then runs a small Optuna search.
# Here we tune the N-BEATSx from notebook 01, with its training recipe, on **accuracy only**:
#
# | | |
# |---|---|
# | Model | N-BEATSx with the 01 recipe and covariates |
# | Grid | blocks {1111, 2222} × MLP width {256, 512} × learning rate {2e-4, 4.83e-4, 1e-3} |
# | Seeds | 3 per cell (36 fits) |
# | Tuned on | the **validation window** (14 weeks before the test window) |
# | Scored on test | once, for the cell we pick |
#
# This notebook
#
# 1. fits every cell of the grid with 3 seeds and scores it on the validation window,
# 2. measures the **noise floor**: how much the score moves when only the seed changes,
# 3. keeps a **shortlist** of cells that the noise floor cannot separate from the best, and
# 4. refits the best cell on the full training window and scores it on test once.
#
# Price response is left for round 2 (notebook 05). See `docs/gfm_learnings.md` §6.

# %%
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import gd_helpers as gd
import gd_plots as gplt
import gd_price as gp

bc = gd.bc
gd.status()
gplt.use_style()

# The grid: 2 block layouts × 2 widths × 3 learning rates = 12 cells; × 3 seeds = 36 fits.
# Each fit costs about one 01 fit (see `minutes_per_fit` in 01_recipe), so plan for ~36 of
# them on the GPU box. A smoke run fits two cells only.
FULL_GRID = {
    "n_blocks": [[1, 1, 1, 1], [2, 2, 2, 2]],
    "mlp_size": [256, 512],
    "learning_rate": [2e-4, 4.83e-4, 1e-3],
}
SMOKE_GRID = {"n_blocks": [[2, 2, 2, 2]], "mlp_size": [512], "learning_rate": [4.83e-4, 1e-3]}
GRID = SMOKE_GRID if gd.SMOKE else FULL_GRID
MAX_SHORTLIST = 3

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## Intuition: a score is only a finding if it beats the seed noise
#
# Refit the same model with a different random seed and the score moves: the starting weights
# and the order of the mini-batches change. That movement is the **noise floor**. Two settings
# whose scores differ by less than it are a tie, however good the better one looks. In the
# research, seed noise was bigger than many of the effects we were chasing (§3), so every cell
# here gets three seeds before we compare it with anything.

# %% [markdown]
# ## 1. Data and the starting point
#
# The settings come from `01_recipe`. The training window is split again: the model trains on
# the **inner** training weeks and is scored on the 14 validation weeks that follow.
# `gp.prepare(cfg, cutoff=bc.VALID_START)` builds that split, with the reference price and the
# statics from the inner weeks only. RMSSE is scaled by the inner training weeks too, so nothing
# after `VALID_START` is used until section 5.

# %%
recipe_cfg = gd.load_config("01_recipe")
base_params = recipe_cfg["params"]
raw = gd.train_panel()
_, inner, valid, tune_data = gp.prepare(recipe_cfg, panel=raw, cutoff=bc.VALID_START)
panel, train, test, full_data = gp.prepare(recipe_cfg, panel=raw)    # for section 5 only
info = gplt.series_table(panel)

n_series = tune_data.fit_df["unique_id"].nunique()
max_steps = gd.scaled_max_steps(n_series, batch_size=base_params["batch_size"])
print(f"inner train → {inner['ds'].max()} | validation {valid['ds'].min()} → "
      f"{valid['ds'].max()} | {n_series:,} series → {max_steps:,} steps per fit")
pd.Series(base_params)

# %% [markdown]
# ## 2. The grid
#
# Every combination of the grid values, each as a small set of overrides on the 01 settings.
# The **incumbent** is the cell with the 01 settings; it is always fitted, because its seeds
# set the noise floor.

# %%
def grid_cells(grid: dict) -> list[dict]:
    """Every combination of the grid values, as dicts of overrides."""
    names = list(grid)
    cells = []
    for values in itertools.product(*grid.values()):
        cells.append(dict(zip(names, values)))
    return cells


def cell_label(cell: dict) -> str:
    """A short name for a cell, e.g. ``b2222 m512 lr4.8e-04``."""
    blocks = "".join(str(b) for b in cell["n_blocks"])
    return f"b{blocks} m{cell['mlp_size']} lr{cell['learning_rate']:.1e}"


cells = grid_cells(GRID)
incumbent = {name: base_params[name] for name in GRID}
if incumbent not in cells:
    cells.append(incumbent)
cells_by_label = {cell_label(c): c for c in cells}
incumbent_label = cell_label(incumbent)
print(f"{len(cells)} cells × {len(gd.SEEDS)} seeds = {len(cells) * len(gd.SEEDS)} fits | "
      f"incumbent: {incumbent_label}")

# %% [markdown]
# Each cell's runs are cached under a hash of its full parameter set, with a summary and a
# small loss curve per seed and no saved model. If the GPU run crashes, rerunning this cell
# picks up where it stopped.

# %%
def fit_cell(cell: dict) -> list[dict]:
    """Three seeds of N-BEATSx with the cell's overrides, trained on the inner window."""
    params = {**base_params, **cell}

    def build(seed):
        return gd.build_nbeatsx(params, tune_data, max_steps, seed, "NBEATSx")

    return gd.fit_seeds(build, tune_data, f"02_cell_{gd.params_key(params)}")


runs = {label: fit_cell(cell) for label, cell in cells_by_label.items()}

# %% [markdown]
# ## 3. Validation scores and the noise floor
#
# Each seed is scored on its own: the median RMSSE over the evaluation series. A cell's score
# is the mean of its seed medians, and `seed_sd` is how far they spread.

# %%
def seed_medians(seed_runs: list[dict], actual, history) -> pd.Series:
    """Median RMSSE of each seed's forecast on ``actual``, scaled by ``history``."""
    fcst = gd.seed_forecasts(seed_runs)
    rmsse = gd.per_series(gd.score(fcst, actual, history))
    seed_cols = [c for c in rmsse.columns if not c.endswith("_mean")]
    return rmsse[seed_cols].median()


seed_scores = {}
rows = []
for label, cell in cells_by_label.items():
    scores = seed_medians(runs[label], valid, inner)
    health = gd.summaries(runs[label])
    seed_scores[label] = scores
    rows.append({"cell": label,
                 "blocks": "".join(str(b) for b in cell["n_blocks"]),
                 "mlp_size": cell["mlp_size"],
                 "learning_rate": cell["learning_rate"],
                 "median_rmsse_valid": scores.mean(),
                 "seed_sd": scores.std(),
                 "best_seed": scores.min(),
                 "oscillation": health["oscillation"].median(),
                 "best_frac": health["best_frac"].median(),
                 "minutes_per_fit": health["fit_seconds"].mean() / 60})
cell_table = pd.DataFrame(rows).set_index("cell").sort_values("median_rmsse_valid")
noise_floor = cell_table.loc[incumbent_label, "seed_sd"]
print(f"noise floor (seed sd of the incumbent): {noise_floor:.4f}")
cell_table.round(4)

# %% [markdown]
# ### The grid as a heatmap
#
# Learning rate down the side, architecture across the top; the number in each square is the
# cell's mean validation score with its seed spread underneath.

# %%
heat = cell_table.assign(arch="b" + cell_table["blocks"] + " m"
                         + cell_table["mlp_size"].astype(str))
means = heat.pivot(index="learning_rate", columns="arch", values="median_rmsse_valid")
sds = heat.pivot(index="learning_rate", columns="arch", values="seed_sd")
fig, ax = plt.subplots(figsize=(1.8 * means.shape[1] + 3, 0.9 * means.shape[0] + 1.6))
im = ax.imshow(means.to_numpy(), cmap="RdYlGn_r", aspect="auto")
ax.set_xticks(range(means.shape[1]), means.columns)
ax.set_yticks(range(means.shape[0]), [f"{lr:.2e}" for lr in means.index])
for (r, c), v in np.ndenumerate(means.to_numpy()):
    ax.text(c, r, f"{v:.3f}\n±{sds.to_numpy()[r, c]:.3f}", ha="center", va="center", fontsize=8)
ax.grid(False)
ax.set(xlabel="architecture (blocks per stack, MLP width)", ylabel="learning rate",
       title="Median validation RMSSE, mean of seeds (± seed sd)")
fig.colorbar(im, ax=ax, label="median RMSSE")
fig.tight_layout()
plt.show()

# %% [markdown]
# ### One setting at a time
#
# The original lab moved one setting at a time. The grid gives the same view for free: average
# the cells over every value of the other settings, and see how far each setting moves the
# score. A spread below the noise floor is not a finding.

# %%
effects = []
for name in GRID:
    column = "blocks" if name == "n_blocks" else name
    by_value = cell_table.groupby(column)["median_rmsse_valid"].mean()
    spread = by_value.max() - by_value.min()
    effects.append({"setting": name, "best value": by_value.idxmin(), "spread": spread,
                    "spread / noise floor": spread / noise_floor})
pd.DataFrame(effects).set_index("setting").round(4)

# %% [markdown]
# ## 4. The shortlist
#
# Cells whose score is within one noise floor of the best (at most three) go forward. They
# cannot be told apart on accuracy, so round 2 (notebook 05) can choose among them on price
# response. The best cell is refitted for the test window below.

# %%
best_score = cell_table["median_rmsse_valid"].min()
close = cell_table[cell_table["median_rmsse_valid"] <= best_score + noise_floor]
shortlist = close.head(MAX_SHORTLIST)
best_label = shortlist.index[0]
best_params = {**base_params, **cells_by_label[best_label]}
print(f"best cell: {best_label} | shortlist: {list(shortlist.index)}")
shortlist[["median_rmsse_valid", "seed_sd", "oscillation", "best_frac"]].round(4)

# %% [markdown]
# The shortlist and the incumbent side by side: every seed's validation curve, how much it
# oscillates late in training, and the spread of the seed scores.

# %%
compare = list(shortlist.index)
if incumbent_label not in compare:
    compare.append(incumbent_label)
gplt.plot_arm_comparison({label: runs[label] for label in compare},
                         metrics={label: seed_scores[label] for label in compare})

# %% [markdown]
# Per-series validation errors of the same cells (seed-mean forecasts):

# %%
valid_wide = []
for label in compare:
    fcst = gd.seed_forecasts(runs[label])[["unique_id", "ds", "NBEATSx_mean"]]
    fcst = fcst.rename(columns={"NBEATSx_mean": label})
    valid_wide.append(gd.per_series(gd.score(fcst, valid, inner)))
valid_wide = pd.concat(valid_wide, axis=1)
gplt.plot_error_dist(valid_wide, info)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ### Reading the result
#
# * Compare each gap in the heatmap with the noise floor printed above. A cell that wins by
#   less than one noise floor is a tie with the incumbent, not a better model.
# * The one-at-a-time table says which settings are worth search budget in round 2. A spread of
#   a few noise floors is a real but small effect.
# * Check `best_frac` before trusting a cell: if its kept weights come from the very end of
#   training, it would have liked more steps, and its score flatters the budget, not the model.
#
# *Reference run: full panel, GPU box. Exact numbers vary a little with hardware; the reading
# should not.*

# %% [markdown]
# ## 5. Refit the best cell and score it on test, once
#
# The best cell is refitted on the whole training window (same series, so the same step
# budget) with three seeds, and the seed-mean forecast goes on the board as `NBEATSx_tuned`.

# %%
def nbeatsx_tuned(seed):
    return gd.build_nbeatsx(best_params, full_data, max_steps, seed, "NBEATSx_tuned")


tuned_runs = gd.fit_seeds(nbeatsx_tuned, full_data, "02_nbeatsx_tuned")
tuned = gd.seed_forecasts(tuned_runs)
board_fcst = tuned[["unique_id", "ds", "NBEATSx_tuned_mean"]].rename(
    columns={"NBEATSx_tuned_mean": "NBEATSx_tuned"})
test_seeds = seed_medians(tuned_runs, test, train)
print(f"test median RMSSE per seed: {test_seeds.round(4).to_dict()}")

gd.save_forecasts(board_fcst, "02_tuned")
gd.record(board_fcst, test, train, lab="02 HPO round 1 (3-seed mean)",
          fit_seconds={"NBEATSx_tuned": gd.summaries(tuned_runs)["fit_seconds"].mean()},
          notes=f"best cell on validation: {best_label}; fit_seconds is per seed")
gd.median_board()

# %% [markdown]
# ### Marginal wins over the 01 recipe model
#
# Series by series, does the tuned model beat the untuned recipe model from notebook 01?

# %%
recipe_fcst = pd.read_parquet(gd.FORECAST_DIR / "01_recipe.parquet")
recipe_fcst = recipe_fcst[["unique_id", "ds", "NBEATSx_recipe_mean"]]
both = board_fcst.merge(recipe_fcst, on=["unique_id", "ds"])
rmsse = gd.per_series(gd.score(both, test, train))
gplt.plot_wins(rmsse, info, model="NBEATSx_tuned", baseline="NBEATSx_recipe_mean")

# %%
shortlist_entries = []
for label, row in shortlist.iterrows():
    shortlist_entries.append({"params": {**base_params, **cells_by_label[label]},
                              "median_rmsse_valid": float(row["median_rmsse_valid"]),
                              "seed_sd": float(row["seed_sd"])})
gd.save_config("02_shortlist", {
    "base_params": base_params,
    "grid": GRID,
    "max_steps": max_steps,
    "seeds": gd.SEEDS,
    "noise_floor": float(noise_floor),
    "shortlist": shortlist_entries,
    "best_params": best_params,
    "futr_exog": recipe_cfg["futr_exog"],
    "y_shift": recipe_cfg["y_shift"],
})

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## Discussion
#
# * **Why not a big search?** Every cell costs three full fits. A grid of 12 cells is already
#   36 fits; an Optuna study of 50 trials would be 150. The noise floor says how fine a
#   difference any search can resolve, so spend the budget on seeds before trials.
# * **One at a time hides interactions.** A higher learning rate may suit the small network and
#   hurt the large one. The grid shows that; a one-at-a-time sweep from the default cannot.
# * **Tuning on one window overfits that window.** The research's phase 4 defaults were picked
#   on the test window (§1). Compare the gap between the validation and test scores of the best
#   cell: if test is much worse, the choice was luck.
# * The research found that the most accurate trials had the **weakest** price response (§6).
#   That is why round 2 picks from a shortlist instead of taking the single best cell.
#
# **Try it:** add `"dropout_prob_theta": [0.1, 0.257]` to `FULL_GRID` and see whether its
# spread beats the noise floor. It doubles the number of fits.
