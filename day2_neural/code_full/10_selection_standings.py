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
# # Day 2 · 10 (full data) — Lab: model selection, ensembles and final standings (20 min)
#
# Every notebook saved its test forecasts for the 2,000 evaluation series. Here we put them
# side by side, with the selected model (`gd.selected_config()`: 05 or 05b) as the incumbent,
# and ask better questions than "which has the lowest RMSSE?".
#
# Goals
#
# 1. **Win rates**: on what share of series does model A beat model B?
# 2. **Bradley–Terry and Elo**: one strength number per model from every series-level duel.
# 3. **Ties**: which models are within the seed noise (notebook 02) of the best?
# 4. **Ensembles**: mean, median, top 3 and per-class selection, chosen on half the series.
# 5. **Final standings**, with accuracy **and** price response side by side.
#
# **In the chain:** reads the selected config, `00_baselines`, `02_shortlist`, `04_price`,
# `05_selected` and `05b_synthetic`, writes `10_final`.

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## Intuition: a league table for models
#
# ![Rating difference against win chance](../slides/img/elo_curve.png)
#
# Treat every series as a match between two models: the one with the lower RMSSE wins. The
# **win rate** ignores the margin, so one disaster cannot dominate. **Bradley–Terry** and
# **Elo** turn all the matches into one rating, as in chess: +100 points is about a 64% chance
# of winning a series. Two models whose medians differ by less than the seed noise are
# **tied**: retraining with a different seed could swap them.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import gd_helpers as gd
import gd_plots as gplt
import gd_price as gp

bc = gd.bc
gd.status()
gplt.use_style()

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 1. Every saved forecast, on the evaluation series
#
# Models that forecast only a subset of the series (e.g. a foundation model run on a CPU) are
# left out: a fair standing needs every model on the same series. As in 06–09, the selected
# model is the column `GFM_selected`. If 05b won, 05's own model stays in as `GFM_05`. The
# table lists each model's median RMSSE.

# %%
panel = gd.train_panel()
train, test = bc.split(panel, cutoff=bc.TEST_START)
info = gplt.series_table(panel)
eval_ids = gd.eval_ids()

selected_cfg = gd.selected_config()
selected_col = "GFM_selected"
synthetic_won = selected_cfg.get("use") == "05b"

fc = gd.only(bc.load_forecasts(), eval_ids)
earlier_run = [c for c in fc.columns if c.startswith(("ENS_", "SEL_"))]
fc = fc.drop(columns=earlier_run)
if synthetic_won:            # 05b's column becomes GFM_selected; 05's is kept as GFM_05
    fc = fc.drop(columns=selected_cfg["forecast_column"])
    fc = fc.rename(columns={"GFM_selected": "GFM_05"})
    fc = fc.merge(gd.selected_forecasts(selected_cfg), on=["unique_id", "ds"], how="left")
all_models = bc.model_columns(fc)
coverage = fc.groupby("unique_id")[all_models].apply(lambda g: g.notna().all()).sum()
coverage = coverage / len(eval_ids)
models = list(coverage[coverage >= 0.99].index)
left_out = coverage[coverage < 0.99].round(3)
if len(left_out):
    print("left out (forecast only a share of the series):", left_out.to_dict())
fc = fc[["unique_id", "ds"] + models].dropna(subset=models, how="any")
print(f"{len(models)} models on {fc['unique_id'].nunique():,} series: {models}")

# %%
metrics = gd.score(fc, test, train)
R = gd.per_series(metrics)[models].dropna()
R.median().sort_values().round(3)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 2. Pairwise win rates
#
# The win matrix: cell (row A, column B) is the share of series where A has the lower RMSSE.
# Ties (e.g. two models that both forecast zero) count half. Rows are sorted by average win
# rate, so the strongest model is at the top. Blue cells (above 0.5) are wins for the row, red
# cells losses; pale cells near 0.5 are coin flips.

# %%
def win_matrix(R: pd.DataFrame) -> pd.DataFrame:
    """Share of series on which each row model beats each column model (ties count half)."""
    a = R.to_numpy()
    ties = (a[:, :, None] == a[:, None, :]).mean(0)
    wins = (a[:, :, None] < a[:, None, :]).mean(0) + 0.5 * ties
    np.fill_diagonal(wins, np.nan)
    return pd.DataFrame(wins, index=R.columns, columns=R.columns)


W = win_matrix(R)
order = W.mean(axis=1).sort_values(ascending=False).index
W = W.loc[order, order]
gplt.plot_heatmap(W, "Pairwise win rate (series level, RMSSE)", cmap="RdBu",
                  label="row beats column", vmin=0.2, vmax=0.8, xrotation=60,
                  figsize=(2 + 0.75 * len(order), 1 + 0.55 * len(order)))

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 3. Bradley–Terry and Elo
#
# Both turn the win matrix into one rating per model, on a scale where 1000 is average.
#
# * **Bradley–Terry**: P(A beats B) = s_A / (s_A + s_B), fitted by maximum likelihood on every
#   pairwise result (the classic MM algorithm).
# * **Elo**: replay the series in a random order and update the ratings after each one. The
#   order matters, so we average over shuffles. All duels of one series are scored at once,
#   which keeps the replay fast on 2,000 series.

# %%
def bradley_terry(R: pd.DataFrame, iters: int = 500) -> pd.Series:
    """Bradley–Terry strengths on an Elo-like scale (1000 = average)."""
    a = R.to_numpy()
    ties = (a[:, :, None] == a[:, None, :]).sum(0)
    wins = (a[:, :, None] < a[:, None, :]).sum(0) + 0.5 * ties
    np.fill_diagonal(wins, 0)
    games = wins + wins.T
    s = np.ones(a.shape[1])
    for _ in range(iters):
        s_new = wins.sum(1) / (games / (s[:, None] + s[None, :])).sum(1)
        s_new = s_new / np.exp(np.log(s_new).mean())
        if np.allclose(s, s_new, rtol=1e-10):
            break
        s = s_new
    return pd.Series(400 * np.log10(s) + 1000, index=R.columns, name="bradley_terry")


def elo(R: pd.DataFrame, k: float = 4.0, n_shuffles: int = 10, seed: int = 0) -> pd.Series:
    """Elo ratings from replaying the series in random orders (all duels of a series at once)."""
    rng = np.random.default_rng(seed)
    a = R.to_numpy()
    total = np.zeros(a.shape[1])
    for _ in range(n_shuffles):
        r = np.full(a.shape[1], 1000.0)
        for row in a[rng.permutation(len(a))]:
            expected = 1 / (1 + 10 ** ((r[None, :] - r[:, None]) / 400))
            result = (row[:, None] < row[None, :]) + 0.5 * (row[:, None] == row[None, :])
            r = r + k * (result - expected).sum(axis=1)     # the diagonal is 0.5 - 0.5
        total += r
    return pd.Series(total / n_shuffles, index=R.columns, name="elo")


ratings = pd.concat([R.median().rename("median_rmsse"), R.mean().rename("mean_rmsse"),
                     W.mean(axis=1).rename("avg_win_rate"), bradley_terry(R), elo(R)], axis=1)
ratings = ratings.sort_values("median_rmsse")
ratings.round(3)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ### Reading the result
#
# * When the median order and the rating order disagree, one model wins **often by a little**
#   and the other **rarely by a lot**.
# * The mean is pulled by the hardest (mostly intermittent) series; the median and the ratings
#   count every series once. A model that ranks well on the median but poorly on the mean is
#   strong on ordinary series and fragile on a few.
# * Look at where `GFM_selected` sits: rating ≈ __ against LightGBM's ≈ __.
#   <!-- TODO: fill after GPU run -->
#
# *Exact numbers vary with seed and hardware.*

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 4. Who is tied with the best?
#
# Notebook 02 measured how much the median RMSSE moves when only the seed changes: the
# **noise floor** (the incumbent's seed-to-seed standard deviation on validation). A model
# within one noise floor of the best cannot be told apart from it. This is the
# **equivalence band** drawn in the final chart.

# %%
shortlist_path = gd.CONFIG_DIR / "02_shortlist.json"
noise_floor = None
if shortlist_path.exists():
    noise_floor = gd.load_config("02_shortlist").get("noise_floor")
best_median = ratings["median_rmsse"].min()
if noise_floor is None:
    print("no 02 noise floor: only the best model counts as best")
    ratings["tied_with_best"] = ratings["median_rmsse"] == best_median
else:
    print(f"noise floor (02): {noise_floor:.4f} median RMSSE")
    ratings["tied_with_best"] = ratings["median_rmsse"] <= best_median + noise_floor
ratings[["median_rmsse", "mean_rmsse", "bradley_terry", "tied_with_best"]].round(3)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 5. Ensembles
#
# Choosing on the test window and then scoring on it is cheating. As in the original lab we
# use a **cross-sectional split**: half the series choose which models go in, the other half
# is scored. The choice uses the median RMSSE. We also add the research's pairing, the
# selected GFM averaged with LightGBM: LightGBM was its most accurate model, N-BEATSx its best
# on price response (§5, [100k set]).

# %%
rng = np.random.default_rng(bc.RANDOM_SEED)
ids = R.index.to_numpy()
sel_ids = set(rng.choice(ids, size=len(ids) // 2, replace=False))
R_sel = R[R.index.isin(sel_ids)]
top3 = list(R_sel.median().nsmallest(3).index)
by_class = R_sel.join(info["sale_class"]).groupby("sale_class")[models].median()
class_choice = by_class.idxmin(axis=1).to_dict()
print("top 3 on the selection half:", top3)
print("per-class choice:", class_choice)

clipped = fc[models].clip(lower=0)
ens = fc[["unique_id", "ds"]].copy()
ens["ENS_mean_all"] = clipped.mean(axis=1)
ens["ENS_median_all"] = clipped.median(axis=1)
ens["ENS_mean_top3"] = clipped[top3].mean(axis=1)
if selected_col in models and "LightGBM" in models:      # the research's pairing (§5)
    ens["ENS_gfm_lightgbm"] = clipped[[selected_col, "LightGBM"]].mean(axis=1)
cls = fc["unique_id"].map(info["sale_class"])
ens["SEL_by_class"] = clipped[top3[0]]
for sale_class, model in class_choice.items():
    ens.loc[cls == sale_class, "SEL_by_class"] = clipped.loc[cls == sale_class, model]
ens_models = bc.model_columns(ens)

# %%
both = fc.merge(ens, on=["unique_id", "ds"])
held_out = both[~both["unique_id"].isin(sel_ids)]
held_metrics = gd.score(held_out, test, train)
held_scores = bc.summarise(held_metrics, held_out, test)
held_scores = held_scores.sort_values("median_rmsse").reset_index(drop=True)
held_scores[["model", "median_rmsse", "mean_rmsse", "wape", "n_series"]].round(3).head(12)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ### Reading the result
#
# * Averaging lets the models' mistakes partly cancel. Does the top-3 average beat every
#   single model on the held-out half, and by more than the noise floor (≈ __ against ≈ __)?
#   <!-- TODO: fill after GPU run -->
# * Per-class selection often fails to carry over to the held-out half. Selection overfits;
#   averaging does not.
# * The ensembles go on the board scored on **all** series, half of which chose them, so their
#   board numbers are optimistic. Quote the held-out table.
#
# *Exact numbers vary with seed and hardware.*

# %%
gd.save_forecasts(ens, "10_ensembles")
gd.record(ens, test, train, lab="10 ensembles",
          notes=f"top3={top3}; chosen on half the series (median RMSSE)")
gd.median_board()

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 6. Price response next to accuracy
#
# An accurate model can still respond the wrong way to price. Notebooks 04, 05 and 05b report
# the response with the same numbers every time: the median arc elasticity at ±20%, the share
# of series that respond the wrong way (perverse), the share that are flat, and how many series
# could be measured. This cell gathers those reports from the configs.

# %%
def find_reports(obj, path: str = "") -> list[dict]:
    """Every elasticity report (a dict with ``median_e20``) inside a config, with its path."""
    found = []
    if isinstance(obj, dict):
        if "median_e20" in obj and not isinstance(obj["median_e20"], dict):
            found.append({"path": path or "(top level)", **obj})
        for key, value in obj.items():
            if isinstance(value, dict):
                found += find_reports(value, f"{path}/{key}".strip("/"))
    return found


REPORT_COLS = ["median_e20", "share_perverse", "share_flat", "n_estimable", "invariance"]
reported = []
for name in ["04_price", "05_selected", "05b_synthetic"]:
    if (gd.CONFIG_DIR / f"{name}.json").exists():
        for rep in find_reports(gd.load_config(name)):
            reported.append({"config": name, **rep})
reported = pd.DataFrame(reported).reindex(columns=["config", "path"] + REPORT_COLS)
print(f"{len(reported)} elasticity reports found in the configs")
reported.round(3)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# Where a notebook kept its per-series sweeps, we recompute the report from the seed-mean
# sweep, so every model is measured the same way. Each board model is paired with the runs (or
# table) its sweeps came from. The strength chart (F6) then shows, per model, the share of
# series in each band, from highly elastic to perverse.

# %%
def sweep_sources() -> dict:
    """Board model -> where its per-seed sweeps are: a run folder, or a cached table."""
    sources = {}
    if (gd.CONFIG_DIR / "04_price.json").exists():
        for arm in gd.load_config("04_price").get("arms", {}):
            sources[f"NBEATSx_{arm}_mean"] = ("run", f"04_{arm}")
        sources["LightGBM"] = ("table", "04_lgbm_sweep")
    column_05 = "GFM_05" if synthetic_won else "GFM_selected"
    column_05b = "GFM_selected" if synthetic_won else "GFM_synthetic"
    if (gd.CONFIG_DIR / "05_selected.json").exists():
        cfg05 = gd.load_config("05_selected")
        for row in cfg05.get("confirm", []):
            if row.get("role") == cfg05.get("role"):
                sources[column_05] = ("run", f"05_confirm_{row['key']}")
    if (gd.CONFIG_DIR / "05b_synthetic.json").exists():
        sources[column_05b] = ("run", "05b_synthetic")
    return sources


def seed_mean_elasticity(kind: str, name: str) -> pd.DataFrame | None:
    """Per-series arc elasticities from the seed-mean sweep, or ``None`` if none was saved."""
    if kind == "table":
        paths = [gd.TABLE_DIR / f"{name}.parquet"]
    else:
        paths = sorted((gd.RUNS_DIR / name).glob("seed*/sweep.parquet"))
    paths = [p for p in paths if p.exists()]
    if not paths:
        return None
    sweeps = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    if not {"unique_id", "shift", "units"} <= set(sweeps.columns):
        return None
    sweep = sweeps.groupby(["unique_id", "shift"], as_index=False)["units"].mean()
    swept = [0.0] + list(gp.SHIFTS)            # LightGBM's sweep also has ±2% and ±30%
    sweep = sweep[np.isclose(sweep["shift"].abs().to_numpy()[:, None], swept).any(axis=1)]
    return gp.arc_elasticity(sweep)


strength = {}
response = {}
for model, (kind, name) in sweep_sources().items():
    el = seed_mean_elasticity(kind, name)
    if el is None or model not in models:
        continue
    strength[model] = gp.e20(el)
    response[model] = gp.elasticity_report(el)
response = pd.DataFrame(response).T.reindex(columns=REPORT_COLS)
print(f"per-series sweeps found for: {list(strength) or 'none'}")
response.round(3)

# %%
if strength:
    gplt.plot_elasticity_strength(strength,
                                  title="Strength of price response (±20%, seed mean)")
else:
    print("No per-series sweeps saved yet: the reported table above is the response evidence.")

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ### Reading the result
#
# * Compare the GFM's median elasticity (≈ __) with the empirical ≈ −3.6 on this data.
#   <!-- TODO: fill after GPU run --> The research's −0.4 to −0.7 came from a different panel
#   [100k set] and is not the target here.
# * If LightGBM appears, expect a response that shrinks as the shock grows and goes flat
#   outside the prices it has seen: an artefact of tree splits, not of demand (§5).
# * Remember the research's sharpest negative result: elasticity tuned on one window did not
#   carry to the next (it shrank 7–18×, §6). Treat any one-window response as provisional.
#
# *Exact numbers vary with seed and hardware.*

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 7. The final model
#
# The rule, fixed before looking: **keep the selected GFM unless another single model beats
# its median RMSSE by more than the noise floor.** The GFM's price response was checked in
# 04–05b; a model that is only tied on accuracy gives no reason to drop that. The rule reads
# the test window, but it can only overturn the incumbent on a clear win, so it is a check,
# not a fresh search.

# %%
singles = ratings.copy()
best_single = singles["median_rmsse"].idxmin()
if selected_col in singles.index and singles.loc[selected_col, "tied_with_best"]:
    final_model = selected_col
else:
    final_model = best_single
print(f"final model: {final_model} (best single model: {best_single})")

# %%
baseline_names = gd.load_config("00_baselines")["baselines"]
standings = ratings.join(response, how="left")
standings["role"] = "model"
standings.loc[standings.index.isin(baseline_names), "role"] = "baseline"
standings.loc[final_model, "role"] = "final"
standings.round(3)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# The class heatmap (F3): median RMSSE by sales class (columns) for the final model, the top
# five and the two yardsticks. Greener is better; read down a column to compare models.

# %%
contenders = list(dict.fromkeys([final_model] + list(ratings.index[:5])
                                + ["LightGBM", "SeasonalNaive"]))
contenders = [m for m in contenders if m in R.columns]
gplt.plot_class_heatmap(R, info, models=contenders)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ### The final model against the yardsticks
#
# First against SeasonalNaive (F4). Bars above 0.5 are classes where the final model wins
# more series than it loses; the volume bars weight each series by its sales.

# %%
gplt.plot_wins(R, info, model=final_model, baseline="SeasonalNaive")

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# Then against LightGBM, the strongest Day 1 model (or against the selected GFM, if LightGBM
# is the final model).

# %%
rival = "LightGBM" if final_model != "LightGBM" else selected_col
gplt.plot_wins(R, info, model=final_model, baseline=rival)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 8. Final standings
#
# One bar per model: median RMSSE on all evaluation series, with the mean as a black dot. The
# shaded **equivalence band** runs from the best single model's median to one noise floor
# above it: bars that end inside it are tied with the best. Green is the final model, grey the
# Day 1 baselines, orange the ensembles. Ensembles are scored on all series here, so they are
# flattered (see section 5).

# %%
all_scores = gd.per_series(gd.score(fc.merge(ens, on=["unique_id", "ds"]), test, train))
board = pd.DataFrame({"median_rmsse": all_scores.median(), "mean_rmsse": all_scores.mean()})
board = board.sort_values("median_rmsse", ascending=False)
colours = []
for m in board.index:
    if m in ens_models:
        colours.append(gplt.COLOURS["bad"])
    elif m == final_model:
        colours.append(gplt.COLOURS["good"])
    elif m in baseline_names:
        colours.append("0.55")
    else:
        colours.append(gplt.MODEL_COLOURS[0])
fig, ax = plt.subplots(figsize=(11, 0.35 * len(board) + 1.5))
ax.barh(board.index, board["median_rmsse"], color=colours)
ax.scatter(board["mean_rmsse"], board.index, color="black", s=12, zorder=3, label="mean")
if noise_floor:
    ax.axvspan(best_median, best_median + noise_floor, color=gplt.COLOURS["band"], alpha=0.2,
               label="within the noise floor of the best single model")
ax.set(xlabel="RMSSE (bar = median, dot = mean; lower is better)",
       title="Final standings · green = final model, grey = Day 1 baselines, "
             "orange = ensembles")
ax.legend(loc="lower right")
fig.tight_layout()
plt.show()

# %%
held_out_table = held_scores.set_index("model")[["median_rmsse", "mean_rmsse"]].round(4)
held_out_table = held_out_table.to_dict("index")
gd.save_config("10_final", {
    "models": models,
    "left_out": left_out.to_dict(),
    "n_series": int(R.shape[0]),
    "noise_floor": noise_floor,
    "final_model": final_model,
    "best_single_model": best_single,
    "final_rule": "selected GFM unless a single model beats its median RMSSE by more than "
                  "the 02 noise floor",
    "standings": standings.round(4).reset_index(names="model").to_dict("records"),
    "tied_with_best": list(ratings.index[ratings["tied_with_best"]]),
    "ensembles": {"top3": top3, "class_choice": class_choice,
                  "selection_half_seed": bc.RANDOM_SEED, "held_out": held_out_table},
    "elasticity_reported": reported.round(4).to_dict("records"),
    "elasticity_from_sweeps": list(strength),
})

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## Discussion
#
# * Which models are tied with the best? If several are, accuracy alone cannot choose: price
#   response, the cost of training and serving, and the seed spread can.
# * Does the final model win where the sales are (the volume bars), or only on count?
# * Does the GFM + LightGBM average keep the best of both? How would you sweep its price
#   response?
#
# ### What we learnt on Day 2
#
# <!-- TODO: fill the ≈ __ numbers after the GPU run -->
#
# * **Training is most of the battle.** Scaled steps, warm-up, clipping and best-weights
#   retention took the research from 3 usable runs in 6 to 6 in 6 (§3, [100k set]); 01 tests
#   the recipe here. Seed noise then sets the bar: a gap below the 02 noise floor is a tie, not
#   a finding [method].
# * **Accuracy.** The final model reaches a median RMSSE of ≈ __ (mean ≈ __), against ≈ __ for
#   SeasonalNaive and ≈ __ for LightGBM on the same 2,000 series.
# * **Price response is a separate test.** An accurate model can ignore price or respond the
#   wrong way. In the research the most accurate trials had the weakest elasticity (§6), and
#   LightGBM, the most accurate model, had an artefactual response (§5) [100k set]. Here the
#   final model's median elasticity is ≈ __ against an empirical ≈ −3.6, with ≈ __% perverse.
# * **Tune on validation, test once.** The research's early tuning was scored on its test
#   window (§1). Once it tuned properly, accuracy carried to a new window but elasticity did
#   not (§6). So trust accuracy gains across windows more than elasticity gains.
# * **New work, not settled practice.** Synthetic discounts (05b), per-cluster models (06),
#   GFM-based reconciliation (08) and conformal bands (09) go beyond the research. Read their
#   results as leads.
# * **Average rather than select.** In the original lab, averaging beat picking a model per
#   segment. Here the top-3 average reaches ≈ __ on the held-out half.
#
# **Try it:** change the selection half (`bc.RANDOM_SEED` in the split) and see whether the
# top 3 and the per-class choice stay the same.
