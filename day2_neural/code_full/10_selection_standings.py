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
# side by side and ask better questions than "which has the lowest mean RMSSE?":
#
# 1. **Win rates**: on what share of series does model A beat model B?
# 2. **Bradley–Terry and Elo**: one strength number per model from every series-level duel.
# 3. **Ties**: which models are within the seed noise (notebook 02) of the best?
# 4. **Ensembles**: mean, median, top 3 and per-class selection, chosen on half the series.
# 5. **Final standings**, with accuracy **and** price response side by side.

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
# ## 1. Every saved forecast, on the evaluation series
#
# Models that forecast only a subset of the series (e.g. a foundation model run on a CPU) are
# left out: a fair standing needs every model on the same series. As in notebooks 06–09, the
# selected model (05 or 05b, `gd.selected_forecasts()`) is the column `GFM_selected`. If 05b
# won, 05's own model stays in as `GFM_05`.

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
# ## Intuition: a league table for models
#
# Treat every series as a match between two models: the one with the lower RMSSE wins. The
# **win rate** ignores the margin, so one disaster cannot dominate. **Bradley–Terry** and
# **Elo** turn all the matches into one rating, as in chess: +100 points is about a 64% chance
# of winning a series. Two models whose medians differ by less than the seed noise are
# **tied**: retraining with a different seed could swap them.

# %% [markdown]
# ## 2. Pairwise win rates
#
# Cell (row A, column B) is the share of series where A has the lower RMSSE. Ties (e.g. two
# models that both forecast zero) count half.

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
fig, ax = plt.subplots(figsize=(2 + 0.75 * len(order), 1 + 0.55 * len(order)))
im = ax.imshow(W.to_numpy(), cmap="RdBu", vmin=0.2, vmax=0.8)
ax.set_xticks(range(len(order)), order, rotation=60, ha="right")
ax.set_yticks(range(len(order)), order)
for (i, j), v in np.ndenumerate(W.to_numpy()):
    if i != j:
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
ax.grid(False)
fig.colorbar(im, ax=ax, label="row beats column")
ax.set_title("Pairwise win rate (series level, RMSSE)")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Bradley–Terry and Elo
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
# When the median order and the rating order disagree, one model wins **often by a little**
# and the other **rarely by a lot**. The mean is pulled by the hardest (mostly intermittent)
# series; the ratings count every series once.

# %% [markdown]
# ## 4. Who is tied with the best?
#
# Notebook 02 measured how much the median RMSSE moves when only the seed changes: the
# **noise floor**. A model within it of the best cannot be told apart from the best.

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
# ## 5. Ensembles
#
# Choosing on the test window and then scoring on it is cheating. As in the original lab we
# use a **cross-sectional split**: half the series choose which models go in, the other half
# is scored. The choice uses the median RMSSE.

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
held_out = ~fc["unique_id"].isin(sel_ids)
both = fc.merge(ens, on=["unique_id", "ds"])
held_metrics = gd.score(both[held_out.to_numpy()], test, train)
held_scores = bc.summarise(held_metrics, both[held_out.to_numpy()], test)
held_scores = held_scores.sort_values("median_rmsse").reset_index(drop=True)
held_scores[["model", "median_rmsse", "mean_rmsse", "wape", "n_series"]].round(3).head(12)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ### Reading the result
#
# * Averaging lets the models' mistakes partly cancel. Does the top-3 average beat every
#   single model on the held-out half, and by more than the noise floor?
# * Per-class selection chooses on half the series and often does not carry over: selection
#   overfits, averaging does not.
#
# The ensembles go on the board scored on **all** series, half of which chose them, so their
# board numbers are optimistic. Quote the held-out table.

# %%
gd.save_forecasts(ens, "10_ensembles")
gd.record(ens, test, train, lab="10 ensembles",
          notes=f"top3={top3}; chosen on half the series (median RMSSE)")
gd.median_board()

# %% [markdown]
# ## 6. Price response next to accuracy
#
# An accurate model can still respond the wrong way to price. Notebooks 04, 05 and 05b report
# the response with the same numbers every time: the median arc elasticity at ±20%, the share
# of series that respond the wrong way (perverse), and how many series could be measured.

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
# Where a notebook kept its per-series sweeps, we recompute the report here, from the seed-mean
# sweep, so every model is measured the same way. Each board model is paired with the runs (or
# table) its sweeps came from.

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
    return gp.arc_elasticity(sweep)


strength = {}
response = {}
for model, (kind, name) in sweep_sources().items():
    el = seed_mean_elasticity(kind, name)
    if el is None or model not in models:
        continue
    top = el[el["shift"] == el["shift"].max()]
    strength[model] = top.set_index("unique_id")["e_arc"]
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
# ## 7. The final model
#
# The rule, fixed before looking: **keep the selected GFM unless another single model beats
# its median RMSSE by more than the noise floor.** The GFM's price response was checked in
# 04–05b; a model that is only tied on accuracy gives no reason to drop that.

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

# %%
contenders = list(dict.fromkeys([final_model] + list(ratings.index[:5])
                                + ["LightGBM", "SeasonalNaive"]))
contenders = [m for m in contenders if m in R.columns]
gplt.plot_class_heatmap(R, info, models=contenders)

# %% [markdown]
# ### The final model against the yardsticks

# %%
gplt.plot_wins(R, info, model=final_model, baseline="SeasonalNaive")

# %% [markdown]
# Against LightGBM, the strongest Day 1 model (or against the selected GFM, if LightGBM is
# the final model).

# %%
rival = "LightGBM" if final_model != "LightGBM" else selected_col
gplt.plot_wins(R, info, model=final_model, baseline=rival)

# %% [markdown]
# ## 8. Final standings
#
# Median RMSSE on all evaluation series (mean as a dot). The shaded band is the noise floor
# above the best model: bars that end inside it are tied with the best. Ensembles are scored on
# all series here, so they are flattered (see section 5).

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
# * Which models are tied with the best? If several are, accuracy alone cannot choose: the
#   price response, the cost of training and serving, and the seed spread can.
# * Does the final model win where the sales are (the volume-weighted bars), or only on count?
# * The research used LightGBM for accuracy and N-BEATSx for price response. Does the
#   GFM + LightGBM average keep the best of both, and how would you sweep its price response?
#
# **Try it:** change the selection half (`bc.RANDOM_SEED` in the split) and see whether the
# top 3 and the per-class choice stay the same.
