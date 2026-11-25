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
# # Day 2 · 11 (full data) — Check: every series, not just the sample (10 min)
#
# Notebooks 00–10 score on the original labs' fixed sample of 2,000 series, stratified by sales
# class. The v1 labs now score on **every** series that can be scored: 17,238 of the 17,241
# eligible ones (3 have no seasonal-naive scale). This notebook rescores our main models on all
# of them, so the two sets of results can be compared directly, and checks how well the sample
# stood in for the whole panel.
#
# Goals
#
# 1. Forecast all 17,241 series with the baselines, the foundation models and the main GFMs.
# 2. Score them on every scorable series, next to the same models on the 2,000-series sample.
# 3. Compare with the v1 labs' all-series results for the models both have.
#
# **In the chain:** reads `00_baselines`, the selected config, `01_recipe`, `08_reconcile`
# (its forecast cap) and `10_final`; writes `11_all_series`. Its scores go to their own board
# (`leaderboard_all_series.csv`) and its forecasts to `results/day2_full/all_series/`, so the
# 2,000-series board and notebook 10 are unchanged.

# %%
from dataclasses import replace

import lightgbm as lgb
import numpy as np
import pandas as pd
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean
from mlforecast.target_transforms import LocalStandardScaler
from neuralforecast.models import NHITS
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, Naive, SeasonalNaive

import gd_helpers as gd
import gd_plots as gplt
import gd_price as gp

bc = gd.bc
gd.status()
gplt.use_style()

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## Intuition: a poll and a census
#
# The 2,000-series sample is a poll: drawn so that every sales class is represented in its
# proper share, it should give the same median as the census of all 17,241. WAPE is different:
# it adds up every unit of error and divides by every unit sold, so a few best sellers carry
# most of the weight, and a sample of 2,000 can contain more or fewer of them than its share.
# Expect the medians to agree closely and WAPE to move by a point or two.

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 1. Settings
#
# Which models, and how each gets its forecasts for every series:
#
# | Model | Forecasts for all series from | Seeds |
# |---|---|---|
# | Naive, SeasonalNaive, AutoETS, LightGBM | refitted as in 00 (CPU) | – |
# | Chronos2_zs, Chronos2_zs_cov, Timestral_zs_cov, T0gamma_zs_cov | zero-shot (07, 07b) | – |
# | GFM_selected | the saved model (seed 1) and two refits (seeds 2 and 3) | 3 |
# | NHITS_recipe | three refits with 01's recipe | 3 |
# | NBEATSx_recipe, GFM_quantile | the saved model | 1 |
# | ENS_mean_top3 | the mean of 10's top three | – |
#
# `GFM_synthetic` is left out: its saved model predates 05b's final refit, so it is not the
# model on the board. Each refit is checked against the run it repeats, on the 2,000 series.

# %%
OUT = gd.OUT_DIR / "all_series"
OUT.mkdir(parents=True, exist_ok=True)
BOARD = gd.OUT_DIR / "leaderboard_all_series.csv"

cfg = gd.selected_config()
cfg01 = gd.load_config("01_recipe")
cfg10 = gd.load_config("10_final")
CAP = gd.load_config("08_reconcile")["item_cap"]["multiple_of_train_max"]    # 3, as in 08
raw = gd.train_panel()
panel, train, test, data = gp.prepare(cfg, panel=raw, synthetic_seed=cfg.get("synthetic_seed"))
info = gplt.series_table(panel)
all_ids = sorted(panel["unique_id"].unique().to_list())
eval_ids = gd.eval_ids()
covariates = cfg["futr_exog"]
print(f"{len(all_ids):,} series forecast | {len(eval_ids):,} in the sample | "
      f"covariates {covariates}")


def everywhere(d: gd.NixtlaData) -> gd.NixtlaData:
    """The same training data, with the forecast made for every original series."""
    return replace(d, predict_df=gd.only(d.fit_df, all_ids),
                   futr_df=bc.to_nixtla(test, d.futr_exog).drop(columns="y"))


def clipped(fcst: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Forecasts clipped at 0 (sales cannot be negative), as float32 to keep the files small."""
    out = fcst[["unique_id", "ds"] + cols].copy()
    out["ds"] = pd.to_datetime(out["ds"])
    for c in cols:
        out[c] = out[c].clip(lower=0).astype("float32")
    return out


# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 2. Baselines, refitted on every series
#
# The same four reference models as 00. The local models now fit every series, and LightGBM,
# which already trained on all of them, forecasts all of them.

# %%
def baselines_everywhere() -> pd.DataFrame:
    """00's four reference models, forecasting every series."""
    base_train, base_test = bc.split(raw, cutoff=bc.TEST_START)
    local = StatsForecast(models=[Naive(), SeasonalNaive(season_length=bc.SEASON),
                                  AutoETS(season_length=1)], freq=bc.FREQ, n_jobs=gd.N_JOBS)
    stats = local.forecast(df=bc.to_nixtla(base_train), h=bc.H).reset_index()
    exog = ["promo", "sale_price", "discount_depth", "price_ratio_to_rolling_mean"]
    mlf = MLForecast(
        models={"LightGBM": lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05,
                                              num_leaves=63, random_state=bc.RANDOM_SEED,
                                              verbosity=-1)},
        freq=bc.FREQ, lags=[1, 2, 4, 13, 52],
        lag_transforms={1: [RollingMean(4), RollingMean(13)]},
        date_features=["month", "week"], target_transforms=[LocalStandardScaler()],
    )
    mlf.fit(bc.to_nixtla(base_train, exog), static_features=[])
    future = bc.to_nixtla(base_test, exog).drop(columns="y")
    lgbm = mlf.predict(h=bc.H, X_df=future, ids=all_ids)
    out = stats.merge(lgbm, on=["unique_id", "ds"])
    return clipped(out, ["Naive", "SeasonalNaive", "AutoETS", "LightGBM"])


with bc.Timer() as t:
    baselines = gd.cached("11_baselines", baselines_everywhere)
print(f"baselines: {baselines['unique_id'].nunique():,} series ({t.seconds:.0f}s)")

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 3. Foundation models, zero-shot on every series
#
# The same calls as 07 and 07b: each series' full training history as context, and the
# selected config's price and promotion inputs for the covariate runs. Chronos-2 with
# covariates was already forecast for every series in 08 (`08_bottom_chronos`).

# %%
hist = bc.to_nixtla(train, covariates)
future = bc.to_nixtla(test, covariates).drop(columns="y")


def chronos_plain() -> pd.DataFrame:
    """Chronos-2 on sales history alone."""
    fc = gd.chronos_forecast(gd.chronos_pipeline(), hist[["unique_id", "ds", "y"]],
                             alias="Chronos2_zs")
    return clipped(fc, ["Chronos2_zs"])


def newer(kind: str, alias: str):
    """Timestral or t0-gamma with the price covariates (07b's main setup)."""
    def compute() -> pd.DataFrame:
        fc = gd.foundation_forecast(kind, gd.foundation_model(kind), hist, future, alias=alias,
                                    covariates=covariates)
        return clipped(fc, [alias])
    return compute


chronos_cov = pd.read_parquet(gd.TABLE_DIR / "08_bottom_chronos.parquet")      # from 08
chronos_cov = chronos_cov.rename(columns={"Base": "Chronos2_zs_cov"})
foundation = [gd.cached("11_chronos2_zs", chronos_plain),
              clipped(chronos_cov, ["Chronos2_zs_cov"]),
              gd.cached("11_timestral", newer("timestral", "Timestral_zs_cov")),
              gd.cached("11_t0gamma", newer("t0", "T0gamma_zs_cov"))]
print({fc.columns[-1]: f"{fc['unique_id'].nunique():,} series" for fc in foundation})

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 4. GFMs: the saved models and the refits
#
# A saved model forecasts every series directly. `GFM_selected` keeps one saved seed, so its
# other two seeds are refitted here with 05's exact settings; `NHITS_recipe` kept none, so all
# three are refitted with 01's. Each refit forecasts every series as it trains, and is cached.

# %%
def saved_model(name: str, d: gd.NixtlaData, column: str, alias: str):
    """Forecasts for every series from the saved model ``name``."""
    def compute() -> pd.DataFrame:
        fc = gd.predict(gd.load_model(name), everywhere(d))
        return clipped(fc.rename(columns={column: alias}), [alias])
    return compute


recipe_panel_train, recipe_panel_test = bc.split(raw, cutoff=bc.TEST_START)
recipe_data = gd.nixtla_data(recipe_panel_train, recipe_panel_test, futr_exog=bc.FUTR_EXOG_COLS,
                             predict_ids=all_ids)
saved = {
    "GFM_selected_s1": gd.cached("11_saved_05_selected",
                                 saved_model("05_selected", data, "GFM_selected",
                                             "GFM_selected_s1")),
    "NBEATSx_recipe": gd.cached("11_saved_01_recipe",
                                saved_model("01_nbeatsx_recipe", recipe_data, "NBEATSx_recipe",
                                            "NBEATSx_recipe")),
    "GFM_quantile": gd.cached("11_saved_09_quantile",
                              saved_model("09_quantile", data, "GFM_quantile-median",
                                          "GFM_quantile")),
}

# %%
selected_runs_name = f"05_confirm_{next(c['key'] for c in cfg['confirm'] if c['selected'])}"


def selected_model(seed):
    return gd.build_nbeatsx(cfg["params"], data, cfg["max_steps"], seed, "GFM_selected")


def nhits_recipe(seed):
    p = gd.NBEATSX_DEFAULTS
    return NHITS(h=bc.H, input_size=p["input_size"], futr_exog_list=recipe_data.futr_exog,
                 stat_exog_list=recipe_data.stat_exog, scaler_type="robust",
                 batch_size=p["batch_size"], learning_rate=p["learning_rate"],
                 random_seed=seed, alias="NHITS_recipe", **gd.recipe_kwargs(cfg01["max_steps"]))


selected_refits = gd.fit_seeds(selected_model, everywhere(data), "11_gfm_selected_all",
                               seeds=gd.SEEDS[1:])
nhits_refits = gd.fit_seeds(nhits_recipe, recipe_data, "11_nhits_recipe_all")
print(pd.DataFrame([r["summary"] for r in selected_refits + nhits_refits])
      [["name", "seed", "fit_seconds", "best_frac"]])

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ### Do the refits repeat the runs they copy?
#
# Each refit's forecasts on the 2,000 sample series, against the cached run with the same
# seed: the median RMSSE of each, and the largest gap in forecasts. A GPU fit is not exactly
# repeatable, so expect gaps of the order of the seed noise (≈ 0.005), not zero.

# %%
def repeat_check(refits: list[dict], original: str, alias: str) -> pd.DataFrame:
    """Median RMSSE on the sample: refit against the cached original, seed by seed."""
    rows = []
    for run in refits:
        seed = run["summary"]["seed"]
        orig = gd.load_runs(original, seeds=[seed])[0]["forecast"]
        new = gd.only(run["forecast"], eval_ids)
        both = orig[["unique_id", "ds", alias]].merge(
            new[["unique_id", "ds", alias]], on=["unique_id", "ds"], suffixes=("_orig", "_refit"))
        both[f"{alias}_orig"] = both[f"{alias}_orig"].clip(lower=0)
        both[f"{alias}_refit"] = both[f"{alias}_refit"].clip(lower=0)
        med = gd.per_series(gd.score(both, test, train)).median()
        rows.append({"model": alias, "seed": seed, "original": med[f"{alias}_orig"],
                     "refit": med[f"{alias}_refit"],
                     "gap": med[f"{alias}_refit"] - med[f"{alias}_orig"]})
    return pd.DataFrame(rows)


saved_check = gd.only(saved["GFM_selected_s1"], eval_ids).merge(
    gd.load_runs(selected_runs_name, seeds=[gd.SEEDS[0]])[0]["forecast"],
    on=["unique_id", "ds"])
saved_gap = float((saved_check["GFM_selected_s1"]
                   - saved_check["GFM_selected"].clip(lower=0)).abs().max())
print(f"saved 05_selected against its run (seed {gd.SEEDS[0]}): largest forecast gap "
      f"{saved_gap:.4f} units")
repeats = pd.concat([repeat_check(selected_refits, selected_runs_name, "GFM_selected"),
                     repeat_check(nhits_refits, "01_nhits_recipe", "NHITS_recipe")])
repeats.round(4)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# Every refit repeats its original's median RMSSE on the sample to four decimals, and the saved
# model matches its run to within 0.01 units: the refitted seeds are the same models.

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 5. Every model on every series
#
# The 3-seed means, the single-seed models, and the mean of 10's top three.

# %%
def seed_mean(frames: list[pd.DataFrame], alias: str) -> pd.DataFrame:
    """The mean of several seeds' forecasts (each clipped at 0 first, as the board does)."""
    out = None
    for i, f in enumerate(frames):
        f = clipped(f, [alias]).rename(columns={alias: f"s{i}"})
        out = f if out is None else out.merge(f, on=["unique_id", "ds"])
    cols = [c for c in out.columns if c.startswith("s")]
    return out.assign(**{alias: out[cols].mean(axis=1)})[["unique_id", "ds", alias]]


gfm_selected = seed_mean([saved["GFM_selected_s1"].rename(columns={"GFM_selected_s1":
                                                                   "GFM_selected"})]
                         + [r["forecast"] for r in selected_refits], "GFM_selected")
nhits = seed_mean([r["forecast"] for r in nhits_refits], "NHITS_recipe")

everything = baselines
for f in foundation + [gfm_selected, nhits, saved["NBEATSx_recipe"], saved["GFM_quantile"]]:
    everything = everything.merge(f, on=["unique_id", "ds"], how="outer")
top3 = [m for m in cfg10["ensembles"]["top3"] if m in everything.columns]
if len(top3) < len(cfg10["ensembles"]["top3"]):
    print("not forecast here, left out of the top 3:",
          sorted(set(cfg10["ensembles"]["top3"]) - set(top3)))
everything["ENS_mean_top3"] = everything[top3].mean(axis=1)
MODELS = bc.model_columns(everything)
missing = everything[MODELS].isna().sum()
print(f"{everything['unique_id'].nunique():,} series × {len(MODELS)} models | top 3: {top3} | "
      f"missing forecasts: {int(missing.sum())}")
everything.to_parquet(OUT / "11_all_series.parquet", index=False)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 6. The cap, as in 08
#
# On every series a few of the N-BEATSx forecasts explode: up to ≈ 8,000 times the item's
# largest training week. They sit mostly outside the 2,000-series sample, so 00–10 barely saw
# them; 08, which sums every item, did, and caps each item's forecast at `CAP` (3) times its
# largest training week. We apply the same cap here, to **every** model, and show the scores
# with and without it. Real sales went above that ceiling in ≈ 0.5% of test weeks.
#
# The likely cause is the robust window scaler, which also scales the price and promotion
# inputs: an input that never moved in the 14 input weeks has no spread, so when it moves in
# the forecast weeks its scaled value is enormous, and N-BEATSx's exogenous stack passes it
# straight to the forecast. The standard scaler, which would avoid it, made the model clearly
# worse in the research (§3 of `docs/gfm_learnings.md`), so the cap is the better fix.

# %%
train_max = gd.only(bc.to_nixtla(train), all_ids).groupby("unique_id")["y"].max()
ceiling = everything["unique_id"].map(train_max) * CAP
capped = everything.copy()
touched = {}
for m in MODELS:
    over = capped[m] > ceiling
    touched[m] = {"item-weeks capped": int(over.sum()),
                  "items capped": int(capped.loc[over, "unique_id"].nunique()),
                  "of them in the sample": int(capped.loc[over & capped["unique_id"].isin(
                      set(eval_ids)), "unique_id"].nunique())}
    capped[m] = capped[m].clip(upper=ceiling)
capped["ENS_mean_top3"] = capped[top3].mean(axis=1)          # the mean of the capped members
pd.DataFrame(touched).T.sort_values("item-weeks capped", ascending=False)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 7. Scores: all series against the sample
#
# Median and mean RMSSE and WAPE for each model, capped, on every scorable series and on the
# 2,000 sample series, then the same on all series without the cap. RMSSE is scaled by each
# series' 52-week seasonal-naive error; a series whose scale is zero cannot be scored and drops
# out, as in the original labs.

# %%
all_train = gd.only(train, all_ids)


def summary(fcst: pd.DataFrame, ids) -> pd.DataFrame:
    """Median and mean RMSSE, WAPE and series count per model, on ``ids``."""
    metrics = bc.score(gd.only(fcst, ids), gd.only(test, ids), gd.only(all_train, ids))
    out = bc.summarise(metrics, gd.only(fcst, ids), gd.only(test, ids)).set_index("model")
    return out[["median_rmsse", "mean_rmsse", "wape", "n_series"]]


summary_all = summary(capped, all_ids)
summary_sample = summary(capped, eval_ids)
summary_raw = summary(everything, all_ids)
scores = pd.concat({"all series": summary_all, "sample": summary_sample,
                    "all series, no cap": summary_raw.drop(columns="n_series")}, axis=1)
scores = scores.sort_values(("all series", "median_rmsse"))
scores.round(3)

# %%
board = summary_all.join(summary_raw.add_suffix("_no_cap").drop(columns="n_series_no_cap"))
board = board.reset_index().sort_values("median_rmsse")
board["lab"] = "11 all series (capped)"
board.to_csv(BOARD, index=False)
gaps = pd.DataFrame({"median RMSSE: all − sample": scores[("all series", "median_rmsse")]
                     - scores[("sample", "median_rmsse")],
                     "WAPE: all − sample": scores[("all series", "wape")]
                     - scores[("sample", "wape")],
                     "WAPE: cap effect": scores[("all series", "wape")]
                     - scores[("all series, no cap", "wape")]})
gaps.round(3)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ### Reading the result
#
# * **The sample was a good stand-in.** With the cap, every model's median RMSSE on all series
#   is within 0.03 of its sample value (mostly within 0.01), and the order of the models is the
#   same except for one swap (`GFM_quantile` and `NBEATSx_recipe`, 0.001 apart on the sample).
#   t0-gamma is still the best single model (0.496 on all series), and the top-3 average
#   better still (0.488).
# * **WAPE moves more**, by up to ≈ 5 points (Naive), because it follows the best sellers.
# * **Without the cap, the N-BEATSx models break down on all series.** `GFM_selected`'s WAPE
#   goes from 0.258 to 0.404 and its mean RMSSE from 1.29 to 2.75; `GFM_quantile`'s WAPE from
#   0.293 to 0.582. Their medians do not move: a few hundred items carry it all. In the sample
#   the cap changes nothing visible, which is why 00–10 did not need it.
# * **The cap touches the other models too, but lightly.** It trims a few hundred item-weeks of
#   Chronos-2, Timestral and AutoETS (WAPE down by ≤ 0.008) and nothing of LightGBM and the
#   naive models.
#
# *Exact numbers vary with seed and hardware.*

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 8. Against the v1 labs, on the same series
#
# The v1 labs' all-series results (`day2_neural/v1/slides/reference_results.json` on the
# Day 2 slides branch, 25 Nov), for the models both sets of labs run, with the same windows,
# eligibility filter and WAPE. Their numbers are copied here, not read from that branch. The v1
# labs do not cap, so both columns of ours are shown: none of the shared models is changed
# much by the cap.

# %%
V1 = pd.DataFrame([  # model, mean RMSSE, median RMSSE, WAPE (17,238 series)
    ("ENS_mean_top3", 1.164, 0.498, 0.219), ("Chronos2_zs_cov", 1.294, 0.511, 0.229),
    ("NHITSx", 1.303, 0.572, 0.275), ("LightGBM", 1.354, 0.569, 0.291),
    ("NHITS", 1.389, 0.638, 0.334), ("Chronos2_zs", 1.432, 0.630, 0.311),
    ("AutoETS", 1.482, 0.695, 0.402), ("Naive", 1.516, 0.745, 0.420),
    ("SeasonalNaive", 1.709, 0.913, 0.446),
], columns=["v1 model", "v1 mean_rmsse", "v1 median_rmsse", "v1 wape"])
SAME = {"ENS_mean_top3": "ENS_mean_top3", "Chronos2_zs_cov": "Chronos2_zs_cov",
        "LightGBM": "LightGBM", "Chronos2_zs": "Chronos2_zs", "AutoETS": "AutoETS",
        "Naive": "Naive", "SeasonalNaive": "SeasonalNaive", "NHITS_recipe": "NHITSx"}
ours = summary_all.loc[list(SAME), ["median_rmsse", "mean_rmsse", "wape"]]
ours["wape (no cap)"] = summary_raw.loc[list(SAME), "wape"]
ours["v1 model"] = [SAME[m] for m in ours.index]
versus = ours.reset_index().merge(V1, on="v1 model")
versus["WAPE gap (ours − v1)"] = versus["wape"] - versus["v1 wape"]
versus.round(3)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ### Reading the result
#
# * **The two sets of labs measure the same thing.** Without the cap, LightGBM, AutoETS, Naive,
#   SeasonalNaive and Chronos-2 without covariates match the v1 labs to three decimals, and
#   Chronos-2 with covariates to 0.001. So any gap elsewhere is the models, not the scoring.
# * **Training explains the rest.** `NHITS_recipe` and `NHITSx` are both NHITS with the price
#   inputs. Ours trains for ≈ 10,000 steps with the research recipe (Huber loss, warm-up,
#   clipping, best weights) over 3 seeds, the v1 lab's for 500 steps with library defaults, to
#   suit a laptop. Ours is 3 WAPE points better (0.244 against 0.275), 0.04 in median RMSSE.
# * `ENS_mean_top3` is a different ensemble on each side: ours averages three foundation models
#   (0.203), the v1 labs' two Chronos-2 variants and NHITSx (0.219).
#
# *Exact numbers vary with seed and hardware.*

# %%
gd.save_config("11_all_series", {
    "n_series_forecast": len(all_ids),
    "n_series_scored": int(summary_all["n_series"].max()),
    "models": MODELS,
    "top3": top3,
    "left_out": {"GFM_synthetic": "saved model predates 05b's final refit"},
    "seeds": {"GFM_selected": gd.SEEDS, "NHITS_recipe": gd.SEEDS,
              "NBEATSx_recipe": gd.SEEDS[:1], "GFM_quantile": gd.SEEDS[:1]},
    "saved_gap_05_selected": saved_gap,
    "cap": {"multiple_of_train_max": CAP, "applied_to": "every model", "touched": touched},
    "repeat_check": repeats.round(4).to_dict("records"),
    "scores_all": summary_all.round(4).to_dict("index"),
    "scores_sample": summary_sample.round(4).to_dict("index"),
    "scores_all_no_cap": summary_raw.round(4).to_dict("index"),
    "versus_v1": versus.round(4).to_dict("records"),
    "board": str(BOARD.relative_to(gd.ROOT)),
})

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## Discussion
#
# * The sample was drawn to match the panel's sales classes. Did that make it a good stand-in
#   for the median? For WAPE?
# * WAPE weights the best sellers most. Which would you report to a buyer, and which to a
#   planner with 17,000 items?
#
# **Try it:** score the models on class A–I series only (`info`), where most of the units are
# sold. Do the gaps between models grow or shrink?
