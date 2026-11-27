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
# # Day 2 · 12 (full data) — Lab: does the realized price give the sales away? (10 min)
#
# Every price-aware model today reads the test window's **realized** selling price as its price
# plan. In production a planned `sale_price` is given for the future, and it is usually close
# to the price realized, so that is a fair stand-in. But this dataset has a quirk: a week with
# **no sales** is recorded at exactly the **regular price** (84% of test-window zero-sale weeks
# are priced at regular; weeks at regular have no sales 66% of the time, other weeks 1.7%). A
# real plan would not carry that fingerprint. The Day 1 framework (`shared/README.md`) treats
# realized future prices as leaky for this reason.
#
# So we separate two things: the value of knowing the price, which a production plan gives,
# and the value of the zero-sales fingerprint, which it does not. Same trained models, same
# test window; only the future price changes:
#
# | Plan | Test-window `sale_price` | Stands for |
# |---|---|---|
# | `realized` | as recorded | every notebook so far |
# | `no_fingerprint` | as recorded, but zero-sale weeks at regular take last week's price | a plan |
# | `no_change` | the last training week's price, all 14 weeks | no future prices |
#
# `no_fingerprint` stands for a production plan: close to the realized price, without the
# zero-sales fingerprint. `no_change` is a strict lower bound: knowing no future prices.
#
# The promotion flag stays as recorded in every plan, and every derived input
# (`discount_depth`, `price_index`, the rolling ratio) is rebuilt from the plan's price.
#
# **In the chain:** reads the selected config (`gd.selected_config()`), its saved model
# `05c_selected` and `02_shortlist`; writes `12_price_plan`. Nothing is retrained except
# LightGBM (minutes on CPU): the GFM and the foundation models only forecast.

# %%
import numpy as np
import pandas as pd
import polars as pl

import gd_helpers as gd
import gd_price as gp

bc = gd.bc
gd.status()

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 1. The rule, written before the results
#
# * **The fingerprint matters** for a model if `no_fingerprint` is worse than `realized` by more
#   than the 02 noise floor in median RMSSE. Then part of that model's score came from reading
#   zero-sales weeks off the price, not from forecasting.
# * `realized` − `no_change` is the value of knowing the future price. That part a production
#   price plan does give, so it is not a leak.
# * Every forecast is clipped at 0 and capped at 3× the item's largest training week, as on
#   the board.

# %%
cfg = gd.selected_config()
noise_floor = gd.load_config("02_shortlist")["noise_floor"]
CAP = gd.load_config("08_reconcile")["item_cap"]["multiple_of_train_max"]
covariates = cfg["futr_exog"]
PLANS = ["realized", "no_fingerprint", "no_change"]
raw = gd.train_panel()
eval_ids = gd.eval_ids()
test_start = pl.lit(bc.TEST_START).cast(pl.Date)


def plan_panel(plan: str) -> pl.DataFrame:
    """The raw panel with the test window's ``sale_price`` replaced by ``plan``'s."""
    df = raw.sort(["unique_id", "ds"])
    in_test = pl.col("ds") >= test_start
    if plan == "realized":
        return df
    if plan == "no_fingerprint":
        at_regular = (pl.col("sale_price") - pl.col("regular_price")).abs() < 1e-9
        fingerprint = in_test & (pl.col("y") == 0) & at_regular
        price = pl.when(fingerprint).then(None).otherwise(pl.col("sale_price"))
        return df.with_columns(price.forward_fill().over("unique_id").alias("sale_price"))
    if plan == "no_change":
        last = (df.filter(~in_test).group_by("unique_id")
                .agg(pl.col("sale_price").last().alias("_last")))
        return (df.join(last, on="unique_id", how="left")
                .with_columns(pl.when(in_test).then(pl.col("_last"))
                              .otherwise(pl.col("sale_price")).alias("sale_price"))
                .drop("_last"))
    raise ValueError(plan)


prepared = {p: gp.prepare(cfg, panel=plan_panel(p)) for p in PLANS}
_, train, test, _ = prepared["realized"]
history = gd.only(bc.to_nixtla(train), eval_ids).groupby("unique_id")["y"].max()

t = gd.only(bc.to_nixtla(test, ["sale_price"]), eval_ids)
changed = {p: float((gd.only(bc.to_nixtla(prepared[p][2], ["sale_price"]), eval_ids)["sale_price"]
                     .to_numpy() != t["sale_price"].to_numpy()).mean()) for p in PLANS}
print(f"{len(eval_ids):,} series | covariates {covariates} | share of test weeks whose price "
      f"the plan changes: " + ", ".join(f"{p} {v:.1%}" for p, v in changed.items()))

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 2. Forecasts under each plan
#
# The GFM is the saved first seed of the selected model (`05c_selected`, with its scaler floor
# and input clip). Chronos-2, t0-gamma and Timestral are zero-shot with the main setup's
# covariates, as in 07b. LightGBM is 00's model, fitted once on the training weeks (identical
# for every plan) and asked to forecast under each plan.

# %%
def capped(fc: pd.DataFrame, col: str) -> pd.DataFrame:
    out = fc[["unique_id", "ds", col]].copy()
    out["ds"] = pd.to_datetime(out["ds"])
    out[col] = out[col].clip(lower=0, upper=out["unique_id"].map(history) * CAP)
    return out


def gfm(plan: str):
    def compute() -> pd.DataFrame:
        data = prepared[plan][3]
        fc = gd.predict(gd.load_model(cfg["saved_model"]), data)
        return capped(fc.rename(columns={gd.bc.model_columns(fc)[0]: "GFM"}), "GFM")
    return compute


_foundation = {}


def foundation(name: str, plan: str):
    def compute() -> pd.DataFrame:
        _, tr, te, _ = prepared[plan]
        hist = gd.only(bc.to_nixtla(tr, covariates), eval_ids)
        futr = gd.only(bc.to_nixtla(te, covariates).drop(columns="y"), eval_ids)
        if name == "Chronos2":
            pipe = _foundation.setdefault(name, gd.chronos_pipeline())
            fc = gd.chronos_forecast(pipe, hist, futr, alias=name)
        else:
            kind = {"Timestral": "timestral", "T0gamma": "t0"}[name]
            model = _foundation.setdefault(name, gd.foundation_model(kind))
            fc = gd.foundation_forecast(kind, model, hist, futr, alias=name,
                                        covariates=covariates)
        return capped(fc, name)
    return compute


LGBM_EXOG = ["promo", "sale_price", "discount_depth", "price_ratio_to_rolling_mean"]
_lgbm = {}


def lightgbm(plan: str):
    def compute() -> pd.DataFrame:
        import lightgbm as lgb
        from mlforecast import MLForecast
        from mlforecast.lag_transforms import RollingMean
        from mlforecast.target_transforms import LocalStandardScaler

        if "model" not in _lgbm:             # 00's LightGBM, fitted once on the training weeks
            tr = bc.split(gp.clean_prices(raw, **gp.PRICE_CLEAN), cutoff=bc.TEST_START)[0]
            mlf = MLForecast(
                models={"LightGBM": lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05,
                                                      num_leaves=63, verbosity=-1,
                                                      random_state=bc.RANDOM_SEED)},
                freq=bc.FREQ, lags=[1, 2, 4, 13, 52],
                lag_transforms={1: [RollingMean(4), RollingMean(13)]},
                date_features=["month", "week"], target_transforms=[LocalStandardScaler()])
            mlf.fit(bc.to_nixtla(tr, LGBM_EXOG), static_features=[])
            _lgbm["model"] = mlf
        te = bc.split(gp.clean_prices(plan_panel(plan), **gp.PRICE_CLEAN),
                      cutoff=bc.TEST_START)[1]
        future = gd.only(bc.to_nixtla(te, LGBM_EXOG).drop(columns="y"), eval_ids)
        fc = _lgbm["model"].predict(h=bc.H, X_df=future, ids=eval_ids)
        return capped(fc, "LightGBM")
    return compute


MODELS = {"GFM": gfm, "T0gamma": lambda p: foundation("T0gamma", p),
          "Chronos2": lambda p: foundation("Chronos2", p),
          "Timestral": lambda p: foundation("Timestral", p), "LightGBM": lightgbm}
forecasts = {}
for name, make in MODELS.items():
    for plan in PLANS:
        with bc.Timer() as tm:
            forecasts[(name, plan)] = gd.cached(f"12_{name}_{plan}", make(plan))
        if gd.TRAIN:
            print(f"{name:9s} {plan:15s} {tm.seconds:6.1f}s")

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ### A check: the realized plan repeats the board
#
# Under `realized` each model should match its board forecasts (the GFM against 05c's run of
# the same seed, the others against 07, 07b and 00), up to GPU noise.

# %%
board = bc.load_forecasts()
board["ds"] = pd.to_datetime(board["ds"])
seed1 = gd.load_runs(cfg["selected_runs"], seeds=[gd.SEEDS[0]])[0]["forecast"]
seed1 = capped(seed1.rename(columns={cfg["forecast_column"]: "GFM_seed1"}), "GFM_seed1")
refs = {"GFM": seed1, "T0gamma": capped(board, "T0gamma_zs_cov"),
        "Chronos2": capped(board, "Chronos2_zs_cov"),
        "Timestral": capped(board, "Timestral_zs_cov"), "LightGBM": capped(board, "LightGBM")}
check = {}
for name, ref in refs.items():
    ref = gd.only(ref, eval_ids).rename(columns={ref.columns[-1]: "board"})
    both = forecasts[(name, "realized")].merge(ref, on=["unique_id", "ds"])
    med = gd.per_series(gd.score(both, test, train)).median()
    check[name] = {"here": med[name], "board": med["board"]}
pd.DataFrame(check).T.round(4)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 3. Scores under each plan

# %%
rows = []
for (name, plan), fc in forecasts.items():
    col = f"{name} | {plan}"
    f = fc.rename(columns={name: col})
    metrics = bc.summarise(gd.score(f, test, train), f, test).iloc[0]
    truth = bc.to_nixtla(test)[["unique_id", "ds", "y"]]
    truth["ds"] = pd.to_datetime(truth["ds"])
    actual = f.merge(truth, on=["unique_id", "ds"])
    zero = actual["y"] == 0
    rows.append({"model": name, "plan": plan, "median_rmsse": metrics["median_rmsse"],
                 "mean_rmsse": metrics["mean_rmsse"], "wape": metrics["wape"],
                 "mean forecast, zero-sale weeks": float(actual.loc[zero, col].mean()),
                 "zero-sale weeks forecast < 1": float((actual.loc[zero, col] < 1).mean())})
scores = pd.DataFrame(rows)
median = scores.pivot(index="model", columns="plan", values="median_rmsse")[PLANS]
median["fingerprint (no_fingerprint − realized)"] = median["no_fingerprint"] - median["realized"]
median["price knowledge (no_change − no_fingerprint)"] = median["no_change"] - \
    median["no_fingerprint"]
median["fingerprint matters"] = median["fingerprint (no_fingerprint − realized)"] > noise_floor
median.round(4)

# %%
scores.set_index(["model", "plan"]).round(3)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ### Reading the result
#
# * **The check passes**: under `realized` every model repeats its board score exactly.
# * **The fingerprint is real but small.** Removing it costs every model 0.005–0.008 of median
#   RMSSE (t0-gamma 0.502 → 0.507, Chronos-2 0.512 → 0.518, LightGBM 0.572 → 0.580): about one
#   noise floor. It changes 4.5% of test weeks' prices, and the order of the models does not
#   change. By the rule it "matters" for every model, but only just.
# * **Knowing the price is worth far more**: 0.11–0.15 of median RMSSE between
#   `no_fingerprint` and `no_change`. That is the value a production price plan gives, and it is
#   most of what the covariates were credited with.
# * **Without any future price** every model scores 0.65–0.69, level with the best Day 1
#   LightGBM that uses no future information (0.664 on the same series and window).
# * The GFM here is the saved first seed (0.526), the one with the remaining blow-up, hence its
#   WAPE.
#
# *Exact numbers vary with seed and hardware.*

# %%
gd.save_config("12_price_plan", {
    "plans": PLANS,
    "share_changed": changed,
    "noise_floor": noise_floor,
    "check_realized_vs_board": {k: {kk: round(float(vv), 4) for kk, vv in v.items()}
                                for k, v in check.items()},
    "median_rmsse": median[PLANS].round(4).to_dict("index"),
    "fingerprint_matters": median["fingerprint matters"].to_dict(),
    "scores": scores.round(4).to_dict("records"),
})

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## Discussion
#
# The realized selling price does carry a fingerprint of zero-sales weeks, and every model reads
# it a little, but it is worth about one noise floor. What the price-aware models gain over the
# Day 1 benchmarks comes mainly from knowing the price, which a production plan supplies. In
# production the planned `sale_price` is close to, but not exactly, the price realized, so the
# honest Day 2 number is the `no_fingerprint` one: ≈ 0.005–0.008 above the board.
#
# Two caveats. The promotion flag is the recorded one for all 14 weeks; the Day 1 framework
# assumes a promotion plan only four weeks ahead. And the models were trained on history that
# contains the fingerprint, so they may have learned to rely on it; removing it at forecast time
# measures that reliance, which is small.
