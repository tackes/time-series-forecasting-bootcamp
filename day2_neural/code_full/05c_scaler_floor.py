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
# # Day 2 · 05c (full data) — Lab: fixing the exploding forecasts at the cause (10 min)
#
# On every series (08, 11) a few hundred N-BEATSx item forecasts explode, up to ≈ 8,000 times
# the item's largest training week. 08 and 11 cap them. Here we test a fix at the cause.
#
# **The cause.** The robust scaler scales each window's inputs by their median and MAD over
# the 14 input weeks. `discount_depth` and `price_index` drift by ≈ 0.001 a week with CPI
# deflation, so in a quiet window their spread is tiny. A 20% cut in the forecast weeks then
# becomes hundreds of scaled units, and the exogenous stack passes it straight to the forecast.
# In the exploding items, `discount_depth`'s largest scaled future value has a median of
# ≈ 600 (11 for the other items), and both price inputs move together on every discount, so
# dropping one would not help. `promo` is exactly flat then, but the scaler's fallback keeps it
# near 1.
#
# **The fix.** Keep the robust scaler, but never let its spread fall below **0.01**, fixed
# before any result. A 20% cut is then at most 20 scaled units. Sales are in units, so their
# spread is almost never that small and the floor leaves them alone.
#
# | | |
# |---|---|
# | Starts from | the selected 05 model, unchanged |
# | Changes | the robust scaler's spread is floored at 0.01 (`gd.floor_robust_scaler`) |
# | Measured | validation accuracy and price probes; exploding item-weeks on every series |
# | Decides | whether the floor replaces the cap, by a rule written before the results |
#
# **Part 2 (sections 6–9)** cleans the prices first, then tests two more arms: cleaned
# prices alone, and cleaned prices with the floor. That part was added after part 1's result
# showed a data error behind the worst explosions; part 1 is unchanged.
#
# **In the chain:** reads `05_selected`, `02_shortlist` and `08_reconcile`, 05b's cached
# validation and probe runs of 05, and 11's all-series forecasts. Writes `05c_scaler_floor`
# and `05c_price_clean`. Later notebooks read neither yet: carrying an arm forward means
# cleaned prices for every price-aware model and a rerun of 05 → 11.

# %%
from dataclasses import replace

import pandas as pd

import gd_helpers as gd
import gd_price as gp

bc = gd.bc
gd.status()

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 1. Settings

# %%
FLOOR = 0.01
cfg = gd.load_config("05_selected")
noise_floor = gd.load_config("02_shortlist")["noise_floor"]
CAP = gd.load_config("08_reconcile")["item_cap"]["multiple_of_train_max"]    # 3, as in 08
price_channel = gd.load_config("04_price").get("price_channel", "price_index")
params, max_steps = cfg["params"], cfg["max_steps"]
PROBE_RUNS = "_p7b"                                     # 05b's cached runs of 05, with probes

raw = gd.train_panel()
panel, train, test, data = gp.prepare(cfg, panel=raw)
_, inner_train, valid, valid_data = gp.prepare(cfg, panel=raw, cutoff=bc.VALID_START)
all_ids = sorted(panel["unique_id"].unique().to_list())
eval_ids = gd.eval_ids()
report_ids, valid_report_ids = gp.report_ids(train), gp.report_ids(inner_train)


def everywhere(d: gd.NixtlaData, window) -> gd.NixtlaData:
    """The same training data, with the forecast made for every original series."""
    return replace(d, predict_df=gd.only(d.fit_df, all_ids),
                   futr_df=bc.to_nixtla(window, d.futr_exog).drop(columns="y"))


def floored_model(d: gd.NixtlaData, alias: str = "GFM_floor", floor: float | None = FLOOR):
    def build(seed):
        model = gd.build_nbeatsx(params, d, max_steps, seed, alias)
        return model if floor is None else gd.floor_robust_scaler(model, floor)
    return build


print(f"floor {FLOOR} | {max_steps:,} steps | seeds {gd.SEEDS} | covariates {data.futr_exog} | "
      f"{len(all_ids):,} series")

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 2. The rule, written before the results
#
# The floor replaces the cap for the GFM **only if all three hold**:
#
# 1. **Accuracy.** Its median validation RMSSE on the sample is no more than the 02 noise
#    floor above 05's.
# 2. **Price response.** On validation, its one-week 20% cut elasticity is no more than 0.1
#    weaker than 05's (≈ −1.8), and its share of right-signed random calendar cuts no more
#    than 1 point lower.
# 3. **Explosions.** On every series, the 3-seed mean forecast goes above `CAP` × the item's
#    largest training week in at most a tenth as many item-weeks as 05's does (11's count).
#
# 1 and 2 are the checks any arm meets (05b's rule). 3 is what the floor is for. It reads the
# test window, because 05's all-series forecasts exist only there (11); it is a count of
# failures, not a score, and no setting is tuned on it.

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 3. Validation: the floor against 05 (3 seeds each)
#
# 05's validation fits are 05b's, cached with the same seeds and data. The floored fits
# forecast every series, so we can also count explosions on validation.

# %%
valid_calendars = gp.promo_calendars(valid_data.futr_df, valid_report_ids)


def valid_extras(nf, d):
    """Sweep and phase 7b probes on the validation window, while the model is in memory."""
    return {"sweep": gp.price_sweep(nf, d, valid_report_ids, channel=price_channel),
            **gp.gfm_probes(nf, d, valid_report_ids, valid_calendars)}


valid_05 = gd.load_runs("05b_valid_05" + PROBE_RUNS)
with bc.Timer() as t:
    valid_floor = gd.fit_seeds(floored_model(valid_data), everywhere(valid_data, valid),
                               "05c_valid_floor", extras=valid_extras)
if gd.TRAIN:
    print(f"validation fits: {t.seconds / 60:.1f} min")


# %%
def response_numbers(runs: list[dict]) -> dict:
    """The rule's response numbers for the seed-mean model of ``runs``."""
    probes = gp.probe_report(gp.seed_mean_q([r["probe7b"] for r in runs], gp.PROBE_KEYS))
    calendar = gp.probe_response(gp.seed_mean_q([r["calendar"] for r in runs], gp.CAL_KEYS))
    return {"cut20_elasticity": gp.headline(probes)["median_elasticity"],
            "cut20_lift": gp.headline(probes)["median_lift"],
            "calendar_right_sign": float(calendar["right_sign"].mean())}


def seed_mean(runs: list[dict], ids) -> pd.DataFrame:
    """The 3-seed mean forecast on ``ids``, each seed clipped at 0 first (as the board does)."""
    fcst = gd.seed_forecasts(runs)
    seeds = [c for c in fcst.columns if c[-5:] != "_mean" and c not in ("unique_id", "ds")]
    alias = runs[0]["summary"]["alias"]
    out = gd.only(fcst, ids)[["unique_id", "ds"]].copy()
    out[alias] = gd.only(fcst, ids)[seeds].clip(lower=0).mean(axis=1).to_numpy()
    return out


def exploding(fcst: pd.DataFrame, column: str, history) -> dict:
    """Item-weeks (and items) whose forecast is above ``CAP`` × the item's largest week."""
    top = gd.only(bc.to_nixtla(history), fcst["unique_id"].unique()).groupby("unique_id")["y"]
    over = fcst[column] > fcst["unique_id"].map(top.max()) * CAP
    return {"item-weeks": int(over.sum()), "items": int(fcst.loc[over, "unique_id"].nunique()),
            "largest multiple": float((fcst[column] / fcst["unique_id"].map(
                top.max().where(top.max() > 0))).max())}


valid_numbers = {}
for name, runs in {"05": valid_05, "floor": valid_floor}.items():
    fc = seed_mean(runs, eval_ids)
    col = fc.columns[-1]
    rmsse = gd.per_series(gd.score(fc, valid, inner_train))
    valid_numbers[name] = {**response_numbers(runs), "median_rmsse": float(rmsse[col].median()),
                           "mean_rmsse": float(rmsse[col].mean())}
valid_numbers = pd.DataFrame(valid_numbers).T
print("floor, validation, every series:",
      exploding(seed_mean(valid_floor, all_ids), "GFM_floor", inner_train))
valid_numbers.round(4)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ### Reading the result
#
# * **Accuracy improves.** The floored model's median validation RMSSE is 0.574 against 05's
#   0.583: better by ≈ 2× the noise floor (0.005), not just within it.
# * **The price response is stronger.** A one-week 20% cut gives an elasticity of −1.95 against
#   −1.77 (lift +55% against +49%), and 99.7% of random calendar cuts lift the forecast.
# * **But explosions remain.** On every series the floored model still has 646 items above
#   the cap on validation, and one forecast at 7 × 10⁷ times its item's largest week. The mean
#   RMSSE (1.13 against 1.08) shows it.
#
# *Exact numbers vary with seed and hardware.*

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 4. The test window: every series, 3 seeds
#
# The floored model on all training weeks, forecasting every series, with the test-window
# probes. 05's numbers come from 11 (forecasts) and 05b (probes), with the same seeds.

# %%
calendars = gp.promo_calendars(data.futr_df, report_ids)


def test_extras(nf, d):
    return {"sweep": gp.price_sweep(nf, d, report_ids, channel=price_channel),
            **gp.gfm_probes(nf, d, report_ids, calendars)}


with bc.Timer() as t:
    test_floor = gd.fit_seeds(floored_model(data), everywhere(data, test), "05c_floor",
                              extras=test_extras)
if gd.TRAIN:
    print(f"test fits: {t.seconds / 60:.1f} min")
gd.summaries(test_floor)[["seed", "fit_seconds", "best_frac", "final_over_best"]].round(3)

# %%
all_05 = pd.read_parquet(gd.OUT_DIR / "all_series" / "11_all_series.parquet",
                         columns=["unique_id", "ds", "GFM_selected"])
all_05["ds"] = pd.to_datetime(all_05["ds"])
all_floor = seed_mean(test_floor, all_ids)
both = all_05.merge(all_floor, on=["unique_id", "ds"])
explosions = pd.DataFrame({"05": exploding(both, "GFM_selected", train),
                           "floor": exploding(both, "GFM_floor", train)}).T
top = gd.only(bc.to_nixtla(train), all_ids).groupby("unique_id")["y"].max()
capped = both.copy()
for m in ["GFM_selected", "GFM_floor"]:
    capped[m] = capped[m].clip(upper=capped["unique_id"].map(top) * CAP)


def summary(fcst: pd.DataFrame, ids) -> pd.DataFrame:
    """Median and mean RMSSE and WAPE per model on ``ids``, as in 11."""
    metrics = bc.score(gd.only(fcst, ids), gd.only(test, ids), gd.only(train, ids))
    out = bc.summarise(metrics, gd.only(fcst, ids), gd.only(test, ids)).set_index("model")
    return out[["median_rmsse", "mean_rmsse", "wape"]]


test_scores = pd.concat({"all, no cap": summary(both, all_ids),
                         "all, capped": summary(capped, all_ids),
                         "sample, no cap": summary(both, eval_ids)}, axis=1)
test_response = pd.DataFrame({"05": response_numbers(gd.load_runs("05b_reference_05"
                                                                  + PROBE_RUNS)),
                              "floor": response_numbers(test_floor)}).T
explosions

# %%
test_scores.round(3)

# %%
test_response.round(4)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 5. The decision

# %%
v = valid_numbers
checks = {
    "accuracy within the noise floor (validation)":
        v.loc["floor", "median_rmsse"] - v.loc["05", "median_rmsse"] <= noise_floor,
    "price response no weaker (validation)":
        (v.loc["floor", "cut20_elasticity"] <= v.loc["05", "cut20_elasticity"] + 0.1)
        and (v.loc["floor", "calendar_right_sign"]
             >= v.loc["05", "calendar_right_sign"] - 0.01),
    "explosions down tenfold (every series, test)":
        explosions.loc["floor", "item-weeks"] <= explosions.loc["05", "item-weeks"] / 10,
}
use = "floor" if all(checks.values()) else "cap"
print(f"the GFM uses: {use}")
pd.Series(checks)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ### Reading the result
#
# * **The rule keeps the cap.** The floor passes on accuracy and price response, but it only
#   halves the explosions (236 item-weeks in 56 items, against 466 in 101), not the tenfold
#   cut the rule asked for.
# * **The largest explosions are a data error.** Item `1120_S1` has `price_index` values of
#   2,000 to 29,000 in some weeks: a price in the wrong units. With the floor, its forecast
#   reaches 5 × 10⁹ units in one seed and carries almost all of the uncapped WAPE (12.6). 29
#   series have a `price_index` above 3 somewhere in training (3 in the sample). No scaler
#   can make sense of such inputs.
# * **One seed (90218) still blows up a normal-priced item** (`6653_S1`, ≈ 550× its largest
#   week). The floor bounds each scaled input, but the exogenous stack can still extrapolate
#   far from a deep discount. Seed 46335 is clean apart from `1120_S1`: uncapped WAPE 0.237.
# * **Many of the remaining "explosions" are moderate** (35–45×) and in items whose sales
#   changed sharply, for example a collapse after a big week. A 3× cap is the right guard for
#   those.
# * **Capped, the floor is the better model:** all-series median RMSSE 0.511 against 0.540,
#   WAPE 0.251 against 0.258; on the sample 0.523 against 0.546. Uncapped, its sample WAPE
#   is 6.99, so its board row must be read with the cap in mind.
#
# *Exact numbers vary with seed and hardware.*

# %%
fit_seconds = gd.summaries(test_floor)["fit_seconds"].mean()
sample_floor = seed_mean(test_floor, eval_ids)
gd.save_forecasts(sample_floor, "05c_floor")
gd.record(sample_floor, test, train, lab=f"05c scaler floor ({len(gd.SEEDS)}-seed mean)",
          fit_seconds=fit_seconds, notes="fit_seconds is per seed")
gd.median_board("05c")

# %%
gd.save_config("05c_scaler_floor", {
    **cfg,
    "scale_floor": FLOOR,
    "seeds": gd.SEEDS,
    "use": use,
    "rule": list(checks),
    "checks": {k: bool(val) for k, val in checks.items()},
    "valid": valid_numbers.round(4).to_dict("index"),
    "explosions": explosions.to_dict("index"),
    "test_scores": {f"{a} | {b}": s for (a, b), s in
                    test_scores.round(4).to_dict().items()},
    "test_response": test_response.round(4).to_dict("index"),
    "runs": {"valid": "05c_valid_floor", "test": "05c_floor",
             "valid_05": "05b_valid_05" + PROBE_RUNS, "test_05": "05b_reference_05" + PROBE_RUNS},
    "forecasts": "05c_floor",
    "forecast_column": "GFM_floor",
})

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 6. Part 2: cleaning the prices
#
# `price_index` is this week's price divided by the series' usual price (its median before
# the cutoff): 1 is the usual price, 0.8 is 20% off, 3 is triple. Two kinds of bad price:
#
# * **placeholders**: weeks priced £0.01–0.05 (229 weeks in 79 series) with normal sales.
#   They look like 99% discounts. When they are most of a series' weeks, as in `1120_S1`,
#   the placeholder becomes the usual price and every real price looks thousands of times
#   too high;
# * **out-of-range prices**: more than 6× above or below the usual price, most likely in the
#   wrong units.
#
# `gp.clean_prices` replaces both with the median of the good prices in the 9 weeks around
# them (the series' usual price if there are none), then rebuilds `discount_depth` and
# `price_index`. The thresholds (£0.05, 6×) were fixed before any fit. Sales are untouched,
# so the scored series are the same. Setting them to missing is not an option: N-BEATSx
# cannot take missing future inputs. Shifting prices by +1 is not one either: a £0.60 item
# is a real cheap item, and the shift would shrink cheap items' discounts more than dear
# ones'.

# %%
CLEAN = {"k": 6.0, "floor": 0.05}
cfg_clean = {**cfg, "price_clean": CLEAN}
replaced = gp.bad_price_weeks(raw, **CLEAN).to_pandas()
_, _, _, data_clean = gp.prepare(cfg_clean, panel=raw)
_, _, _, valid_clean = gp.prepare(cfg_clean, panel=raw, cutoff=bc.VALID_START)
ranges = pd.DataFrame({
    "raw": [float(panel["price_index"].min()), float(panel["price_index"].max()),
            int((panel["discount_depth"] > 0.8).sum())],
    "cleaned": [float(data_clean.fit_df["price_index"].min()),
                float(data_clean.fit_df["price_index"].max()),
                int((data_clean.fit_df["discount_depth"] > 0.8).sum())]},
    index=["price_index min", "price_index max", "weeks with discount_depth > 0.8"])
print(f"{len(replaced):,} weeks replaced in {replaced['unique_id'].nunique():,} series "
      f"({replaced['unique_id'].isin(set(eval_ids)).sum():,} weeks in the sample)")
ranges.round(3)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 7. The rule for part 2, written before the results
#
# Each arm (**clean**, **clean + floor**) is compared with 05 on the validation window, with
# the same checks as section 2's first two:
#
# 1. median validation RMSSE no more than the 02 noise floor above 05's;
# 2. one-week 20% cut elasticity no more than 0.1 weaker than 05's, and right-signed calendar
#    cuts no more than 1 point fewer.
#
# Of the arms that pass, the one with the lower validation median RMSSE is recommended. The
# cap stays in every case, as a safety net: part 1 showed no setting removes every
# explosion. The exploding item-weeks on every series (test window) are reported for each.

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 8. Validation and test: the two arms (3 seeds each)

# %%
ARMS = {"clean": ("GFM_clean", None), "clean_floor": ("GFM_clean_floor", FLOOR)}
arm_valid, arm_test = {}, {}
with bc.Timer() as t:
    for arm, (alias, floor) in ARMS.items():
        arm_valid[arm] = gd.fit_seeds(floored_model(valid_clean, alias, floor),
                                      everywhere(valid_clean, valid), f"05c_valid_{arm}",
                                      extras=valid_extras)
        arm_test[arm] = gd.fit_seeds(floored_model(data_clean, alias, floor),
                                     everywhere(data_clean, test), f"05c_{arm}",
                                     extras=test_extras)
if gd.TRAIN:
    print(f"part 2 fits: {t.seconds / 60:.1f} min")
pd.concat([gd.summaries(r) for r in list(arm_valid.values()) + list(arm_test.values())])[
    ["name", "seed", "fit_seconds", "best_frac", "final_over_best"]].round(3)

# %%
valid2 = {"05": valid_numbers.loc["05"].to_dict(), "floor": valid_numbers.loc["floor"].to_dict()}
for arm, runs in arm_valid.items():
    fc = seed_mean(runs, eval_ids)
    col = fc.columns[-1]
    rmsse = gd.per_series(gd.score(fc, valid, inner_train))
    valid2[arm] = {**response_numbers(runs), "median_rmsse": float(rmsse[col].median()),
                   "mean_rmsse": float(rmsse[col].mean())}
valid2 = pd.DataFrame(valid2).T
valid2.round(4)

# %%
all_arms = both.copy()
for arm, runs in arm_test.items():
    all_arms = all_arms.merge(seed_mean(runs, all_ids), on=["unique_id", "ds"])
arm_cols = {"05": "GFM_selected", "floor": "GFM_floor",
            **{arm: alias for arm, (alias, _) in ARMS.items()}}
explosions2 = pd.DataFrame({arm: exploding(all_arms, col, train)
                            for arm, col in arm_cols.items()}).T
capped2 = all_arms.copy()
for col in arm_cols.values():
    capped2[col] = capped2[col].clip(upper=capped2["unique_id"].map(top) * CAP)
test_scores2 = pd.concat({"all, no cap": summary(all_arms, all_ids),
                          "all, capped": summary(capped2, all_ids),
                          "sample, no cap": summary(all_arms, eval_ids)}, axis=1)
test_response2 = pd.concat([test_response, pd.DataFrame(
    {arm: response_numbers(runs) for arm, runs in arm_test.items()}).T])
explosions2

# %%
test_scores2.round(3)

# %%
test_response2.round(4)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 9. The decision for part 2

# %%
checks2 = {}
for arm in ARMS:
    checks2[arm] = {
        "accuracy within the noise floor (validation)":
            valid2.loc[arm, "median_rmsse"] - valid2.loc["05", "median_rmsse"] <= noise_floor,
        "price response no weaker (validation)":
            (valid2.loc[arm, "cut20_elasticity"] <= valid2.loc["05", "cut20_elasticity"] + 0.1)
            and (valid2.loc[arm, "calendar_right_sign"]
                 >= valid2.loc["05", "calendar_right_sign"] - 0.01)}
passing = [arm for arm in ARMS if all(checks2[arm].values())]
recommended = min(passing, key=lambda a: valid2.loc[a, "median_rmsse"]) if passing else "05"
print(f"recommended: {recommended} (the cap stays in every case)")
pd.DataFrame(checks2)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ### Reading the result
#
# * **Cleaning alone fails the rule.** Validation median 0.599 against 05's 0.583, 3× the
#   noise floor worse. It also has *more* exploding items (156 against 101): with the fake 99%
#   discounts gone, more input windows are near-flat, which is the mechanism the floor targets.
# * **Clean + floor passes and is recommended.** Validation 0.562 against 0.583 (≈ 4× the
#   noise floor better), one-week cut elasticity −2.03 against −1.77. On the test window,
#   capped, it ties the floor alone (all series 0.514 against 0.511; sample 0.519 against
#   0.523) and beats 05 (0.540 and 0.546).
# * **The cap is still essential.** Uncapped, clean + floor's all-series WAPE is 221: one item,
#   `1120_S1`, carries it. Its prices are now in range but swing 2–5× its input-window level,
#   and even floored at 0.01 that is ≈ 400 scaled units. The floor bounds small moves in
#   near-flat inputs, not large moves; the exogenous stack is linear in its inputs and
#   extrapolates. `1120_S1` sells nothing in 90% of its weeks.
# * **Training is less steady than 05's.** Several fits find their best weights early
#   (`best_frac` 0.3–0.4) and one ends 3.4× worse than its best; the best-weights callback
#   recovers them, but it is a sign the price inputs still kick the training around.
#
# *Exact numbers vary with seed and hardware.*

# %%
for arm, (alias, _) in ARMS.items():
    sample_arm = seed_mean(arm_test[arm], eval_ids)
    gd.save_forecasts(sample_arm, f"05c_{arm}")
    gd.record(sample_arm, test, train, lab=f"05c {arm.replace('_', ' + ')} "
              f"({len(gd.SEEDS)}-seed mean)",
              fit_seconds=gd.summaries(arm_test[arm])["fit_seconds"].mean(),
              notes="fit_seconds is per seed")
gd.median_board("05c")

# %%
gd.save_config("05c_price_clean", {
    **cfg_clean,
    "scale_floor": FLOOR,
    "seeds": gd.SEEDS,
    "replaced": {"weeks": int(len(replaced)), "series": int(replaced["unique_id"].nunique())},
    "recommended": recommended,
    "cap_kept": True,
    "rule": list(next(iter(checks2.values()))),
    "checks": {arm: {k: bool(v) for k, v in c.items()} for arm, c in checks2.items()},
    "valid": valid2.round(4).to_dict("index"),
    "explosions": explosions2.to_dict("index"),
    "test_scores": {f"{a} | {b}": v for (a, b), v in test_scores2.round(4).to_dict().items()},
    "test_response": test_response2.round(4).to_dict("index"),
    "runs": {arm: {"valid": f"05c_valid_{arm}", "test": f"05c_{arm}"} for arm in ARMS},
    "forecasts": {arm: f"05c_{arm}" for arm in ARMS},
    "forecast_column": {arm: alias for arm, (alias, _) in ARMS.items()},
})

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## Discussion
#
# The robust scaler's tiny spreads were one cause of the explosions, and the floor removes it:
# it halves the exploding items and makes the GFM more accurate and more price-responsive. But
# it is not a fix on its own. A handful of series carry corrupted prices, and N-BEATSx's
# exogenous stack can still extrapolate wildly in a rare seed. So the cap stays, as the rule
# says.
#
# Part 2 cleaned the prices: placeholders and out-of-range weeks replaced from their
# neighbours. That is right as data hygiene, but on its own it costs accuracy and adds
# explosions. With the floor it is the best GFM so far on validation (0.562), with the
# strongest price response (−2.03), and it ties the floor alone on the test window.
#
# Neither fix removes the explosions. The remaining ones are large real moves in a price
# input relative to its input window, passed linearly to the forecast. A fix at that level
# would bound the scaled inputs themselves (clip them at, say, ±10), not their spread. That
# is untested. Until then: **clean + floor, with the cap**, is the recommended GFM, and
# carrying it forward means cleaned prices for every price-aware model and a rerun of
# 05 → 11.
