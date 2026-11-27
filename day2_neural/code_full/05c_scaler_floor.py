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
# showed a data error behind the worst explosions; part 1 is unchanged. **Part 3 (sections
# 10–11)** adds a clip on the scaled price inputs, after part 2 showed large real price moves
# still exploding.
#
# **In the chain:** reads `05_selected`, `02_shortlist` and `08_reconcile`, 05b's cached
# validation and probe runs of 05, and 11's all-series forecasts. Writes `05c_scaler_floor`,
# `05c_price_clean` and `05c_input_clip`. `05c_input_clip`'s `use` key makes
# `gd.selected_config()` return the chosen arm, so 06–11 build on it.

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


def floored_model(d: gd.NixtlaData, alias: str = "GFM_floor", floor: float | None = FLOOR,
                  clip: float | None = None):
    def build(seed):
        model = gd.build_nbeatsx(params, d, max_steps, seed, alias)
        return model if floor is None else gd.floor_robust_scaler(model, floor, clip=clip)
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
# probes. 05's numbers come from 11's first run (forecasts, kept in
# `05c_reference_05_all_series`) and 05b (probes), with the same seeds.

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
# 05's 3-seed all-series forecasts, as 11 made them before 05c was carried forward. Kept as a
# fixed table: 11 now forecasts with 05c's model, so its current file is no longer 05.
REFERENCE_05 = gd.TABLE_DIR / "05c_reference_05_all_series.parquet"
if not REFERENCE_05.exists():           # smoke runs: 11's file still holds 05's forecasts
    REFERENCE_05 = gd.OUT_DIR / "all_series" / "11_all_series.parquet"
all_05 = pd.read_parquet(REFERENCE_05, columns=["unique_id", "ds", "GFM_selected"])
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
# ## 10. Part 3: clipping the scaled price inputs
#
# The floor bounds the *spread*, so a small move in a near-flat input stays small. It does not
# bound a large move: `1120_S1`'s price at 5× its input-window level is still ≈ 400 scaled
# units. Clipping bounds the scaled inputs themselves. The target is never clipped.
#
# **How far to clip** was chosen from the inputs alone, before any fit. With the floor, a
# normal 20% cut is already ≈ 20 units. On the cleaned test-window inputs, the scaled
# `price_index` has a 99th percentile of ≈ 43 and a 99.9th of ≈ 75 (`discount_depth` ≈ 23 and
# ≈ 43; `promo` never above 6). A clip of ±10 would flatten 40% of series' real price
# signal. **±50** touches ≈ 0.5% of item-weeks and cuts the worst (≈ 440) about ninefold.
#
# **The rule**, as section 7's: within the noise floor of 05 on validation, and a price
# response no weaker. Of every arm that passes (clean + floor from part 2, or clean + floor +
# clip), the lower validation median RMSSE is recommended. The cap stays either way.

# %%
CLIP = 50.0
alias_clip = "GFM_clean_floor_clip"
with bc.Timer() as t:
    clip_valid = gd.fit_seeds(floored_model(valid_clean, alias_clip, FLOOR, CLIP),
                              everywhere(valid_clean, valid), "05c_valid_clean_floor_clip",
                              extras=valid_extras)
    clip_test = gd.fit_seeds(floored_model(data_clean, alias_clip, FLOOR, CLIP),
                             everywhere(data_clean, test), "05c_clean_floor_clip",
                             extras=test_extras)
if gd.TRAIN:
    print(f"part 3 fits: {t.seconds / 60:.1f} min")
gd.summaries(clip_valid + clip_test)[["name", "seed", "fit_seconds", "best_frac",
                                       "final_over_best"]].round(3)

# %%
fc = seed_mean(clip_valid, eval_ids)
rmsse = gd.per_series(gd.score(fc, valid, inner_train))
valid3 = pd.concat([valid2, pd.DataFrame({"clean_floor_clip": {
    **response_numbers(clip_valid), "median_rmsse": float(rmsse[alias_clip].median()),
    "mean_rmsse": float(rmsse[alias_clip].mean())}}).T])
all3 = all_arms.merge(seed_mean(clip_test, all_ids), on=["unique_id", "ds"])
arm_cols3 = {**arm_cols, "clean_floor_clip": alias_clip}
explosions3 = pd.DataFrame({arm: exploding(all3, col, train)
                            for arm, col in arm_cols3.items()}).T
capped3 = all3.copy()
for col in arm_cols3.values():
    capped3[col] = capped3[col].clip(upper=capped3["unique_id"].map(top) * CAP)
test_scores3 = pd.concat({"all, no cap": summary(all3, all_ids),
                          "all, capped": summary(capped3, all_ids),
                          "sample, no cap": summary(all3, eval_ids)}, axis=1)
test_response3 = pd.concat([test_response2, pd.DataFrame(
    {"clean_floor_clip": response_numbers(clip_test)}).T])
valid3.round(4)

# %%
explosions3

# %%
test_scores3.round(3)

# %%
test_response3.round(4)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 11. The decision for part 3

# %%
checks3 = {}
for arm in ["clean_floor", "clean_floor_clip"]:
    checks3[arm] = {
        "accuracy within the noise floor (validation)":
            valid3.loc[arm, "median_rmsse"] - valid3.loc["05", "median_rmsse"] <= noise_floor,
        "price response no weaker (validation)":
            (valid3.loc[arm, "cut20_elasticity"] <= valid3.loc["05", "cut20_elasticity"] + 0.1)
            and (valid3.loc[arm, "calendar_right_sign"]
                 >= valid3.loc["05", "calendar_right_sign"] - 0.01)}
passing3 = [arm for arm in checks3 if all(checks3[arm].values())]
recommended3 = min(passing3, key=lambda a: valid3.loc[a, "median_rmsse"]) if passing3 else "05"
print(f"recommended: {recommended3} (the cap stays in every case)")
pd.DataFrame(checks3)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ### Reading the result
#
# * **The rule recommends clean + floor**, but it is a tie: validation 0.562 against 0.567 for
#   clean + floor + clip, 0.005 apart, inside the noise floor. Both beat 05 (0.583).
# * **The clip does what it was for.** The largest forecast falls from 2 × 10⁸ times the
#   item's largest week to 300×; `1120_S1` no longer explodes. Capped, the clipped arm has
#   the best all-series WAPE (0.248) and ties on median RMSSE (0.513).
# * **One explosion is left, and it is not the inputs.** In seed 46335, `6653_S1`, a best
#   seller, is forecast at ≈ 900× its largest week, with every scaled input inside ±5 and
#   ordinary promotions. That is a rare blow-up inside the network, which no input fix can
#   reach; the other two seeds are clean (uncapped all-series WAPE 0.235 and 0.268).
# * **The price response is a little weaker with the clip** (−1.85 against −2.03), still
#   stronger than 05's −1.77.
#
# *Exact numbers vary with seed and hardware.*

# %%
# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ### The choice: clean + floor + clip
#
# The rule's tie-break picks clean + floor, by 0.005 of validation median RMSSE, inside the
# noise floor: on accuracy the two are the same model. We carry **clean + floor + clip**
# forward instead, and say so here. The reason is not a score: it is that the calculations
# training relies on should be stable. With the clip, no input can be scaled beyond ±50, so
# no window can push the exogenous stack, its gradients or its forecast to extremes (the worst
# forecast is 300× the item's largest week, against 2 × 10⁸). Its training was also the
# steadiest of the cleaned arms. The cost is a slightly weaker price response (−1.85 against
# −2.03, still stronger than 05's −1.77). `docs/stability_choices.md` sets out each choice.
#
# The cap is part of the carried-forward model: its saved forecasts, which 06–10 read, and
# its board row are capped at 3× the item's largest training week, as 08 and 11 cap every
# item. The other 05c arms are diagnostics and stay uncapped on the board.
#
# The first seed is refitted once more and saved as `05c_selected`, for 08, 09 and 11, which
# need the model itself. The saved model notes its floor and clip, and `gd.load_model`
# reapplies them.

# %%
CHOSEN = "clean_floor_clip"
saved = gd.fit_seed(floored_model(data_clean, alias_clip, FLOOR, CLIP), gd.SEEDS[0],
                    everywhere(data_clean, test), "05c_selected", save_as="05c_selected")
check = gd.only(saved["forecast"], eval_ids).merge(
    gd.only(clip_test[0]["forecast"], eval_ids), on=["unique_id", "ds"], suffixes=("", "_run"))
print("saved model against the seed-1 run, median RMSSE on the sample:",
      gd.per_series(gd.score(check[["unique_id", "ds", alias_clip, alias_clip + "_run"]], test,
                             train)).median().round(4).to_dict())

# %%
# The carried-forward model includes the cap (section 11): its saved forecasts, which 06-10
# read, and its board row are capped at CAP x the item's largest training week. The other
# 05c arms are diagnostics and stay uncapped.
sample_clip = seed_mean(clip_test, eval_ids)
sample_clip[alias_clip] = sample_clip[alias_clip].clip(
    upper=sample_clip["unique_id"].map(top) * CAP)
gd.save_forecasts(sample_clip, "05c_clean_floor_clip")
gd.record(sample_clip, test, train,
          lab=f"05c clean + floor + clip, capped ({len(gd.SEEDS)}-seed mean)",
          fit_seconds=gd.summaries(clip_test)["fit_seconds"].mean(),
          notes="fit_seconds is per seed")
gd.save_config("05c_input_clip", {
    **cfg_clean,
    "scale_floor": FLOOR,
    "input_clip": CLIP,
    "seeds": gd.SEEDS,
    "recommended": recommended3,
    "use": CHOSEN,
    "chosen_by": "the user, over the rule's tie-break (0.005 apart, inside the noise floor)",
    "choice_reason": "stable training calculations: no scaled input beyond ±50, worst "
                     "forecast 300x not 2e8x; see docs/stability_choices.md",
    "saved_model": "05c_selected",
    "cap_kept": True,
    "rule": list(next(iter(checks3.values()))),
    "checks": {arm: {k: bool(v) for k, v in c.items()} for arm, c in checks3.items()},
    "valid": valid3.round(4).to_dict("index"),
    "explosions": explosions3.to_dict("index"),
    "test_scores": {f"{a} | {b}": v for (a, b), v in test_scores3.round(4).to_dict().items()},
    "test_response": test_response3.round(4).to_dict("index"),
    "runs": {"valid": "05c_valid_clean_floor_clip", "test": "05c_clean_floor_clip"},
    "forecasts": "05c_clean_floor_clip",
    "forecast_column": alias_clip,
})
gd.median_board("05c")

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
# Part 3 clipped the scaled price inputs at ±50, which bounds the large real price moves the
# floor leaves alone. It removes the input-driven explosions; what is left is a rare blow-up
# inside the network in one seed, with ordinary inputs. That is why the cap stays: it is the
# guard for a failure no input treatment can prevent.
#
# **The rule's pick is clean prices + floor; clean + floor + clip ties it, and is the one we
# carry forward**, with the cap, for stable training calculations (section 11). From here on
# `gd.selected_config()` returns it, and every model in the final comparison uses the cleaned
# prices (`gp.PRICE_CLEAN`): LightGBM (00, 04), the GFMs (06, 08, 09, 11) and the foundation
# models (07, 07b, 11). The GFM development notebooks 01–05 keep the raw prices they were
# run with.
