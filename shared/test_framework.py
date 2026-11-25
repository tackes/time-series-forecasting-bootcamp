"""Checks for the shared scoring code. Run from the repo root: python -m pytest shared/test_framework.py"""

from functools import partial

import numpy as np
import pandas as pd
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import nd, rmsse

from shared.evaluation import apc, apc_blocks, score, to_wide
from shared.leaderboard import to_long
from shared.prepare_data import DATA, TEST_CUTOFF, teaching_sample


def weekly(start, n):
    return pd.date_range(start, periods=n, freq="W-WED")


def two_series(n=70):
    """Two series of different volume, so pooled and averaged percentages differ."""
    dates = weekly("2024-01-03", n)
    rng = np.random.default_rng(0)
    big = rng.integers(500, 1500, n).astype(float)
    small = rng.integers(0, 10, n).astype(float)
    return dates, pd.concat([pd.DataFrame({"unique_id": "big", "ds": dates, "y": big}),
                             pd.DataFrame({"unique_id": "small", "ds": dates, "y": small})])


def forecasts(dates, actuals, cutoff_index, shifts):
    """One forecast per model: the actual shifted by a constant, so error and bias are known."""
    rows = []
    for model, shift in shifts.items():
        target = actuals[actuals["ds"].isin(dates[cutoff_index + 1: cutoff_index + 15])]
        rows.append(target.assign(cutoff=dates[cutoff_index], model=model, yhat=target["y"] + shift)[["unique_id", "cutoff", "ds", "model", "yhat"]])
    return pd.concat(rows)


def test_wape_bias_rmsse_by_hand():
    dates = weekly("2024-01-03", 60)
    actuals = pd.DataFrame({"unique_id": "a", "ds": dates, "y": np.arange(60.0) % 7})
    fcst = pd.DataFrame({"unique_id": "a", "cutoff": dates[52], "ds": dates[53:56], "model": "m", "yhat": [3.0, 3.0, 3.0]})
    row = score(fcst, actuals).iloc[0]
    y = actuals.set_index("ds").loc[dates[53:56], "y"].to_numpy()
    assert np.isclose(row["wape"], np.abs(3 - y).sum() / y.sum())
    assert np.isclose(row["accuracy"], 1 - row["wape"])
    assert np.isclose(row["bias"], (3 - y).sum() / y.sum())
    scale = ((actuals["y"] - actuals["y"].shift(52)) ** 2)[:53].mean()
    assert np.isclose(row["mean_rmsse"], np.sqrt(((3 - y) ** 2).mean() / scale))


def test_wape_and_bias_are_portfolio_ratios_not_sku_averages():
    dates, actuals = two_series()
    fcst = forecasts(dates, actuals, 52, {"over": 5.0, "under": -3.0})
    out = score(fcst, actuals).set_index("model")
    y = fcst.merge(actuals, on=["unique_id", "ds"]).groupby("model").y.sum()
    assert np.isclose(out.loc["over", "wape"], 5.0 * 28 / y["over"])
    assert np.isclose(out.loc["over", "bias"], 5.0 * 28 / y["over"])
    assert np.isclose(out.loc["under", "bias"], -3.0 * 28 / y["under"])

    per_sku = evaluate(to_wide(fcst.merge(actuals, on=["unique_id", "ds"])), [nd], models=["over"])
    assert not np.isclose(per_sku["over"].mean(), out.loc["over", "wape"], rtol=0.05)   # averaging SKU WAPE is a different number


def test_grouped_scores_match_utilsforecast_on_the_group():
    dates, actuals = two_series()
    fcst = forecasts(dates, actuals, 52, {"a": 2.0, "b": -1.0})
    fcst["horizon"] = (fcst.ds - fcst.cutoff).dt.days // 7
    out = score(fcst, actuals, by=["horizon"], rmsse=False).query("model == 'a' and horizon == 3").iloc[0]
    wide = to_wide(fcst[fcst.horizon == 3].merge(actuals, on=["unique_id", "ds"])).assign(group="all")
    assert np.isclose(out["wape"], evaluate(wide, [nd], models=["a"], id_col="group").a.iloc[0])


def test_rmsse_is_utilsforecast_rmsse_with_undefined_scales_left_out():
    dates, actuals = two_series(n=70)
    short = pd.DataFrame({"unique_id": "short", "ds": dates[-30:], "y": np.arange(30.0)})
    flat = pd.DataFrame({"unique_id": "flat", "ds": dates, "y": 4.0})
    actuals = pd.concat([actuals, short, flat])
    fcst = forecasts(dates, actuals, 52, {"m": 1.0})
    out = score(fcst, actuals).iloc[0]

    train = actuals[actuals.ds <= dates[52]].sort_values(["unique_id", "ds"])
    nixtla = evaluate(to_wide(fcst.merge(actuals, on=["unique_id", "ds"])), [partial(rmsse, seasonality=52)], models=["m"], train_df=train).set_index("unique_id")["m"]
    # utilsforecast gives no usable score when the scale is undefined: 0 up to 0.2.14, NaN or inf from 0.2.15
    assert all(nixtla[s] == 0 or not np.isfinite(nixtla[s]) for s in ["short", "flat"])
    assert out["eligible_series"] == 2                           # here those two are left out, whatever the version
    assert np.isclose(out["mean_rmsse"], nixtla[["big", "small"]].mean())


def test_rmsse_scale_uses_only_history_up_to_the_cutoff():
    dates, actuals = two_series(n=70)
    fcst = forecasts(dates, actuals, 55, {"m": 1.0})
    later = actuals.assign(y=np.where(actuals.ds > dates[55 + 14], 999.0, actuals.y))   # after the forecast window
    assert np.isclose(score(fcst, actuals).mean_rmsse.iloc[0], score(fcst, later).mean_rmsse.iloc[0])


def test_score_uses_rows_every_model_forecasts():
    dates, actuals = two_series()
    fcst = forecasts(dates, actuals, 52, {"a": 2.0, "b": 2.0})
    partial_b = fcst[~((fcst.model == "b") & (fcst.unique_id == "small"))]
    out = score(partial_b, actuals).set_index("model")
    assert (out["series"] == 1).all() and np.isclose(out.loc["a", "wape"], out.loc["b", "wape"])


def test_apc_blocks_and_missing_origin():
    dates = weekly("2024-01-03", 40)
    old_cut, new_cut = dates[10], dates[14]
    fcst = pd.concat([pd.DataFrame({"unique_id": "a", "cutoff": cut, "ds": dates[dates > cut][:14], "model": "m", "yhat": level})
                      for cut, level in [(old_cut, 10.0), (new_cut, 12.0)]])
    assert len(apc_blocks(fcst, [new_cut])) == 2           # 10 overlapping weeks, 2 complete blocks of 4
    out = apc(fcst, [new_cut])
    assert np.isclose(out["apc"].iloc[0], 0.2) and out["apc_coverage"].iloc[0] == 1
    assert apc(fcst[fcst["cutoff"] == new_cut], [new_cut]).empty    # one origin: unavailable, not zero


def test_apc_zero_base_excluded():
    dates = weekly("2024-01-03", 40)
    old_cut, new_cut = dates[10], dates[14]
    old = pd.DataFrame({"unique_id": "a", "cutoff": old_cut, "ds": dates[dates > old_cut][:14], "model": "m", "yhat": 0.0})
    new = old.assign(cutoff=new_cut, ds=dates[dates > new_cut][:14], yhat=5.0)
    out = apc(pd.concat([old, new]), [new_cut])
    assert out["zero_base"].iloc[0] == 2 and pd.isna(out["apc"].iloc[0]) and out["apc_coverage"].iloc[0] == 0


def test_apc_common_support_uses_the_same_blocks_for_every_model():
    dates = weekly("2024-01-03", 40)
    old_cut, new_cut = dates[10], dates[14]
    rows = []
    for model, level in [("a", 10.0), ("b", 0.0)]:
        old = pd.DataFrame({"unique_id": "s", "cutoff": old_cut, "ds": dates[dates > old_cut][:14], "model": model, "yhat": 10.0})
        old.loc[old.index[4:8], "yhat"] = level   # old weeks 5-8 are new weeks 1-4: the first block
        new = old.assign(cutoff=new_cut, ds=dates[dates > new_cut][:14], yhat=12.0)
        rows += [old, new]
    out = apc(pd.concat(rows), [new_cut]).set_index("model")
    assert out.loc["a", "apc_coverage"] == out.loc["b", "apc_coverage"] == 0.5
    assert out.loc["b", "zero_base"] == 1 and out.loc["a", "zero_base"] == 0
    assert np.isclose(out.loc["a", "apc"], 0.2) and np.isclose(out.loc["b", "apc"], 0.2)


def test_apc_single_origin_model_does_not_change_the_common_population():
    dates = weekly("2024-01-03", 40)
    old_cut, new_cut = dates[10], dates[14]
    frames = [pd.DataFrame({"unique_id": "s", "cutoff": cut, "ds": dates[dates > cut][:14], "model": "a", "yhat": level})
              for cut, level in [(old_cut, 10.0), (new_cut, 12.0)]]
    single = pd.DataFrame({"unique_id": "s", "cutoff": new_cut, "ds": dates[dates > new_cut][:14], "model": "c", "yhat": 5.0})
    out = apc(pd.concat(frames + [single]), [new_cut])
    assert list(out.model) == ["a"] and out["apc_coverage"].iloc[0] == 1


def test_existing_baselines_match_shared_format():
    from statsforecast import StatsForecast
    from statsforecast.models import Naive

    old = pd.read_parquet(DATA.parent / "results" / "baseline_test.parquet")
    wide = old[["unique_id", "ds", "Naive", "HistoricAverage", "SeasonalNaive"]]
    long = to_long(wide, TEST_CUTOFF, "01_baselines")
    assert set(long.columns) >= {"unique_id", "cutoff", "ds", "model", "yhat"}
    assert long["ds"].min() == TEST_CUTOFF + pd.Timedelta(weeks=1)

    panel = pd.read_parquet(DATA / "retail_prepared.parquet", columns=["unique_id", "ds", "y"])
    ids = teaching_sample()[:50]
    train = panel[panel["unique_id"].isin(ids) & (panel["ds"] <= TEST_CUTOFF)]
    new = StatsForecast(models=[Naive()], freq="W-WED", n_jobs=1).forecast(df=train, h=14)
    merged = new.merge(old, on=["unique_id", "ds"], suffixes=("_new", "_old"))
    assert len(merged) > 0 and np.allclose(merged["Naive_new"], merged["Naive_old"])


def test_sample_is_deterministic():
    assert teaching_sample() == teaching_sample() and len(teaching_sample()) == 2000
