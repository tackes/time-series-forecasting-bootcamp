"""Tests for the phase 7b price probes in gd_price. Run from the repo root:

    GD_SMOKE=1 .venv/bin/python -m pytest day2_neural/code_full/tests -q
"""

import os
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

os.environ.setdefault("GD_SMOKE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gd_helpers as gd  # noqa: E402
import gd_price as gp  # noqa: E402

bc = gd.bc
H = 14
START = date(2025, 4, 16)                 # first test week (W-WED), as in the panel


def _panel(n_hist: int = 20, n_series: int = 2, seed: int = 0) -> pl.DataFrame:
    """Small panel with varying prices, price features built by bootcamp's own function."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_series):
        for i in range(n_hist + H):
            price = 10.0 * (1 - 0.2 * (rng.random() < 0.2))
            rows.append({"unique_id": f"s{s}", "ds": START + timedelta(weeks=i - n_hist),
                         "y": 50.0, "sale_price": price, "promo": 0.0})
    df = bc._add_price_features(pl.DataFrame(rows))
    return df.with_columns((pl.col("sale_price") / 10.0).alias("price_index"))


def _split(df: pl.DataFrame):
    cols = ["unique_id", "ds", "y", "promo", "discount_depth", "price_index",
            "price_ratio_to_rolling_mean"]
    pdf = df.select(cols).to_pandas()
    pdf["ds"] = pd.to_datetime(pdf["ds"])
    hist = pdf[pdf["ds"] < pd.Timestamp(START)]
    futr = pdf[pdf["ds"] >= pd.Timestamp(START)].drop(columns="y")
    return hist, futr


def _constant_elasticity(e: float = -2.0):
    """A model whose forecast is 100 × price_index^e in each week, with no memory."""
    def predict(futr: pd.DataFrame) -> pd.DataFrame:
        return futr[["unique_id", "ds"]].assign(yhat=100 * futr["price_index"] ** e)
    return predict


def test_rebuild_matches_bootcamp_features():
    hist, futr = _split(_panel())
    assert gp.rebuild_check(hist, futr) < 1e-9


def test_pulse_moves_one_week_and_rebuilds_depth():
    hist, futr = _split(_panel(seed=3))
    tl = gp.price_timeline(hist, futr)
    flat = tl.assign(price_index=1.0)                 # flat price: the depth is easy to check
    plan = pd.DataFrame({"unique_id": ["s0"], "step": [4], "shock": [-0.2]})
    out = gp.shocked_futr(flat, plan, rebuild=True)
    s0 = out[out["unique_id"] == "s0"].reset_index(drop=True)
    assert np.isclose(s0.loc[4, "price_index"], 0.8)
    assert np.isclose(s0.drop(index=4)["price_index"], 1.0).all()
    assert np.isclose(s0.loc[4, "discount_depth"], 1 - 0.8 / 0.95)     # 4-week mean 0.95
    assert np.isclose(s0.loc[5, "price_ratio_to_rolling_mean"], 1 / 0.95)
    plain = gp.shocked_futr(flat, plan, rebuild=False)
    assert np.isclose(plain[plain["unique_id"] == "s0"]["discount_depth"].iloc[4], 0.0)


def test_promo_variant_sets_the_flag_on_cuts_only():
    hist, futr = _split(_panel())
    tl = gp.price_timeline(hist, futr)
    plan = pd.DataFrame({"unique_id": ["s0", "s1"], "step": [4, 4], "shock": [-0.2, 0.2]})
    out = gp.shocked_futr(tl, plan, rebuild=True, promo=True)
    week = out[out["step"] == 4].set_index("unique_id")["promo"]
    assert week["s0"] == 1.0 and week["s1"] == 0.0


def test_price_probes_recover_a_constant_elasticity():
    hist, futr = _split(_panel())
    raw = gp.price_probes(_constant_elasticity(-2.0), hist, futr, ["s0", "s1"])
    report = gp.probe_report(raw).set_index(["mode", "shock"])
    for mode in ("sustained_price", "pulse_price", "pulse_all"):
        q1 = 0.8 ** -2.0
        expected = gp.arc_one_side(1.0, q1, -0.2)
        assert np.isclose(report.loc[(mode, -0.2), "median_elasticity"], expected)
        assert np.isclose(report.loc[(mode, -0.2), "median_lift"], q1 - 1)
    assert np.isclose(report.loc[("pulse_all", -0.2), "median_spill"], 0.0)
    assert ("pulse_all_promo", 0.2) not in report.index      # a rise is not a promotion
    assert report["share_right_sign"].eq(1.0).all()
    assert gp.headline(report.reset_index())["median_lift"] == pytest.approx(0.8 ** -2 - 1)


def test_promo_calendars_respect_gap_events_and_planned_promotions():
    hist, futr = _split(_panel(n_series=4))
    futr.loc[(futr["unique_id"] == "s1") & (futr["ds"] == futr["ds"].max()), "promo"] = 1.0
    cal = gp.promo_calendars(futr, ["s0", "s1", "s2", "s3"], draws=3)
    again = gp.promo_calendars(futr, ["s3", "s2", "s1", "s0"], draws=3)
    key = ["draw", "unique_id", "step"]
    pd.testing.assert_frame_equal(cal.sort_values(key).reset_index(drop=True),
                                  again.sort_values(key).reset_index(drop=True))
    steps = futr.sort_values(["unique_id", "ds"]).groupby("unique_id").cumcount()
    week_of = dict(zip(steps, pd.to_datetime(futr.sort_values(["unique_id", "ds"])["ds"])))
    events = gp.event_weeks(futr["ds"])
    assert not any(week_of[s] in events for s in cal["step"])         # weeks 1, 2 and 4
    assert not ((cal["unique_id"] == "s1") & (cal["step"] == H - 1)).any()
    for _, g in cal.groupby(["draw", "unique_id"]):
        assert len(g) <= gp.CAL_CUTS
        assert (np.diff(sorted(g["step"])) >= gp.CAL_GAP).all()
    assert set((-cal["shock"]).round(2)) <= set(gp.CAL_DEPTHS)


def test_calendar_probe_reads_each_cut_and_the_depth_seen():
    hist, futr = _split(_panel(n_series=3, seed=5))
    ids = ["s0", "s1", "s2"]
    cal = gp.promo_calendars(futr, ids, draws=2)
    raw = gp.calendar_probe(_constant_elasticity(-2.0), hist, futr, cal)
    assert len(raw) == len(cal)
    resp = gp.probe_response(raw)
    assert np.allclose(resp["lift"], (1 + resp["shock"]) ** -2.0 - 1)
    assert (raw["seen_depth"] < -raw["shock"] + 1e-9).all()          # rolling mean dilutes it
    report = gp.calendar_report(raw)
    assert set(report["depth"]) <= set(gp.CAL_DEPTHS)


def test_seed_mean_q_averages_forecasts_not_elasticities():
    a = pd.DataFrame({"unique_id": ["x"], "mode": ["pulse_all"], "shock": [-0.2],
                      "q0": [10.0], "q1": [20.0]})
    b = a.assign(q0=30.0, q1=30.0)
    m = gp.seed_mean_q([a, b], gp.PROBE_KEYS)
    assert m.loc[0, "q0"] == 20.0 and m.loc[0, "q1"] == 25.0


def test_implied_lift():
    assert gp.implied_lift(-1.0, 0.5) == pytest.approx(1.0)
    assert np.allclose(gp.implied_lift(0.0, [0.1, 0.3]), 0.0)


def test_seed_mean_q_keeps_the_calendar_shock():
    a = pd.DataFrame({"draw": [0], "unique_id": ["x"], "step": [4], "shock": [-0.2],
                      "q0": [10.0], "q1": [12.0], "seen_depth": [0.15]})
    m = gp.seed_mean_q([a, a.assign(q1=14.0)], gp.CAL_KEYS)
    assert m.loc[0, "shock"] == -0.2 and m.loc[0, "q1"] == 13.0
    assert len(gp.calendar_report(m)) == 1
