"""Tests for gd_price.clean_prices. Run from the repo root:

    GD_SMOKE=1 .venv/bin/python -m pytest day2_neural/code_full/tests -q
"""

import os
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

os.environ.setdefault("GD_SMOKE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gd_helpers as gd  # noqa: E402
import gd_price as gp  # noqa: E402

bc = gd.bc
CUTOFF = date(2025, 4, 16)


def _series(uid: str, prices: list[float]) -> pl.DataFrame:
    """One series of weekly prices ending just after ``CUTOFF``, features built by bootcamp."""
    n = len(prices)
    weeks = [CUTOFF + timedelta(weeks=i - n + 2) for i in range(n)]
    df = pl.DataFrame({"unique_id": [uid] * n, "ds": weeks, "y": [10.0] * n,
                       "sale_price": prices, "regular_price": prices})
    return bc._add_price_features(df)


def _clean(df: pl.DataFrame) -> pl.DataFrame:
    return gp.clean_prices(df, cutoff=CUTOFF).sort("ds")


def test_a_penny_placeholder_is_replaced_and_discount_depth_rebuilt():
    prices = [2.0] * 10 + [0.01] + [2.0] * 5
    raw = _series("a", prices)
    assert raw["discount_depth"].max() > 0.99                        # a fake 99% discount
    out = _clean(raw)
    assert out["sale_price"][10] == pytest.approx(2.0)
    assert out["discount_depth"].max() == pytest.approx(0.0)
    np.testing.assert_allclose(out["price_index"], 1.0)
    assert len(gp.bad_price_weeks(raw, cutoff=CUTOFF)) == 1


def test_a_series_whose_usual_price_is_the_placeholder():
    prices = [0.01] * 8 + [20.0, 25.0, 0.01, 30.0, 0.01, 25.0]      # like 1120_S1
    out = _clean(_series("b", prices))
    assert out["sale_price"].min() >= 20.0
    assert out["price_index"].max() <= 6 and out["price_index"].min() >= 1 / 6


def test_real_cuts_and_cheap_items_are_left_alone():
    prices = [0.60] * 6 + [0.30] + [0.60] * 4 + [0.12] + [0.60] * 3   # 50% and 80% cuts
    raw = _series("c", prices)
    out = _clean(raw)
    np.testing.assert_allclose(out["sale_price"], raw.sort("ds")["sale_price"])
    assert gp.bad_price_weeks(raw, cutoff=CUTOFF).is_empty()


def test_a_unit_error_is_replaced_from_its_neighbours():
    prices = [3.0] * 6 + [3.3, 300.0, 3.3] + [3.0] * 5                 # 100x: wrong units
    out = _clean(_series("d", prices))
    assert out["sale_price"][7] == pytest.approx(3.0)
    assert out["price_index"].max() < 1.2


def test_no_good_price_before_the_cutoff_falls_back_to_later_weeks():
    prices = [0.01] * 12 + [20.0, 25.0]                              # real prices start late
    out = _clean(_series("e", prices))
    assert out["sale_price"].min() >= 20.0
    assert out["price_index"].max() <= 6 and out["price_index"].min() >= 1 / 6
