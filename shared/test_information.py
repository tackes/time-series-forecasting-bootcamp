"""What each driver frame may know on the cutoff date. Run from the repo root: python -m pytest shared/test_information.py

- Historical drivers and their X_df do not change when everything after the cutoff is scrambled.
- Future frames have one row per series and forecast week, the expected columns and no missing values.
- Planned drivers never read the realized selling price outside promotion weeks; the leakage diagnostics do.
- A partial plan equals the full plan in its planned weeks and assumes no promotion after them.
"""

import numpy as np
import pandas as pd
import pytest

from shared.features import (HISTORICAL, PROMO_CALENDAR, add_drivers, day2_covariates, future_drivers, historical_drivers,
                             historical_future, in_horizon, partial_plan, realized_drivers)
from shared.prepare_data import DEV_CUTOFFS, H, TEST_CUTOFF, load_panel

CUTOFFS = [DEV_CUTOFFS[0], TEST_CUTOFF]
PLANNED = PROMO_CALENDAR + ["discount", "price_index", "regular_price", "planned_price"]


@pytest.fixture(scope="module")
def raw():
    return load_panel()


@pytest.fixture(scope="module")
def panel(raw):
    return add_drivers(raw)


def scrambled(raw, after, columns):
    """Copy of the raw panel with the given columns randomized in weeks after `after`."""
    out = raw.copy()
    rng = np.random.default_rng(0)
    late = out.ds > after
    if "y" in columns:
        out.loc[late, "y"] = rng.integers(0, 5000, late.sum()).astype(float)
    if "is_promotional" in columns:
        out.loc[late, "is_promotional"] = rng.random(late.sum()) < 0.3
    if "regular_price" in columns:
        out.loc[late, "regular_price"] *= rng.uniform(0.8, 1.25, late.sum())
    if "sale_price" in columns:
        out.loc[late, "sale_price"] = out.loc[late, "regular_price"] * rng.uniform(0.5, 1.5, late.sum())
    return out


def same(a, b, cols):
    a, b = a.sort_values(["unique_id", "ds"]), b.sort_values(["unique_id", "ds"])
    assert (a[["unique_id", "ds"]].to_numpy() == b[["unique_id", "ds"]].to_numpy()).all()
    return np.allclose(a[cols].to_numpy(float), b[cols].to_numpy(float), equal_nan=True)


@pytest.mark.parametrize("cutoff", CUTOFFS)
def test_historical_drivers_use_nothing_after_the_cutoff(raw, panel, cutoff):
    everything = scrambled(raw, cutoff, ["y", "is_promotional", "regular_price", "sale_price"])
    hist, hist_s = historical_drivers(panel), historical_drivers(add_drivers(everything))
    assert same(hist[hist.ds <= cutoff], hist_s[hist_s.ds <= cutoff], HISTORICAL)
    assert same(historical_future(hist, cutoff), historical_future(hist_s, cutoff), HISTORICAL)


@pytest.mark.parametrize("cutoff", CUTOFFS)
def test_future_frames_have_the_right_rows_and_columns(panel, cutoff):
    weeks = pd.date_range(cutoff + pd.Timedelta(weeks=1), periods=H, freq="W-WED")
    series = set(panel.unique_id)
    frames = {
        "future": (in_horizon(future_drivers(panel), cutoff), PLANNED),
        "historical": (historical_future(historical_drivers(panel), cutoff), HISTORICAL),
        "partial": (in_horizon(partial_plan(panel, cutoff, 4), cutoff), PLANNED),
        "realized": (in_horizon(realized_drivers(panel), cutoff), PROMO_CALENDAR + ["realized_discount", "realized_index"]),
    }
    for name, (frame, cols) in frames.items():
        assert len(frame) == len(series) * H, name
        assert set(frame.ds) == set(weeks), name
        assert set(frame.unique_id) == series, name
        assert not frame.duplicated(["unique_id", "ds"]).any(), name
        assert set(cols) <= set(frame.columns), name
        planned_cols = [c for c in cols if c not in HISTORICAL[:-1]]     # lagged columns may be missing for short histories
        assert frame[planned_cols].notna().all().all(), name


def test_planned_drivers_ignore_the_realized_price_outside_promotions(raw, panel):
    out = raw.copy()
    rng = np.random.default_rng(0)
    rows = ~out.is_promotional
    out.loc[rows, "sale_price"] = out.loc[rows, "regular_price"] * rng.uniform(0.5, 1.5, rows.sum())
    other = add_drivers(out)
    assert same(future_drivers(panel), future_drivers(other), PLANNED)
    assert same(partial_plan(panel, TEST_CUTOFF, 4), partial_plan(other, TEST_CUTOFF, 4), PLANNED)
    # the diagnostics are built from the realized price, so they must change
    assert not same(realized_drivers(panel), realized_drivers(other), ["realized_discount", "realized_index"])
    assert not same(day2_covariates(panel, TEST_CUTOFF), day2_covariates(other, TEST_CUTOFF), ["day2_discount_depth", "day2_price_index"])


def test_discount_is_zero_outside_promotions(panel):
    assert (panel.loc[panel.promo == 0, "discount"] == 0).all()
    assert panel.discount.between(0, 0.9).all()
    assert np.allclose(panel.planned_price, panel.regular_price * (1 - panel.discount))


def test_partial_plan(panel):
    full = in_horizon(future_drivers(panel), TEST_CUTOFF)
    part = in_horizon(partial_plan(panel, TEST_CUTOFF, 4), TEST_CUTOFF)
    planned = full.ds <= TEST_CUTOFF + pd.Timedelta(weeks=4)
    assert same(full[planned], part[part.ds <= TEST_CUTOFF + pd.Timedelta(weeks=4)], PLANNED)
    late = part[part.ds > TEST_CUTOFF + pd.Timedelta(weeks=4)]
    assert (late.promo == 0).all() and (late.discount == 0).all()
    last_regular = panel[panel.ds <= TEST_CUTOFF].groupby("unique_id").regular_price.last()
    assert np.allclose(late.regular_price, late.unique_id.map(last_regular))
