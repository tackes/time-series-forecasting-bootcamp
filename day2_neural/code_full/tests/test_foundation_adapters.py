"""Tests for the Timestral / t0-gamma adapter in gd_helpers, with stub models (no download).

    GD_SMOKE=1 .venv/bin/python -m pytest day2_neural/code_full/tests -q
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

os.environ.setdefault("GD_SMOKE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gd_helpers as gd  # noqa: E402

H = 3


def _frames():
    """Two series; ``b`` starts two weeks late. One covariate, known ahead."""
    weeks = pd.date_range("2025-01-01", periods=6, freq=gd.bc.FREQ)
    hist = pd.DataFrame({"unique_id": ["a"] * 6 + ["b"] * 4,
                         "ds": list(weeks) + list(weeks[2:]),
                         "y": [1.0, 2, 3, 4, 5, 6] + [10.0, 20, 30, 40],
                         "x": [0.0, 1, 0, 1, 0, 1] + [5.0, 5, 5, 5]})
    future_weeks = pd.date_range(weeks[-1] + pd.Timedelta(weeks=1), periods=H, freq=gd.bc.FREQ)
    future = pd.DataFrame({"unique_id": ["a"] * H + ["b"] * H, "ds": list(future_weeks) * 2,
                           "x": [9.0] * H + [7.0] * H})
    return hist, future, future_weeks


class _StubTimestral:
    """Forecasts each series' last value; quantile knots like the real model."""
    def forecast(self, context, prediction_length, known_covariates=None):
        self.known = known_covariates
        last = torch.tensor(context[:, -1:]).repeat(1, prediction_length)        # (B, H)
        levels = (0.05, 0.1, 0.5, 0.9, 0.95)
        q = torch.stack([last * (1 + k - 0.5) for k in levels], dim=-1)[:, None]  # (B,1,H,Q)
        return SimpleNamespace(quantiles=q, quantile_levels=levels)


class _StubT0:
    def predict(self, context, h, quantile_levels, future_covariates=None, **kwargs):
        self.kwargs = kwargs
        last = torch.tensor(context[:, -1:]).repeat(1, h)
        q = torch.stack([last - 1, last, last + 1], dim=-1)                        # (B, H, 3)
        return SimpleNamespace(quantiles=q)


def test_panel_arrays_pads_late_starts_and_fills_covariates():
    hist, future, future_weeks = _frames()
    ids, context, known, weeks = gd.panel_arrays(hist, future, ["x"], H)
    assert ids == ["a", "b"] and context.shape == (2, 6) and known.shape == (2, 1, 6 + H)
    assert np.isnan(context[1, :2]).all() and context[1, 2] == 10.0
    np.testing.assert_allclose(known[1, 0, :2], 5.0)            # filled across the late start
    np.testing.assert_allclose(known[0, 0, 6:], 9.0)            # the future values
    assert list(weeks) == list(future_weeks)


def test_foundation_forecast_long_frame_for_both_models():
    hist, future, future_weeks = _frames()
    stub = _StubTimestral()
    out = gd.foundation_forecast("timestral", stub, hist, future, alias="TS", h=H,
                                 covariates=["x"])
    assert list(out.columns) == ["unique_id", "ds", "TS", "TS-lo-80", "TS-hi-80"]
    assert len(out) == 2 * H and set(out["ds"]) == set(future_weeks)
    np.testing.assert_allclose(out[out["unique_id"] == "b"]["TS"], 40.0)
    assert (out["TS-lo-80"] <= out["TS"]).all() and (out["TS"] <= out["TS-hi-80"]).all()
    assert stub.known.shape == (2, 1, 6 + H)

    t0 = _StubT0()
    out = gd.foundation_forecast("t0", t0, hist, None, alias="T0", h=H)
    np.testing.assert_allclose(out[out["unique_id"] == "a"]["T0"], 6.0)
    assert t0.kwargs["symmetric_averaging"] and t0.kwargs["clip_nonnegative"]
