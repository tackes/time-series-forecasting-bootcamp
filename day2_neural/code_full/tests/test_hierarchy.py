"""Tests for the hierarchy helpers used by 08. Run from the repo root:

    GD_SMOKE=1 .venv/bin/python -m pytest day2_neural/code_full/tests -q
"""

import os
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

os.environ.setdefault("GD_SMOKE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gd_helpers as gd  # noqa: E402
import gd_price as gp  # noqa: E402

LEVELS = [["total"], ["total", "department"]]


def _items() -> pl.DataFrame:
    """Three items in two departments over four weeks; item c never sells before the cutoff."""
    weeks = [date(2025, 1, 1) + timedelta(weeks=k) for k in range(4)]
    spec = {"a": ("D1", [10, 10, 10, 10], [1, 0, 1, 0], [0.2, 0.0, 0.1, 0.0]),
            "b": ("D1", [30, 30, 30, 30], [0, 0, 1, 1], [0.0, 0.0, 0.3, 0.3]),
            "c": ("D2", [0, 0, 5, 5], [1, 1, 1, 1], [0.5, 0.5, 0.5, 0.5])}
    rows = []
    for uid, (dept, ys, promos, depths) in spec.items():
        for k, ds in enumerate(weeks):
            rows.append({"total": "Total", "department": dept, "unique_id": uid, "ds": ds,
                         "y": float(ys[k]), "promo": float(promos[k]),
                         "discount_depth": depths[k], "price_index": 1.0 - depths[k]})
    return pl.DataFrame(rows)


def test_aggregate_covariates_are_sales_weighted():
    cutoff = date(2025, 1, 15)                 # weights from the first two weeks
    out = gp.aggregate_covariates(_items(), LEVELS, cutoff=cutoff)
    d1 = out[out["unique_id"] == "Total/D1"].sort_values("ds")
    # weights 10 and 30: week 0 promo (1*10 + 0*30) / 40, depth (0.2*10) / 40
    np.testing.assert_allclose(d1["promo"].iloc[0], 0.25, atol=1e-6)
    np.testing.assert_allclose(d1["discount_depth"].iloc[2], (0.1 * 10 + 0.3 * 30) / 40,
                               atol=1e-6)
    # an item with no sales before the cutoff carries (almost) no weight in the total
    total = out[out["unique_id"] == "Total"].sort_values("ds")
    np.testing.assert_allclose(total["promo"].iloc[0], 0.25, atol=1e-6)
    # a node whose items never sold falls back to their plain mean
    d2 = out[out["unique_id"] == "Total/D2"]
    np.testing.assert_allclose(d2["discount_depth"], 0.5)
    assert set(gp.EVENT_COLS) <= set(out.columns)
    assert set(out["unique_id"]) == {"Total", "Total/D1", "Total/D2"}


def test_split_by_share_keeps_parent_totals():
    ds = pd.Timestamp("2025-04-16")
    parent = pd.DataFrame({"unique_id": ["P", "Q"], "ds": [ds, ds], "m": [100.0, 9.0]})
    child = pd.DataFrame({"unique_id": ["a", "b", "c", "d", "e"], "ds": [ds] * 5,
                          "Base": [1.0, 3.0, 0.0, 0.0, 0.0]})
    parent_of = pd.Series({"a": "P", "b": "P", "c": "Q", "d": "Q", "e": "Q"})
    out = gd.split_by_share(parent, child, parent_of, "m").set_index("unique_id")["m"]
    np.testing.assert_allclose(out[["a", "b"]], [25.0, 75.0])
    np.testing.assert_allclose(out[["c", "d", "e"]], [3.0, 3.0, 3.0])    # equal split


def test_rolling_one_step_never_sees_the_origin():
    weeks = pd.date_range("2025-01-01", periods=30, freq="7D")
    frame = pd.DataFrame({"unique_id": "a", "ds": weeks, "y": np.arange(30.0), "x": 1.0})
    origins = weeks[10:13]

    def predict(hist, futr):
        assert hist["ds"].max() < futr["ds"].min()
        assert "y" not in futr.columns and len(futr) == gd.bc.H
        last = hist.groupby("unique_id")["y"].last()
        return futr.assign(yhat=futr["unique_id"].map(last))[["unique_id", "ds", "yhat"]]

    out = gd.rolling_one_step(predict, frame, origins)
    assert list(out["ds"]) == list(origins)
    np.testing.assert_allclose(out["yhat"], [9.0, 10.0, 11.0])     # the week before each origin


def test_floor_residual_sd_only_touches_flat_nodes():
    ds = pd.date_range("2025-01-01", periods=4, freq="7D")
    frame = pd.DataFrame({"unique_id": ["a"] * 4 + ["b"] * 4, "ds": list(ds) * 2,
                          "y": [0.0] * 4 + [1.0, 2.0, 3.0, 4.0], "Base": [0.0] * 4 + [1.0] * 4})
    out = gd.floor_residual_sd(frame)
    a = out[out["unique_id"] == "a"]
    assert (a["y"] - a["Base"]).std() > 0
    np.testing.assert_allclose(out[out["unique_id"] == "b"]["Base"], 1.0)


def test_mintrace_direct_matches_an_exact_wls_solve():
    """Item variances spread over nine orders of magnitude, as in the panel."""
    from scipy import sparse
    rng = np.random.default_rng(0)
    n_items, n_dept, weeks = 400, 8, 52
    dept = rng.integers(0, n_dept, n_items)
    S = sparse.vstack([sparse.csr_matrix(np.ones((1, n_items))),
                       sparse.csr_matrix((np.ones(n_items), (dept, np.arange(n_items))),
                                         shape=(n_dept, n_items)),
                       sparse.identity(n_items, format="csr")]).tocsr()
    sd = np.exp(rng.uniform(np.log(0.1), np.log(1e4), S.shape[0]))
    resid = rng.normal(size=(S.shape[0], weeks)) * sd[:, None]
    y_insample = rng.uniform(0, 10, (S.shape[0], weeks))
    y_hat = np.abs(rng.normal(size=(S.shape[0], 3))) * 100
    out = gd.MinTraceDirect(method="wls_var").fit_predict(
        S=S, y_hat=y_hat, y_insample=y_insample, y_hat_insample=y_insample - resid)["mean"]
    root_w = 1 / np.sqrt(np.var(resid, axis=1, ddof=1))[:, None]
    dense = S.toarray()
    exact = dense @ np.linalg.lstsq(root_w * dense, root_w * y_hat, rcond=None)[0]
    np.testing.assert_allclose(out, exact, rtol=0, atol=1e-5 * np.abs(exact).max())
    np.testing.assert_allclose(out[0], out[1 + n_dept:].sum(axis=0))      # coherent
