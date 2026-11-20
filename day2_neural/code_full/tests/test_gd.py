"""Unit tests for the code_full helpers. Run from the repo root:

    GD_SMOKE=1 .venv/Scripts/python -m pytest day2_neural/code_full/tests -q
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
LEI_RESULTS = gd.ROOT / "results" / "day2"


def test_paths_point_away_from_lei_results():
    for path in (bc.RESULTS_DIR, bc.FORECAST_DIR, bc.LEADERBOARD_PATH):
        assert gd.OUT_DIR in Path(path).parents or Path(path) == gd.OUT_DIR
        assert LEI_RESULTS not in Path(path).parents
    assert bc.DATA_DIR == gd.ROOT / "data"
    assert (bc.DATA_DIR / "sales_weekly.parquet").exists()


def test_config_round_trip():
    cfg = {"model": "NBEATSx", "lr": np.float64(4.83e-4), "steps": np.int64(10), "d": date(2025, 1, 1)}
    gd.save_config("test_roundtrip", cfg)
    back = gd.load_config("test_roundtrip")
    assert back == {"model": "NBEATSx", "lr": 4.83e-4, "steps": 10, "d": "2025-01-01"}
    (gd.CONFIG_DIR / "test_roundtrip.json").unlink()


def test_missing_config_says_what_to_do():
    with pytest.raises(FileNotFoundError, match="run the notebook"):
        gd.load_config("does_not_exist")


def test_scaled_max_steps(monkeypatch):
    monkeypatch.setattr(gd, "SMOKE", False)
    assert gd.scaled_max_steps(17_241) == int(174 * 17_241 / 256)
    assert gd.scaled_max_steps(10) == 2_000            # floor
    assert gd.scaled_max_steps(10_000_000) == 250_000  # ceiling


def test_warmup_then_step():
    f = [gd.warmup_then_step(s, warmup_steps=10, decay_every=100) for s in (0, 9, 10, 109, 110, 210)]
    assert f == pytest.approx([0.1, 1.0, 1.0, 1.0, 0.5, 0.25])


def test_recipe_kwargs_on_and_off():
    on, off = gd.recipe_kwargs(1000), gd.recipe_kwargs(1000, recipe=False)
    assert on["gradient_clip_val"] == 1.0 and isinstance(on["callbacks"][0], gd.BestWeights)
    assert "callbacks" not in off and "lr_scheduler" not in off
    assert on["enable_checkpointing"] is False and on["logger"] is False
    assert gd.recipe_kwargs(1000)["callbacks"][0] is not on["callbacks"][0]  # fresh per model


def test_arc_elasticity_recovers_constant_elasticity():
    # q = p^-1 at every shift: arc elasticity is exactly -1 by construction
    rows = [{"unique_id": "a", "shift": s, "units": 1 / (1 + s)}
            for s in (0.0, -0.1, 0.1, -0.2, 0.2)]
    el = gp.arc_elasticity(pd.DataFrame(rows))
    assert el.loc[el["shift"] == 0.2, "e_arc"].item() == pytest.approx(-1.0, abs=1e-9)
    assert (el["e_down"] < 0).all() and (el["e_up"] < 0).all()


def test_categorise():
    cats = gp.categorise(pd.Series([-4.0, -1.5, -0.5, -0.01, 0.3])).astype(str).tolist()
    assert cats == ["Highly elastic", "Elastic", "Inelastic", "Flat", "Perverse"]


def test_pareto_and_knees():
    df = pd.DataFrame({"accuracy": [0.70, 0.72, 0.75, 0.90, 0.80],
                       "elasticity": [-0.05, -0.30, -0.35, -0.45, -0.10]})
    front = gd.pareto_front(df)
    assert list(front.index) == [0, 1, 2, 3]            # row 4 is dominated by row 1
    assert gd.knee_by_gain(front) == 1
    assert gd.knee_by_distance(front) == 1


def test_week_start_and_events():
    assert gp.week_start(date(2024, 3, 29)) == date(2024, 3, 27)   # Good Friday 2024 -> Wed
    assert gp.week_start(date(2024, 3, 27)) == date(2024, 3, 27)
    ev = gp.event_dates(2025)
    assert ev["ev_easter"] == date(2025, 4, 18) and ev["ev_mothers_day"] == date(2025, 5, 11)
    weeks = pl.DataFrame({"ds": [date(2024, 3, 20) + timedelta(weeks=k) for k in range(3)]})
    out = gp.add_events(weeks)
    assert out["ev_easter"].to_list() == [0.0, 1.0, 0.0]


def test_copies_use_training_weeks_only():
    ds = [bc.TEST_START - timedelta(weeks=30 - k) for k in range(44)]   # 30 train + 14 test
    df = pl.DataFrame({"unique_id": "x", "ds": ds, "sale_class": "D",
                       "is_discount": [k < 12 or k >= 30 for k in range(44)]})
    copies = gp.copies_per_series(df)
    assert copies["discount_weeks"].item() == 12                  # test weeks not counted
    assert copies["copies"].item() == min(round(4 * 1.5 * 5), gp.COPY_CAP)
    big = gp.oversample(df, copies)
    assert big["unique_id"].n_unique() == 1 + gp.COPY_CAP
    assert gp.drop_copies(big)["unique_id"].unique().to_list() == ["x"]


def test_empirical_elasticity_recovers_planted_slope():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(40):
        level = rng.uniform(1, 4)
        for k in range(60):
            p = np.exp(rng.normal(0, 0.15))
            ly = level - 0.5 * np.log(p) + 0.02 * k + rng.normal(0, 0.05)
            rows.append({"unique_id": f"s{i}", "ds": bc.TEST_START - timedelta(weeks=61 - k),
                         "sale_class": "D", "y": np.expm1(ly), "price_index": p})
    est = gp.empirical_elasticity(pl.DataFrame(rows), [f"s{i}" for i in range(40)])
    assert est.loc[est["sale_class"] == "all", "elasticity"].item() == pytest.approx(-0.5, abs=0.05)
