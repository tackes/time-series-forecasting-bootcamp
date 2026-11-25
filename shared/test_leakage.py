"""Information-availability checks for the notebooks.

Each test reruns a notebook's modeling cells after changing something the decision date should not see:

- notebooks 03 (including its CatBoost and categorical sections), 03b and 03c: sales, realized prices after the cutoff, and promotions beyond the 4-week plan are scrambled.
  Historical-only and planned_scenario forecasts must not move; oracle and leaky_diagnostic forecasts must.
- notebook 07: sales after a decision date are scrambled. The selection made on that date must not move.

Run from the repo root: python -m pytest shared/test_leakage.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
FORECAST_CELL = "forecasts = pd.concat(forecasts)"
SAVED = ROOT / "results" / "day1" / "forecasts"


@pytest.fixture(autouse=True, scope="module")
def saved_forecasts_unchanged():
    """The reruns below skip every cell that calls save_forecasts; fail if a saved file changed anyway."""
    before = {f.name: f.stat().st_mtime_ns for f in SAVED.glob("*.parquet")}
    yield
    assert {f.name: f.stat().st_mtime_ns for f in SAVED.glob("*.parquet")} == before


def notebook_source(name, stop):
    cells = json.loads((ROOT / "day1_ml" / name).read_text(encoding="utf8"))["cells"]
    code = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]
    code = [c for c in code if "google.colab" not in c and "plt." not in c and "save_forecasts(" not in c]
    return "\n".join(code[: next(i for i, c in enumerate(code) if stop in c) + 1])


def run(source):
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    namespace = {"__name__": "notebook"}
    exec(source, namespace)
    return namespace


def run_exogenous(name, scramble, stop=FORECAST_CELL, result="forecasts", plan_weeks=4):
    source = notebook_source(name, stop).replace("for cutoff in ALL_CUTOFFS:", "for cutoff in [TEST_CUTOFF]:")
    source = source.replace("from shared.prepare_data import load_panel,", "from shared.prepare_data import TEST_CUTOFF, load_panel,")
    if scramble:
        source = source.replace(
            'actuals = panel[["unique_id", "ds", "y"]]',
            "rng = np.random.default_rng(1)\n"
            "late = panel.ds > TEST_CUTOFF\n"
            "panel.loc[late, 'y'] = rng.integers(0, 5000, late.sum())\n"
            "if 'rel_price' in panel:\n"
            "    panel['rel_price'] = panel['rel_price'].astype(float)\n"
            "    panel.loc[late, 'rel_price'] = rng.uniform(0.3, 1.5, late.sum())\n"
            "    panel.loc[late, 'discount'] = (1 - panel.loc[late, 'rel_price']).clip(lower=0)\n"
            f"beyond = panel.ds > TEST_CUTOFF + pd.Timedelta(weeks={plan_weeks})\n"
            "panel.loc[beyond, 'promo'] = rng.integers(0, 2, beyond.sum())\n"
            'actuals = panel[["unique_id", "ds", "y"]]',
        )
    return run(source)[result].set_index(["unique_id", "ds", "model"]).yhat


def check_exogenous(name, **cells):
    normal = run_exogenous(name, scramble=False, **cells)
    scrambled = run_exogenous(name, scramble=True, **cells)
    for model in normal.index.get_level_values("model").unique():
        a, b = normal.xs(model, level="model"), scrambled.xs(model, level="model")
        same = np.allclose(a, b.reindex(a.index), rtol=1e-4)
        assert same != model.endswith(("oracle", "leaky")), model


def test_exogenous_lab_uses_no_future_information():
    check_exogenous("03_exogenous_features.ipynb")


def test_catboost_models_use_no_future_information():
    check_exogenous("03_exogenous_features.ipynb", stop="boost_forecasts = pd.concat(boost_forecasts)", result="boost_forecasts")


def test_categorical_lightgbm_models_use_no_future_information():
    check_exogenous("03_exogenous_features.ipynb", stop="strong_forecasts = pd.concat(strong_forecasts)", result="strong_forecasts")


def test_behavior_features_use_no_future_information():
    # promotions are scrambled in every forecast week: these models may not use a promotion plan
    check_exogenous("03c_behavior_features.ipynb", stop="behavior_forecasts, fit_seconds, last_model = run_experiments(experiments)",
                    result="behavior_forecasts", plan_weeks=0)


def test_spike_lab_uses_no_future_information():
    check_exogenous("03b_promotion_spikes.ipynb")


def selection_forecasts(scramble_after=None):
    source = notebook_source("07_model_selection.ipynb", "grid_runs = ")
    if scramble_after is not None:
        late = f"(panel.ds > pd.Timestamp('{scramble_after}'))"
        source = source.replace(
            'actuals = panel[["unique_id", "ds", "y"]]',
            f"panel.loc[{late}, 'y'] = np.random.default_rng(1).integers(0, 5000, {late}.sum())\n"
            'actuals = panel[["unique_id", "ds", "y"]]',
        )
    namespace = run(source)
    key = ["unique_id", "cutoff", "ds", "model"]
    hier_runs = namespace["hier_runs"]
    dev = pd.concat([namespace["dev_strategies"], hier_runs[hier_runs.cutoff.isin(namespace["decisions"])]])
    final = pd.concat([namespace["strategies"], hier_runs[hier_runs.cutoff.isin(namespace["targets"])]])
    return dev.set_index(key).yhat, final.set_index(key).yhat


def test_selection_uses_only_evidence_known_on_the_decision_date():
    dev, final = selection_forecasts()
    # sales after 2025-01-01 cannot change what was selected on or before 2025-01-01
    dev_scrambled, _ = selection_forecasts("2025-01-01")
    assert np.allclose(dev, dev_scrambled.reindex(dev.index), rtol=1e-4)
    # sales after the final cutoff cannot change any selection
    _, final_scrambled = selection_forecasts("2025-04-09")
    assert np.allclose(final, final_scrambled.reindex(final.index), rtol=1e-4)
