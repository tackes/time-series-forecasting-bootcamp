"""Information-availability checks for the notebooks.

Each test reruns a notebook's modeling cells at the final cutoff after scrambling something in the weeks after
the cutoff, and checks every forecast against the information its own `information_tier` says it uses:

- "sales and realized price": sales, and the realized selling price outside promotion weeks. Only leakage
  diagnostics may move, and they must.
- "plan after 4 weeks": promotions, regular prices and promotional prices from the fifth forecast week on.
  Forecasts without future drivers and the 4-week partial plan must not move; full future-driver forecasts must.
- "everything": sales, promotions and all prices. Forecasts without future drivers must not move; forecasts
  that read the plan must.
- notebook 07: sales after a decision date are scrambled. The selection made on that date must not move.

Run from the repo root: python -m pytest shared/test_leakage.py (slow, several minutes)
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SAVED = ROOT / "results" / "day1" / "forecasts"
HOOK = "panel = add_drivers(panel)"
SKIP = ("google.colab", "plt.", "save_forecasts(", "leaderboard(", "breakdown(", "origin_wape", "score(")
STATIC = ("history", "historical_drivers")
PLANNED = ("partial_future_drivers", "future_drivers")


@pytest.fixture(autouse=True, scope="module")
def saved_forecasts_unchanged():
    """The reruns below skip every cell that calls save_forecasts; fail if a saved file changed anyway."""
    before = {f.name: f.stat().st_mtime_ns for f in SAVED.glob("*.parquet")}
    yield
    assert {f.name: f.stat().st_mtime_ns for f in SAVED.glob("*.parquet")} == before


def notebook_source(name, stop, skip=SKIP):
    """Code cells up to the one containing `stop`, without plotting, saving and scoring cells."""
    cells = json.loads((ROOT / "day1_ml" / name).read_text(encoding="utf8"))["cells"]
    code = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]
    code = code[: next(i for i, c in enumerate(code) if stop in c) + 1]
    return "\n".join(c for c in code if stop in c or not any(marker in c for marker in skip))


def run(source):
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    namespace = {"__name__": "notebook", "display": lambda *args, **kwargs: None}
    exec(source, namespace)
    return namespace


def scramble(kind):
    """Code that scrambles the raw panel before add_drivers."""
    after = {"sales and realized price": 0, "plan after 4 weeks": 4, "everything": 0}[kind]
    return f'''
rng = np.random.default_rng(1)
late = panel.ds > TEST_CUTOFF + pd.Timedelta(weeks={after})
kind = "{kind}"
if kind in ("sales and realized price", "everything"):
    panel.loc[late, "y"] = rng.integers(0, 5000, late.sum()).astype(float)
if kind == "sales and realized price":
    rows = late & ~panel.is_promotional
    panel.loc[rows, "sale_price"] = panel.loc[rows, "regular_price"] * rng.uniform(0.5, 1.5, rows.sum())
if kind in ("plan after 4 weeks", "everything"):
    panel.loc[late, "is_promotional"] = rng.random(late.sum()) < 0.3
    panel.loc[late, "regular_price"] = panel.loc[late, "regular_price"] * rng.uniform(0.8, 1.25, late.sum())
    panel.loc[late, "sale_price"] = panel.loc[late, "regular_price"] * np.where(panel.loc[late, "is_promotional"], rng.uniform(0.5, 0.95, late.sum()), 1.0)
{HOOK}
'''


def forecasts(name, stop, results, kind=None, extra=""):
    source = notebook_source(name, stop)
    assert HOOK in source, name
    source = source.replace("for cutoff in ALL_CUTOFFS:", "for cutoff in [TEST_CUTOFF]:")
    source = source.replace(HOOK, (scramble(kind) if kind else HOOK) + extra, 1)
    namespace = run(source)
    out = pd.concat([namespace[r] for r in results])
    return out.set_index(["unique_id", "ds", "model"]).yhat, out.drop_duplicates("model").set_index("model").information_tier


def check(name, stop, results, expectations, extra=""):
    """expectations: scramble kind -> {tier: True if forecasts must move, False if they must not, absent: either}."""
    normal, tiers = forecasts(name, stop, results, extra=extra)
    for kind, expected in expectations.items():
        changed, _ = forecasts(name, stop, results, kind, extra=extra)
        for model, tier in tiers.items():
            if tier not in expected:
                continue
            a, b = normal.xs(model, level="model"), changed.xs(model, level="model")
            moved = not np.allclose(a, b.reindex(a.index), rtol=1e-4)
            assert moved == expected[tier], (kind, model, tier)


EVERY_TIER = {
    "sales and realized price": {"history": False, "historical_drivers": False, "partial_future_drivers": False, "future_drivers": False, "leakage_diagnostic": True},
    "plan after 4 weeks": {"history": False, "historical_drivers": False, "partial_future_drivers": False, "future_drivers": True},
    "everything": {"history": False, "historical_drivers": False, "partial_future_drivers": True, "future_drivers": True},
}
WITHOUT_PLAN = {"everything": {"history": False, "historical_drivers": False}}


def test_exogenous_notebook_uses_only_what_each_tier_allows():
    # best_future is chosen on the development scores, which this rerun skips; use the largest feature set
    check("03_exogenous_features.ipynb", "partial_forecasts = pd.concat(partial_forecasts)",
          ["forecasts", "leak_forecasts", "partial_forecasts"], EVERY_TIER,
          extra='\nbest_future = "lgbm_future_price_levels"\n')


def test_catboost_models_use_no_future_information():
    check("03_exogenous_features.ipynb", "boost_forecasts = pd.concat(boost_forecasts)", ["boost_forecasts"], WITHOUT_PLAN,
          extra='\nbest_future = "lgbm_future_price_levels"\n')


def test_spike_models_read_the_plan_and_nothing_else():
    check("03b_promotion_spikes.ipynb", "forecasts = pd.concat(forecasts)", ["forecasts"],
          {"sales and realized price": {"future_drivers": False}, "everything": {"future_drivers": True}})


def test_l1_refits_use_only_what_each_tier_allows():
    check("06_objectives.ipynb", "l1_feature_forecasts = pd.concat(l1_feature_forecasts)", ["l1_feature_forecasts"],
          {"sales and realized price": {"history": False, "historical_drivers": False, "future_drivers": False},
           "everything": {"history": False, "historical_drivers": False, "future_drivers": True}})


def test_behavior_features_use_no_future_information():
    check("03c_behavior_features.ipynb", "behavior_forecasts = pd.concat(behavior_forecasts)", ["behavior_forecasts"], WITHOUT_PLAN)


def selection_forecasts(scramble_after=None):
    source = notebook_source("07_model_selection.ipynb", "grid_runs = ", skip=("google.colab", "plt.", "save_forecasts("))
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
