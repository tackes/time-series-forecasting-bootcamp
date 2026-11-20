"""Information tiers and evaluation windows. Run from the repo root: python -m pytest shared/test_tiers.py"""

import json
from pathlib import Path

import pandas as pd

from shared.leaderboard import TIERS, combine_tiers, load_forecasts
from shared.prepare_data import DEV_CUTOFFS, STABILITY_CUTOFFS, TEST_CUTOFF, completed_runs

ROOT = Path(__file__).resolve().parents[1]


def test_strategy_takes_the_highest_tier_of_its_candidates():
    assert combine_tiers(["historical_only", "historical_only"]) == "historical_only"
    assert combine_tiers(["historical_only", "planned_scenario"]) == "planned_scenario"
    assert combine_tiers(["planned_scenario", "oracle", "historical_only"]) == "oracle"
    assert combine_tiers(["oracle", "leaky_diagnostic"]) == "leaky_diagnostic"
    assert TIERS == ["historical_only", "planned_scenario", "oracle", "leaky_diagnostic"]


def test_saved_forecasts_carry_the_right_tier():
    fcst = load_forecasts()
    tiers = fcst.drop_duplicates("model").set_index("model").information_tier
    assert set(tiers) <= set(TIERS)
    for model, tier in tiers.items():
        if model.endswith("_oracle"):
            assert tier == "oracle", model
        elif model.endswith("_planned") or model.startswith(("Select_planned", "Average_planned")):
            assert tier == "planned_scenario", model
        elif model.endswith("_leaky") or model.startswith("Best_in_hindsight"):
            assert tier == "leaky_diagnostic", model
        else:
            assert tier == "historical_only", model


def test_only_the_leaky_model_uses_price_features():
    cells = json.loads((ROOT / "day1_ml" / "03_exogenous_features.ipynb").read_text(encoding="utf8"))["cells"]
    source = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
    for line in source.splitlines():
        if line.startswith(("basic_cols", "calendar_cols")):
            assert "price" not in line and "discount" not in line, line
    assert 'leaky_cols = ["promo", "rel_price", "discount"]' in source


def test_evidence_is_complete_on_the_decision_date():
    week = pd.Timedelta(weeks=14)
    for decision in list(DEV_CUTOFFS) + list(STABILITY_CUTOFFS) + [TEST_CUTOFF]:
        runs = completed_runs(decision)
        assert all(run + week <= decision for run in runs)
        assert all(run + week > decision for run in DEV_CUTOFFS if run not in runs)
    assert completed_runs(TEST_CUTOFF) == list(DEV_CUTOFFS)
    assert max(DEV_CUTOFFS) + week == TEST_CUTOFF
