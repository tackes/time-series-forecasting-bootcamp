"""Information labels, scenario eligibility, evaluation windows and the saved artifacts.

Run from the repo root: python -m pytest shared/test_tiers.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from shared.leaderboard import FOLDER, INFORMATION, SCENARIOS, TIERS, combine_tiers, eligible, load_forecasts, scenario_pool
from shared.prepare_data import DEV_CUTOFFS, STABILITY_CUTOFFS, TEST_CUTOFF, completed_runs

ROOT = Path(__file__).resolve().parents[1]
LEAKAGE = "leakage_diagnostic"


def test_strategy_takes_the_highest_tier_of_its_candidates():
    assert TIERS == ["history", "historical_drivers", "partial_future_drivers", "future_drivers", LEAKAGE]
    assert set(INFORMATION) == set(TIERS)
    assert combine_tiers(["history", "history"]) == "history"
    assert combine_tiers(["history", "historical_drivers"]) == "historical_drivers"
    assert combine_tiers(["historical_drivers", "future_drivers", "history"]) == "future_drivers"
    assert combine_tiers(["future_drivers", LEAKAGE]) == LEAKAGE


def test_scenario_eligibility():
    assert [t for t in TIERS if eligible(t, "no_future_drivers")] == ["history", "historical_drivers"]
    assert [t for t in TIERS if eligible(t, "future_drivers")] == ["history", "historical_drivers", "partial_future_drivers", "future_drivers"]
    for scenario in SCENARIOS:
        assert not eligible(LEAKAGE, scenario)


def test_scenario_pool_keeps_leakage_out():
    tier_of = pd.Series({"a": "history", "b": "future_drivers", "c": LEAKAGE, "d": "historical_drivers", "e": "partial_future_drivers"})
    assert scenario_pool(tier_of, "no_future_drivers") == ["a", "d"]
    assert scenario_pool(tier_of, "future_drivers") == ["a", "b", "d", "e"]
    assert scenario_pool(tier_of, "future_drivers", ["c", "b"]) == ["b"]


def expected_tier(model):
    """Tier implied by a model's name, for individual models and the strategies of notebook 07."""
    if model.endswith("_leaky") or model.startswith("Best_in_hindsight"):
        return LEAKAGE
    if model.startswith("lgbm_partial_plan"):
        return "partial_future_drivers"
    if model.startswith(("lgbm_future_", "lgbm_adj_")) or "_future" in model or "_P8_" in model:
        return "future_drivers"
    return None


def test_saved_forecasts_carry_the_right_tier():
    fcst = load_forecasts()
    tiers = fcst.drop_duplicates("model").set_index("model").information_tier
    experiments = fcst.drop_duplicates("model").set_index("model").experiment
    assert set(tiers) <= set(TIERS)
    for model, tier in tiers.items():
        expected = expected_tier(model)
        if expected:
            assert tier == expected, model
        elif experiments[model] in ("03c_behavior_features",) or model.startswith(("lgbm_hist_", "catboost_hist", "catboost_exog")):
            assert tier == "historical_drivers", model
        else:
            assert tier in ("history", "historical_drivers"), model
        if experiments[model] in ("01_baselines", "02_ml_baseline", "04_global_vs_specialized", "04_partition_granularity",
                                  "04_limited_history", "05_horizon_strategies", "06_objectives", "06_raw_quantile", "06_max_scaled"):
            assert tier == "history", model


def test_selections_choose_only_eligible_candidates():
    """Every SKU forecast of a Select_ or Tour_ strategy at the final cutoff equals one of its eligible candidates."""
    fcst = load_forecasts()
    final = fcst[fcst.cutoff == TEST_CUTOFF]
    tiers = fcst.drop_duplicates("model").set_index("model").information_tier
    individual = final[final.experiment != "07_model_selection"]
    wide = individual.pivot(index=["unique_id", "ds"], columns="model", values="yhat")
    for name, rows in final[final.model.str.startswith(("Select_", "Tour_"))].groupby("model"):
        strategy_tier = tiers[name]
        assert strategy_tier != LEAKAGE, name
        allowed = [m for m in wide.columns if tiers[m] != LEAKAGE and TIERS.index(tiers[m]) <= TIERS.index(strategy_tier)]
        values = rows.set_index(["unique_id", "ds"]).yhat.reindex(wide.index)
        matches = np.isclose(wide[allowed].to_numpy(), values.to_numpy()[:, None], rtol=1e-5, atol=1e-3).any(axis=1)
        assert matches.all(), (name, (~matches).sum())


def test_evidence_is_complete_on_the_decision_date():
    week = pd.Timedelta(weeks=14)
    for decision in list(DEV_CUTOFFS) + list(STABILITY_CUTOFFS) + [TEST_CUTOFF]:
        runs = completed_runs(decision)
        assert all(run + week <= decision for run in runs)
        assert all(run + week > decision for run in DEV_CUTOFFS if run not in runs)
    assert completed_runs(TEST_CUTOFF) == list(DEV_CUTOFFS)
    assert max(DEV_CUTOFFS) + week == TEST_CUTOFF


@pytest.mark.parametrize("name", ["day2_handoff", "day2_benchmarks"])
def test_day2_artifacts_match_the_forecasts_and_the_leaderboard(name):
    artifact = pd.read_parquet(FOLDER / f"{name}.parquet")
    board = pd.read_csv(FOLDER / "leaderboard.csv").set_index("model")
    fcst = load_forecasts()
    key = ["unique_id", "cutoff", "ds", "model"]
    assert set(artifact.cutoff) == {TEST_CUTOFF, *STABILITY_CUTOFFS}
    assert (artifact.groupby(["model", "cutoff"]).size() == 2000 * 14).all()
    joined = artifact.merge(fcst, on=key, suffixes=("", "_saved"), validate="one_to_one")
    assert len(joined) == len(artifact)
    assert np.allclose(joined.yhat, joined.yhat_saved)
    assert (joined.information_tier == joined.information_tier_saved).all()
    tiers = artifact.drop_duplicates("model").set_index("model").information_tier
    assert (board.loc[tiers.index, "information_tier"] == tiers).all()
    if name == "day2_benchmarks":
        handoff = pd.read_parquet(FOLDER / "day2_handoff.parquet")
        assert set(artifact.model) <= set(handoff.model)


def test_leaderboard_flags_match_the_tiers():
    board = pd.read_csv(FOLDER / "leaderboard.csv")
    assert (board.without_future_drivers == board.information_tier.map(lambda t: eligible(t, "no_future_drivers"))).all()
    assert (board.with_future_drivers == board.information_tier.map(lambda t: eligible(t, "future_drivers"))).all()
    assert (board.information == board.information_tier.map(INFORMATION)).all()
