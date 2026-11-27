"""Tests for carrying 05c's stability settings forward. Run from the repo root:

    GD_SMOKE=1 .venv/bin/python -m pytest day2_neural/code_full/tests -q
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("GD_SMOKE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gd_helpers as gd  # noqa: E402

bc = gd.bc


def _data() -> gd.NixtlaData:
    """Two short series with one future covariate."""
    weeks = pd.date_range("2024-01-03", periods=40, freq=bc.FREQ)
    rng = np.random.default_rng(0)
    fit = pd.DataFrame({"unique_id": np.repeat(["a", "b"], 40), "ds": list(weeks) * 2,
                        "y": rng.poisson(20, 80).astype(float) + 1,
                        "promo": rng.integers(0, 2, 80).astype(float)})
    futr_weeks = pd.date_range(weeks[-1] + pd.Timedelta(weeks=1), periods=bc.H, freq=bc.FREQ)
    futr = pd.DataFrame({"unique_id": np.repeat(["a", "b"], bc.H),
                         "ds": list(futr_weeks) * 2, "promo": 0.0})
    return gd.NixtlaData(fit_df=fit, predict_df=fit, futr_df=futr, static_df=None,
                         futr_exog=["promo"])


PARAMS = {"input_size": 8, "n_blocks": [1, 1, 1, 1], "mlp_size": 8, "mlp_depth": 2}


def test_build_nbeatsx_applies_the_floor_and_clip_from_params():
    plain = gd.build_nbeatsx(PARAMS, _data(), 2, 1, "plain")
    stable = gd.build_nbeatsx({**PARAMS, "scale_floor": 0.01, "input_clip": 50.0}, _data(), 2,
                              1, "stable")
    assert gd._stability_of(plain) is None
    assert gd._stability_of(stable) == {"scale_floor": 0.01, "input_clip": 50.0}
    floor_only = gd.build_nbeatsx({**PARAMS, "scale_floor": 0.01, "input_clip": None},
                                  _data(), 2, 1, "floor")
    assert gd._stability_of(floor_only) == {"scale_floor": 0.01, "input_clip": None}


def test_a_saved_model_reloads_with_its_floor_and_clip(monkeypatch, tmp_path):
    from neuralforecast import NeuralForecast

    monkeypatch.setattr(gd, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(gd.bc, "ACCELERATOR", "cpu", raising=False)
    data = _data()
    model = gd.build_nbeatsx({**PARAMS, "scale_floor": 0.01, "input_clip": 50.0}, data, 2, 1,
                             "stable")
    nf = NeuralForecast(models=[model], freq=bc.FREQ)
    nf.fit(df=data.fit_df, val_size=bc.H)
    gd.save_model(nf, "m")
    assert json.loads((tmp_path / "m" / gd.STABILITY_FILE).read_text())["input_clip"] == 50.0
    assert gd._stability_of(gd.load_model("m").models[0]) == {"scale_floor": 0.01,
                                                             "input_clip": 50.0}


def test_selected_config_prefers_05c_choice(monkeypatch, tmp_path):
    monkeypatch.setattr(gd, "CONFIG_DIR", tmp_path)
    base = {"params": {"input_size": 14}, "max_steps": 10, "saved_model": "05_selected"}
    (tmp_path / "05_selected.json").write_text(json.dumps(base))
    assert gd.selected_config()["selected_from"] == "05"
    clip = {"use": "clean_floor_clip", "price_clean": {"k": 6.0, "floor": 0.05},
            "scale_floor": 0.01, "input_clip": 50.0}
    (tmp_path / "05c_input_clip.json").write_text(json.dumps(clip))
    cfg = gd.selected_config()
    assert cfg["selected_from"] == "05c" and cfg["saved_model"] == "05c_selected"
    assert cfg["params"] == {"input_size": 14, "scale_floor": 0.01, "input_clip": 50.0}
    assert cfg["price_clean"] == {"k": 6.0, "floor": 0.05}
    assert cfg["forecast_column"] == "GFM_clean_floor_clip"
    (tmp_path / "05c_input_clip.json").write_text(json.dumps({**clip, "use": None}))
    assert gd.selected_config()["selected_from"] == "05"
