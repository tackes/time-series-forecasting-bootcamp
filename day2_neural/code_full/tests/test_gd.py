"""Unit tests for the code_full helpers. Run from the repo root:

    GD_SMOKE=1 .venv/Scripts/python -m pytest day2_neural/code_full/tests -q
"""

import json
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
ORIGINAL_RESULTS = gd.ROOT / "results" / "day2"      # the original labs' outputs


def test_paths_point_away_from_original_results():
    for path in (bc.RESULTS_DIR, bc.FORECAST_DIR, bc.LEADERBOARD_PATH):
        assert gd.OUT_DIR in Path(path).parents or Path(path) == gd.OUT_DIR
        assert ORIGINAL_RESULTS not in Path(path).parents
    assert bc.DATA_DIR == gd.ROOT / "data"
    assert (bc.DATA_DIR / "sales_weekly.parquet").exists()


def test_config_round_trip():
    cfg = {"model": "NBEATSx", "lr": np.float64(4.83e-4), "steps": np.int64(10),
           "d": date(2025, 1, 1)}
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
    steps = (0, 9, 10, 109, 110, 210)
    f = [gd.warmup_then_step(s, warmup_steps=10, decay_every=100) for s in steps]
    assert f == pytest.approx([0.1, 1.0, 1.0, 1.0, 0.5, 0.25])


def test_recipe_kwargs_on_and_off():
    on, off = gd.recipe_kwargs(1000), gd.recipe_kwargs(1000, recipe=False)
    assert on["gradient_clip_val"] == 1.0 and isinstance(on["callbacks"][0], gd.BestWeights)
    assert "callbacks" not in off and "lr_scheduler" not in off
    assert on["enable_checkpointing"] is False and on["logger"] is False
    assert gd.recipe_kwargs(1000)["callbacks"][0] is not on["callbacks"][0]  # fresh per model


def test_one_device_per_fit():
    # devices="auto" starts distributed training on a multi-GPU box and hangs in a notebook
    assert gd.recipe_kwargs(1000)["devices"] == 1
    assert gd.recipe_kwargs(1000, recipe=False)["devices"] == 1


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
    overall = est.loc[est["sale_class"] == "all", "elasticity"].item()
    assert overall == pytest.approx(-0.5, abs=0.05)


def test_save_config_writes_nan_as_null():
    cfg = {"a": float("nan"), "b": [np.float64("inf"), 1.0], "c": {"d": np.float32("nan")}}
    path = gd.save_config("test_nan", cfg)

    def no_constants(name):
        raise ValueError(f"non-strict JSON constant {name}")

    back = json.loads(path.read_text(), parse_constant=no_constants)
    assert back == {"a": None, "b": [None, 1.0], "c": {"d": None}}
    path.unlink()


def _middle_class_panel(n: int = 3) -> pl.DataFrame:
    """A few scored smoke-panel series from classes that get discount copies (B-H)."""
    panel = gd.train_panel()
    ids = (panel.filter(pl.col("unique_id").is_in(gd.eval_ids())
                        & pl.col("sale_class").is_in(list("DEFG")))
           ["unique_id"].unique().sort().head(n).to_list())
    return panel.filter(pl.col("unique_id").is_in(ids))


@pytest.mark.parametrize("deflate", [False, True])
def test_prepare_valid_cutoff_reference_price_uses_pre_cutoff_weeks(deflate):
    panel = _middle_class_panel()
    later = pl.col("ds") >= bc.VALID_START
    shocked = panel.with_columns(pl.when(later).then(pl.col("sale_price") * 10)
                                 .otherwise(pl.col("sale_price")).alias("sale_price"))
    cfg = {"features": {"deflate": deflate}, "futr_exog": bc.FUTR_EXOG_COLS}
    base, inner, valid, data = gp.prepare(cfg, panel=panel, cutoff=bc.VALID_START)
    moved = gp.prepare(cfg, panel=shocked, cutoff=bc.VALID_START)[0]

    before = pl.col("ds") < bc.VALID_START
    key = ["unique_id", "ds"]
    a = base.filter(before).sort(key)["price_index"].to_numpy()
    b = moved.filter(before).sort(key)["price_index"].to_numpy()
    assert np.allclose(a, b)                       # later prices do not move the reference
    if not deflate:                                # reference = median pre-cutoff sale price
        ref = panel.filter(before).group_by("unique_id").agg(pl.col("sale_price").median())
        check = base.join(ref.rename({"sale_price": "ref"}), on="unique_id")
        assert np.allclose(check["price_index"], check["sale_price"] / check["ref"])
    assert inner["ds"].max() < bc.VALID_START
    assert valid["ds"].min() == bc.VALID_START
    assert valid["ds"].n_unique() == bc.H
    futr_weeks = pd.to_datetime(data.futr_df["ds"]).dt.date
    assert futr_weeks.min() == bc.VALID_START
    assert futr_weeks.nunique() == bc.H


def test_prepare_valid_cutoff_counts_copies_pre_cutoff():
    panel = _middle_class_panel().with_columns(
        pl.when(pl.col("ds") >= bc.VALID_START).then(0.5).otherwise(0.0).alias("discount_depth"))
    cfg = {"futr_exog": bc.FUTR_EXOG_COLS,
           "oversample": {"rule": "rolling", "cap": gp.COPY_CAP, "multiplier": 5}}
    # every discount week sits after VALID_START: tuning sees none, so no copies
    _, inner, _, tune = gp.prepare(cfg, panel=panel, cutoff=bc.VALID_START)
    assert tune.fit_df["unique_id"].nunique() == inner["unique_id"].n_unique()
    assert pd.to_datetime(tune.fit_df["ds"]).dt.date.max() < bc.VALID_START
    # the final fit's training weeks include them (14 weeks: the 10-25 tier), so copies appear
    _, train, _, final = gp.prepare(cfg, panel=panel)
    assert final.fit_df["unique_id"].nunique() > train["unique_id"].n_unique()


def _sweep(shifts, elasticity_at):
    """A one-series sweep whose arc elasticity at shift m is ``elasticity_at(m)``."""
    rows = [{"unique_id": "a", "shift": 0.0, "units": 100.0}]
    for m in shifts:
        e = elasticity_at(m)
        # q_up / q_dn chosen so (q_up - q_dn) / mean / (2m) == e
        ratio = (1 + e * m) / (1 - e * m)
        rows.append({"unique_id": "a", "shift": -m, "units": 100.0})
        rows.append({"unique_id": "a", "shift": m, "units": 100.0 * ratio})
    return pd.DataFrame(rows)


def test_elasticity_report_reads_twenty_percent_not_the_largest_shift():
    sweep = _sweep((0.05, 0.10, 0.20, 0.30), lambda m: -1.0 if m == 0.20 else -9.0)
    el = gp.arc_elasticity(sweep)
    report = gp.elasticity_report(el)
    assert report["median_e20"] == pytest.approx(-1.0)
    assert report["median_e30"] == pytest.approx(-9.0)
    assert gp.e20(el).item() == pytest.approx(-1.0)
    assert gp.e20(sweep).item() == pytest.approx(-1.0)          # raw sweep accepted too
    only_small = gp.arc_elasticity(_sweep((0.05, 0.10), lambda m: -2.0))
    assert gp.report_shift(only_small) == pytest.approx(0.10)   # falls back to the largest


def test_seed_mean_sweep_averages_units():
    runs = [{"sweep": pd.DataFrame({"unique_id": ["a"], "shift": [0.2], "units": [u]})}
            for u in (1.0, 3.0)]
    assert gp.seed_mean_sweep(runs)["units"].item() == 2.0


def test_cached_resumes_in_a_train_run(monkeypatch):
    name = "test_cached_resume"
    path = gd.TABLE_DIR / f"{name}.parquet"
    path.unlink(missing_ok=True)
    calls = []

    def compute():
        calls.append(1)
        return pd.DataFrame({"x": [len(calls)]})

    monkeypatch.setattr(gd, "TRAIN", True)
    monkeypatch.setattr(gd, "REFRESH", False)
    assert gd.cached(name, compute)["x"].item() == 1
    assert gd.cached(name, compute)["x"].item() == 1        # read back, not recomputed
    assert len(calls) == 1
    monkeypatch.setattr(gd, "REFRESH", True)
    assert gd.cached(name, compute)["x"].item() == 2        # GD_REFRESH=1 recomputes
    monkeypatch.setattr(gd, "TRAIN", False)
    assert gd.cached(name, compute)["x"].item() == 2        # LOAD reads
    path.unlink()
    with pytest.raises(FileNotFoundError, match="GD_TRAIN=1"):
        gd.cached(name, compute)


def test_seed_medians_one_value_per_seed():
    ids = gd.eval_ids()[:3]
    weeks = [bc.TEST_START + timedelta(weeks=k) for k in range(2)]
    hist_weeks = [bc.TEST_START - timedelta(weeks=60 - k) for k in range(60)]
    rng = np.random.default_rng(0)
    history = pl.DataFrame({"unique_id": np.repeat(ids, 60), "ds": hist_weeks * 3,
                            "y": rng.uniform(1, 10, 180)})
    actual = pl.DataFrame({"unique_id": np.repeat(ids, 2), "ds": weeks * 3,
                           "y": [5.0] * 6})
    runs = []
    for seed, value in ((1, 5.0), (2, 8.0)):
        fcst = pd.DataFrame({"unique_id": np.repeat(ids, 2), "ds": weeks * 3, "M": value})
        runs.append({"summary": {"alias": "M", "seed": seed}, "forecast": fcst})
    meds = gd.seed_medians(runs, actual, history)
    assert list(meds.index) == ["M_s1", "M_s2"]
    assert meds["M_s1"] == pytest.approx(0.0)
    assert meds["M_s2"] > 0


def test_build_nbeatsx_loss_override():
    from neuralforecast.losses.pytorch import MAE, HuberLoss, MQLoss

    data = gd.NixtlaData(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None,
                         futr_exog=["promo"], stat_exog=[])
    default = gd.build_nbeatsx({}, data, max_steps=10, seed=0, alias="a")
    assert isinstance(default.loss, HuberLoss) and isinstance(default.valid_loss, MAE)
    quant = gd.build_nbeatsx({}, data, max_steps=10, seed=0, alias="q",
                             loss=MQLoss(level=[80]), valid_loss=MQLoss(level=[80]))
    assert isinstance(quant.loss, MQLoss) and isinstance(quant.valid_loss, MQLoss)
    assert quant.alias == "q" and quant.max_steps == 10


def test_atomic_write_keeps_old_file_on_crash(tmp_path):
    path = tmp_path / "table.parquet"
    pd.DataFrame({"a": [1]}).to_parquet(path)
    with pytest.raises(RuntimeError):
        with gd._atomic(path) as tmp:
            tmp.write_text("half a file")
            raise RuntimeError("crash mid-write")
    assert pd.read_parquet(path)["a"].tolist() == [1]          # the old file is untouched
    with gd._atomic(path) as tmp:
        pd.DataFrame({"a": [2]}).to_parquet(tmp)
    assert pd.read_parquet(path)["a"].tolist() == [2]
    assert [p.name for p in tmp_path.iterdir()] == ["table.parquet"]   # no temp file left


def _fake_run(runs_dir, name, seed, alias="NBEATSx"):
    run_dir = runs_dir / name / f"seed{seed}"
    run_dir.mkdir(parents=True)
    pd.DataFrame({"unique_id": ["a"], "ds": [pd.Timestamp("2025-01-01")], alias: [1.0]}
                 ).to_parquet(run_dir / "forecast.parquet")
    pd.DataFrame({"step": [1], "valid_loss": [1.0]}).to_csv(run_dir / "trajectory.csv")
    summary = {"name": name, "seed": seed, "alias": alias, "fit_seconds": 60.0,
               "max_steps": 20, "n_train_series": 150, "best_frac": 0.5, "best_valid": 1.0}
    (run_dir / "summary.json").write_text(json.dumps(summary))


def test_load_runs_and_list_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(gd, "RUNS_DIR", tmp_path)
    for seed in reversed(gd.SEEDS):
        _fake_run(tmp_path, "05_trial_abc", seed)
    (tmp_path / "05_trial_abc" / "seed999").mkdir()          # unfinished: no summary.json
    _fake_run(tmp_path, "02_cell_xyz", gd.SEEDS[0])

    runs = gd.load_runs("05_trial_abc")
    assert [r["summary"]["seed"] for r in runs] == gd.SEEDS    # fit_seeds order
    assert set(runs[0]) == {"summary", "trajectory", "forecast"}
    assert list(gd.seed_forecasts(runs).columns)[-1] == "NBEATSx_mean"

    table = gd.list_runs("05_trial")
    assert len(table) == len(gd.SEEDS) and set(table["name"]) == {"05_trial_abc"}
    assert len(gd.list_runs()) == len(gd.SEEDS) + 1
    with pytest.raises(FileNotFoundError):
        gd.load_runs("05_trial_missing")
    with pytest.raises(FileNotFoundError):
        gd.load_runs("02_cell_xyz", seeds=gd.SEEDS)


def test_run_study_resumes_interrupted_and_out_of_memory_trials(tmp_path):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.ERROR)
    storage = f"sqlite:///{(tmp_path / 'study.db').as_posix()}"
    ran = []

    def objective(trial):
        x = trial.suggest_float("x", 0, 1)
        ran.append(round(x, 3))
        if x == 0.25 and ran.count(0.25) == 1:
            raise torch_oom()
        return x

    def torch_oom():
        return gd.torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")

    # a crash leaves a trial RUNNING (here: started by hand and never finished)
    study = optuna.create_study(study_name="s", storage=storage)
    gd.run_study(study, objective, 1, first_trials=[{"x": 0.1}])
    stale = study.ask({"x": optuna.distributions.FloatDistribution(0, 1)})
    stale_x = stale.params["x"]

    study = optuna.load_study(study_name="s", storage=storage)
    study.enqueue_trial({"x": 0.25})                          # will run out of memory once
    gd.run_study(study, objective, 4)
    complete = study.get_trials(states=[optuna.trial.TrialState.COMPLETE])
    assert len(complete) == 4
    xs = [round(t.params["x"], 3) for t in complete]
    assert round(stale_x, 3) in xs                            # the interrupted trial was redone
    assert 0.25 in xs and ran.count(0.25) == 2                # retried once after the OOM
    assert len(study.get_trials(states=[optuna.trial.TrialState.FAIL])) == 2
