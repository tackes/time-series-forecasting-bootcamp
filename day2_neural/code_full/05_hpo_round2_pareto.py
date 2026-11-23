# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# # Day 2 · 05 (full data) — Lab: tuning for accuracy *and* price response (30 min)
#
# Notebook 02 tuned for accuracy alone. Here we tune for two things at once: accuracy, and a
# price response with the right sign and a plausible size. The research found these pull
# against each other (§6, [100k set]). So there is no single best setting, only a **front** of
# settings where you cannot improve one without giving up some of the other.
#
# Goals
#
# 1. Fix the accuracy limit, the plausibility bound and the selection rule **before** any trial.
# 2. Run a two-objective Optuna search on the validation window, 3 seeds per trial.
# 3. Find the Pareto front and its knee, two ways, and choose by a rule on validation numbers.
# 4. Confirm on the test window, once, for a pre-registered set of settings.
#
# The research's most important lesson (`docs/gfm_learnings.md` §6, [100k set]): **accuracy
# transferred from tuning to test; elasticity did not.** It shrank 7–18×, and no setting could
# be told apart from zero. This notebook checks whether that holds here.
#
# **In the chain:** reads `04_price`, `02_shortlist` and `01_recipe`, writes `05_selected`.

# %%
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

import gd_helpers as gd
import gd_plots as gplt
import gd_price as gp

bc = gd.bc
gd.status()
gplt.use_style()
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)  # TPE options

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## Intuition: a front, not a winner
#
# Choosing a car on fuel economy and speed is like this. Some cars are worse on both, and you
# drop them. The rest form a **front**: to gain speed you must give up economy. The **knee** is
# the car where the trade turns bad, where a little more speed starts to cost a lot of economy.
#
# Here each trial is a car: accuracy along the bottom, price response up the side, lower is
# better on both. A trial is on the front if no other trial beats it on both.

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 1. Settings carried forward
#
# The features, covariates and oversampling come from `04_price`, and the noise floor from
# `02_shortlist`. The **incumbent** is the model 04 selected. The cell also prints a rough GPU
# budget from 01's minutes per fit.

# %%
cfg01 = gd.load_config("01_recipe")
cfg02 = gd.load_config("02_shortlist")
cfg04 = gd.load_config("04_price")
noise_floor = cfg02["noise_floor"]
incumbent_params = cfg04["params"]
max_steps = cfg04["max_steps"]
data_cfg = {"features": cfg04["features"], "futr_exog": cfg04["futr_exog"],
            "oversample": cfg04["oversample"], "y_shift": cfg04.get("y_shift", gd.Y_SHIFT)}
band = tuple(cfg04["empirical_band"])

N_TRIALS = 30 if not gd.SMOKE else 3   # 30 trials × 3 seeds = 90 fits, plus 3 + up to 12 confirm
CONFIRM_SEEDS = gd.SEEDS               # 3 seeds; add two more if the GPU budget allows
SAMPLER_SEED = bc.RANDOM_SEED
STUDY_NAME = "05_pareto"
STUDY_PATH = gd.OUT_DIR / "05_study.db"

minutes = cfg01.get("minutes_per_fit", {}).get("NBEATSx_recipe_mean")
if minutes:
    n_fits = (N_TRIALS + 1) * len(gd.SEEDS) + 4 * len(CONFIRM_SEEDS)
    print(f"budget: about {n_fits} fits × {minutes:.1f} min ≈ {n_fits * minutes / 60:.1f} h")
print(f"data from 04: {data_cfg}")

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 2. Two windows: tune on validation, confirm on test
#
# This is the fix for the research's early mistake of tuning on the test window (§1, [method]).
# Tuning fits use weeks before `VALID_START` and are scored on the 14 weeks after it. Every
# per-series statistic, including which series get discount copies, comes from those inner
# weeks only; `gp.prepare(cfg, cutoff=bc.VALID_START)` builds the tuning data that way. The
# test window is used once, in section 7.

# %%
raw = gd.train_panel()
_, inner_train, valid, inner_data = gp.prepare(data_cfg, panel=raw, cutoff=bc.VALID_START)
panel, train, test, full_data = gp.prepare(data_cfg, panel=raw)
info = gplt.series_table(panel)
valid_report = gp.report_ids(inner_train)
test_report = gp.report_ids(train)
print(f"tuning: train to {inner_train['ds'].max()}, validation {valid['ds'].min()} to "
      f"{valid['ds'].max()}, {inner_data.fit_df['unique_id'].nunique():,} training series")
print(f"test:   train to {train['ds'].max()}, test {test['ds'].min()} to {test['ds'].max()}")
print(f"report series: {len(valid_report)} (validation), {len(test_report)} (test)")


# %%
def fit_config(params: dict, data: gd.NixtlaData, prefix: str, seeds: list[int],
               report: list[str]) -> list[dict]:
    """Fit one setting for each seed (cached by a hash of ``params``), with a price sweep."""
    def build(seed):
        return gd.build_nbeatsx(params, data, max_steps, seed, "GFM_selected")

    def extras(nf, d):
        return {"sweep": gp.price_sweep(nf, d, report, channel=cfg04["price_channel"])}

    name = f"{prefix}_{gd.params_key(params)}"
    with bc.Timer() as t:
        runs = gd.fit_seeds(build, data, name, seeds=seeds, extras=extras)
    if gd.TRAIN:
        print(f"{name}: {t.seconds / 60:.1f} min")
    return runs


def evaluate(runs: list[dict], window, history) -> dict:
    """Accuracy of the seed-mean forecast and elasticity of the seed-mean sweep, on one window.

    ``e20`` is the median arc elasticity at ±20% of the seed-mean sweep (as 04 reports it);
    ``seed_e20`` keeps each seed's own median, for the spread charts.
    """
    fcst = gd.seed_forecasts(runs)
    rmsse = gd.per_series(gd.score(fcst, window, history))
    mean_col = [c for c in rmsse.columns if c.endswith("_mean")][0]
    seed_e = []
    for run in runs:
        seed_e.append(float(gp.e20(run["sweep"]).median()))
    mean_e = float(gp.e20(gp.seed_mean_sweep(runs)).median())
    return {"accuracy": float(rmsse[mean_col].median()),
            "mean_rmsse": float(rmsse[mean_col].mean()),
            "seed_rmsse": gd.seed_medians(runs, window, history).tolist(),
            "seed_e20": seed_e,
            "e20": mean_e,
            "forecast": fcst.rename(columns={mean_col: "seed_mean"})}


# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 3. The incumbent, and the rules fixed before any trial
#
# First the 04 model on the validation window, with 3 seeds. Its accuracy sets the limit. The
# plausibility bound comes from the empirical elasticity of the weeks before `VALID_START`,
# so no validation or test week shapes the objective.

# %%
incumbent_runs = fit_config(incumbent_params, inner_data, "05_trial", gd.SEEDS, valid_report)
incumbent = evaluate(incumbent_runs, valid, inner_train)

ACC_LIMIT = 1.10 * incumbent["accuracy"]
valid_empirical = gp.empirical_elasticity(inner_train, gp.report_ids(inner_train))
is_all = valid_empirical["sale_class"] == "all"
empirical_all = float(valid_empirical.loc[is_all, "elasticity"].iloc[0])
if np.isfinite(empirical_all):
    PLAUSIBLE_BOUND = 2 * abs(empirical_all)
else:
    PLAUSIBLE_BOUND = 2.0
print(f"incumbent: median RMSSE {incumbent['accuracy']:.4f}, median e20 {incumbent['e20']:.3f}")
print(f"accuracy limit {ACC_LIMIT:.4f} | empirical elasticity before VALID_START "
      f"{empirical_all:.2f} | plausibility bound |e| <= {PLAUSIBLE_BOUND:.2f}")

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# **The rules, fixed here, before any trial runs:**
#
# * **Objective 1, accuracy:** median RMSSE of the seed-mean forecast on the validation window.
#   It never sees elasticity.
# * **Objective 2, elasticity:** the median arc elasticity at ±20% of the **seed-mean sweep**
#   (units averaged over seeds for each series and shift, then the arc elasticity, as in 04;
#   single channel, report series). Set to 0 if fewer than two seeds finished, if it is not
#   finite, or if `|e|` is above the plausibility bound: **twice the empirical elasticity** of
#   the weeks before `VALID_START`. (The research used 2, but its empirical value was about
#   −0.5; here it is ≈ −3.6.)
# * **Constraint:** a trial is feasible only if its accuracy is at most **1.10 × the
#   incumbent's**.
# * **Confirm set:** the accuracy-optimal feasible point, the knee by both methods, and the
#   incumbent as a control. Each is refitted on the full training window and scored on the test
#   window once, with `CONFIRM_SEEDS`. This reports what transferred; it chooses nothing.
# * **Selection, on the validation numbers:** a knee wins if its validation elasticity is
#   negative in at least 2 of 3 seeds **and** its validation median RMSSE is within the 02
#   noise floor of the accuracy-optimal point's. The gain knee is checked first, then the
#   distance knee. Otherwise the accuracy-optimal point wins.
#
# Weight decay was in the research's search space. It is left out here: the library's default
# optimiser has none, and adding one would change how every later notebook builds the model.

# %%
SELECTION_RULE = ("on the validation window: knee (gain first, then distance) if e20 < 0 in "
                  ">= 2/3 of seeds and median RMSSE <= accuracy-optimal + 02 noise floor; "
                  "else accuracy-optimal")
BLOCKS = ["1111", "2222", "1221", "3333"]
COVERAGE_LRS = [1e-4, 5e-4]


def trial_params(trial) -> dict:
    """The incumbent's settings with the searched ones replaced."""
    params = dict(incumbent_params)
    blocks = trial.suggest_categorical("n_blocks", BLOCKS)
    params["n_blocks"] = [int(b) for b in blocks]
    params["mlp_size"] = trial.suggest_categorical("mlp_size", [256, 512, 768])
    params["mlp_depth"] = trial.suggest_categorical("mlp_depth", [2, 3])
    params["learning_rate"] = trial.suggest_float("learning_rate", 2e-5, 1e-3, log=True)
    params["dropout_prob_theta"] = trial.suggest_float("dropout_prob_theta", 0.05, 0.30)
    return params


def elasticity_objective(res: dict) -> float:
    """Objective 2: the seed-mean sweep's median e20, or 0 if implausible or not finite."""
    e = res["e20"]
    if len(res["seed_e20"]) < 2 or not np.isfinite(e) or abs(e) > PLAUSIBLE_BOUND:
        return 0.0
    return e


def objective(trial):
    """Fit 3 seeds on the inner window; return (accuracy, elasticity)."""
    params = trial_params(trial)
    runs = fit_config(params, inner_data, "05_trial", gd.SEEDS, valid_report)
    res = evaluate(runs, valid, inner_train)
    trial.set_user_attr("key", gd.params_key(params))
    trial.set_user_attr("params", json.dumps(params))
    trial.set_user_attr("e20_raw", res["e20"])
    trial.set_user_attr("seed_e20", res["seed_e20"])
    trial.set_user_attr("constraint", res["accuracy"] - ACC_LIMIT)
    return res["accuracy"], elasticity_objective(res)


def constraint(trial) -> list[float]:
    """Optuna's feasibility check: values <= 0 are feasible."""
    return [trial.user_attrs.get("constraint", 0.0)]


# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 4. The search
#
# Forced **coverage cells** run first: every block layout at two learning rates, so the search
# does not miss a region by chance. Then Optuna's TPE sampler takes over. The study is stored
# in SQLite and every fit is cached by a hash of its settings, so a crash resumes where it
# stopped.

# %%
def run_study() -> None:
    """Create or resume the study and run the trials still to do."""
    study = optuna.create_study(study_name=STUDY_NAME,
                                storage=f"sqlite:///{STUDY_PATH.as_posix()}",
                                directions=["minimize", "minimize"], load_if_exists=True)
    # A fresh seed per restart: a fixed seed would re-draw the same points after a crash.
    study.sampler = optuna.samplers.TPESampler(seed=SAMPLER_SEED + len(study.trials),
                                               multivariate=True, constraints_func=constraint)
    if not study.trials:
        cells = []
        for blocks in BLOCKS:
            for lr in COVERAGE_LRS:
                cells.append({"n_blocks": blocks, "learning_rate": lr,
                              "mlp_size": incumbent_params["mlp_size"],
                              "mlp_depth": incumbent_params["mlp_depth"],
                              "dropout_prob_theta": incumbent_params["dropout_prob_theta"]})
        for cell in cells[:N_TRIALS]:
            study.enqueue_trial(cell)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if N_TRIALS > len(done):
        study.optimize(objective, n_trials=N_TRIALS - len(done))


def trials_table() -> pd.DataFrame:
    """One row per finished trial: objectives, feasibility and settings."""
    study = optuna.load_study(study_name=STUDY_NAME,
                              storage=f"sqlite:///{STUDY_PATH.as_posix()}")
    rows = []
    for t in study.get_trials(states=[optuna.trial.TrialState.COMPLETE]):
        row = {"trial": t.number, "accuracy": t.values[0], "elasticity": t.values[1],
               "e20_raw": t.user_attrs["e20_raw"], "constraint": t.user_attrs["constraint"],
               "seed_e20": json.dumps(t.user_attrs["seed_e20"]), "key": t.user_attrs["key"],
               "params": t.user_attrs["params"]}
        row.update(t.params)
        rows.append(row)
    out = pd.DataFrame(rows)
    out["feasible"] = out["constraint"] <= 0
    return out


if gd.TRAIN:
    with bc.Timer() as t:
        run_study()
    print(f"study: {t.seconds / 60:.1f} min")
    # the study may have grown since the table was cached, so rebuild it from the study
    (gd.TABLE_DIR / "05_trials.parquet").unlink(missing_ok=True)
trials = gd.cached("05_trials", trials_table)
show_cols = ["trial", "accuracy", "elasticity", "e20_raw", "feasible", "n_blocks", "mlp_size",
             "mlp_depth", "learning_rate", "dropout_prob_theta"]
trials.sort_values("accuracy")[show_cols].round(4)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 5. The front and the knee
#
# **How to read the Pareto chart.** Each dot is one trial: median validation RMSSE along the
# bottom, median e20 up the side (more negative is a stronger response). Blue dots are
# feasible and grey dots broke the accuracy limit. The red line joins the front. The open
# diamond is the incumbent.
#
# The knee is found two ways, and both are marked:
#
# * **gain** (star): the most elasticity gained per unit of accuracy lost, against the
#   accuracy-optimal point;
# * **distance** (plus): the point furthest from the straight line joining the front's two
#   ends, with both axes scaled 0–1.

# %%
feasible = trials[trials["feasible"]]
front = gd.pareto_front(feasible) if len(feasible) else feasible
knees = {}
if len(front) >= 2:
    gain_knee = gd.knee_by_gain(front)
    if pd.notna(gain_knee):
        knees["gain"] = gain_knee
    knees["distance"] = gd.knee_by_distance(front)
gplt.plot_pareto(trials, front, knees, incumbent=(incumbent["accuracy"], incumbent["e20"]))
front[show_cols].round(4)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ### Reading the result
#
# * Dots at exactly 0 on the vertical axis had an implausible or undefined response, and were
#   scored as "no response".
# * A steep front means a little accuracy buys a lot of response. A flat one means the trade is
#   poor. Front size ≈ __ of ≈ __ trials; knee e20 ≈ __. <!-- TODO: fill after GPU run -->
# * If the two knee methods disagree, the research's rule was that a person decides. Here the
#   rule from section 3 decides between them, on the validation numbers.
#
# *Reference run: full panel, GPU box, 3 seeds per trial. Exact numbers vary with seed and
# hardware; the reading should not.*

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 6. Applying the selection rule, on the validation numbers
#
# The rule from section 3, applied as written to each trial's validation accuracy and its
# per-seed validation e20. The test window has not been used yet.

# %%
roles = {}
if len(front):
    roles["accuracy_optimal"] = front.index[0]          # the front is sorted by accuracy
for method, idx in knees.items():
    roles[f"knee_{method}"] = idx

valid_rows = []
for role, idx in roles.items():
    seed_e = json.loads(trials.loc[idx, "seed_e20"])
    valid_rows.append({"role": role, "key": trials.loc[idx, "key"],
                       "valid_rmsse": trials.loc[idx, "accuracy"],
                       "valid_e20": trials.loc[idx, "e20_raw"],
                       "valid_share_seeds_negative": float(np.mean(np.array(seed_e) < 0)),
                       "valid_seed_e20": seed_e})
valid_rows.append({"role": "incumbent", "key": gd.params_key(incumbent_params),
                   "valid_rmsse": incumbent["accuracy"], "valid_e20": incumbent["e20"],
                   "valid_share_seeds_negative":
                       float(np.mean(np.array(incumbent["seed_e20"]) < 0)),
                   "valid_seed_e20": incumbent["seed_e20"]})
valid_table = pd.DataFrame(valid_rows).set_index("role")

if "accuracy_optimal" in valid_table.index:
    selected_role = "accuracy_optimal"
else:
    selected_role = "incumbent"            # no feasible trial: keep the 04 model
for knee in ("knee_gain", "knee_distance"):
    if knee not in valid_table.index or selected_role != "accuracy_optimal":
        continue
    if valid_table.loc[knee, "key"] == valid_table.loc["accuracy_optimal", "key"]:
        continue
    negative = valid_table.loc[knee, "valid_share_seeds_negative"] >= 2 / 3
    limit = valid_table.loc["accuracy_optimal", "valid_rmsse"] + noise_floor
    close = valid_table.loc[knee, "valid_rmsse"] <= limit
    if negative and close:
        selected_role = knee
selected_key = valid_table.loc[selected_role, "key"]
print(f"selected on validation: {selected_role} ({selected_key})")
valid_table.round(4)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 7. The confirm run: refit on the full training window, score the test window once
#
# Each setting in the confirm set is refitted on all training weeks and scored on test. This
# reports what transferred; it does not change the choice above. The chart after the table
# puts validation against test, for accuracy and for elasticity. On the dashed diagonal a
# setting scored the same on both windows.

# %%
confirm_params = {role: json.loads(trials.loc[idx, "params"]) for role, idx in roles.items()}
confirm_params["incumbent"] = incumbent_params
selected_params = confirm_params[selected_role]

confirm_runs = {}
confirm_nf = {}
for role, params in confirm_params.items():
    key = gd.params_key(params)
    if key not in confirm_runs:
        confirm_runs[key] = fit_config(params, full_data, "05_confirm", CONFIRM_SEEDS,
                                       test_report)
        confirm_nf[key] = confirm_runs[key][0]["nf"]   # first seed, kept to save if selected

rows = []
confirm_eval = {}
for role, params in confirm_params.items():
    key = gd.params_key(params)
    res = evaluate(confirm_runs[key], test, train)
    confirm_eval[role] = res
    rows.append({"role": role, "key": key,
                 "valid_rmsse": valid_table.loc[role, "valid_rmsse"],
                 "test_rmsse": res["accuracy"],
                 "valid_e20": valid_table.loc[role, "valid_e20"], "test_e20": res["e20"],
                 "valid_share_seeds_negative":
                     valid_table.loc[role, "valid_share_seeds_negative"],
                 "test_share_seeds_negative": float(np.mean(np.array(res["seed_e20"]) < 0)),
                 "test_seed_e20": res["seed_e20"], "selected": role == selected_role})
confirm = pd.DataFrame(rows).set_index("role")
confirm.round(4)

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, metric in zip(axes, ["rmsse", "e20"]):
    x = confirm[f"valid_{metric}"]
    y = confirm[f"test_{metric}"]
    ax.scatter(x, y, s=50, color=gplt.MODEL_COLOURS[0])
    for role in confirm.index:
        ax.annotate(role, (x[role], y[role]), fontsize=8, xytext=(4, 4),
                    textcoords="offset points")
    lim = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lim, lim, color="0.4", ls="--", lw=1, label="same on both windows")
    ax.set(xlabel=f"validation {metric}", ylabel=f"test {metric}")
    ax.legend()
axes[0].set_title("Accuracy: validation against test")
axes[1].set_title("Elasticity at ±20%: validation against test")
axes[1].axhline(0, color=gplt.COLOURS["bad"], ls=":", lw=1)
fig.tight_layout()
plt.show()

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ### Reading the result
#
# * **Accuracy:** if the points keep their order from left to right, tuning on validation
#   transferred to the test window.
# * **Elasticity:** in the research the tuned elasticities shrank 7–18× on the new window, and
#   no setting could be told apart from zero (§6, [100k set]). Points above the dashed line here
#   (closer to zero on test) mean the response measured during tuning was optimistic.
# * Remember the windows differ: validation has no event week, and test opens with Easter.
# * Knee e20: validation ≈ __, test ≈ __. <!-- TODO: fill after GPU run -->
#
# *Exact numbers vary with seed and hardware; the reading should not.*

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# The confirm set on test, in the strength, by-class and arm comparison charts from notebook 04.

# %%
strength = {}
for role, res in confirm_eval.items():
    mean_sweep = gp.seed_mean_sweep(confirm_runs[confirm.loc[role, "key"]])
    strength[role] = gp.e20(mean_sweep)
empirical = pd.DataFrame(cfg04["empirical"])
gplt.plot_elasticity_strength(strength, title="Confirm set: strength of response on test")
gplt.plot_elasticity_by_class(strength, info, empirical=empirical)

# %%
confirm_arms = {role: confirm_runs[confirm.loc[role, "key"]] for role in confirm.index}
seed_rmsse = {role: res["seed_rmsse"] for role, res in confirm_eval.items()}
seed_e20 = {role: res["seed_e20"] for role, res in confirm_eval.items()}
gplt.plot_arm_comparison(confirm_arms, metrics=seed_rmsse, elasticity=seed_e20, band=band)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 8. The selected setting on the board
#
# The setting chosen in section 6. Its seed-mean test forecast goes on the board as
# `GFM_selected`, and its first seed is saved for the later notebooks.

# %%
selected_fcst = confirm_eval[selected_role]["forecast"]
selected_fcst = selected_fcst[["unique_id", "ds", "seed_mean"]].rename(
    columns={"seed_mean": "GFM_selected"})
fit_seconds = gd.summaries(confirm_runs[selected_key])["fit_seconds"].mean()
gd.save_forecasts(selected_fcst, "05_selected")
gd.record(selected_fcst, test, train,
          lab=f"05 selected ({len(CONFIRM_SEEDS)}-seed mean)",
          fit_seconds=fit_seconds, notes=f"{selected_role}; fit_seconds is per seed")
gd.median_board()

# %%
def save_selected() -> None:
    """Save the first confirm seed of the selected setting as ``05_selected``.

    If the confirm runs were loaded from disk (a resumed run), that seed is refitted once.
    """
    nf = confirm_nf.get(selected_key)
    if nf is not None:
        gd.save_model(nf, "05_selected")
        return

    def build(seed):
        return gd.build_nbeatsx(selected_params, full_data, max_steps, seed, "GFM_selected")

    gd.fit_seed(build, CONFIRM_SEEDS[0], full_data, f"05_selected_{selected_key}",
                save_as="05_selected")


if gd.TRAIN:
    save_selected()

# %%
trial_no = None
if selected_role != "incumbent":
    trial_no = int(trials.loc[roles[selected_role], "trial"])
gd.save_config("05_selected", {
    "params": selected_params,
    "features": data_cfg["features"],
    "futr_exog": data_cfg["futr_exog"],
    "oversample": data_cfg["oversample"],
    "y_shift": data_cfg["y_shift"],
    "max_steps": max_steps,
    "seeds": CONFIRM_SEEDS,
    "trial": trial_no,
    "role": selected_role,
    "selection_rule": SELECTION_RULE,
    "acc_limit": ACC_LIMIT,
    "plausible_bound": PLAUSIBLE_BOUND,
    "n_trials": int(len(trials)),
    "confirm": confirm.reset_index().to_dict("records"),
    "median_rmsse": confirm.loc[selected_role, "test_rmsse"],
    "median_e20": confirm.loc[selected_role, "test_e20"],
    "median_rmsse_valid": confirm.loc[selected_role, "valid_rmsse"],
    "median_e20_valid": confirm.loc[selected_role, "valid_e20"],
    "empirical_all_valid": empirical_all,
    "saved_model": "05_selected",
    "forecasts": "05_selected",
    "forecast_column": "GFM_selected",
})

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## Discussion
#
# * Did accuracy keep its order from validation to test? Did elasticity? If elasticity shrank,
#   was it the window (a different 14 weeks, perhaps with Easter in it) or the settings?
# * The rule was fixed before the trials. What would have gone wrong if we had looked at the
#   front first and then decided what "good enough" accuracy meant?
# * Three seeds per trial is the minimum. The research found a 3-seed mean picked extremes that
#   did not hold up, and advised 6+ seeds for elasticity claims (§6). How many would you want
#   before quoting an elasticity to a pricing team?
#
# **Try it:** change `CONFIRM_SEEDS` to five seeds and rerun only section 7. The choice cannot
# change, because it was made on validation. Does the selected setting's test elasticity move,
# and does its seed spread shrink?
