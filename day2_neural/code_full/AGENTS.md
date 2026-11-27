# Working on `day2_neural/code_full`

Notes for anyone (person or coding agent) editing these notebooks.

## What this folder is
A full-data version of the Day 2 labs. Each notebook trains on **every eligible series** (≈17,241) and scores on the **fixed 2,000-series evaluation sample** used by `day2_neural/code`. Each one builds on the config written by the one before. The research behind the choices is summarised in `docs/gfm_learnings.md`.

## Ground rules
- **`day2_neural/v1/code/` is read-only** (it was `day2_neural/code/` before the v1 update). `bootcamp.py` is imported, never edited. `gd_helpers.py` points its paths at `results/day2_full/`; nothing is ever written to `results/day2/`.
- **Same evaluation.**
  - Score with `gd.record` / `gd.score`. They restrict `bc.record` / `bc.score` to the eval ids.
  - Sort and discuss the board by **median** RMSSE, with the mean alongside.
  - Show it with `gd.median_board("<NN>")`, passing the notebook's own number ("07b"): a
    notebook's board shows only its own and earlier notebooks' rows, never later ones.
- **Tune on validation, test once.**
  - HPO and model choice use the window from `bc.VALID_START`. Pass `cutoff=` explicitly to `bc.split`.
  - Per-series statistics use training weeks only: scaling, statics, oversampling counts, the reference price and clusters.
- **Configs carry ideas forward.** Each notebook writes `results/day2_full/configs/NN_<name>.json` and reads the previous one rather than hardcoding settings.
- **Train once, load on the day.**
  - `GD_TRAIN=1` fits and saves; the default only loads.
  - Heavy work sits inside `gd.fit_seed(s)` or `gd.cached(...)`, which cache per run, so a crash resumes.
  - A TRAIN run reuses any cached table or run already on disk. `GD_REFRESH=1` makes `gd.cached` recompute its tables; delete a `runs/` folder to refit it.
  - Each `gd.cached` computation stands alone: it must not read globals that another `cached` call sets.
- **Saving and size.**
  - No Lightning checkpoints or logs, and one device per fit (`gd.trainer_kwargs()`). `BestWeights` keeps the best weights in memory.
- **One GPU.** Build every neural model's trainer settings from `gd.trainer_kwargs()` (or `gd.recipe_kwargs`), never `bc.trainer_kwargs()` directly. Lightning's default trains on every visible GPU, which hangs inside a notebook on a multi-GPU box.
  - HPO trials save summaries, not models.
  - `save_as=` (which uses `nf.save(save_dataset=False)`) only for models a later notebook or a live demo loads, one seed each: `01_nbeatsx_recipe`, `05_selected`, `05b_synthetic`, `09_quantile`. Models go to `models/` (gitignored, hosted outside the repo). A model is saved the moment it is fitted, never held for later: 05 saves each confirm setting's first seed (`05_confirm_<key>`) and copies the winner to `05_selected`.
  - `fit_seed` frees each fitted model once its results are on disk (each one holds a copy of the training data). Need the model? Save it with `save_as=` and `gd.load_model` it.
  - Every file is written atomically, so a crash never leaves a half-written file for a LOAD run.
- **Pulling results into a notebook.** `gd.list_runs(prefix)` lists every cached fit; `gd.load_runs(name)` reads all seeds of one, in the shape `fit_seeds` returns. Use them in demo cells rather than refitting.
- **HPO studies resume.** Run Optuna studies with `gd.run_study`: it requeues a trial a crash interrupted, retries a trial that ran out of GPU memory, and loads finished seeds from the cache.
- **Figures are displayed.**
  - Call `gplt.use_style()` once, and use the `gd_plots` functions (F1–F12; see its docstring).
  - Each `.ipynb` stays under ~3 MB.
- **3 seeds minimum** for any quoted number (`gd.SEEDS`).
- **Measuring price response.** Two methods, both in `gd_price`:
  - the sweep (`price_sweep`, `e20`): moves `price_index` alone over all 14 weeks. 04 and 05
    chose with it, so it stays in their configs.
  - the phase 7b probes (`price_probes`, `promo_calendars`, `calendar_probe`): move the price
    and rebuild every input derived from it (`discount_depth`, `price_ratio_to_rolling_mean`)
    over history and horizon, as a real discount week has them. Use these for anything new.
    The headline is a one-week 20% cut (`pulse_all`, test week 5, after the event weeks). Any
    model works through a `predict(futr)` function (`gp.gfm_predict` for a GFM). Check
    `gp.rebuild_check` is ≈ 0 before trusting a probe on new data.
  - Probes need the model in memory: compute them as `fit_seed` extras, or from a saved model.
- **Chronos-2.** Load it with `gd.chronos_pipeline()` and forecast long frames with
  `gd.chronos_forecast` (07 and 08 share them), so every notebook uses the same model and
  settings.
- **Hierarchy (08).** Upper nodes get their items' sales-weighted price and promotion inputs
  from `gp.aggregate_covariates` (weights from training weeks only). MinT's past errors come
  from `gd.rolling_one_step`; `gd.floor_residual_sd` guards a node whose errors are exactly
  zero.
  - Use `gd.MinTraceDirect`, never the library's `MinTraceSparse`, for sparse MinT. The
    library's iterative solve ignores non-convergence and gave forecasts in the millions under
    `wls_var` here; `MinTraceDirect` solves exactly.
  - An arm whose chosen method is BottomUp and whose items are already on the board
    (Chronos-2) is not added again.
- **Newer foundation models (07b).** Timestral and t0-gamma load with `gd.foundation_model`
  and forecast long frames with `gd.foundation_forecast`, beside Chronos-2. Their main
  setup (the price covariates) was fixed before the results and goes on the board; the
  other setups stay in 07b's tables. Timestral's weights are not free for commercial use
  above $20M monthly revenue.
- **All series (11).** 11 rescores the main models on all 17,238 scorable series and writes
  its own board (`leaderboard_all_series.csv`) and forecasts (`results/day2_full/all_series/`),
  never `forecasts/` (10 would read them). Every model there is capped at 3× the item's
  largest training week, as in 08: some N-BEATSx forecasts explode outside the sample.
- **Public repo.** No client names, internal system names, people's names or client-identifying figures.

## Style
- Jupytext percent format (`# %%`, `# %% [markdown]`). Edit the `.py`, then sync the `.ipynb`.
- Plumbing goes in `gd_helpers.py` / `gd_price.py` / `gd_plots.py`: short functions with docstrings. Model set-up, features and results stay visible in cells.
- Lines at most 100 characters; imports at the top of the notebook; no tuple-expression one-liners, nested ternaries or dense comprehensions.
- Prose in British English ("learnt"), in the structure of the original labs:
  - title with minutes, a short introduction, then `Goals` (no colon) with capitalised numbered goals, then `**In the chain:** reads <config>, writes <config>.`;
  - `## Intuition:`;
  - numbered sections, with `### Reading the result` after key outputs;
  - `## Discussion` and `**Try it:**`.
- Prose not yet reviewed carries `<!-- DRAFT: Graeme to edit -->`.

## Running
```bash
# CPU smoke test (tiny panel, few steps; writes results/day2_full_smoke):
GD_SMOKE=1 GD_TRAIN=1 BOOTCAMP_ACCELERATOR=cpu jupyter nbconvert --to notebook --execute <nb>.ipynb --output /tmp/smoke.ipynb
# Unit tests:
GD_SMOKE=1 python -m pytest day2_neural/code_full/tests -q
```
See `docs/progress.md` for the GPU commands.

## The config chain
Every notebook reads the config(s) before it and writes its own to `results/day2_full/configs/`. Keys that `gd_price.prepare(cfg)` understands are marked ★. Passing any config to `prepare` rebuilds exactly that notebook's training data.

| Config | Written by | Keys |
|---|---|---|
| `00_baselines` | 00 | `n_train_series`, `n_eval_series`, `examples` (tier → ids), `baselines`, `median_rmsse`, `mean_rmsse` |
| `01_recipe` | 01 | `params` (NBEATSx, see `gd.NBEATSX_DEFAULTS`), `recipe`, ★`y_shift`, ★`futr_exog`, `stat_exog`, `max_steps`, `seeds`, `minutes_per_fit`, `median_rmsse`, `seed_spread`, `saved_model` |
| `02_shortlist` | 02 | `base_params`, `grid`, `max_steps`, `seeds`, `noise_floor`, `shortlist` (list of `{params, median_rmsse_valid, seed_sd}`), `best_params`, ★`futr_exog`, ★`y_shift` |
| `03_features` | 03 | `params`, ★`features` (`deflate`, `events`, `fourier_k`, `hierarchy_level`), ★`futr_exog`, ★`y_shift`, `max_steps`, `seeds`, `median_rmsse` |
| `04_price` | 04 | `params`, ★`features`, ★`futr_exog`, ★`y_shift`, ★`oversample` (`rule`, `cap`, `multiplier`) or null, `price_channel`, `elasticity` (arm → test report), `elasticity_valid` (arm → validation report), `median_rmsse`, `median_rmsse_valid`, `empirical`, `empirical_band`, `selected_arm` (chosen on validation), `max_steps`, `seeds` |
| `05_selected` | 05 | `params`, ★`features`, ★`futr_exog`, ★`oversample`, ★`y_shift`, `max_steps`, `seeds`, `trial`, `role` (chosen on validation), `selection_rule`, `confirm`, `saved_model`, `forecasts`, `forecast_column` |
| `05b_synthetic` | 05b | as `05_selected`, plus ★`synthetic` (`cuts_per_copy`), `use` (`"05"` or `"05b"`, chosen on validation with the phase 7b probes), `checks`, `checks_test`, `probes`, `calendar`, `probe_runs` (runs holding 3-seed probes of 05 and 05b), `forecasts`, `forecast_column` |
| `05c_scaler_floor` | 05c | as `05_selected`, plus `scale_floor` (0.01), `use` (`"floor"` or `"cap"`), `checks`, `valid`, `explosions` (item-weeks above the cap, every series), `test_scores`, `test_response`, `runs`, `forecasts`, `forecast_column` (`GFM_floor`) |
| `05c_price_clean` | 05c | as `05_selected`, plus ★`price_clean` (`k`, `floor`: `gp.clean_prices`), `scale_floor`, `replaced`, `recommended` (`"clean"`, `"clean_floor"` or `"05"`), `cap_kept`, `checks`, `valid`, `explosions`, `test_scores`, `test_response`, `runs`, `forecasts`, `forecast_column` |
| `05c_input_clip` | 05c | as `05c_price_clean`, plus `input_clip` (50: `gd.floor_robust_scaler(model, floor, clip=)`), `recommended` (the rule's pick: `"clean_floor"`), ★`use` (`"clean_floor_clip"`: carried forward, `chosen_by`, `choice_reason`), `saved_model` (`05c_selected`), `checks`, `valid`, `explosions`, `test_scores`, `test_response`, `runs`, `forecasts`, `forecast_column` |
| `07b_new_foundation` | 07b | `models`, `setups`, `main_setup`, `board_columns`, `chronos_gap_to_07`, `median_rmsse_valid` / `median_rmsse_test` (model → setup), `wilcoxon`, `hard_weeks`, `history`, `probes`, `calendar`, `headline`, `pairs`, `seconds`, `licences` |
| `06_clusters`, `07_foundation`, `08_reconcile`, `09_intervals`, `10_final` | 06–10 | free-form, documented in the notebook |
| `11_all_series` | 11 | `models`, `cap`, `repeat_check`, `scores_all`, `scores_sample`, `scores_all_no_cap`, `versus_v1`, `left_out` |
| `12_price_plan` | 12 | `plans` (`realized`, `no_fingerprint`, `no_change`), `share_changed`, `check_realized_vs_board`, `median_rmsse` per model and plan, `fingerprint_matters`, `scores`. Report only |

Notebooks after 05 call `gd.selected_config()`. It returns 05c's chosen arm if `05c_input_clip` has a `use` (05's config plus `price_clean`, `scale_floor` and `input_clip` in `params`, `saved_model: 05c_selected`, `selected_runs`), else `05b_synthetic` if its `use` is `"05b"`, else `05_selected`; `selected_from` says which. `gd.build_nbeatsx` applies `scale_floor`/`input_clip` from `params`; `gd.save_model` writes `stability.json` next to a patched model and `gd.load_model` reapplies it.
- **Cleaned prices everywhere after 05:** every model in the final comparison reads prices cleaned with `gp.PRICE_CLEAN` (LightGBM in 00, 04 and 11 via `gp.clean_prices`; everything built from `gp.prepare(selected_config())`). 01–05 keep raw prices. Why: `docs/stability_choices.md`. The raw-price caches of 00, 04's LightGBM and 06–11 are in git at `b76e787` (and locally in `results/day2_full_archive_raw_prices/`).
- **Forecasts:** read them with `gd.selected_forecasts()`, which returns the column as `GFM_selected` whichever notebook won. 05b saves its own file and column (`GFM_synthetic`). `gd.floor_robust_scaler(model, floor)` floors the robust scaler's spread on a built model (05c); it patches the instance, so a model loaded from disk needs it applied again, named in its config's `forecasts` and `forecast_column` keys.
- **Tuning data:** use `gp.prepare(cfg, cutoff=bc.VALID_START)`. It returns the pre-validation training rows and the validation window, with every per-series statistic computed before the cutoff.
- **Cached runs go stale.** Runs and tables are cached by name, not by config. If an upstream config changes, delete the cached `runs/` and `tables/` entries of every later notebook before retraining. On the GPU box, always run the notebooks in order.
