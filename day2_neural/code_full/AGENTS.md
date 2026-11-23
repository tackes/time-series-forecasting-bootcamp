# Working on `day2_neural/code_full`

Notes for anyone (person or coding agent) editing these notebooks.

## What this folder is
A full-data version of the Day 2 labs. Each notebook trains on **every eligible series** (≈17,241) and scores on the **fixed 2,000-series evaluation sample** used by `day2_neural/code`. Each one builds on the config written by the one before. The research behind the choices is summarised in `docs/gfm_learnings.md`.

## Ground rules
- **`day2_neural/code/` is read-only.** `bootcamp.py` is imported, never edited. `gd_helpers.py` points its paths at `results/day2_full/`; nothing is ever written to `results/day2/`.
- **Same evaluation.**
  - Score with `gd.record` / `gd.score`. They restrict `bc.record` / `bc.score` to the eval ids.
  - Sort and discuss the board by **median** RMSSE, with the mean alongside.
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
| `05b_synthetic` | 05b | as `05_selected`, plus ★`synthetic` (`cuts_per_copy`), `use` (`"05"` or `"05b"`, chosen on validation), `checks`, `checks_test`, `forecasts`, `forecast_column` |
| `06_clusters`, `07_foundation`, `08_reconcile`, `09_intervals`, `10_final` | 06–10 | free-form, documented in the notebook |

Notebooks after 05 call `gd.selected_config()`, which returns `05b_synthetic` if its `use` is `"05b"` and `05_selected` otherwise.
- **Forecasts:** read them with `gd.selected_forecasts()`, which returns the column as `GFM_selected` whichever notebook won. 05b saves its own file and column (`GFM_synthetic`), named in its config's `forecasts` and `forecast_column` keys.
- **Tuning data:** use `gp.prepare(cfg, cutoff=bc.VALID_START)`. It returns the pre-validation training rows and the validation window, with every per-series statistic computed before the cutoff.
- **Cached runs go stale.** Runs and tables are cached by name, not by config. If an upstream config changes, delete the cached `runs/` and `tables/` entries of every later notebook before retraining. On the GPU box, always run the notebooks in order.
