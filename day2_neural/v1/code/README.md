# Packt Nov 2026 · Day 2 · Deep Learning, Foundation Models and Advanced Topics

Materials for Day 2 of the Packt Nov 2026 Time Series Bootcamp (28–29 Nov 2026; 4 hours). They use the weekly retail dataset in
`../data` (see `../README.md`) and continue the Day 1 leaderboard: same test window, same score.
Interactive figures: https://time-series-modeling.github.io/workshops/packt-2026-11/day2

## Agenda and files

| Time | Min | Segment | Type | Material |
|---|---|---|---|---|
| 0:00 | 10 | Recap and handoff: Day 1 leaderboard | talk + notebook | `00_recap_leaderboard` |
| 0:10 | 25 | Deep learning for global models; ML vs NNs: scaling, evaluation, losses | talk | slides |
| 0:35 | 20 | Default training with GFMs: with/without decomposition, wrangling, evaluative splitting | lab | `01_lab_gfm_default_training` |
| 0:55 | 30 | Hyperparameter optimisation: what is worth tuning | lab | `02_lab_hpo` |
| 1:25 | 15 | Break | | |
| 1:40 | 30 | Exogenous variables and responsiveness: price/promo response curves | lab | `03_lab_exogenous_response` |
| 2:10 | 10 | Foundation models: zero-shot vs fine-tuned, covariates, cost | talk | slides |
| 2:20 | 30 | Foundation models: fine-tune or not | lab | `04_lab_foundation_models` |
| 2:50 | 20 | Hierarchy and coherence: reconciliation types and performance | lab | `05_lab_hierarchy_coherence` |
| 3:10 | 15 | Uncertainty: proper scoring rules, conformal prediction | talk + demo | `06_demo_uncertainty` |
| 3:25 | 20 | Model selection and ensembling: win rates, Elo / Bradley–Terry, by shape and volume | lab | `07_lab_model_selection` |
| 3:45 | 10 | Wrap-up: final standings, take-home | | `07_lab_model_selection` (last section) |

* `08_bonus_delay_embeddings` — **bonus, not in the schedule**: why can these series be forecast at all? Delay
  embeddings (Takens), phase portraits, "similar states, similar futures" against shuffled series, nearest windows from
  other items, recurrence plots, τ/m choice and S-map surrogates. Interactive version:
  https://time-series-modeling.github.io/workshops/packt-2026-11/day2/embedding (about 5 minutes on 4 cores with all series).
* `slides/day2_slides.md` — the 29-slide lecture deck (about 65 minutes of talk); `slides/day2_slides_full.md` is the
  longer 53-slide version with extra background slides. Both are [Marp](https://marp.app) Markdown, with speaker
  notes in the HTML comments. Export with `npx @marp-team/marp-cli slides/day2_slides.md --pdf` (or `--pptx`), or open it in
  VS Code with the Marp extension. `slides/day2_slides.pptx` is an editable copy (native text, tables and
  notes) that imports into PowerPoint or Google Slides.
  Rebuild it with `node slides/build_pptx.js slides/day2_slides.md slides/day2_slides.pptx` (needs `pptxgenjs`;
  see the note at the top of the script) after `python slides/make_figures.py` has refreshed the figures.
* Every content slide ends with a one-line **Takeaway**; the last slide is the take-home summary.
* Every topic has an **Intuition** slide (plain language, an analogy and a simple picture before any formula),
  and every lab has a **What we found** slide that reads the reference-run results and explains why they came
  out that way. The notebooks carry the same material as "Intuition" and "Reading the result" cells.
  The pictures are in `slides/img/`, made by `python slides/make_figures.py`. The result charts read
  `slides/reference_results.json`, which `python slides/collect_results.py RUN_DIR slides/reference_results.json`
  collects from a run's executed notebooks (see the script's docstring); rerun both after rerunning the labs.
* Each notebook exists as `.ipynb` and as a [jupytext](https://jupytext.readthedocs.io) percent-format `.py`
  (the same convention as the book chapters). Edit the `.py`, then `jupytext --sync` or
  `jupytext --to ipynb *.py`.
* `bootcamp.py` — the shared harness: data loading, the split, scoring and the leaderboard.

## The competition

| | |
|---|---|
| Test window | final 14 weeks, from 2025-04-16 (`bc.TEST_START`) |
| Validation window | the 14 weeks before that (`bc.VALID_START`), for tuning and early stopping |
| Frequency | `W-WED` |
| Primary score | mean RMSSE, scaled by the in-sample 52-week seasonal-naive error |
| Also reported | median RMSSE, mean and median MASE, sMAPE (0–1 scale), WAPE and bias % (volume-weighted), fit seconds |
| Metric code | `utilsforecast.losses`, the evaluation library of the Nixtla stack the labs use; RMSSE, MASE and MAE cross-checked against `sktime` on 1,000 series (identical to 1e-13). Lab 06 scores intervals with the same library (coverage, Winkler score, scaled multi-quantile loss) |
| Series | items with ≥ 66 training weeks, positive prices and non-constant sales; all 17,241 eligible series by default (of 18,692) |

Every lab calls `bc.record(forecasts, test, train, lab=...)`, which scores the forecasts and upserts one row per
model into `../results/day2/leaderboard.csv`, and `bc.save_forecasts(...)`, which stores them in
`../results/day2/forecasts/` for the model-selection lab. If a `../results/day1/leaderboard.csv` exists, it is
shown alongside the Day 2 rows.

Run the notebooks in order. `00` must run first (it creates the reference entries); `07` reads everything the
others saved, so it works with whichever labs were run.

## Metrics and saved forecasts

### Saved forecasts: compute any metric later

Every lab writes what it predicted, with the actuals, to `../results/day2/raw/<window>/<lab>.parquet`
(`bc.save_raw`; read back with `bc.load_raw(window)`). Columns: `unique_id`, `ds`, `cutoff` (backtests only), `y`,
then one column per model plus `<model>-lo-<level>` / `<model>-hi-<level>` where the model gives intervals.

| Window | Files |
|---|---|
| `test` (the leaderboard window) | `00_day1_reference`, `01_gfm_default`, `02_hpo`, `03_exogenous`, `04_foundation` (with 80/95 % intervals), `05_hierarchy_all_levels` (every node from total to item, every reconciler), `06_uncertainty` (four interval methods), `07_ensembles` |
| `valid` (the 14 weeks before) | `02_hpo_grid` (every NHITS setting and seed of the sensitivity study), `04_foundation` (foundation models with intervals, SeasonalNaive, NHITS) |
| `cv` (rolling origins on the training data) | `01_gfm_default` (NHITS, 2 folds), `06_uncertainty_lightgbm` (LightGBM, 3 folds) |

`../results/day2/forecasts/` keeps the test point forecasts that Lab 07 pools. The reference run's whole
`results/day2` folder (raw forecasts, leaderboard, `metrics_test.csv`; 211 MB) is attached to the release
[day2-reference-run-2026-11-21](https://github.com/emptymalei/tswpt-workshops/releases/tag/day2-reference-run-2026-11-21):
unzip it into `Packt_conference_2026-11/public/` to score the reference forecasts without running anything.

### The metric suite

`bc.metric_suite(df, train)` scores one window and returns one row per model. Every number comes from
`utilsforecast.losses` or `sktime`; the harness only averages per-series values or takes ratios of them. The
leaderboard (`bc.record`) keeps its short list: mean and median RMSSE and MASE, sMAPE, WAPE and bias %.

| Family | Columns | Library | Why |
|---|---|---|---|
| Scaled, per series | `mean_rmsse` (primary), `median_rmsse`, `mean_mase`, `median_mase` | utilsforecast | Comparable across items of any size. RMSSE (squared) does not reward forecasting zero on intermittent items, MASE (absolute) does; the median shows the typical item |
| Relative to SeasonalNaive | `avgrelmae`, `avgrelrmse` (geometric mean over series of the MAE / RMSE ratio), `share_beats_baseline` | utilsforecast `mae`, `rmse` | "How much better than the naive plan" in one number; the geometric mean is the recommended way to average ratios (Davydenko & Fildes, 2013) |
| Percentage | `mean_smape` (0–1 scale), `mean_maape`, `mean_rmsle` | utilsforecast, sktime | All three are defined when sales are 0. MAAPE (arctangent of the percentage error) was designed for intermittent demand |
| Pooled, volume-weighted | `pooled_wape`, `pooled_mae`, `pooled_rmse` (units per week) | utilsforecast | Best sellers count most: the view of total volume |
| Bias | `bias_pct` (sum of ŷ − y over sum of y; + = over-forecast), `share_series_over` | utilsforecast `bias`, `cfe` | Accuracy and bias are different questions: excess stock vs lost sales |
| Demand type | `rmsse_smooth`, `rmsse_erratic`, `rmsse_intermittent`, `rmsse_lumpy` | utilsforecast | Syntetos–Boylan classes from `bc.series_info`; the intermittent items carry much of the mean |
| Intervals (where present) | `coverage_80/95`, `scaled_winkler_80/95`, `scaled_width_80/95`, `scaled_crps` | utilsforecast `coverage`, `winkler_score`, `scaled_mqloss` | Coverage is a diagnostic, the Winkler (interval) score and CRPS are proper scores. CRPS is approximated by twice the scaled pinball loss averaged over the available quantiles (2.5, 10, 50, 90, 97.5 %) |

Pitfalls we hit, and how the harness handles them:

* **sMAPE scale.** utilsforecast's sMAPE is the mean of `|y − ŷ| / (|y| + |ŷ|)`, from 0 to 1: exactly half of the
  "0–200 %" convention of sktime and the M3/M4 papers. We keep it and say so; double it to compare with papers.
* **sktime's RMSLE class.** In sktime 1.3, `MeanSquaredLogError(square_root=True)` ignores `square_root` and returns
  the MSLE. The functional `mean_squared_log_error(..., square_root=True)` is correct, so the harness uses the
  functional API (which is also about 2,000× faster on 17,000 series).
* **Zero scale.** A series that repeats exactly 52 weeks apart has a zero seasonal-naive scale and an infinite
  RMSSE; it is dropped (3 of 17,241 series, hence `n_series` 17,238).
* **Negative forecasts** are clipped at 0 before scoring.
* RMSSE, MASE and MAE from utilsforecast were checked against sktime on 1,000 series: identical to 1e-13. The
  library interval scores reproduce the formulas Lab 06 used to write by hand.

Left out on purpose:

* **MAPE, MdAPE, GMRAE**: undefined or unbounded when actual sales (or the naive error) are 0, which is most weeks
  for intermittent items.
* **utilsforecast `scaled_crps`**: it divides by the sum of actuals in the test window, which explodes for an item
  that happens to sell nothing in those 14 weeks. `scaled_mqloss` (seasonal-naive scale, as in the M5 uncertainty
  track) is used instead.
* **utilsforecast `pis` / `spis`**: despite the name ("periods in stock"), they sum *absolute* errors, so they are
  not the signed bias measure of the inventory literature; `bias_pct` and `share_series_over` cover bias.
* **Tweedie deviance**: utilsforecast requires strictly positive forecasts, and a forecast of exactly 0 in a week
  with sales has infinite deviance, which SeasonalNaive and the reconciled models produce often.

## Setup

```bash
pip install neuralforecast statsforecast mlforecast utilsforecast hierarchicalforecast \
            polars pyarrow pandas matplotlib lightgbm optuna jupytext
pip install timecopilot          # lab 04 only; pulls the foundation-model dependencies
pip install pyEDM teaspoon neurokit2 pyunicorn scikit-learn joblib scipy   # bonus notebook 08 only
```

Lab 04 downloads model weights from Hugging Face on first use (Chronos-2 about 0.5 GB, TimesFM 2.5 about 1 GB). Ask participants to run
its first cell before the break.

### Size and speed knobs

Set these environment variables before starting Jupyter; every notebook must use the same `BOOTCAMP_N_SERIES`.

| Variable | Default | Meaning |
|---|---|---|
| `BOOTCAMP_N_SERIES` | `0` | `0` = all 17,241 eligible series; a positive number draws a stratified sample of that size for slow machines |
| `BOOTCAMP_MAX_STEPS` | `500` | training steps for every neural model |
| `BOOTCAMP_ACCELERATOR` | `auto` | `cpu`, `gpu` or `auto` |
| `FM_N_SERIES` | `0` | lab 04 only: `0` = all eligible series; a positive number runs the foundation models on a random subset of that size |

Measured on a 4-core laptop CPU with no GPU, on all 17,241 series (the default):

| Notebook | Run time |
|---|---|
| `00` recap | 1 min |
| `01` default GFMs | 2.5 min |
| `02` HPO (4 sensitivity parameters + 3 seeds + 8 Optuna trials; `RUN_ALL = True` adds ~12 min) | 18 min |
| `03` exogenous | 3 min |
| `04` foundation models (TimesFM ≈ 70 min, Chronos-2 fine-tuning ≈ 30 min, both windows; `FM_N_SERIES=1000` ≈ 12 min) | 1 h 55 min |
| `05` hierarchy (19,334 nodes, sparse reconcilers) | 1.5 min |
| `06` uncertainty | 3 min |
| `07` selection + standings | 1 min |

The committed `.ipynb` files contain the outputs of that run, so instructors can see the expected results.
Notebook `04` was run with `timecopilot` 0.0.42 and needs internet access to Hugging Face on first use; in the
room, have participants on a CPU set `FM_N_SERIES=1000` unless they have two hours to spare.
With a GPU, `BOOTCAMP_MAX_STEPS=1500` gives results closer to `nbeats_weekly_phase3.py`. On a slow laptop,
`BOOTCAMP_N_SERIES=2000` runs every lab on a stratified sample in a fraction of the time (set it for every notebook).
