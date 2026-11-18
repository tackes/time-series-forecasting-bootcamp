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

* `slides/day2_slides.md` — the 28-slide lecture deck (about 65 minutes of talk); `slides/day2_slides_full.md` is the
  longer 52-slide version with extra background slides. Both are [Marp](https://marp.app) Markdown, with speaker
  notes in the HTML comments. Export with `npx @marp-team/marp-cli slides/day2_slides.md --pdf` (or `--pptx`), or open it in
  VS Code with the Marp extension. `slides/day2_slides.pptx` is an editable copy (native text, tables and
  notes) that imports into PowerPoint or Google Slides.
  Rebuild it with `node slides/build_pptx.js slides/day2_slides.md slides/day2_slides.pptx` (needs `pptxgenjs`;
  see the note at the top of the script) after `python slides/make_figures.py` has refreshed the figures.
* Every content slide ends with a one-line **Takeaway**; the last slide is the take-home summary.
* Every topic has an **Intuition** slide (plain language, an analogy and a simple picture before any formula),
  and every lab has a **What we found** slide that reads the reference-run results and explains why they came
  out that way. The notebooks carry the same material as "Intuition" and "Reading the result" cells.
  The pictures are in `slides/img/`, made by `python slides/make_figures.py` (the result charts use the
  reference-run numbers listed at the top of that script; update them if you rerun the labs).
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
| Also reported | median RMSSE, MASE, sMAPE, WAPE, fit seconds |
| Series | items with ≥ 66 training weeks, positive prices and non-constant sales; a 2,000-series sample stratified by `sale_class` |

Every lab calls `bc.record(forecasts, test, train, lab=...)`, which scores the forecasts and upserts one row per
model into `../results/day2/leaderboard.csv`, and `bc.save_forecasts(...)`, which stores them in
`../results/day2/forecasts/` for the model-selection lab. If a `../results/day1/leaderboard.csv` exists, it is
shown alongside the Day 2 rows.

Run the notebooks in order. `00` must run first (it creates the reference entries); `07` reads everything the
others saved, so it works with whichever labs were run.

## Setup

```bash
pip install neuralforecast statsforecast mlforecast utilsforecast hierarchicalforecast \
            polars pyarrow pandas matplotlib lightgbm optuna jupytext
pip install timecopilot          # lab 04 only; pulls the foundation-model dependencies
```

Lab 04 downloads model weights from Hugging Face on first use (Chronos-2 about 0.5 GB). Ask participants to run
its first cell before the break.

### Size and speed knobs

Set these environment variables before starting Jupyter; every notebook must use the same `BOOTCAMP_N_SERIES`.

| Variable | Default | Meaning |
|---|---|---|
| `BOOTCAMP_N_SERIES` | `2000` | sample size; `0` = all ~17,000 eligible series |
| `BOOTCAMP_MAX_STEPS` | `500` | training steps for every neural model |
| `BOOTCAMP_ACCELERATOR` | `auto` | `cpu`, `gpu` or `auto` |

The defaults are sized so that the core cells of each lab finish inside its slot on a recent laptop CPU.
Measured on a 4-core CPU with no GPU at the default size:

| Notebook | Run time |
|---|---|
| `00` recap | 20 s |
| `01` default GFMs | 3 min |
| `02` HPO (4 sensitivity parameters + 8 Optuna trials; `RUN_ALL = True` adds ~6 min) | 13 min |
| `03` exogenous | 1.5 min |
| `04` foundation models | not measured: needs the Hugging Face download (see below) |
| `05` hierarchy | 1 min |
| `06` uncertainty | 1 min |
| `07` selection + standings | 20 s |

The committed `.ipynb` files contain the outputs of that run, so instructors can see the expected results.
Notebook `04` is committed without outputs: it was written against `timecopilot` 0.0.40 and Chronos-2's
fine-tuning API, but the Hugging Face model download was blocked where it was built, so it has not been run
end to end. Do a dry run on a machine with internet access before the workshop.
With a GPU, `BOOTCAMP_N_SERIES=0 BOOTCAMP_MAX_STEPS=1500` gives results closer to `nbeats_weekly_phase3.py`.
