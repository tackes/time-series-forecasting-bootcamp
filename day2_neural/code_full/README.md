# Day 2, full data (`code_full`)

These notebooks are the Day 2 labs rerun on **every eligible series** (≈17,241), scored on the same **2,000-series evaluation sample** as `../code`, so the numbers line up with the original labs.

Each notebook builds on the one before: it reads the previous notebook's config from `results/day2_full/configs/` and writes its own. The models were trained in advance on a GPU. By default every notebook **loads** the saved results, so it reads top to bottom in minutes on a laptop.

| # | Notebook | Starts from | Adds |
|---|---|---|---|
| 00 | `00_recap_full` | `../v1/code/00` | full panel, eval ids, dropped-series report, example series, median-led board, baseline error and win charts |
| 01 | `01_gfm_recipe` | `../v1/code/01` | NBEATSx and NHITS with the research training recipe off vs on, seed spread, training health |
| 02 | `02_hpo_round1` | `../v1/code/02` | accuracy-only grid, 3 seeds per cell, noise floor, shortlist |
| 03 | `03_calendar_hierarchy` | `../v1/code/03` (part) | calendar and event features (Easter, Anzac Day, Christmas, Mother's Day), hierarchy statics |
| 04 | `04_price_response` | `../v1/code/03` (part) | single-channel vs compound price sweeps, CPI deflation, oversampling, probes, elasticity charts |
| 05 | `05_hpo_round2_pareto` | new | two-objective search (accuracy, price response), Pareto front and knee, confirm run |
| 05b | `05b_synthetic_discounts` | new | synthetic price changes in the training data |
| 06 | `06_cluster_gfms` | new | pooled vs pooled with a cluster static vs one model per cluster |
| 07 | `07_foundation` | `../v1/code/04` | Chronos-2 against the selected GFM |
| 08 | `08_hierarchy_reconcile` | `../v1/code/05` | the selected GFM as the item-level base, reconciled up the hierarchy |
| 09 | `09_uncertainty` | `../v1/code/06` | quantile GFM, conformal intervals by class and horizon |
| 10 | `10_selection_standings` | `../v1/code/07` | final board on accuracy and price response |

**Helpers:**
- `gd_helpers.py`: paths, data, configs, training recipe, fitting and caching.
- `gd_price.py`: price, events, oversampling, sweeps, elasticity.
- `gd_plots.py`: figures F1–F12.

**Research notes:** `../../docs/gfm_learnings.md`. **Contributor notes:** `AGENTS.md`.
**Results so far:** `../../docs/day2_full_findings.md`. **Why prices are cleaned and the GFM's
inputs floored and clipped** (a plain-language explainer): `../../docs/stability_choices.md`.

**Set-up**
```bash
uv venv --python 3.11 .venv
uv pip install -r day2_neural/code_full/requirements.txt
```

**Saved models:** download them into `models/day2_full/` (link to be added) to predict live. They aren't needed to read the notebooks.

**Data:** `data/cpi_weekly_deflator.csv` is a weekly CPI deflator, interpolated from the national all-groups CPI and set to 1 in the last data week (2025-07-16).
