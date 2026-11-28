# Day 1: machine learning forecasting

Four-hour, instructor-led session. Global LightGBM models with Nixtla MLForecast, statistical baselines with StatsForecast, evaluation with utilsforecast, on a 2,000-series weekly retail sample with a 14-week horizon.

## Workshop sequence

The session takes the notebooks in numerical order.

| step | notebook | live material | about |
|---|---|---|---|
| 1 | `01_baselines` | all | 15 min |
| 2 | `02_ml_baseline` | all | 25 min |
| 3 | `03_exogenous_features` | sections 1 to 10: price and promotion features, future drivers, historical drivers, results | 35 min |
| 4 | `04_global_vs_specialized` | up to "Optional" | 20 min |
| 5 | `05_horizon_strategies` | all | 20 min |
| 6 | `06_objectives` | sections 1 to 7 | 25 min |
| 7 | `07_model_selection` | sections 1 to 5 | 45 min |
| 8 | `08_final_leaderboard` | value of information, leaderboards, Day 2 handoff | 15 min |

Notebooks 02 to 05 keep the L2 objective of the 02 baseline, so each changes only what it is about. Notebook 06 introduces the other objectives (L1, Tweedie, quantile), relates them to the skewness and intermittency of the data, and refits the main notebook 03 models with L1 (section 7), so model selection can choose between both.

About 200 minutes, leaving time for a break and questions.

### Information and business scenarios

Every forecast is labeled with what it needs to know on the cutoff date: sales history and calendar only, historical exogenous features, partial or full future-driver visibility, or leakage diagnostic (`shared/README.md`). A retailer forecasting its own sales knows its price and promotion plans, so models that read them are legitimate in that scenario; a manufacturer without those plans can use only history. Notebook 07 selects within each scenario, and notebook 08 shows both what future drivers are worth and the best deployable forecast in each scenario. Leakage diagnostics are never deployable.

Each notebook reads the forecasts that earlier notebooks saved in `results/day1/forecasts/`, so any notebook can be opened on its own and in this order. Sections headed "Optional" are not part of the live path; they hold further experiments for after the session:

- `03b_promotion_spikes`: what happens around promotions, and spike-adjusted training with the promotion plan.
- `03c_behavior_features`: demand-distribution, promotion-response, price-response and hierarchy-demand features on top of the historical drivers of 03.
- `03d_catboost_and_categorical_features`: CatBoost against LightGBM on the same features, with item attributes as native categories.
- Optional sections of 03, 04 and 07: a four-week promotion plan, the leakage diagnostics kept for Day 2 comparisons, partitioned models, and in 07 sections A to F (selection levels and costs, penalties, candidate pools, hindsight bound, ensembles). Notebook 07's executed outputs can be read without running them.

## Running the notebooks

- **Colab**: open a notebook and run the first cell. It clones the repository, installs the pinned packages and moves to the repository root. The repository must be public.
- **Locally**: install `requirements.txt` (Python 3.12), start Jupyter in the repository root or in `day1_ml/`, and run the cells in order.

Full run times from a fresh kernel on the development machine (all sections, including the optional ones): 01 under a minute, 02 under a minute, 03 about three, 03b about one, 03c about three and a half, 03d about nine, 04 about three and a half, 05 about one and a half, 06 about two, 07 about 19 (about four for sections 1 to 5) and 08 about one. The whole sequence takes about 45 minutes.

Results, the leaderboard and the Day 2 handoff files are described in `results/day1/README.md`; the shared code in `shared/README.md`.
