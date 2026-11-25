# Day 1: machine learning forecasting

Four-hour, instructor-led session. Global LightGBM models with Nixtla MLForecast, statistical baselines with StatsForecast, evaluation with utilsforecast, on a 2,000-series weekly retail sample with a 14-week horizon.

## Workshop sequence

The notebooks keep their original numbers; the session takes them in this order.

| step | notebook | live material | about |
|---|---|---|---|
| 1 | `01_baselines` | all | 15 min |
| 2 | `02_ml_baseline` | all | 25 min |
| 3 | `06_objectives` | sections 1 to 7 | 25 min |
| 4 | `04_global_vs_specialized` | up to "Optional"; then the categorical section of `03_exogenous_features` | 25 min |
| 5 | `03_exogenous_features` | sections 1 to 5 (plan in the notebook) | 30 min |
| 6 | `05_horizon_strategies` | all | 20 min |
| 7 | `07_model_selection` | sections 1 to 5 | 45 min |
| 8 | `08_final_leaderboard` | leaderboard, Day 2 handoff, accuracy against stability | 15 min |

About 200 minutes, leaving time for a break and questions.

Each notebook reads the forecasts that earlier notebooks saved in `results/day1/forecasts/`, so any notebook can be opened on its own and in this order. Sections headed "Optional" are not part of the live path; they hold further experiments for after the session:

- `03b_promotion_spikes`: what happens around promotions, and spike-adjusted training.
- `03c_behavior_features`: demand-distribution, promotion, price and hierarchy-demand features.
- Optional sections of 03, 04, 06 and 07: CatBoost, partitioned models, max-scaled Tweedie, hierarchy-level selection, candidate pools, ensembles.

## Running the notebooks

- **Colab**: open a notebook and run the first cell. It clones the repository, installs the pinned packages and moves to the repository root. The repository must be public.
- **Locally**: install `requirements.txt` (Python 3.12), start Jupyter in the repository root or in `day1_ml/`, and run the cells in order.

Run times on one laptop CPU: most notebooks take one to three minutes. With their optional sections, 03 takes about ten, 03c about three, 04 about five and 07 about 21 (two for the live sections).

Results, the leaderboard and the Day 2 handoff files are described in `results/day1/README.md`; the shared code in `shared/README.md`.
