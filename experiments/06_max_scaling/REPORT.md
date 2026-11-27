# Max scaling for Tweedie

Research experiment, formerly optional section 8 of `day1_ml/06_objectives.ipynb`. It was moved here when that notebook was restructured for teaching. Code: `max_scaling.py`. Tables: `results/`. The forecasts in `results/day1/forecasts/06_max_scaled.parquet` came from this code; the script reproduces them (within float32 rounding) without overwriting them. They are not in the leaderboard or in the notebook 07 selection pools.

**Question.** Tweedie needs a non-negative target, so notebook 06 compares it only in raw units, because the z-score scaler makes the target negative. Does a non-negative per-series scale help Tweedie, and can it then compete with L1 on the same scale?

**Answer.** No. Max-scaled Tweedie is worse than raw Tweedie, and on the same scale L1 beats it at every origin.

## Setup

- **max scaling** `y / max(y)` per series, using the training rows at each origin. It keeps zeros at zero and puts every training value in [0, 1]. A series that never sold is divided by 1. MLForecast's `LocalMinMaxScaler` computes `(y - min) / (max - min)`, which differs from this when a series' minimum is above zero, so the script defines a small custom transform.
- Same lags, rolling means, calendar features, LightGBM parameters and ten origins as notebook 06. Only the scale and the objective change.

## Results (development origins; final window in brackets)

| model | scale | objective | WAPE | bias | mean RMSSE |
|---|---|---|---|---|---|
| `lgbm_l1_std` | z-score | L1 | 0.378 (0.369) | -0.039 (-0.003) | 1.71 |
| `lgbm_l1_max` | max | L1 | 0.390 (0.379) | -0.042 (-0.000) | 1.73 |
| `lgbm_tweedie_max` | max | Tweedie 1.3 | 0.491 (0.481) | +0.161 (+0.197) | 1.96 |
| `lgbm_l1_raw` | raw | L1 | 0.409 (0.375) | -0.143 (-0.073) | 1.76 |
| `lgbm_tweedie_raw` | raw | Tweedie 1.3 | 0.432 (0.455) | +0.008 (+0.147) | 1.92 |

- **Tweedie does not benefit from max scaling.** It is worse than raw Tweedie at all seven development origins, by 5.9 points on average, and in the final window (0.481 against 0.455). It also overforecasts more (bias +0.161 against +0.008).
- **On the same scale, L1 still wins clearly.** Max-scaled L1 beats max-scaled Tweedie by 10.1 points on average, and at every origin.
- **For L1 the scale matters little.** Max-scaled L1 is 1.2 points behind z-scored L1 on development, and better at three of seven origins, which is within the spread across origins. It does remove most of z-scored L1's bias on zero-heavy products (-0.10 against -0.70), at a slightly higher WAPE there (`results/dev_segments.csv`).
- **Where Tweedie is better.** This is retrospective, because the split uses the actual target-week sales. In weeks that turned out above three times the product's median, max-scaled Tweedie has the lowest WAPE (0.539 against 0.675 for z-scored L1) and underforecasts least. In every other week it is much worse (0.482 against 0.328), and those weeks are 88% of the rows (`results/dev_high_weeks.csv`). Tweedie forecasts higher everywhere, which helps in spike weeks and costs more in all the others.

A possible reason, not tested: Tweedie's variance assumption depends on the units of the target, and dividing each series by its own maximum changes that relationship series by series. Other variance powers are untested.
