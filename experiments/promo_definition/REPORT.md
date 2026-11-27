# Promotion definition experiment

Exploratory. Nothing in `day1_ml/`, `shared/` or `results/day1/` was changed. Code: `promo_definition.py`; tables, forecasts and plots: `results/`.

**Question.** Does a business rule (count a week as a promotion only when it is flagged *and* discounted by at least a threshold) improve the full-plan forecast?

**Answer.** Slightly and fairly consistently, but by little more than refitting with a different random seed changes the result. Part of the gain also comes from adding extra features rather than from the rule itself. The clearest value is diagnostic: the rule finds products whose "promotion" flag is really a permanent price state.

## 1. The definitions

Discount: `1 - sale_price / regular_price`, undefined when `regular_price <= 0` (2 rows in the sample up to the final cutoff). An undefined discount fails the rule. The rule matches the planned-discount proxy that the control already uses (asserted in code). The table covers the 2,000-series teaching sample, weeks up to 2025-04-09. That is the period behind the figures in the request, which it reproduces (95.3%, 27.6%, 78 / 567 / 1,355).

| definition | promo share | original promo weeks reclassified | always / never / sometimes | volume-weighted promo share | non-promo weeks with discount ≥ threshold |
|---|---|---|---|---|---|
| original flag | 23.1% | – | 78 / 567 / 1,355 | 50.2% | – |
| flag and discount ≥ 5% | 21.3% | 7.8% | 63 / 609 / 1,328 | 48.6% | 17.7% |
| **flag and discount ≥ 10%** | **18.3%** | **20.8%** | **44 / 686 / 1,270** | **46.0%** | **11.7%** |
| flag and discount ≥ 15% | 13.7% | 40.8% | 20 / 781 / 1,199 | 41.5% | 7.8% |
| flag and discount ≥ 20% | 10.2% | 55.7% | 12 / 893 / 1,095 | 36.9% | 5.2% |

Full-dataset figures and configuration D's definitions are in `results/definitions.csv`.

**Leakage check (`results/realized_discount_vs_zero_sales.csv`).**
- **Flagged weeks:** a discount of at least 10% occurs in 79.2% of weeks with sales and 79.8% of zero-sales weeks. The adjusted flag carries no target information.
- **Unflagged weeks:** the same discount occurs in 16.2% of weeks with sales but only 0.009% of zero-sales weeks. A zero-sales week gets the regular price, so a realized discount there reveals that sales were positive.

**Why D was not modelled.** Built from realized prices, configuration D's future flag would leak the target. Built from the planned-price proxy, it is identical to B, because the planned discount is zero outside flagged promotions. No planned price exists for unflagged weeks, so D cannot be tested without breaking the information rules.

## 2. Setup

- **Control (A):** `lgbm_l1_future_price_levels` from notebook 06: L1 objective, standardized target, the same lags, categorical attributes, parameters and 10 cutoffs. It reproduces the saved forecasts exactly (280,000 rows, max |difference| 0.0).
- **B:** the six promotion-calendar features recomputed from the adjusted flag (`promo`, both promotion lags, `promo_freq_13`, `weeks_since_promo`, `promo_x_gap`). The price features are unchanged.
- **C:** the original calendar plus the adjusted calendar (16 features).
- **Future flags** come from the promotion calendar and the planned discount, the same full-plan assumption as the control.

The first run used notebook 06's random seed. The differences were small, so I added two references, neither a candidate:
- **Seed references:** every configuration refitted with three more LightGBM seeds (`random_state` only), each compared with the control fitted on the same seed.
- **Placebos:** the same 20.8% of promotion weeks removed at random instead of by discount, used in place of the original calendar (placebo B) or next to it (placebo C).

## 3. Results

**Development, seed 0 (the comparison as specified)** (`results/summary.csv`)

| model | dev WAPE | change (pts) | origins better (of 7) | dev bias | mean / median RMSSE | dev APC | final WAPE | final bias |
|---|---|---|---|---|---|---|---|---|
| A control | 0.2913 | – | – | +0.002 | 1.586 / 0.729 | 0.118 | 0.2678 | +0.023 |
| B 10% | 0.2889 | −0.24 | 5 | −0.002 | 1.585 / 0.727 | 0.116 | 0.2666 | +0.019 |
| C 10% | 0.2874 | −0.39 | 6 | −0.004 | 1.581 / 0.726 | 0.114 | 0.2663 | +0.020 |
| C 20% | 0.2864 | −0.48 | 7 | −0.007 | 1.579 / 0.726 | 0.114 | 0.2676 | +0.017 |
| placebo C | 0.2881 | −0.32 | 5 | −0.003 | 1.583 / 0.727 | 0.114 | 0.2669 | +0.027 |

The control refitted with seeds 1 to 3 scores a development WAPE between 0.2885 and 0.2910, 0.03 to 0.28 points better than seed 0. So the specified control sits at the unlucky end of the seed spread, which inflates the seed-0 improvements.

**Averaged over four seeds, each compared with the same-seed control** (`results/summary_over_seeds.csv`)

| config | mean dev WAPE | mean change (pts) | change range | seeds better (of 4) | mean final WAPE |
|---|---|---|---|---|---|
| A control | 0.2901 | – | – | – | 0.2680 |
| B 5% | 0.2901 | −0.00 | −0.09 to +0.17 | 3 | 0.2693 |
| **B 10%** | **0.2887** | **−0.14** | −0.38 to +0.06 | 3 | 0.2677 |
| B 15% | 0.2885 | −0.17 | −0.27 to +0.03 | 3 | 0.2667 |
| B 20% | 0.2896 | −0.06 | −0.24 to +0.18 | 3 | 0.2682 |
| placebo B | 0.2913 | +0.11 | −0.10 to +0.37 | 1 | 0.2752 |
| C 5% | 0.2892 | −0.10 | −0.28 to +0.03 | 2 | 0.2670 |
| C 10% | 0.2879 | −0.22 | −0.40 to +0.01 | 3 | 0.2681 |
| C 15% | 0.2873 | −0.29 | −0.42 to −0.18 | 4 | 0.2668 |
| C 20% | 0.2867 | −0.34 | −0.48 to −0.17 | 4 | 0.2665 |
| placebo C | 0.2886 | −0.16 | −0.32 to −0.01 | 4 | 0.2686 |

What the table shows:

- **B at 10%.** 0.14 points better on average (0.5% relative), better at 3 of 4 seeds. The random placebo makes the model worse by 0.11 points, so the rule beats an arbitrary change to the flag by about 0.25 points. Against the control, the gain is smaller than the seed spread.
- **C.** Improves at 15% and 20% for every seed, by 0.29 to 0.34 points. But the random placebo C also improves at every seed, by 0.16 points. About half of C's gain comes from giving the model a second, redundant calendar, not from the business rule. Net of the placebo, the rule adds about 0.06 points at 10% and 0.13 to 0.18 points at 15 to 20%.
- **Threshold.** The response is flat between 10% and 20%, and 5% does nothing. C 20% is best on development, but among eight configurations separated by about 0.1 points that is not a reliable choice.
- **Bias.** Every adjusted configuration lowers the forecasts slightly (development bias falls by 0.2 to 0.8 points). **Stability (APC)** is unchanged or very slightly better.
- **Final window.** Differences are within ±0.15 points for the 10 to 20% configurations; it was not used for any choice.

**Where the gain comes from** (development WAPE change against the control in points, averaged over seeds; `results/dev_segments_*.csv`)

| segment | rows | B 10% | C 10% | C 20% | placebo C |
|---|---|---|---|---|---|
| always-promotional products | 7,644 | −0.00 | −0.04 | −0.20 | +0.01 |
| sometimes-promotional products | 132,790 | −0.21 | −0.30 | −0.43 | −0.22 |
| never-promotional products | 55,566 | +0.13 | +0.07 | +0.06 | +0.09 |
| promo weeks kept at 10% | 39,742 | −0.14 | −0.14 | −0.25 | −0.17 |
| promo weeks reclassified at 10% | 9,741 | −0.11 | −0.33 | −0.20 | −0.01 |
| non-promo weeks | 146,517 | −0.14 | −0.29 | −0.44 | −0.15 |
| realized discount ≥ 10% | 59,952 | −0.04 | −0.08 | −0.14 | −0.12 |

The gains are in sometimes-promotional products, including their non-promotional weeks. Never-promotional products get slightly worse; their features did not change, so that comes from the global model's fit. The reclassified weeks themselves gain little under B.

## 4. Product examples (`results/product_*.png`)

Chosen on the definition alone: in each volume tercile, the sometimes-promotional product with the most reclassified weeks in the development horizons, plus the always-promotional product with the most. Development MAE (control / B 10% / C 10%) is shown for information; it played no part in the choice.

- **14681_S1, high volume** (1,721 / 1,670 / 1,643). Flagged promotional for over a year, but the regular price was cut to the selling price (3.50), so the "promotion" is a permanent price change. The adjusted flag turns off and the forecast moves up slightly.
- **2197_S1, mid volume** (309 / 330 / 307). Flagged promotional almost every week at about 9% below regular. The adjusted flag keeps only the occasional deeper weeks. B is worse here.
- **3947_S1, low volume** (90 / 78 / 88). Flagged promotional through a delisting with zero sales. Every model misses the restart.
- **10211_S1, always promotional** (179 / 192 / 187). A constant 5% below regular price, every week. The adjusted flag is off throughout; B and C are slightly worse.

The rule mostly corrects flags that describe a lasting price state rather than an event. In the sample up to the final cutoff:
- **Long runs:** 44% of reclassified weeks sit in promotional runs longer than a year, and another 20% in runs of 14 to 52 weeks.
- **Short promotions:** only 23% sit in promotions of 1 to 4 weeks, and the rule reclassifies just 14% of the weeks in such short promotions.

## 5. Recommendation

**Do not add this as a modelling experiment in Notebook 03.**
- The best honest estimate of the gain is 0.1 to 0.2 points of WAPE once seed noise and the redundant-feature effect are removed.
- It needs a threshold choice that the data does not pin down.
- It lowers the forecasts slightly.
- Teaching it properly would need the seed and placebo controls, which is a lot of notebook for a small effect.

**Worth a short optional note in Notebook 03's driver section:** the finding that some "promotional" flags are permanent price states (examples 14681_S1 and 10211_S1). That is a useful lesson about checking business definitions before modelling. A simpler alternative to test later would be one feature such as "weeks since the regular price changed", which addresses the cause directly.

**Method point for the course:** single-seed differences under about 0.3 points of WAPE are within LightGBM's seed noise here. That applies to some other Day 1 comparisons too.
