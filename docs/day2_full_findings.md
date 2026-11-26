<!-- DRAFT: Graeme to edit -->
# Day 2 on the full panel: what we did and what we found

*Written 25 Nov 2026, after the GPU runs of 23–25 Nov. The numbers come from
`results/day2_full/configs/` and the executed notebooks in `day2_neural/code_full/`. The
running log, with timings and decisions, is `docs/progress.md`.*

## In one paragraph
We rebuilt the Day 2 labs to train on every eligible series (17,241) instead of a sample, and
scored everything on the original labs' fixed 2,000-series evaluation sample. A global neural
model (NBEATSx) trained with a careful recipe beats every baseline, LightGBM included. But the
most accurate models are **zero-shot foundation models fed the price and promotion plan**,
and the best of them, **t0-gamma, is the final model** (median RMSSE 0.502, against 0.546 for
our best trained GFM and 0.568 for LightGBM). Averaging the three foundation models is better
still (0.495 on held-out series). Every model responds to a price cut in the right direction,
but every one **under-responds**: for a one-week 20% cut the data imply a lift of about +125%,
the GFM forecasts +49% and t0-gamma +30%. So use the foundation models to forecast and the
GFM to ask what a new price would do, and treat neither as a price-response model.

## How we worked
- **One evaluation for everything.** Median RMSSE (mean alongside) on the same 2,000 series
  and the same 14-week test window as the original labs. 10 compares on the 1,996 series
  every model covers.
- **Tune on validation, test once.** Every choice (settings, features, arms, the upper model
  in 08, the main setup in 07b) is made on the validation window, with a rule written down
  before the results. The test window only reports.
- **A noise floor.** The same model trained with 3 seeds moves by about 0.005 in median
  RMSSE (02). A smaller gap is a tie. In 07b a difference also needs a paired Wilcoxon test
  at p < 0.05 to count as real.
- **Configs carry decisions forward.** Each notebook reads the config before it and writes
  its own. Train once on the GPU, then load from the cache on the day.
- **Price response measured as the research's phase 7b does.** Shock the price, rebuild every
  input derived from it (discount depth, ratio to the rolling mean) and read the forecast
  lift. Also five random promotion calendars per series.

## The final standings (test, 1,996 series)
| Model | Notebook | Median RMSSE | Mean | Price lift, 20% cut |
|---|---|---|---|---|
| **t0-gamma, zero-shot + price** (final) | 07b | **0.502** | 1.03 | +30% |
| Chronos-2, zero-shot + price | 07 | 0.512 | 1.04 | +37% |
| Timestral, zero-shot + price | 07b | 0.521 | 1.00 | +50% |
| NHITS with the recipe | 01 | 0.537 | 1.03 | |
| GFM selected (NBEATSx, tuned, CPI, oversampling) | 05 | 0.546 | 1.04 | +49% |
| GFM at the bottom of the hierarchy | 08 | 0.552 | 1.05 | |
| LightGBM | 00 | 0.568 | 1.12 | +28% |
| Chronos-2, zero-shot, no covariates | 07 | 0.645 | 1.16 | |
| AutoETS | 00 | 0.707 | 1.25 | |
| SeasonalNaive | 00 | 0.929 | 1.49 | |
| *implied by the data (elasticity ≈ −3.6)* | | | | *+125%* |

On the held-out half, the average of the top three (t0-gamma, Chronos-2, Timestral) scores
0.495, against 0.507 for t0-gamma alone: the only ensemble that beats the best single model
by more than the noise floor.

## Every series, not just the sample (11)
The v1 labs score on all 17,238 scorable series, not the 2,000-series sample. 11 rescores our
main models on all of them.

| Model | Median RMSSE, all / sample | WAPE, all / sample | v1 labs, all series |
|---|---|---|---|
| Top-3 average | 0.488 / 0.486 | 0.203 / 0.203 | 0.498 / 0.219 (their own top 3) |
| t0-gamma | 0.496 / 0.502 | 0.208 / 0.205 | |
| Chronos-2 + price | 0.510 / 0.512 | 0.221 / 0.216 | 0.511 / 0.229 |
| NHITS, our recipe (3 seeds) | 0.534 / 0.538 | 0.244 / 0.251 | NHITSx 0.572 / 0.275 |
| GFM selected (3 seeds) | 0.540 / 0.546 | 0.258 / 0.257 | |
| LightGBM | 0.569 / 0.569 | 0.291 / 0.295 | 0.569 / 0.291 |
| SeasonalNaive | 0.913 / 0.929 | 0.446 / 0.438 | 0.913 / 0.446 |

All forecasts capped at 3× the item's largest training week, as in 08.

- **The sample was a good stand-in.** Medians are within 0.03 (mostly 0.01) and the order of
  the models holds. WAPE moves by up to ≈ 5 points, because it follows the best sellers.
- **Same data, same scoring as the v1 labs:** the baselines, LightGBM and Chronos-2 match them
  to three decimals. Our neural models are better because they train ≈ 20× longer with the
  research recipe, over 3 seeds; the v1 labs use 500 steps and library defaults to suit a
  laptop.
- **The N-BEATSx models explode on a few hundred items outside the sample** (up to ≈ 8,000×
  the item's largest week). Uncapped, the selected GFM's WAPE on all series is 0.404, not
  0.258. The likely cause is the robust window scaler, which also scales the price and
  promotion inputs: an input that never moved in the 14 input weeks has no spread. The
  standard scaler avoids it but was clearly worse in the research, so we cap, in 08 and 11.

## What each notebook found
- **00 Baselines.** LightGBM 0.569 is the bar to beat; SeasonalNaive 0.929.
- **01 The recipe.** Training is most of the battle: NBEATSx off the shelf scores 0.773 (seed
  spread 0.052), with the recipe 0.568 (spread 0.012). NHITS with the same recipe is better
  (0.539) at half the cost; we kept NBEATSx for its exogenous stack.
- **02 HPO round 1.** The recipe was already the best of 12 cells; nothing else was within
  the noise floor.
- **03 Calendar and hierarchy features.** Both hurt: Fourier and event inputs make the model
  overfit (validation 0.70–0.72 against 0.61).
- **04 Price.** CPI deflation helps accuracy and the price response; oversampling promotion
  weeks steepens the response further (chosen arm, validation 0.593).
- **05 HPO round 2.** The accuracy-optimal trial wins (test 0.546). Tuning moved accuracy,
  not the price response, which barely changes across the Pareto front.
- **05c Scaler floor.** Flooring the robust scaler's spread at 0.01 halves the exploding
  forecasts and improves accuracy (validation 0.574 against 0.583), but the rule written in
  advance needed a tenfold cut, so the cap stays. Cleaning placeholder and out-of-range
  prices as well gives the best validation score (0.562); recommended with the cap, not yet
  carried forward.
- **05b Synthetic discounts.** Training copies with planted discounts did not help: tied on
  the one-week cut, worse on validation (0.593 against 0.583). 05 goes forward.
- **06 Clusters.** One pooled model beats per-cluster models (0.546 against 0.570); they lose
  most in the small clusters.
- **07 Chronos-2.** Zero-shot with the price and promotion covariates it beat every trained
  model (0.512), with no training. The covariates are worth ≈ 0.13 of RMSSE.
- **07b Newer foundation models.** t0-gamma beats Chronos-2 by twice the noise floor (p ≈ 0);
  Timestral does not differ from it. Event dummies and tree-style inputs change little.
  Every model needs the full history (≈ 0.60 with 13 or 26 weeks). Timestral is best on
  promotion weeks and weeks above anything in the history; t0-gamma wins on the
  higher-selling classes A–I. Timestral's weights need a separate licence above $20M monthly
  revenue; the other two are Apache-2.0.
- **08 Hierarchy.** With price-aware upper nodes and two item models (GFM, Chronos-2), both
  choose **BottomUp**: the item models are good enough that the sum of items is best or close
  at every level. The weighted MinT methods win the upper levels with GFM items
  (`MinT-shrink-sub` total 0.660 against 0.843) but lose a little at the items. A few
  exploding item forecasts are capped at 3× the item's largest training week.
- **09 Intervals.** At 80%, conformal by step hits its target (coverage 0.791); the quantile
  GFM over-covers (0.864). At 95%, conformal's interval score is several times worse: a few
  large misses dominate.
- **10 Selection.** The rule fixed in advance (keep the GFM unless a model beats it by more
  than the noise floor) picks t0-gamma. Per-class selection does not carry over to the
  held-out half; averaging does.

## Price response
| One-week 20% cut, inputs rebuilt | Lift | Calendar cuts with the right sign |
|---|---|---|
| Timestral | +50% | 90% |
| GFM selected | +49% | 99% |
| Chronos-2 | +37% | 90% |
| t0-gamma | +30% | 89% |
| LightGBM | +28% | 98% |
| implied by the data | +125% | |

- Every model is right-signed and too weak. The best gives back about 40% of the lift the
  data imply.
- The trained GFMs are the most **consistent** (99% of random cuts lift the forecast). The
  foundation models learn each series' response from that series alone, and about 1 in 10
  cuts does not lift their forecast.
- The **most accurate model responds least**. Accuracy rewards reading the usual pattern of
  promotions; a probe asks what a new price would do.
- Moving the price alone understates the response. LightGBM gives +5% with the price alone,
  +28% with the discount depth rebuilt. That is why the probes rebuild every input.

## Things that went wrong, and the fixes
- **Exploding item forecasts.** On every series (08, 11) a few hundred N-BEATSx item forecasts
  explode, which put the bottom-up total 47% too high and the GFM's all-series WAPE from 0.26
  to 0.40. Fixed by capping each item at 3× its largest training week, applied to every model.
  Part of the cause is the robust scaler: `discount_depth` and `price_index` barely move in a
  quiet input window (CPI drift of ≈ 0.001), so a discount becomes hundreds of scaled units.
  **05c** floors the scaler's spread at 0.01. That halves the exploding items (101 → 56) and
  makes the GFM better (capped median RMSSE on all series 0.511 against 0.540; one-week cut
  elasticity −1.95 against −1.77), but it is not a cure. The worst remaining case is a data
  error (`1120_S1` has `price_index` up to 29,000; 29 series have one above 3), and one seed
  still blows up a normal-priced item. Part 2 cleaned the prices (placeholders and weeks
  beyond 6× of the usual price, replaced from their neighbours). Clean prices plus the floor
  is the best GFM on validation (0.562 against 0.583) and ties the floor on test (capped
  0.514), but still needs the cap: a large real price move is still passed linearly to the
  forecast. Clipping the scaled inputs is untested.
- **A silent library bug in sparse MinT.** `hierarchicalforecast`'s `MinTraceSparse` solves its
  system iteratively and ignores whether it converged. With this panel's spread of error
  variances, `wls_var` returned forecasts in the millions and took 37 min per reconcile.
  `gd.MinTraceDirect` solves the same equations exactly in ≈ 16 s.
- **Two-GPU hang.** Lightning's default trains on every visible GPU, which hung inside a
  notebook. Every fit now uses one device.

## Open questions
- **Why does every model under-respond to price?** The data's elasticity (≈ −3.6) comes from
  short, deep promotional cuts; the models may be learning the promotion flag more than the
  price. 05b's planted discounts did not close the gap.
- **The hierarchy rule.** It judges each method by its worst level, so the methods that win
  the department and class totals lose to BottomUp on a 0.3–0.9% item-level gap. A planner
  who cares most about department totals would choose differently.
- **The upper-model rule.** ETS won although Chronos-2 was ≈ 28% better below the total,
  because it was 4% worse at the total itself.
- **Fine-tuning.** Chronos-2 fine-tuning is built but off (`GD_FM_FINETUNE`); t0-gamma and
  Timestral were only run zero-shot.
- **NHITS.** It beat NBEATSx with the same recipe at half the cost, but was not carried
  forward. It is worth a full HPO round.
