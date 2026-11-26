<!-- DRAFT: Graeme to edit -->
# Day 2 on the full panel: what we did and what we found

*Written 25 Nov 2026, after the GPU runs of 23–25 Nov; updated 26 Nov after 05c and the
rerun of 00 and 06–11 on cleaned prices (`docs/stability_choices.md`). The numbers come from
`results/day2_full/configs/` and the executed notebooks in `day2_neural/code_full/`. The
running log, with timings and decisions, is `docs/progress.md`. What 05c changed in the data
and the model, and why, is explained for a general data-science reader in
`docs/stability_choices.md`.*

## In one paragraph
We rebuilt the Day 2 labs to train on every eligible series (17,241) instead of a sample, and
scored everything on the original labs' fixed 2,000-series evaluation sample. A global neural
model (NBEATSx) trained with a careful recipe beats every baseline, LightGBM included. But the
most accurate models are **zero-shot foundation models fed the price and promotion plan**,
and the best of them, **t0-gamma, is the final model** (median RMSSE 0.501, against 0.520 for
our best trained GFM and 0.572 for LightGBM). Averaging the three foundation models is better
still (0.495 on held-out series). Making the GFM's calculations stable (cleaned prices, a
floor on the scaler's spread, a clip on the scaled inputs, 05c) took it from 0.546 to 0.520,
level with Timestral. Every model responds to a price cut in the right direction, but every
one **under-responds**: for a one-week 20% cut the data imply a lift of about +125%, the GFM
forecasts +50% and t0-gamma +30%. So use the foundation models to forecast and the GFM to ask
what a new price would do, and treat neither as a price-response model.

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
| **t0-gamma, zero-shot + price** (final) | 07b | **0.501** | 1.03 | +30% |
| Chronos-2, zero-shot + price | 07 | 0.512 | 1.04 | +37% |
| GFM selected (05 + cleaned prices, floor, clip, cap) | 05c | 0.520 | 1.05 | +50% |
| Timestral, zero-shot + price | 07b | 0.521 | 1.00 | +50% |
| GFM at the bottom of the hierarchy | 08 | 0.526 | 1.06 | |
| NHITS with the recipe | 01 | 0.537 | 1.03 | |
| GFM 05 (NBEATSx, tuned, CPI, oversampling; raw prices) | 05 | 0.546 | 1.04 | +49% |
| LightGBM (cleaned prices) | 00 | 0.572 | 1.12 | +28% |
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
| Top-3 average | 0.488 / 0.485 | 0.203 / 0.203 | 0.498 / 0.219 (their own top 3) |
| t0-gamma | 0.496 / 0.502 | 0.208 / 0.205 | |
| Chronos-2 + price | 0.510 / 0.512 | 0.221 / 0.216 | 0.511 / 0.229 |
| GFM selected (05c, 3 seeds) | 0.513 / 0.520 | 0.248 / 0.399 | |
| Timestral | 0.516 / 0.521 | 0.217 / 0.227 | |
| NHITS, our recipe (3 seeds) | 0.532 / 0.534 | 0.246 / 0.249 | NHITSx 0.572 / 0.275 |
| LightGBM | 0.570 / 0.572 | 0.292 / 0.297 | 0.569 / 0.291 |
| SeasonalNaive | 0.913 / 0.929 | 0.446 / 0.438 | 0.913 / 0.446 |

Every model reads cleaned prices, and every forecast is capped at 3× the item's largest
training week, as in 08. Before 05c the selected GFM scored 0.540 / WAPE 0.258 here.

- **The sample was a good stand-in.** Medians are within 0.03 (mostly 0.01) and the order of
  the models holds. WAPE moves by up to ≈ 5 points, because it follows the best sellers.
- **Same data, same scoring as the v1 labs:** the baselines and Chronos-2 match them to three
  decimals, LightGBM (on cleaned prices) to 0.001. Our neural models are better because they
  train ≈ 20× longer with the research recipe, over 3 seeds; the v1 labs use 500 steps and
  library defaults to suit a laptop.
- **The GFM's sample WAPE (0.399) is far above its all-series WAPE (0.248).** Its one
  remaining blow-up (one seed, `6653_S1`) is a best seller in the sample; even capped at 3×
  its largest week it adds millions of units to a sample that sells far less than all series.
  Uncapped, the GFM's all-series WAPE is 1.46. See *Things that went wrong*.

## What each notebook found
- **00 Baselines.** LightGBM 0.572 is the bar to beat (0.569 on raw prices); SeasonalNaive
  0.929.
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
- **05c Stable calculations.** Cleaning placeholder and out-of-range prices, flooring the
  robust scaler's spread at 0.01 and clipping the scaled price inputs at ±50 make the GFM
  better (validation 0.567 against 0.583; test 0.520 against 0.546) and stop the input-driven
  explosions (worst forecast 300× the item's largest week, against 2 × 10⁸ without the clip).
  The rule's tie-break preferred clean + floor (0.562, a tie); we carry the clipped version
  forward for its stable training calculations. Every later model uses the cleaned prices,
  and the cap stays. Why and how: `docs/stability_choices.md`.
- **05b Synthetic discounts.** Training copies with planted discounts did not help: tied on
  the one-week cut, worse on validation (0.593 against 0.583). 05 goes forward.
- **06 Clusters.** One pooled model beats per-cluster models (0.520 against 0.542) and the
  cluster static (0.526); they lose most in the small clusters.
- **07 Chronos-2.** Zero-shot with the price and promotion covariates it beats every trained
  model (0.512 against the GFM's 0.520), with no training. The covariates are worth ≈ 0.13 of
  RMSSE. Cleaning the prices did not change it.
- **07b Newer foundation models.** t0-gamma beats Chronos-2 by twice the noise floor (p ≈ 0);
  Timestral does not differ from it, nor from the GFM. Event dummies and tree-style inputs
  change little.
  Every model needs the full history (≈ 0.60 with 13 or 26 weeks). Timestral is best on
  promotion weeks and weeks above anything in the history; t0-gamma wins on the
  higher-selling classes A–I. Timestral's weights need a separate licence above $20M monthly
  revenue; the other two are Apache-2.0.
- **08 Hierarchy.** With price-aware upper nodes and two item models (GFM, Chronos-2), both
  choose **BottomUp**: the item models are good enough that the sum of items is best or close
  at every level. The weighted MinT methods win the upper levels with GFM items
  (`MinT-shrink-sub` total 0.680 against 0.853) but lose a little at the items. The GFM's
  saved seed is the one with the `6653_S1` blow-up, so its bottom-up total is ≈ 4% too high
  even capped (Chronos-2: ≈ 2%).
- **09 Intervals.** At 80%, conformal by step hits its target (coverage 0.791); the quantile
  GFM over-covers less than before (0.850, against 0.864 for 05's) and has the best interval
  score. At 95%, conformal's interval score is several times worse: a few
  large misses dominate.
- **10 Selection.** The rule fixed in advance (keep the GFM unless a model beats it by more
  than the noise floor) picks t0-gamma (0.501 against 0.520). Per-class selection does not
  carry over to the held-out half; averaging does.

## Price response
| One-week 20% cut, inputs rebuilt | Lift | Calendar cuts with the right sign |
|---|---|---|
| GFM selected (05c) | +50% | 99% |
| Timestral | +50% | 90% |
| GFM 05 (raw prices) | +49% | 99% |
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
  explode, which put the bottom-up total 47% too high and the GFM's all-series WAPE from 0.26 to
  0.40. First guarded by capping each item at 3× its largest training week, applied to every
  model. 05c then found the causes. `discount_depth` and `price_index` barely move in a quiet
  input window (CPI drift of ≈ 0.001), so the robust scaler turns a discount into hundreds of
  scaled units; flooring its spread at 0.01 halves the exploding items (101 → 56). The worst
  remaining cases were data errors: £0.01 placeholder prices and prices in the wrong units
  (`1120_S1` had a `price_index` up to 29,000), which cleaning replaces from neighbouring weeks.
  A large real price move still reached the forecast, so the scaled price inputs are also
  clipped at ±50, which takes the worst forecast to 300× the item's largest week. What is left
  is a rare blow-up inside the network (one seed, one best seller, ordinary inputs), which only
  the cap guards against. Clean + floor + clip, with the cap, is carried forward, and 00 and
  06–11 were rerun on cleaned prices (`docs/stability_choices.md`).
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
  the department and class totals lose to BottomUp on a 0.1–1.6% item-level gap. A planner
  who cares most about department totals would choose differently.
- **The upper-model rule.** ETS won although Chronos-2 was ≈ 28% better below the total,
  because it was 5% worse at the total itself.
- **Fine-tuning.** Chronos-2 fine-tuning is built but off (`GD_FM_FINETUNE`); t0-gamma and
  Timestral were only run zero-shot.
- **NHITS.** It beat NBEATSx with the same recipe at half the cost, but was not carried
  forward. It is worth a full HPO round, with 05c's stability settings.
- **The last blow-up.** One seed of the selected GFM still blows up one best seller with
  ordinary inputs. Is it the trend or seasonality stack extrapolating, and would a different
  seed policy (e.g. dropping a seed whose validation forecasts explode) be fair?
