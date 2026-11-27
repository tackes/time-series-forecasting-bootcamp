<!-- DRAFT: Graeme to edit -->
# Making the forecasts numerically stable: what we changed, how it works, what it does

*Written 26 Nov 2026 for readers who know data science but not neural forecasting. The
experiments are in `day2_neural/code_full/05c_scaler_floor.py`; the numbers come from its
configs (`05c_scaler_floor`, `05c_price_clean`, `05c_input_clip`) and from the rerun of 00 and
06–11. The running log is `docs/progress.md`; the overall results are in
`docs/day2_full_findings.md`.*

## The short version
When we scored our trained neural model (the "GFM") on all 17,238 series instead of the usual
2,000-series sample, a few hundred of its forecasts were absurd: up to 8,000 times the most the
item had ever sold in a week. We traced this to three things, fixed each, and kept one safety
net:

| Change | Where it acts | What it does | Setting |
|---|---|---|---|
| **1. Clean the prices** | the data | replaces placeholder and wrong-unit prices | £0.05, 6× |
| **2. Floor the scaler** | model input | stops tiny price wobbles being blown up | 0.01 |
| **3. Clip the inputs** | model input | caps how extreme a scaled price input can be | ±50 |
| **4. Cap the forecast** (kept) | model output | caps a forecast at 3× the item's best week | 3× |

The model became more accurate as well as safer: median RMSSE 0.546 → **0.520** on the sample,
0.540 → 0.513 on all series, and a slightly stronger price response. Every model in the final
comparison now uses the cleaned prices. The final model (t0-gamma, a zero-shot foundation
model) is unchanged.

## 1. What the reader needs to know first
**The task.** Forecast weekly unit sales for 14 weeks ahead, for ≈ 17,000 products. Each week
also has inputs known in advance: the planned price and whether a promotion runs. These are
"future covariates": the model sees them for the forecast weeks too, which is what lets it
answer "what if we cut the price?".

**The model.** N-BEATSx, a neural network trained on all products at once (a *global* model,
hence "GFM"). It does not see a product's whole history at once. It sees **windows**: 14 weeks
of history plus the 14 weeks to forecast, cut from every product at many points in time. A
training step takes a batch of 256 windows, forecasts them, measures the error (Huber loss)
and nudges the weights to reduce it. ≈ 11,700 steps make one model; we train 3 seeds.

**Scaling each window.** Products sell anything from 1 to 600,000 units a week, so before a
window reaches the network every input in it is rescaled, separately for each window:

```
z = (x − median of x over the 14 input weeks) / spread of x over those weeks
```

The library (neuralforecast) uses a *robust* scaler: the spread is the median absolute
deviation (MAD). If the MAD is 0 it falls back to the standard deviation, and if that is 0 too
it uses 1; then it adds 10⁻⁶. Sales and prices both go through it. The forecast comes out in
scaled units and is converted back with the sales' own median and spread.

**How price reaches the forecast.** N-BEATSx has an *exogenous* block that turns the future
covariates into part of the forecast. For each forecast week it is a weighted sum:

```
forecast contribution (week t) = Σ over inputs  θ_input × z_input(week t)
```

The weights θ come from a small network that reads the input window, so they differ from
window to window, but the contribution is **linear in the scaled future inputs**. Double a
scaled input and you double its contribution. That is what makes it useful for price
questions, and what made it fragile.

**The prices we feed it.**
- `price_index`: this week's price ÷ the product's usual price (its median before the
  forecast period). 1 is the usual price, 0.8 is 20% off, 3 is triple.
- `discount_depth`: how far this week's price is below its 4-week rolling mean (0.2 = 20% cut).
- `promo`: 1 if a promotion runs, else 0.
- Prices are **deflated by CPI** (divided by a weekly price index) so that £1 in 2023 and £1 in
  2025 mean the same. CPI rises ≈ 0.07% a week, so a shelf price that never changes becomes a
  real price that drifts down very slowly.

## 2. What went wrong
On the 2,000-series sample everything looked fine. On all series, 101 items had forecasts
above 3× their largest training week, the worst ≈ 8,000×. Summed into the store total (08),
they made it 47% too high; the GFM's all-series WAPE went from 0.26 to 0.40.

We split the exploding items by what their inputs did, and found three causes.

**Cause A: placeholder prices.** 229 weeks in 79 products are priced £0.01–0.05 with ordinary
sales. They are not 99% discounts; they are placeholders in the source data. They create fake
`discount_depth` values near 1. In one product (`1120_S1`) most weeks are £0.01, so £0.01
became its *usual* price, and its real prices of £20–300 came out as a `price_index` of
2,000–30,000. Other products have prices more than 6× above or below their usual price, most
likely recorded in the wrong units. 196 products in all.

**Cause B: near-flat inputs blown up by the scaler.** Take a product whose shelf price did not
change for 14 weeks. After CPI deflation its real price falls ≈ 0.07% a week, so each week sits
≈ 0.1% below its own 4-week rolling mean, and `discount_depth` hovers just above zero:

| Week | 1 | 2 | 3 | … | 14 | forecast week 1 |
|---|---|---|---|---|---|---|
| `discount_depth` | 0.0010 | 0.0013 | 0.0009 | … | 0.0011 | 0.20 (a 20% cut) |

The spread of the 14 input weeks is ≈ 0.0004 (the median for the exploding items): not zero,
so no fallback applies. The scaled value of the cut is `(0.20 − 0.001) / 0.0004 ≈ +500`.
`price_index` moves on the same discount; its spread in those windows was ≈ 0.001, so the cut
is ≈ −180 there. An ordinary window's inputs sit between about −5 and +5. In the exploding
items the largest scaled future `discount_depth` had a median of ≈ 600, against 11 for all
other items. Through the linear exogenous block, inputs of several hundred give a forecast
contribution a hundred times the usual size, and both price inputs move together on every
discount, so dropping one would not help. `promo` was not the problem: when it is exactly
flat, the scaler's fallback keeps it near 1.

**Cause C: large real price moves.** Even with clean prices and sensible spreads, a real price
move several times the window's own level (a product going from its usual price to 5× it) is
hundreds of scaled units, and the linear block extrapolates it.

## 3. The four changes
### Change 1: clean the prices (the data)
**What it does.** A week's price is *bad* if it is at most £0.05, or more than 6× above or
below the product's usual price (the median of its good prices before the forecast period).
A bad price is replaced by the median of the good prices in the 9 weeks around it (or the usual
price if there are none). Then `discount_depth`, the rolling ratio and `price_index` are
rebuilt from the cleaned price. `regular_price` is cleaned the same way.

**Example.** A product priced £2.00 every week except one at £0.01: the £0.01 week becomes
£2.00, its fake 99.5% discount becomes 0, and its `price_index` is 1 throughout. In `1120_S1`
the placeholders are replaced by its real prices, and its `price_index` goes from up to 29,000
to between 0.85 and 5.3.

**What it changes in the data.** 828 product-weeks in 196 products (40 weeks in the sample).
`price_index` now runs from 0.13 to 5.9 instead of 0.0002 to 30,000, and no week has a
`discount_depth` above 0.8 (there were 249). Sales are untouched, so the scored series and the
evaluation are exactly the same.

**Why not set bad prices to missing?** N-BEATSx cannot take missing future inputs, so they
would have to be filled anyway. It is better to fill them once, in the data, the same way for
every model.

**Why not shift prices by +1 to keep them away from zero?** No price is at or below zero (the
data preparation already drops products with one), and a price below £1 is a real cheap
product. Ratios only explode when the denominator is a placeholder, which the rule replaces. A
+1 shift would distort real discounts unevenly: a 20% cut on a £2 product would look like 13%,
while on a £50 product it stays ≈ 20%. That biases the price response, and depends on the
currency.

**Why 6×?** At 3× the rule would flag 674 products, including genuine deep clearance cuts. At
6× it catches the placeholders and unit errors and leaves real cuts alone. Both thresholds were
fixed before any model was fitted with them.

**Where.** `gd_price.clean_prices` with `gp.PRICE_CLEAN = {"k": 6.0, "floor": 0.05}`; applied by
`gp.prepare` when a config has `price_clean`, and directly for LightGBM and NHITS.

### Change 2: floor the scaler's spread (the model's input transform)
**What it does.** The spread used to scale each window is never allowed below 0.01:
`z = (x − median) / max(spread, 0.01)`.

**Example.** The near-flat window above: spread 0.0004 → 0.01, so the 20% cut scales to
`0.199 / 0.01 ≈ 20` instead of 500 (and `price_index`'s ≈ −180 becomes ≈ −20). A change in
a raw input can now move its scaled value by at most 100× that change, against up to 10⁶×
before.

**What it does not touch.** Sales are in units, so their spread is essentially never below
0.01; the floor leaves the target alone. In a window where a price genuinely moves, the spread
is already above 0.01 and nothing changes.

**Where.** `gd.floor_robust_scaler(model, floor=0.01)` replaces the library's scaling function
on the model. The setting travels in the model's `params` (`scale_floor`), so every refit and
every loaded model gets it.

### Change 3: clip the scaled inputs (the model's input transform)
**What it does.** After scaling, every covariate is limited to ±50:
`z_clipped = min(max(z, −50), 50)`. The target (sales) is never clipped, and the forecast is
converted back with the sales' own statistics as before.

**How ±50 was chosen.** From the inputs alone, before fitting anything. With the floor, a
normal 20% cut is ≈ 20 units. Across all test-window inputs, the scaled `price_index` has a
99th percentile of ≈ 43 and a 99.9th of ≈ 75 (`discount_depth` ≈ 23 and ≈ 43; `promo` never
above 6). ±10 would flatten the real price signal of 40% of products; ±50 touches ≈ 0.5% of
product-weeks and cuts the most extreme value (≈ 440) about ninefold.

**How clipping changes training (the loss surface).** Training adjusts each weight in the
direction that reduces the loss, by an amount proportional to the gradient. For the weights
that read a covariate, the gradient is roughly *the error signal × the input value*.
- **Without the clip**, one window with an input of 10⁵ produces a gradient thousands of times
  larger than an ordinary window's. Our training recipe clips the total gradient to a fixed
  length (1.0), which keeps the step size safe, but the *direction* is still set by that one
  window: the other 255 windows in the batch contribute almost nothing to that step.
- **The curvature** of the loss (how sharply it bends) grows with the square of the inputs.
  Huge inputs make the loss surface a steep, narrow ravine in some directions: a learning rate
  that suits ordinary windows overshoots there, so training oscillates or jumps out of a good
  region. Bounding the inputs bounds the curvature, so one learning rate suits every window: a
  smoother surface.
- **With the Huber loss.** Huber limits the error side of the gradient (a big miss counts
  linearly, not squared). The clip limits the input side. Together they bound how far any
  single window can move the weights.
- **The cost.** Beyond ±50 the model sees the same value, so it cannot tell a scaled 60 from
  600, and those windows teach it nothing about how far beyond 50 to go. That affects ≈ 0.5% of
  product-weeks, mostly deep cuts in otherwise quiet windows, and shows as a slightly weaker
  price response: a one-week 20% cut gives an elasticity of −1.85 on validation with the clip,
  −2.03 without (the old model: −1.77).

**What it does to the model.** The forecast is now a bounded function of the inputs: however
far a price moves, the exogenous block sees at most ±50. The worst forecast fell from 2 × 10⁸
to 300 times the product's largest week. Training is steadier: every clipped fit ended within
6% of its best validation loss, against up to 85% worse for the unclipped version and 3.4× for
cleaned prices alone.

**Why the floor and the clip together.** The floor bounds the denominator; the clip bounds the
result. Without the floor, every small move in a near-flat window would hit ±50 and lose its
size. With it, small moves keep their size and the clip only catches large ones.

**Where.** `gd.floor_robust_scaler(model, floor=0.01, clip=50)`; `input_clip` in `params`.

### Change 4: keep the cap (the model's output)
**What it does.** Each forecast is limited to 3× the product's largest training week. Real
sales exceeded that in ≈ 0.5% of test weeks, so it rarely cuts a real spike.

**Why it stays.** With changes 1–3 in place, one failure is left: in one of the three seeds, a
best seller (`6653_S1`) is forecast at ≈ 900× its largest week although every scaled input is
within ±5 and its promotions are ordinary. That is a rare instability inside the network, which
no input treatment can reach. The cap is the guard for it. It is applied to the selected GFM's
saved forecasts and to every model in 08 and 11.

**Its limit.** For a best seller, 3× its largest week is still millions of units too many. That
one product makes the GFM's WAPE on the sample 0.40 (0.25 on all series) and the GFM's
bottom-up store total in 08 ≈ 4% too high, although its median and mean errors are fine.

## 4. How data now flows through the GFM
```mermaid
flowchart LR
  A[Raw weekly panel] --> B[Clean prices<br/>£0.05 / 6× rule]
  B --> C[CPI deflation,<br/>price features rebuilt]
  C --> D[Windows: 14 weeks in,<br/>14 weeks out]
  D --> E[Robust scaling per window<br/>spread ≥ 0.01]
  E --> F[Clip covariates ±50]
  F --> G[N-BEATSx]
  G --> H[Unscale with sales'<br/>median and spread]
  H --> I[Cap at 3× largest<br/>training week]
  I --> J[Forecast]
```
Steps B, the floor in E, F and I are new. LightGBM and the foundation models (Chronos-2,
t0-gamma, Timestral) go through B and C only: they scale their inputs in their own ways.

## 5. How we decided
Each change was tested as an *arm*: the selected model's settings with one thing changed,
trained with the same 3 seeds on the weeks before the validation window, and judged on that
window by a rule written before the results (accuracy within the noise floor of the old model,
a price response no weaker). The test window was scored once afterwards.

| Arm (all with the cap) | Validation median | 20% cut elasticity | Items above 3× | Worst |
|---|---|---|---|---|
| old model (raw prices) | 0.583 | −1.77 | 101 | 8,000× |
| floor | 0.574 | −1.95 | 56 | 1 × 10⁷× |
| clean | 0.599 | −1.75 | 156 | 15,000× |
| clean + floor | 0.562 | −2.03 | 56 | 2 × 10⁸× |
| **clean + floor + clip** (chosen) | 0.567 | −1.85 | 52 | 300× |

- Cleaning alone made the model *worse* and added explosions: with the fake discounts gone,
  more windows were near-flat, which is cause B. It only pays with the floor.
- The rule's tie-break preferred clean + floor, by 0.005, inside the noise floor (the amount a
  model moves between seeds). On accuracy the two are the same model. **We chose clean + floor
  + clip** because its calculations are stable: bounded inputs, bounded gradients, steadier
  training, and a worst forecast about a million times smaller. That was a deliberate choice
  over the tie-break, recorded in `05c_input_clip` (`use`, `chosen_by`, `choice_reason`).

## 6. What changed in the results
| Test window | Before | After |
|---|---|---|
| Selected GFM, sample, median RMSSE | 0.546 | **0.520** |
| Selected GFM, all series (capped): median / WAPE | 0.540 / 0.258 | **0.513 / 0.248** |
| Selected GFM, lift for a one-week 20% cut | +49% | +50% |
| LightGBM, sample | 0.569 | 0.572 |
| Foundation models (Chronos-2, t0-gamma, Timestral) | | within 0.002 |
| Final model (10) | t0-gamma 0.502 | t0-gamma 0.501 |
| Top-3 average, held-out half | 0.495 | 0.495 |

The GFM moved from fifth to third on the board, level with Timestral and 0.008 behind
Chronos-2. The foundation models and LightGBM barely noticed the cleaning: they had not been
leaning on the bad weeks. Clean prices are still the right input for them.

**Scope.** Every model in the final comparison uses the cleaned prices: LightGBM (00, 04, 11),
the GFMs (06, 08, 09, 11), Chronos-2 (07, 08, 11), t0-gamma and Timestral (07b, 11) and NHITS
(11). The development notebooks 01–05, where the model's recipe, features and hyperparameters
were chosen, keep the raw prices they were run with.

## 7. What it does not fix
- **The last blow-up.** One seed, one best seller, ordinary inputs. Only the cap guards against
  it, and for a best seller the cap is loose.
- **Under-response to price.** Every model still forecasts a much smaller lift than the data
  imply (+50% against ≈ +125% for a 20% cut). Stability was not the cause of that.

## 8. Where it lives in the code
| What | Where |
|---|---|
| price cleaning | `gd_price.clean_prices`, `gp.bad_price_weeks`, `gp.PRICE_CLEAN` |
| floor and clip | `gd.floor_robust_scaler`; `scale_floor`, `input_clip` in `params` |
| applying them on every fit | `gd.build_nbeatsx` reads `scale_floor` and `input_clip` |
| saved models | `gd.save_model` writes `stability.json`; `gd.load_model` reapplies it |
| the selected setting | `gd.selected_config()` reads `05c_input_clip` (`use`) |
| the experiments | `day2_neural/code_full/05c_scaler_floor.py` |
| tests | `tests/test_price_clean.py`, `test_scaler_floor.py`, `test_stability.py` |

## Glossary
- **GFM**: global forecasting model, one network trained on every product.
- **Covariate**: an input other than past sales (price, promotion); *future* covariates are
  known for the forecast weeks.
- **Window**: a slice of one product's history (14 weeks) plus the weeks to forecast (14).
- **Robust scaler, MAD**: rescales each window by its median and median absolute deviation.
- **Exogenous block**: the part of N-BEATSx that turns future covariates into forecast.
- **Seed**: the random start of a training run; we train 3 and average them.
- **Noise floor**: how much a model's median RMSSE moves between seeds (≈ 0.005); smaller gaps
  are ties.
- **RMSSE**: forecast error scaled by the product's own seasonal-naive error; 1 means no better
  than repeating last year.
- **WAPE**: total absolute error ÷ total sales; dominated by the best sellers.
- **Elasticity**: % change in sales ÷ % change in price; −2 means a 10% cut lifts sales 20%.
