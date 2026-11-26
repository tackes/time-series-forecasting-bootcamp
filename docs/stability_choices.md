<!-- DRAFT: Graeme to edit -->
# Stabilising the calculations training relies on: price cleaning, scaler floor, input clip

*Written 26 Nov 2026. The experiments are in `day2_neural/code_full/05c_scaler_floor.py`;
the numbers come from its configs (`05c_scaler_floor`, `05c_price_clean`, `05c_input_clip`).*

## Why
On every series (08, 11), a few hundred N-BEATSx item forecasts exploded, up to ≈ 8,000×
the item's largest training week. The sample of 2,000 series barely showed it, which is why
00–10 did not need a cap. We tested three causes, each as an arm on the validation window, and
carried forward the combination that makes the calculations training relies on stable:
scaling, gradients and forecasts that cannot reach extreme values whatever the input.

| Choice | What it fixes | Setting |
|---|---|---|
| Clean the prices | placeholder and out-of-range prices | £0.05, 6× (`gp.PRICE_CLEAN`) |
| Floor the scaler's spread | tiny spreads turning small price moves into huge inputs | 0.01 |
| Clip the scaled inputs | large real price moves reaching the model at any size | ±50 |
| Cap the forecasts (kept) | a rare blow-up inside the network | 3× the item's largest week |

Every setting was fixed before the fits it was tested in.

## 1. Clean the prices
**What `price_index` is.** This week's price divided by the series' usual price, its median
price before the cutoff. 1 is the usual price, 0.8 is 20% off, 3 is triple the usual price.
`discount_depth` is the cut against the 4-week rolling mean of the price.

**What was wrong.** 229 weeks in 79 series are priced £0.01–0.05 with ordinary sales: these are
placeholders, not 99% discounts. They gave 249 weeks a `discount_depth` above 0.8. In
`1120_S1` most weeks are £0.01, so the placeholder became the usual price and its real prices
of £20–300 came out as a `price_index` of 2,000–30,000. Other series have prices more than 6×
above or below their usual price, most likely in the wrong units: 196 series in all.

**The rule.** A week's price is bad if it is at most £0.05, or more than 6× above or below the
series' usual price (the median of its good prices). A bad price is replaced by the median of
the good prices in the 9 weeks around it, or the usual price if there are none. Then
`discount_depth`, the rolling ratio and `price_index` are rebuilt. 828 weeks in 196 series are
replaced (40 weeks in the sample); sales are untouched, so the scored series are the same.

**Why not set them to missing.** N-BEATSx cannot take missing future inputs, so they would
have to be filled anyway, and a model-agnostic fill is better done once, in the data.

**Why not shift prices by +1.** No price is at or below zero: the data preparation drops any
series that has one. A price below £1 is a real cheap item, and an index below 1 is any
discount. Ratios only explode when the denominator is a placeholder, which the rule replaces. A
+1 shift would shrink real discounts unevenly: a 20% cut on a £2 item would look like 13%, on a
£50 item still ≈ 20%. That biases the price response, and depends on the currency.

**Why 6×.** At 3× the rule would catch 674 series, including real deep clearance cuts. 6×
catches the placeholders and unit errors and leaves real cuts alone.

**Scope.** Every model in the final comparison uses cleaned prices: LightGBM (00, 04, 11), the
GFMs (06, 08, 09, 11), Chronos-2 (07, 08, 11), t0-gamma and Timestral (07b, 11), NHITS (11).
The GFM development notebooks 01–05 keep the raw prices they were run with; their choices
(recipe, features, hyperparameters) were made there.

**Result on its own.** Cleaning alone made the GFM worse (validation 0.599 against 0.583) and
added explosions (156 items against 101): with the fake discounts gone, more input windows
are near-flat, which is the problem the floor addresses.

## 2. Floor the scaler's spread
neuralforecast's robust scaler scales each input over the 14 input weeks:
`z = (x − median) / spread`, the spread being the MAD, with fallbacks to the standard deviation
and then 1 when it is zero, plus 10⁻⁶. With CPI deflation, a price that did not move drifts by
≈ 0.001 a week, so its spread is tiny but not zero, and no fallback applies. A 20% cut in the
forecast weeks then becomes `0.2 / 0.0004 ≈ 500` scaled units, and N-BEATSx's exogenous stack
passes it almost linearly to the forecast.

The floor keeps the spread at 0.01 or more (`gd.floor_robust_scaler`). How much a scaled input
can move per unit of raw input is then at most 100, against up to 10⁶ before. A 20% cut is at
most 20 units. Sales are in units, so their spread is almost never below 0.01 and the floor
leaves the target alone.

**Result.** The exploding items halve (101 → 56) and accuracy improves (validation 0.574
against 0.583). With cleaned prices, it is the best GFM on validation (0.562). But a *large*
real move still gets through: `1120_S1`'s cleaned price runs 2–5× its input-window level,
≈ 400 scaled units even with the floor, and its forecast reached 2 × 10⁸ times its largest
week.

## 3. Clip the scaled inputs
The clip bounds the scaled covariates themselves: `z_c = min(max(z, −50), 50)`. The target is
never clipped, and the forecast is unscaled with the target's own statistics.

**How ±50 was chosen.** From the inputs alone, before any fit. With the floor, the scaled
`price_index` in the forecast weeks has a 99th percentile of ≈ 43 and a 99.9th of ≈ 75
(`discount_depth` ≈ 23 and ≈ 43; `promo` never above 6). A clip of ±10 would flatten the real
price signal of 40% of series. ±50 touches ≈ 0.5% of item-weeks and cuts the worst (≈ 440)
about ninefold.

### How the clip changes the loss surface
Take the first layer of the exogenous stack, `h = W z + b`, and a loss `L` of the forecast.
- **Gradients.** `∂L/∂W = δ zᵀ`, where `δ` is the error signal coming back from the loss. The
  gradient grows with the input. One window with `z ≈ 10⁵` gives a gradient thousands of times
  larger than an ordinary window's, so it dominates its batch. The recipe clips the gradient
  norm at 1.0, which keeps the step size safe but not its direction: the step is rescaled as a
  whole, so that one window still decides where the weights move, and the other 255 windows in
  the batch contribute almost nothing. With the clip, no window's input exceeds 50, so no
  window can take over a batch.
- **Curvature.** For a linear layer the curvature of the loss in the weights grows with
  `z zᵀ`, so with the square of the input. Huge inputs make the loss surface very steep in a
  few directions: a learning rate that suits ordinary windows overshoots there, and training
  oscillates or jumps out of a good region. Bounding `z` bounds that curvature (by ≈ 50² per
  input), so one learning rate suits every window: a smoother surface.
- **With the Huber loss.** Huber bounds `δ` for large errors, the residual side of the
  gradient. The clip bounds `z`, the input side. Together they bound the product, so a single
  bad window can move the weights only by a bounded amount.
- **What it costs.** Beyond ±50 the input's gradient is zero: the model cannot tell a scaled
  60 from 600, and those windows teach it nothing about how far beyond 50 to go. The function
  has a kink at ±50, which gradient descent handles as it does ReLU's. It affects ≈ 0.5% of
  item-weeks, mostly deep cuts in otherwise quiet windows, and shows in the price response: a
  one-week 20% cut gives an elasticity of −1.85 with the clip, −2.03 without it (05: −1.77).

### What it does to the model
- **Forecasts are bounded functions of the inputs.** However far a price moves, the
  exogenous stack sees at most ±50, so its contribution to the forecast is bounded. The worst
  forecast fell from 2 × 10⁸ to 300 times the item's largest week.
- **Training is steadier.** On validation, every clipped fit ended within 6% of its best
  validation loss (`final_over_best` ≤ 1.06), against up to 1.85 for clean + floor and 3.4
  for cleaned prices alone.
- **Accuracy is unchanged within the noise floor**: validation 0.567 against 0.562 for clean
  + floor; test, capped, all series 0.513 against 0.514, with the best WAPE (0.248).

**Why the floor and the clip together.** The floor bounds the denominator, the clip the
result. Without the floor, every small move in a near-flat window would hit ±50 and lose its
size; with it, small moves keep their size and the clip only catches large ones.

## 4. Keep the cap
One explosion is left with every fix in place: in one of three seeds, `6653_S1`, a best seller,
is forecast at ≈ 900× its largest week with every scaled input inside ±5 and ordinary
promotions. That is a rare blow-up inside the network, which no input treatment can reach. So
every model's forecast is still capped at 3× the item's largest training week (08's rule).
Real sales went above that ceiling in ≈ 0.5% of test weeks.

## 5. The choice of clean + floor + clip
The rule written before part 3 recommends the arm with the lower validation median RMSSE:
clean + floor (0.562) over clean + floor + clip (0.567). The gap, 0.005, is inside the noise
floor, so on accuracy they are the same model. We carry the clipped arm forward because its
calculations are stable: no scaled input beyond ±50, bounded gradients and curvature, steadier
training, and a worst forecast about a million times smaller. Its price response is a little
weaker, and still stronger than 05's. This was a deliberate choice over the rule's tie-break,
recorded in `05c_input_clip` (`use`, `chosen_by`, `choice_reason`).

## Summary
| Arm (with the cap) | Validation median | 20% cut elasticity | Items above 3× | Worst forecast |
|---|---|---|---|---|
| 05 (raw prices) | 0.583 | −1.77 | 101 | 8,000× |
| floor | 0.574 | −1.95 | 56 | 1 × 10⁷× |
| clean | 0.599 | −1.75 | 156 | 15,000× |
| clean + floor | 0.562 | −2.03 | 56 | 2 × 10⁸× |
| **clean + floor + clip** (carried forward) | 0.567 | −1.85 | 52 | 300× |
