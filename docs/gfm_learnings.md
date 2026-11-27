# What we learnt training global forecasting models (GFMs)

This is a summary of a research programme on global N-BEATSx models for retail demand and price response. It is written for the Day 2 `code_full` notebooks, which refer to it by tag.

**Tags**
- **[single store]**: dataset A, one large store with many thousands of weekly item series, similar in shape to the bootcamp data.
- **[100k set]**: dataset B, a much larger multi-store set of weekly item-store series. The research moved to it part-way through.
- **[method]**: holds whatever the data.

Each finding is marked **accepted**, **rejected** or **experimental**.

---

## 1. How the research progressed

| Phase | Data | What we tried | Verdict |
|---|---|---|---|
| 0–1 | [single store], daily, ~2,700-series department subset | A hand-built N-BEATSx in Darts, then NeuralForecast | Darts too slow; daily data pulled the model toward noise. **Rejected** |
| 1→2 | [single store], weekly | NeuralForecast NBEATSx / NHITS, relative-price features, Huber loss | RMSSE ≈ 0.76 → ≈ 0.50. **Accepted** |
| 2 | [single store] | Oversampling | Stronger, less often perverse elasticity. **Accepted**, in graduated form |
| 3 | [single store] | CPI deflation; drop series with price ≤ 0 | Perverse (positive) elasticity largely removed. **Accepted** |
| 4 | [single store] | NBEATSx vs NHITS, TSMixerx and others; oversampling variants | NBEATSx chosen: NHITS gave poor elasticity with oversampling. **Accepted** |
| 5 | [100k set] | Scaling up: step budget, filters, scalers, targets, LightGBM benchmark | See §3 |
| 6a | [100k set] | Training stability | Warm-up, best-weights retention, gradient clip. **Accepted** |
| 6b–c | [100k set], 50k sample | Optuna HPO: single-objective, then two-objective Pareto | **Accepted** method; results did not fully transfer |
| 7 | [100k set] | Calendar events; synthetic discounts in training; price probes | **Experimental** |
| 8 | [100k set] | Hierarchy: GFM at item level, simple models above, reconciliation | **Experimental** |

**A lesson learnt the hard way [method].** Until phase 6b, HPO trials were scored on the final test window: the code defined validation splits but never used them. The phase 4 defaults below were therefore chosen on test data. From phase 6b onwards the split was weeks 1–74 train, 75–88 tuning, 89–102 final test, and the test window was touched once. `code_full` tunes on `bc.VALID_START` only.

---

## 2. Data selection [single store]

**Early daily scripts (phases 0–1):**
- Five departments, about 2,700 of the ~18,000 series.
- Target shifted by one (`y + 1`) so that zeros are safe for scaling and ratios.
- Any series with a zero-sales run of 180 days or more was dropped.
- At least 365 days of history required (added later).
- Series with no class or item group were dropped.
- No price filter.

**Weekly build (phases 1–4):**
- Wednesday weeks (`W-WED`, as in `bootcamp.py`).
- Units summed per week; the promo flag is "any promotional day"; price is the units-weighted sale price.
- Only active series with at least 26 weeks were kept.
- From phase 3, series with any price ≤ 0 were dropped.

**What the later research said [100k set]:**
- Do **not** filter sparse series: removing them made the error 6–11× worse.
- Evaluate on a common floor instead (historical mean ≥ 2 units/week), so phases are comparable.

**Decision for `code_full`:**
- Train on the original labs' eligible set (≈17,241 series). Their filters are close to the research's final rule: price ≤ 0 anywhere, fewer than 66 training weeks, and constant series.
- Notebook 00 shows what the early filters would also have dropped.

---

## 3. The training recipe

### Accepted

| Item | Setting | Why | Tag |
|---|---|---|---|
| Model | NBEATSx with fixed stacks: identity, trend (3 polynomials), seasonality (3 harmonics), exogenous | Best price response | [single store] |
| Starting HPs | blocks [2,2,2,2], mlp 512, input 14, dropout 0.257, lr 4.83e-4, batch 256, weight decay 1e-6 | Phase 4 defaults (tuned on test; see §1) | [single store] |
| Target | y + 1, removed before scoring and elasticity | Safe scaling | [method] |
| Scaler | Robust per-window (median / MAD) | The standard scaler was 2.7× worse | [100k set] |
| Loss | Huber to train, MAE to validate | A simpler loss "left half the estate unresponsive" | [method] |
| Step budget | Scale `max_steps` to panel size: ≈ 174 visits per series, floor 10k, ceiling 250k | Fixed budgets under-trained larger panels | [100k set] |
| LR schedule | 5% linear warm-up, then 3 step decays (γ = 0.5) | Validation oscillation down by a third | [100k set] |
| Stopping | No early stopping. Run the full budget, then restore the best validation weights (checked every 500 steps) | Usable runs went from 3/6 to 6/6, and the spread of forecast error fell 80% | [100k set] |
| Gradient clip | 1.0 | 0.1 was erratic; 5.0 lost accuracy | [100k set] |
| Prices | CPI-deflated | Otherwise inflation looks like a price rise | [single store] |
| Oversampling | Rebalance the training data, built from the training window only; the extra training series are dropped before scoring | More price variation for the model to learn from. The promo flag barely correlates with real discounts (r ≈ −0.03) | [single store] + [100k set] |
| Seeds | At least 3 for any quoted number | Seed noise was bigger than many of the effects | [method] |

### Rejected

| Tried | Result | Tag |
|---|---|---|
| Removing low-volume series | Error 6–11× worse | [100k set] |
| Reweighting training by volume | Error > 30× worse, elasticity positive | [100k set] |
| log1p target | Elasticity turned positive | [100k set] |
| Per-series scaling (a competition winner's approach) | Elasticity weakened sharply; ~25% of series became price-insensitive | [100k set] |
| Automatic divergence cut-off | Fired on every run, including the best one | [100k set] |
| Daily resolution | Noisy and slow | [single store] |

---

## 4. Code sketches [method]

These are generic and written for NeuralForecast. The real versions go in `code_full/gd_helpers.py`.

**Step budget**
```python
def scaled_max_steps(n_series, batch_size=256, visits=174, floor=10_000, ceiling=250_000):
    """Steps needed for each series to be sampled about `visits` times."""
    steps = visits * n_series / batch_size
    return int(min(max(steps, floor), ceiling))
```

**Warm-up, then step decay** (pass to the model as `lr_scheduler=LambdaLR` with `lr_scheduler_kwargs`):
```python
def warmup_then_step(step, warmup_steps, decay_every, gamma=0.5):
    """LR multiplier: linear warm-up, then multiply by gamma every `decay_every` steps."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    return gamma ** ((step - warmup_steps) // decay_every)
```

**Best-weights retention** (a Lightning callback; weights are kept in memory, never written to disk):
```python
import copy
import pytorch_lightning as pl

class BestWeights(pl.Callback):
    """Remember the weights with the lowest validation loss and restore them when fitting ends."""

    def __init__(self, metric="ptl/val_loss"):
        self.metric, self.best, self.best_step, self.state = metric, float("inf"), None, None

    def on_validation_end(self, trainer, model):
        loss = trainer.callback_metrics.get(self.metric)
        if loss is not None and float(loss) < self.best:
            self.best, self.best_step = float(loss), trainer.global_step
            self.state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})

    def on_fit_end(self, trainer, model):
        if self.state is not None:
            model.load_state_dict(self.state)
```

**Health metrics** (screen runs on these; never screen on elasticity):
- Oscillation of the validation curve over the second half of training.
- Best step as a fraction of the budget. Above about 0.65 means the run is still under-trained.
- Share of negative predictions. Discard a run if more than 10% are negative.

---

## 5. Measuring price response

### How the method changed
1. **Phases 1–4: a compound sweep.** Price was multiplied by ±5% to ±25%, and the relative-price features were recomputed with it. Results were reported as shares of perverse, elastic and highly elastic series, and per-class medians against a week-on-week historical elasticity.
2. **Phase 5: the compound sweep is wrong [100k set].** Moving price together with its derived features gave a *positive* elasticity (+0.45). Moving the price channel alone gave the right sign on all series. **From then on, only one price channel moves; derived and promo features stay at their observed values.** The original lab 03 moves `price_index` and `discount_depth` together, and `code_full/04` shows both versions.
3. **Phase 6: a standard sweep [method].**
   - Shift sizes ±5%, ±10% and ±20%.
   - For each shift size, predict at the base price, at −m and at +m.
   - Use unclipped predictions with the +1 removed.
   - Midpoint (arc) elasticity: `e = (q_hi − q_lo) / mean(q_hi, q_lo) / (2m)`. Example: 5.6 vs 4.4 units at ±20% gives −0.6.
   - Report only series with mean ≥ 3 units/week (classes A–I). This is a reporting filter, not a training filter.
   - Always report: n_estimable, share perverse (e > 0), share flat (|e| < 0.05), and the invariance ratio (median at 5% ÷ median at 20%; about 1 for a smooth curve).
   - Headline: median arc elasticity at ±20%, overall and by sales class. Plausibility bound |e| ≤ 2.
4. **Phase 7: simulated discounts.**
   - **Probes** are price shocks of −10%, −20%, −30% and +20%:
     - *sustained*: every horizon week;
     - *pulse*: one week, to show lift in that week and spill into the rest.
     - Pass bar: median elasticity ≤ −0.3 with the right sign.
   - **Synthetic training discounts** add synthetic price changes, with a known lift, to the training data. An "echo" check measures how well the model recovers the planted response.
   - **Experimental.** This is `code_full/05b`.

### Benchmarks
- **Empirical:** a two-way within regression of log(y+1) on log price, with series and week effects removed. It gave about −0.4 to −0.7 for classes A–I [100k set].
- **LightGBM:**
  - It was the most accurate model, but its response shrank as the shock grew, and it was flat outside the range of prices it had seen. That is an artefact of tree splits, not demand.
  - Price features carried about 1.5% of split gain, against over 60% for a 13-week rolling mean.
  - Use LightGBM for accuracy and NBEATSx for price response.
- **Price rises are extrapolation.** Very few sales happen above the regular price, so trust price falls more.

### Plots we relied on
1. **Elasticity panel** (three plots side by side):
   - a histogram of per-series arc elasticity (clipped to [−5, 2]), with the overall median, the responsive-only median and the empirical band shaded;
   - a box plot by sales class;
   - median arc / down / up elasticity against shift size, to test invariance.
2. **Training curve:** train and validation loss (log scale) against step, with lines at the end of warm-up and at the step whose weights were kept.
3. **Arm comparison** (2×2):
   - validation curves for every seed;
   - box-and-jitter by arm for oscillation, for RMSSE, and for elasticity against the empirical band.
4. **Median elasticity by class** against the historical or empirical values.
5. **Item plots:** actuals and forecasts, with price as a step line on a second axis and promo ticks.
6. **Probe chart:** lift by shock size and mode, and spill into weeks that weren't shocked.

---

## 6. Hyper-parameter optimisation [100k set unless marked]

### Samples and storage
- Tune on a stratified sample (sales class × cluster × discount-week bucket).
- A 20k-series sample failed the fidelity check: its validation curve oscillated 4.7× more than the full estate's. A 50k-series sample passed.
- Store the study in Optuna with SQLite so it can resume. Cache fits by a hash of (params, seed, population, window, features).
- HPO fits save summaries, not models.

### Single objective (phase 6b)
- Score: RMSSE on price-moving weeks × an oscillation penalty × a cross-seed variation penalty.
- Gradient clip fixed at 1.0.
- Elasticity tracked the learning rate, and the most accurate trials had the weakest elasticity.

### Two objectives (phase 6c)
- Objective 1: the accuracy score, never shown elasticity. Infeasible if the score > 0.80 or RMSSE > 1.20.
- Objective 2: median arc elasticity at ±20% across seeds. It is set to 0 when the trial is pruned, has only one seed, or has |e| > 2.
- Forced coverage cells (architecture × LR) run before TPE takes over.
- Search space: blocks {1111, 2222, 1221, 3333}, mlp {256, 512, 768}, depth {2, 3}, LR 2e-5 to 1e-3, dropout 0.05–0.30, weight decay 1e-7 to 1e-5.
- **The selection rule is fixed before any trial runs:**
  - the accuracy-optimal point;
  - the knee;
  - two spread points;
  - the incumbent as a control.
- **Knee, by two methods:**
  - largest elasticity gained per unit of accuracy lost, relative to the accuracy-optimal point;
  - largest perpendicular distance from the chord between the extremes, both axes scaled 0–1.
  - If the two disagree, a person decides.

```python
def knee_by_gain(front):
    """front: DataFrame of Pareto trials with columns accuracy (lower is better) and elasticity."""
    base = front.loc[front["accuracy"].idxmin()]
    lost = (front["accuracy"] - base["accuracy"]).replace(0, float("nan"))
    gain = (base["elasticity"] - front["elasticity"]) / lost
    return gain.idxmax()
```

### The confirm run: the most important finding
The front was re-run on the full estate with 6 seeds, on the held-back window.
- **Accuracy transferred.** Configuration explained about half the variance, and the accuracy-optimal configuration was the most stable.
- **Elasticity did not transfer.** It shrank 7–18× (the knee went from −0.30 to −0.04), and no configuration was separable from zero.
- **Why:**
  - mostly the window, not the population;
  - Easter fell in the held-back window;
  - extremes had been selected on a noisy 3-seed mean.
- **Each run is all-negative or all-positive** in its price response, and roughly linear in the size of the price change.
- **Practical rules:**
  - use 6+ seeds for elasticity claims;
  - screen out runs with more than 10% negative predictions;
  - re-inject fixed parameters such as `gradient_clip_val` on confirm runs, because Optuna stores only the parameters it searched over.

---

## 7. Features

| Feature | Use | Tag |
|---|---|---|
| sale price (deflated), price ratio to 4-week rolling mean, 1-week % change, discount depth, promo flag | Future exogenous inputs | [single store] |
| Week, month and quarter sin/cos | Calendar | [method] |
| Event dummies (moving holidays, Christmas, retail events) | Fourier terms can't represent moving events such as Easter. **Experimental** | [100k set] |
| Class, subclass, sales class (one-hot), cluster (one-hot) | Statics | [method] |
| Raw integer hierarchy IDs as floats | Unscaled and meaningless as numbers. Use one-hot or embeddings | [method] |
| Item group | Turned out to equal subclass on every row, so the same feature was fed twice | [single store] |

---

## 8. Hierarchy and clusters

- **Store clustering:** clusters on raw features (Ward linkage) beat autoencoder clusters, because the autoencoder discarded intermittency information (ADI, CV²). [100k set]
- **Pooling:** one pooled GFM with cluster as a static. Separate per-cluster NBEATSx models were **not** trained; only LightGBM cluster arms were tried. `code_full/06` is new work.
- **Reconciliation** (phase 8, **experimental**):
  - The GFM forecasts the bottom level (item × store), with the +1 removed and clipped at 0.
  - Upper levels use a dynamic harmonic regression: ARIMA with Fourier terms and events, guarded at 2× the node's historical maximum, with seasonal naive as the fallback.
  - Reconciled with BottomUp, TopDown, MiddleOut and MinT (ols, wls_struct). MiddleOut was preferred.
  - Scored per level: median RMSSE, bias, WAPE, win rate against seasonal naive.

---

## 9. Open questions carried into `code_full`
- How to represent events (Easter) so that the price response holds in windows that contain them.
- Cosine vs step LR schedule; order of search: architecture → schedule → budget.
- Distribution losses (Tweedie, negative binomial), longer inputs (26, 52 weeks).
- Log-log price form (the current form is semi-log); RRP-anchored price features for NBEATSx.
- Conformal intervals per horizon and group.
- Foundation-model benchmark (Chronos-2) against the selected GFM.
