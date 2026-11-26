# Day 2 `code_full`: progress and GPU runs

The summary of what we did and found is `docs/day2_full_findings.md`; this file is
the running log.

## Status
| Step | What | State |
|---|---|---|
| 0 | Pinned env (`day2_neural/code_full/requirements.txt`), `AGENTS.md`, README | done (laptop); GPU box to check |
| 1 | `docs/gfm_learnings.md` | approved (23 Nov) |
| 2 | `gd_helpers.py`, `gd_price.py`, `gd_plots.py` + `tests/` (42 passing) | done |
| 3+ | notebooks 00 → 10 (+05b, 07b) | built, prose drafted, style-reviewed; full GPU run done (23–24 Nov), outputs committed |

## Findings from the build so far
- **Panel:** 18,692 series in the data, 17,241 pass the original labs' filters, 2,000 in the eval sample (`results/eval_ids.csv`). By class: A 35, B 90, C 161, D 266, E 400, F 575, G 829, H 1,277, I 2,231, J 11,377.
- **`bootcamp.py:36`** resolves the base folder to `day2_neural/`, so `load_panel()` cannot find `data/` without `BOOTCAMP_DATA_DIR`. `gd_helpers` redirects `bc.DATA_DIR`; `bootcamp.py` is untouched, so the original labs still need the variable on a fresh clone.
- **`regular_price` is unreliable:** 30% of training weeks are priced above it. The promo flag tracks "more than 10% below regular" well (corr 0.62), unlike on the 100k research set. Oversampling therefore keys on `discount_depth ≥ 0.10` (8.5% of weeks) by default; 04 shows both rules.
- **Empirical elasticity on this data is ≈ −3.6** (two-way within regression, classes A–I, ≥3 units/week, 680 eval series). That is far stronger than the 100k set's −0.4 to −0.7, because this panel's price variation is mostly temporary promo cuts with large short-term lifts. The elasticity charts use a band computed from this data (`gp.empirical_band`), not the research constant. Framing agreed: this panel's data is cleaner than the later research sets, so a strong, well-identified response is expected.
- **CPI deflator:** copied from the research repo's weekly deflator for this dataset, trimmed to the data dates and set to 1 on 2025-07-16. Source confirmed: public ABS all-groups CPI.

## GPU box setup (Linux, once)
The data (`data/`, `results/eval_ids.csv`, the CPI csv) is committed, so a clone is all that is needed.
```bash
git clone https://github.com/tackes/time-series-forecasting-bootcamp.git
cd time-series-forecasting-bootcamp
git checkout graeme/step1-learnings

uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r day2_neural/code_full/requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # 2.6.0+cu124 True
GD_SMOKE=1 python -m pytest day2_neural/code_full/tests -q                        # 25 passed
```
- **Use a fresh venv in the clone.** Saved models only reload under the same `neuralforecast` / `torch` versions. Installing the pins into another project's venv could change packages that project relies on. To check an existing venv instead, run `uv pip install --dry-run -r day2_neural/code_full/requirements.txt` inside it. Reuse it only if nothing core (`neuralforecast`, `torch`, `numpy`, `pandas`) would change.
- **Editor:** open the repo folder and pick `.venv/bin/python` as the interpreter and Jupyter kernel. Use it to look at results. Run the training from a terminal inside `tmux`, so a closed editor or dropped SSH session doesn't kill it.

## GPU run
`run_gpu.sh` runs each notebook twice: a TRAIN pass (fits, caches runs/tables/configs under `results/day2_full`, saves the selected models), then a LOAD pass (re-executes from the saved results so the committed `.ipynb` has clean outputs). It stops at the first failure.
- **Logs:** the executed TRAIN-pass copies go to `results/day2_full/gpu_logs/`, which is gitignored.
- **Timings:** wall-clock minutes per pass are appended to `results/day2_full/gpu_timings.csv`.
- **Crash recovery:** a TRAIN pass that dies from memory (CUDA out of memory, a killed kernel) is retried up to `GD_RETRIES` more times (default 2) and resumes from the cache; any other error stops the chain. Failures are logged as `gpu_logs/<nb>_train_failed<N>.err`, and RAM and GPU memory once a minute in `gpu_logs/memory.csv`.
- **One GPU:** every fit uses one device (`gd.trainer_kwargs()` sets `devices=1`), and `run_gpu.sh` defaults `CUDA_VISIBLE_DEVICES=0`. On a two-GPU box, Lightning's default (`devices="auto"`) started distributed training inside the notebook kernel and hung in 01 with no fits made. Use `CUDA_VISIBLE_DEVICES=1` to run on the other card.
```bash
tmux new -s gd                               # reattach later with: tmux attach -t gd
source .venv/bin/activate
bash day2_neural/code_full/run_gpu.sh 00_recap_full 01_gfm_recipe
# check 01's minutes per fit (printed in the notebook and in configs/01_recipe.json) against the
# budget below, then:
bash day2_neural/code_full/run_gpu.sh 02_hpo_round1 03_calendar_hierarchy 04_price_response     05_hpo_round2_pareto 05b_synthetic_discounts 06_cluster_gfms 07_foundation     07b_new_foundation 08_hierarchy_reconcile 09_uncertainty 10_selection_standings
git add -A results/day2_full day2_neural/code_full && git commit -m "run: code_full on GPU" && git push
```
Models land in `models/day2_full/` (gitignored); upload them to the release or hosting decided in open decision 3.

**Order matters.** Run the notebooks 00 → 10 in order, on a clean `results/day2_full/runs` and `tables`. Runs are cached by name, so a changed upstream config is not noticed (see `AGENTS.md`).

## GPU budget (full-panel fits per notebook)
| Notebook | Fits | Notes |
|---|---|---|
| 00 | 0 | StatsForecast on 2,000 series + one LightGBM (CPU minutes) |
| 01 | 9 | 3 arms × 3 seeds; sets minutes per fit (**budget gate**) |
| 02 | 39 | 12-cell grid × 3 seeds + 3 refits; trim `GRID` if 01's timing is high |
| 03 | 12 | 3 arms × 3 seeds on validation + 3 test refits (base reused from 02) |
| 04 | 18 | 3 arms × 3 seeds on validation (choice) + on test (report); arm C ~2× series + LightGBM |
| 05 | ~96–105 | 30 trials × 3 seeds + incumbent + confirm set; `N_TRIALS` is the main dial |
| 05b | 9 | 05 vs 05b on validation (6, the choice) + 05b on test (3) |
| 06 | ~6–7 equivalents | cluster static × 3 + per-cluster models |
| 07 | 0 (+3 opt-in) | Chronos-2 zero-shot; fine-tune only with `GD_FM_FINETUNE=1` |
| 07b | 0 | Chronos-2, Timestral, t0-gamma zero-shot: 4 setups × 2 windows + history and probes |
| 08 | 0 | two item models; ETS, ARIMAX and Chronos-2 on ~2,100 upper nodes; 52 rolling one-step origins |
| 09 | ~4 | quantile model (1 seed) + 3 calibration fits |
| 10 | 0 | |

Total ≈ 200 full-panel fits. Multiply by 01's minutes per fit to get GPU hours. If that's too many, cut `N_TRIALS` in 05 first, then the 02 grid.

## Known limitations
- **Reference price for HPO:** `bc.load_panel` computes `price_index` against the median price before the test window. Notebooks that tune use `gp.prepare(cfg, cutoff=bc.VALID_START)`, which rebuilds it from pre-validation weeks only. The original labs keep the small leak.
- **Validation window has no event week:** 8 Jan – 9 Apr 2025. Validation can judge the Fourier terms but not the event dummies. The test window holds three events: Good Friday 2025 in test week 1 (from 16 Apr), Anzac Day in week 2 (from 23 Apr) and Mother's Day (11 May) in week 4 (from 7 May).
- **CPI deflator look-ahead:** the weekly deflator is interpolated from monthly CPI, so weeks near a cutoff use CPI published later (effect ≈ 0.3% on prices). Accepted.
- **All model choices use the validation window.** 04 (arm), 05 (Pareto selection, plausibility bound) and 05b (05 vs 05b) choose on validation fits. The test window is the reported outcome only.
- **Single-seed exceptions to the 3-seed rule:**
  - the quantile GFM in 09 (intervals);
  - 08's item-level base (the saved seed);
  - 05's reference model in the 05b comparison.
  Saving more seeds in 05 would remove the last two.

## Decisions (23 Nov)
- Plan, learnings doc, CPI source and the −3.6 framing approved.
- DRAFT tags stay until step 11 (release), then are removed.
- 06's report-only verdict on test is accepted.

## Timing log (GPU run, 23–24 Nov)
One RTX 4060 Ti (16 GB) per fit, 2× in the box, 48 cores, 125 GB RAM. Minutes per fit and fits come
from each run's `summary.json` (`gd.list_runs()`); the train-pass total from `gpu_timings.csv`.
Every LOAD pass took under a minute.

| Notebook | Fits | Minutes per fit | Train pass (min) |
|---|---|---|---|
| 00 | 0 | – | < 1 (CPU) |
| 01 | 9 | 7.0 (NBEATSx 8.5, NHITS 4.2) | 63 |
| 02 | 39 | 7.4 | 261 |
| 03 | 12 | 8.6 | 104 |
| 04 | 18 | 8.8 | 160 |
| 05 | 105 | 8.3 | 873 |
| 05b | 9 | 6.2 | 56 |
| 06 | 18 (smaller per-cluster fits) | 3.2 | 58 |
| 07 | 0 (Chronos-2 zero-shot, ≈3 s per window) | – | < 1 |
| 08 | 0 | – | < 1 |
| 09 | 4 | 6.1 | 24 |
| 10 | 0 | – | < 1 |
| **Total** | **214** | **≈ 7.5** | **1,600 (≈ 26.7 h)** |

No retries were needed: no out-of-memory errors and no interrupted trials. Kernel RAM stayed flat
at ≈ 4.9 GB per notebook, except in 05, where it crept from 6.2 to 7.6 GB over 14 h (the Optuna
study). GPU memory held at ≈ 0.6 GB.

## Findings from the GPU run
Median RMSSE on the 2,000-series eval sample, test window, 3-seed means unless noted.

- **Scoring:** 1,999 of 2,000 series are scored. One series has a zero 52-week seasonal-naive
  scale (NaN in `bc.score`), as in the original labs. 10 compares on the 1,996 series every model
  covers.
- **00 baselines:** LightGBM 0.569, AutoETS 0.708, Naive 0.771, SeasonalNaive 0.929.
- **01 recipe:** the recipe matters most. NBEATSx off the shelf 0.773 (seed spread 0.052) against
  0.568 with the recipe (spread 0.012). NHITS with the same recipe scores 0.539, better than
  NBEATSx, at half the fit time; 01 keeps NBEATSx by design (exogenous stack for the price work).
  The best step lands at 95–97% of `max_steps`, so the recipe may be slightly under-trained.
- **02 HPO round 1:** the best of 12 cells is the recipe itself (validation 0.613); no other cell
  is within the noise floor (seed sd 0.005), so the shortlist has one entry and the test score
  equals 01's. Round 1 confirms the recipe.
- **03 calendar and hierarchy:** both calendar arms are clearly worse on validation (0.717 and
  0.703 against 0.613), and their best step falls at about half the budget: they overfit with
  Fourier and event inputs. Calendar + hierarchy is still worse on test (0.647), which holds three
  event weeks. The base arm goes forward.
- **04 price response:** CPI deflation helps both accuracy and the price response; oversampling
  promo weeks steepens the response further.
  | Arm | Valid | Test | e20 valid | e20 test | Perverse (test) |
  |---|---|---|---|---|---|
  | A features | 0.606 | 0.568 | −0.54 | −0.60 | 1.0% |
  | B + CPI | 0.600 | 0.540 | −1.04 | −1.13 | 0.6% |
  | C + CPI + oversampling (chosen) | 0.593 | 0.553 | −1.33 | −1.25 | 0.1% |
  Every model's response stays far weaker than the empirical −3.6 (band −4.6 to −3.4); LightGBM's
  is −0.66.
- **05 HPO round 2 (30 trials):** the accuracy-optimal trial wins (a smaller net: one block per
  stack, dropout 0.10): validation 0.583, test 0.546, e20 −1.28 on test, negative in every seed.
  The knee settings missed the accuracy limit on validation by ≈ 0.003. Elasticity barely moves
  across the Pareto front (−1.3 to −1.4): tuning moved accuracy, not the price response.
- **05b synthetic discounts:** re-decided on 25 Nov with the phase 7b probes (see below). 05 goes
  forward.
- **06 clusters:** pooled wins (0.546) over the cluster static (0.551) and one model per cluster
  (0.570). Per-cluster models lose most in the small clusters.
- **07 foundation model:** **Chronos-2 zero-shot with the promo and price covariates is the best
  model on the board** (validation 0.575, test 0.512), 0.034 ahead of the selected GFM with no
  training. Without covariates it scores 0.645. Its price response is probed in 07 §8 (below).
- **08 hierarchy:** a few item forecasts from the saved model exploded (one item that averages 128
  units a week was forecast at ≈ 2,900× its test sales), none of them in the eval sample. On its
  own that put the bottom-up store total 47% too high. 08 now caps each item at 3× its largest
  training week (568 item-weeks in 135 items; real sales went past that in ≈ 0.5% of test weeks).
  With the cap, BottomUp is best at every level (total 0.843 against ETS's 0.891, bias +3.5%) and
  is chosen. The likely cause is the robust scaler on spiky series; worth a look in a later
  round.
- **09 intervals:** at 80%, conformal by step hits the target (0.791 coverage, interval score
  3.56); the quantile GFM over-covers (0.864, 3.60). At 95%, conformal's interval score is 4–9×
  worse than the quantile GFM's (31.9 against 7.1): a few large misses dominate the mean.
- **10 standings:** `final_model` was Chronos2_zs_cov (0.512), beating the selected GFM by more
  than the noise floor. Since 07b (25 Nov, below) it is T0gamma_zs_cov (0.502), and the held-out
  top-3 mean (t0-gamma, Chronos-2, Timestral) scores 0.495.
- **Notebook blanks filled (24 Nov):** every `≈ __` / `TODO: fill after GPU run` placeholder now
  holds the number from the executed notebook. The DRAFT tags stay for the prose edit.

## Price response measured with the phase 7b probes (25 Nov)
The old measure (`e20`, the sweep) moved `price_index` alone over all 14 weeks, so a cut arrived
with no discount depth. The new probes in `gd_price` follow the research's phase 7b: shock the
price and **rebuild every input derived from it** (`discount_depth`,
`price_ratio_to_rolling_mean`), sustained or in one week (test week 5, after the event weeks),
with and without the promotion flag. They add **random promotion calendars**: per series, five
draws of three one-week cuts at 10–30%, in weeks with no event and no planned promotion. The
headline is a one-week 20% cut, every input rebuilt (`pulse_all`). The sweep stays in 04/05
(they chose with it). Reruns: 04, 05b (12 refits, 75 min), 07, 10.

| One-week 20% cut (test) | Median RMSSE | Elasticity | Lift | + promo flag | Calendar cuts right-signed | Lift at 10 / 20 / 30% |
|---|---|---|---|---|---|---|
| Chronos2_zs_cov | 0.512 | −1.43 | +37% | +43% | 89.6% | +18 / +45 / +75% |
| GFM_selected (05) | 0.546 | −1.79 | +49% | +59% | 99.4% | +26 / +57 / +89% |
| GFM_synthetic (05b) | 0.546 | −1.73 | +47% | +59% | 99.3% | +26 / +55 / +90% |
| LightGBM | 0.568 | −1.10 | +28% | +43% | 97.6% | +10 / +27 / +44% |
| implied by the data (e ≈ −3.6) | | −3.46 | +125% | | | +47 / +125 / +266% |
| planted in 05b's copies | | −2.08 | +60% | | | +35 / +60 / +150% |

- **Every model is right-signed and too weak:** the best gives back about 40% of the lift the
  data implies. Moving the price alone understates the response: LightGBM's one-week cut gives
  +5% with price alone, +28% with the rebuilt depth.
- **05b re-decided on validation:** the one-week cut is a tie (−1.7697 for both, equal to five
  decimals), 05b has slightly more right-signed calendar cuts (99.6% against 99.4%), but its
  validation RMSSE is 0.593 against 0.583 (gap 0.011, noise floor 0.005). It fails two checks;
  `use` stays "05", so 06–09 did not need rerunning. On test, 05b ties on accuracy and is
  slightly weaker. The 05 reference refit reproduced 05's stored forecasts exactly (gap 0.0).
- **Echo check:** 05 already gives back about the planted lift up to a 20% cut; at 25–30% both
  models fall well short of it (+90% against +150% at 30%). The copies taught 05b nothing new.
- **Chronos-2** reads the price in context: a lift that grows with depth and no spill into other
  weeks, but it is the least consistent (≈ 10% of calendar cuts do not lift its forecast).
- **RMSSE in 10's table** is on the 1,996 series every model covers, so GFM_synthetic shows 0.543
  there (0.546 on all 1,999): the four series dropped sit near its median.
- **Stale cache folders:** the old-method fits (`runs/05b_valid_05`, `runs/05b_valid_05b`,
  `runs/05b_synthetic`, `tables/05b_reference_*`) and the old `tables/08_upper_base` were
  deleted on 25 Nov; the current runs carry the `_p7b` suffix.

## Hierarchy: price-aware upper nodes, two item models, fixed MinT (25 Nov)
08 now gives the upper nodes their items' sales-weighted price, promotion and event inputs
(`gp.aggregate_covariates`) and chooses the upper model on validation from ETS, ARIMAX and
Chronos-2. It reconciles two item models (the selected GFM, one seed, and Chronos-2), adds
`MinT-wls_var` on the full tree and `MinT-shrink-sub` (dense `mint_shrink` to subclass, items
split by share), with MinT's past errors from 52 rolling one-step forecasts.

| Test, median RMSSE | total | department | class | subclass | item |
|---|---|---|---|---|---|
| GFM items, BottomUp | 0.843 | 0.811 | 0.822 | 0.778 | **0.725** |
| GFM items, MinT-wls_var | 0.857 | 0.829 | **0.811** | **0.751** | 0.727 |
| GFM items, MinT-shrink-sub | **0.660** | **0.810** | 0.786 | 0.775 | 0.731 |
| Chronos-2 items, BottomUp | 0.767 | 0.693 | 0.684 | 0.676 | 0.676 |

- **Upper model: ETS.** Chronos-2 with the node inputs beat it by ≈ 28% at department, class
  and subclass on validation, but was 4% worse at the total; the worst-level rule picks ETS.
  ARIMAX was more than twice as bad at the total (1.38 against 0.57).
- **Both item models choose BottomUp.** With GFM items, `MinT-wls_var` is inside the 1% tie
  band (worst level 0.3% behind, at the item level) and the preference order picks BottomUp.
- **Library bug, worked around:** `MinTraceSparse` solves its system with unpreconditioned
  `bicgstab` and ignores the exit code. With this panel's spread of error variances it stopped
  early: `wls_var` forecasts reached the millions and each reconcile took ≈ 37 min.
  `gd.MinTraceDirect` keeps the library's weights and solves exactly (Woodbury identity, one
  2,118 × 2,118 Cholesky factor): ≈ 16 s per item model, and on a 4,000-item test tree it
  agrees with an exact least-squares solve to 1e-14 where the library is off by 3.4×.
  `MinT-ols` moved slightly too (it had not fully converged either).
- **Board:** only `HIER_GFM_BottomUp` goes on it. `HIER_Chronos_BottomUp` would be
  `Chronos2_zs_cov` again, capped, so 08 no longer adds it.
- **Cost:** the first train pass took 93 min: ≈ 74 of them the library's failing solve, and
  ≈ 16 Chronos-2's rolling one-step forecasts. Everything is cached, so with the fixed solve a
  pass now takes ≈ 1 min.

## Newer foundation models: Timestral and t0-gamma (07b, 25 Nov)
Zero-shot on the 2,000 eval series, with Chronos-2 rerun as the reference (it reproduces 07's
forecasts exactly). Four covariate setups; the main one (`price`: `promo`, `discount_depth`,
`price_index`) was fixed before the results and goes on the board.

| Main setup | Validation | Test | Promo weeks* | Above the history* | One-week 20% cut lift | Licence (weights) |
|---|---|---|---|---|---|---|
| t0-gamma | **0.536** | **0.502** | 3.81 | 60.6 | +30% | Apache-2.0 |
| Chronos-2 | 0.575 | 0.512 | 3.67 | 58.8 | +37% | Apache-2.0 |
| Timestral-209M | 0.557 | 0.523 | **3.36** | 52.1 | +50% | Modified MIT ($20M/month cap) |
| GFM_selected | | 0.546 | 3.47 | **51.0** | +49% | |

\* mean absolute error scaled by the 52-week seasonal-naive error (the RMSSE scale).

- **t0-gamma is the new final model in 10** (median 0.502 on 10's 1,996 series). Its lead over
  Chronos-2 is real: twice the noise floor, Wilcoxon p ≈ 0, better on 54% of series.
  Timestral against Chronos-2 is not a real difference (p = 0.88).
- **It uses the price inputs** (validation 0.703 without them), but responds least to a price
  cut of the three; every model gives back well under half the lift the data implies (+125%).
  Agreed framing: t0-gamma for the forecast, with the price-response caveat stated; the GFM
  stays the reference for price questions.
- **The extra inputs matter little:** events and the tree-style columns move each model by
  0.025 or less, in no consistent direction. Every model needs the full history (13 or 26
  weeks: ≈ 0.60).
- **Averages of two** beat both members (t0-gamma + Chronos-2: 0.492), but per-series errors
  correlate at 0.99.
- **Cost:** the train pass took 11 min on one GPU. Main setup ≈ 3 s per window for Chronos-2,
  6 s for t0-gamma, 13 s for Timestral (2,000 series).
- **Board filter:** each notebook's board now shows only its own and earlier notebooks' rows
  (`gd.median_board("<NN>")`), and 07b's significance column is "real difference" (gap above
  the noise floor and p < 0.05).

## Every series: 11 rescores the main models on all 17,238 (25 Nov)
- **Same evaluation as the v1 labs:** the baselines, LightGBM and Chronos-2 match their
  all-series numbers to three decimals, so the panel, filter, windows and WAPE are the same.
- **Our lead is the training:** NHITS with our recipe (≈ 10k steps, Huber, warm-up, clip,
  best weights, 3 seeds) scores WAPE 0.244 against the v1 NHITSx's 0.275 (500 steps, MAE,
  standard scaler, 1 seed).
- **Exploding N-BEATSx forecasts** outside the sample: every seed has some (65–129 items above
  3× their largest week, up to ≈ 8,000×); the saved seed 46335 is worst (uncapped WAPE 0.699).
  NHITS barely has them. Cap at 3× the item's largest training week (08's rule) applied to
  every model in 11; on the sample the cap changes nothing at three decimals, so 00–10 stand.
- **Refits repeat their originals:** GFM_selected seeds 11207, 90218 and the three NHITS seeds
  reproduce the cached runs' sample scores to four decimals.
- **Left out:** GFM_synthetic (its saved model predates 05b's final refit).
- **Cost:** 30 min on one GPU (5 refits, 4–6 min each; foundation models and baselines are
  minutes). Scores in `results/day2_full/leaderboard_all_series.csv`, forecasts in
  `results/day2_full/all_series/`.


## 05c: flooring the robust scaler's spread (26 Nov)
<!-- DRAFT: Graeme to edit -->
- **Diagnosis first (read-only script):** in exploding items, `promo` is often exactly flat
  then moves, but the scaler's fallback keeps it near 1. `discount_depth` and `price_index`
  are near-flat (spread ≈ 0.0004 and 0.001, from CPI drift), so a discount scales to hundreds
  of units. Both move together on every discount, so dropping one would not help.
- **Arm:** 05's settings with the robust scaler's spread floored at 0.01
  (`gd.floor_robust_scaler`), 3 seeds on validation and 3 on the test window, all series.
- **Result:** accuracy and price response pass (validation 0.574 against 0.583; one-week cut
  elasticity −1.95 against −1.77); explosions only halve (466 → 236 item-weeks, 101 → 56
  items), short of the rule's tenfold. **The cap stays.**
- **Found on the way:** `1120_S1` has `price_index` values up to 29,000 (a price in the wrong
  units); 29 series exceed 3 somewhere in training. With the floor it carries almost all of
  the uncapped WAPE. One seed (90218) also blows up a normal-priced item (`6653_S1`, ≈ 550×).
- **Open:** the floor plus the cap beats 05 (capped all-series median 0.511 against 0.540) but
  was not in the rule; cleaning out-of-range prices would help every price-aware model.
- **Cost:** 37 min on one GPU (6 fits of ≈ 6 min).

## 05c part 2: cleaning the prices (26 Nov)
<!-- DRAFT: Graeme to edit -->
- **`price_index`** = the week's price ÷ the series' median price before the cutoff (1 =
  usual, 0.8 = 20% off). Bad prices: £0.01–0.05 placeholders (229 weeks, 79 series; they
  look like 99% discounts) and prices beyond 6× or 1/6 of the usual price (196 series in
  all). `gp.clean_prices` (config key `price_clean`) replaces them with the median of good
  prices in the 9 weeks around; 828 weeks in 196 series, 40 weeks in the sample.
- **Not missing** (N-BEATSx can't take missing future inputs) and **no +1 shift** (no price
  is ≤ 0; a shift would shrink cheap items' discounts more than dear ones').
- **Clean alone:** validation 0.599 against 05's 0.583 (fails), and *more* exploding items
  (156 against 101): fewer fake discounts leave more near-flat windows.
- **Clean + floor (recommended):** validation 0.562, elasticity −2.03 against −1.77; test,
  capped, all series 0.514 / WAPE 0.256 (05: 0.540 / 0.258). Uncapped it still explodes
  (WAPE 221, one item: `1120_S1`, real prices 2–5× its input window). The floor bounds small
  moves, not large ones; clipping the scaled inputs would be the next arm. **Cap stays.**
- **Cost:** 74 min on one GPU (12 fits of ≈ 6 min).

## 05c part 3: clipping the scaled price inputs at ±50 (26 Nov)
<!-- DRAFT: Graeme to edit -->
- **Clip level from the inputs alone:** with the floor, the scaled `price_index` has a 99th
  percentile of ≈ 43; ±10 would flatten 40% of series' real price signal, ±50 touches ≈ 0.5%
  of item-weeks.
- **Result:** validation 0.567 against clean + floor's 0.562 (a tie; the rule picks clean +
  floor). Worst forecast 300× the item's largest week (clean + floor: 2 × 10⁸). Capped,
  all-series WAPE 0.248, the best GFM.
- **The last explosion is inside the network:** seed 46335 forecasts `6653_S1` (a best
  seller) at ≈ 900× with every scaled input within ±5. No input fix can reach it, so the cap
  stays as the guard.
- **Cost:** 36 min on one GPU (6 fits).

## Carrying clean + floor + clip forward; 00 and 06–11 on cleaned prices (26 Nov)
<!-- DRAFT: Graeme to edit -->
- **Choice:** clean + floor + clip, with the cap, over the rule's tie-break (clean + floor,
  0.005 better on validation, inside the noise floor), for stable training calculations.
  Recorded in `05c_input_clip` (`use`, `chosen_by`, `choice_reason`); reasons and the effect
  of clipping on the loss surface in `docs/stability_choices.md`.
- **Plumbing:** `gd.selected_config()` returns 05c's arm (floor and clip in `params`, applied
  by `gd.build_nbeatsx`); saved models carry `stability.json`, reapplied by `gd.load_model`;
  `gp.PRICE_CLEAN` for LightGBM (00, 04, 11) and NHITS (11); `gp.selected_gfm_probes` for
  07/07b; 11 uses 05c's all-series fits instead of refits. The selected model's saved
  forecasts and board row are capped (`05c_selected` saved for 08, 09, 11).
- **Old caches:** the raw-price tables/runs of 00, 04's LightGBM and 06–11 moved to
  `results/day2_full_archive_raw_prices/` (gitignored; in git at `b76e787`).
- **Results (test, sample):** t0-gamma 0.501 (final, unchanged), Chronos-2 0.512, **GFM
  selected 0.520** (05: 0.546), Timestral 0.521, LightGBM 0.572 (0.569 raw). Top-3 average
  0.495 held out. All series, capped: GFM 0.513 / WAPE 0.248 (was 0.540 / 0.258). GFM price
  lift +50% (−1.82), 99.4% right-signed. Foundation models unchanged by cleaning (≤ 0.002).
- **Remaining weak spot:** seed 46335 (also the saved model) blows up `6653_S1`; capped, it
  still makes the sample WAPE 0.399 and 08's GFM bottom-up total ≈ 4% too high.
- **Cost:** ≈ 2.5 h on one GPU (06 ≈ 1 h, 09 25 min, 08 21 min, 11 16 min, 07b 11 min).

## Day 1 comparison and the price-plan leak test, 12 (26 Nov)
<!-- DRAFT: Graeme to edit -->
- **Nothing new from Day 1 on GitHub:** `day1-framework`'s tip (19 Nov) is already in our
  branch; `main`'s only new change is the v1 Day 2 update (it moves `bootcamp.py` to
  `day2_neural/v1/code/`, same scoring; merging needs `gd_helpers.py` repointed).
- **Same scoring as Day 1:** Naive 0.771 and SeasonalNaive 0.929 on the 2,000 series both ways.
  Day 1's best no-future-information LightGBM: 0.664 / WAPE 0.344.
- **The fingerprint:** 85% of zero-sale test weeks are priced exactly at the regular price;
  weeks at regular have no sales 66% of the time, others 1.7%.
- **12's result:** removing it (zero-sale weeks at regular take last week's price; 4.5% of test
  weeks) costs 0.005–0.008 of median RMSSE for every model; no future prices at all costs
  0.11–0.15 more (0.65–0.69, level with Day 1). Rankings unchanged.
- **Cost:** ≈ 5 minutes on one GPU (inference only, LightGBM refit on CPU).
