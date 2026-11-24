# Day 2 `code_full`: progress and GPU runs

## Status
| Step | What | State |
|---|---|---|
| 0 | Pinned env (`day2_neural/code_full/requirements.txt`), `AGENTS.md`, README | done (laptop); GPU box to check |
| 1 | `docs/gfm_learnings.md` | approved (23 Nov) |
| 2 | `gd_helpers.py`, `gd_price.py`, `gd_plots.py` + `tests/` (25 passing) | done |
| 3+ | notebooks 00 → 10 (+05b) | built, prose drafted, style-reviewed; full GPU run done (23–24 Nov), outputs committed |

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
bash day2_neural/code_full/run_gpu.sh 02_hpo_round1 03_calendar_hierarchy 04_price_response     05_hpo_round2_pareto 05b_synthetic_discounts 06_cluster_gfms 07_foundation     08_hierarchy_reconcile 09_uncertainty 10_selection_standings
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
| 08 | 0 | one prediction for all series + AutoETS on ~2,100 upper nodes |
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
- **05b synthetic discounts:** 05b lost on validation by one series' worth on "more right-signed
  series" (98.0% against 98.1%) and passed the other two checks; on test it would have won all
  three (0.543, e20 −1.43, 97.6% right-signed). 05 goes forward. The echo test shows every model
  recovers only a third to a half of a planted price response.
- **06 clusters:** pooled wins (0.546) over the cluster static (0.551) and one model per cluster
  (0.570). Per-cluster models lose most in the small clusters.
- **07 foundation model:** **Chronos-2 zero-shot with the promo and price covariates is the best
  model on the board** (validation 0.575, test 0.512), 0.034 ahead of the selected GFM with no
  training. Without covariates it scores 0.645. Its price response is not measured in 07.
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
- **10 standings:** `final_model` is Chronos2_zs_cov (0.512), beating the selected GFM by more
  than the noise floor. On the held-out half, the mean of the top three (Chronos-2, GFM_synthetic,
  NHITS) scores 0.507, the best of all.
- **Notebook TODOs:** the prose placeholders (`__`, `TODO: fill after GPU run`) are not filled
  in; the numbers above are the source for them.
