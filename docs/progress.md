# Day 2 `code_full`: progress and GPU runs

## Status
| Step | What | State |
|---|---|---|
| 0 | Pinned env (`day2_neural/code_full/requirements.txt`), `AGENTS.md`, README | done (laptop); GPU box to check |
| 1 | `docs/gfm_learnings.md` | approved (23 Nov) |
| 2 | `gd_helpers.py`, `gd_price.py`, `gd_plots.py` + `tests/` (12 passing) | done |
| 3+ | notebooks 00 → 10 (+05b) | built, prose drafted, style-reviewed, review fixes applied; clean CPU smoke chain passes (train, load, executed) |

## Findings from the build so far
- **Panel:** 18,692 series in the data, 17,241 pass the original labs' filters, 2,000 in the eval sample (`results/eval_ids.csv`). By class: A 35, B 90, C 161, D 266, E 400, F 575, G 829, H 1,277, I 2,231, J 11,377.
- **`bootcamp.py:36`** resolves the base folder to `day2_neural/`, so `load_panel()` cannot find `data/` without `BOOTCAMP_DATA_DIR`. `gd_helpers` redirects `bc.DATA_DIR`; `bootcamp.py` is untouched, so the original labs still need the variable on a fresh clone.
- **`regular_price` is unreliable:** 30% of training weeks are priced above it. The promo flag tracks "more than 10% below regular" well (corr 0.62), unlike on the 100k research set. Oversampling therefore keys on `discount_depth ≥ 0.10` (8.5% of weeks) by default; 04 shows both rules.
- **Empirical elasticity on this data is ≈ −3.6** (two-way within regression, classes A–I, ≥3 units/week, 680 eval series). That is far stronger than the 100k set's −0.4 to −0.7, because this panel's price variation is mostly temporary promo cuts with large short-term lifts. The elasticity charts use a band computed from this data (`gp.empirical_band`), not the research constant. Framing agreed: this panel's data is cleaner than the later research sets, so a strong, well-identified response is expected.
- **CPI deflator:** copied from the research repo's weekly deflator for this dataset, trimmed to the data dates and set to 1 on 2025-07-16. Source confirmed: public ABS all-groups CPI.

## GPU run (per notebook, in order)
```bash
git pull && git checkout <branch>
uv pip install -r day2_neural/code_full/requirements.txt      # same versions as the laptop
export BOOTCAMP_ACCELERATOR=gpu
cd day2_neural/code_full
jupytext --to ipynb --update <nb>.py
# 1) train: fits, caches runs/tables/configs under results/day2_full, saves selected models
GD_TRAIN=1 jupyter nbconvert --to notebook --execute <nb>.ipynb --output /tmp/<nb>_train.ipynb \
    --ExecutePreprocessor.timeout=-1
# 2) load: re-execute from the saved results so the committed notebook has clean outputs
jupyter nbconvert --to notebook --execute --inplace <nb>.ipynb --ExecutePreprocessor.timeout=-1
cd ../.. && git add -A results/day2_full day2_neural/code_full && git commit -m "run: <nb> on GPU" && git push
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

## Timing log (fill in from the GPU runs)
| Notebook | Fits | Minutes per fit | Total |
|---|---|---|---|
| 01 | | | |
