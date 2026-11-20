# Day 2 `code_full`: progress and GPU runs

## Status
| Step | What | State |
|---|---|---|
| 0 | Pinned env (`day2_neural/code_full/requirements.txt`), `AGENTS.md`, README | done (laptop); GPU box to check |
| 1 | `docs/gfm_learnings.md` | draft, awaiting approval |
| 2 | `gd_helpers.py`, `gd_price.py`, `gd_plots.py` + `tests/` (12 passing) | done |
| 3+ | notebooks 00 → 10 | in progress |

## Findings from the build so far
- **Panel:** 18,692 series in the data, 17,241 pass Lei's filters, 2,000 in the eval sample (`results/eval_ids.csv`). By class: A 35, B 90, C 161, D 266, E 400, F 575, G 829, H 1,277, I 2,231, J 11,377.
- **Lei's `bootcamp.py:36`** resolves the base folder to `day2_neural/`, so `load_panel()` cannot find `data/` without `BOOTCAMP_DATA_DIR`. `gd_helpers` redirects `bc.DATA_DIR`; Lei's file is untouched. (Worth telling Lei: his labs fail on a fresh clone.)
- **`regular_price` is unreliable:** 30% of training weeks are priced above it. The promo flag tracks "more than 10% below regular" well (corr 0.62), unlike on the 100k research set. Oversampling therefore keys on `discount_depth ≥ 0.10` (8.5% of weeks) by default; 04 shows both rules.
- **Empirical elasticity on this data is ≈ −3.6** (two-way within regression, classes A–I, ≥3 units/week, 680 eval series). That is far stronger than the 100k set's −0.4 to −0.7, because this panel's price variation is mostly temporary promo cuts with large short-term lifts. The elasticity charts use a band computed from this data (`gp.empirical_band`), not the research constant.
- **CPI deflator:** copied from the research repo's weekly deflator for this dataset, trimmed to the data dates and set to 1 on 2025-07-16. *Confirm its source is the public ABS all-groups CPI before release.*

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

## Timing log (fill in from the GPU runs)
| Notebook | Fits | Minutes per fit | Total |
|---|---|---|---|
| 01 | | | |
