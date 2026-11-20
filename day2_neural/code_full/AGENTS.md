# Working on `day2_neural/code_full`

Notes for anyone (person or coding agent) editing these notebooks.

## What this folder is
A full-data version of the Day 2 labs. Each notebook trains on **every eligible series** (≈17,241) and scores on the **fixed 2,000-series evaluation sample** used by `day2_neural/code`. Each one builds on the config written by the one before. The research behind the choices is summarised in `docs/gfm_learnings.md`.

## Ground rules
- **`day2_neural/code/` is read-only.** `bootcamp.py` is imported, never edited. `gd_helpers.py` points its paths at `results/day2_full/`; nothing is ever written to `results/day2/`.
- **Same evaluation.**
  - Score with `gd.record` / `gd.score`. They restrict `bc.record` / `bc.score` to the eval ids.
  - Sort and discuss the board by **median** RMSSE, with the mean alongside.
- **Tune on validation, test once.**
  - HPO and model choice use the window from `bc.VALID_START`. Pass `cutoff=` explicitly to `bc.split`.
  - Per-series statistics use training weeks only: scaling, statics, oversampling counts, the reference price and clusters.
- **Configs carry ideas forward.** Each notebook writes `results/day2_full/configs/NN_<name>.json` and reads the previous one rather than hardcoding settings.
- **Train once, load on the day.**
  - `GD_TRAIN=1` fits and saves; the default only loads.
  - Heavy work sits inside `gd.fit_seed(s)` or `gd.cached(...)`, which cache per run, so a crash resumes.
- **Saving and size.**
  - No Lightning checkpoints or logs (`bc.trainer_kwargs()`). `BestWeights` keeps the best weights in memory.
  - HPO trials save summaries, not models.
  - `save_as=` (which uses `nf.save(save_dataset=False)`) only for configs a later notebook loads. Models go to `models/` (gitignored, hosted outside the repo).
- **Figures are displayed.**
  - Call `gplt.use_style()` once, and use the `gd_plots` functions (F1–F12; see its docstring).
  - Each `.ipynb` stays under ~3 MB.
- **3 seeds minimum** for any quoted number (`gd.SEEDS`).
- **Public repo.** No client names, internal system names, people's names or client-identifying figures.

## Style
- Jupytext percent format (`# %%`, `# %% [markdown]`). Edit the `.py`, then sync the `.ipynb`.
- Plumbing goes in `gd_helpers.py` / `gd_price.py` / `gd_plots.py`: short functions with docstrings. Model set-up, features and results stay visible in cells.
- Lines at most 100 characters; imports at the top of the notebook; no tuple-expression one-liners, nested ternaries or dense comprehensions.
- Prose in British English, in the structure of the original labs:
  - title with minutes, then goals;
  - `## Intuition:`;
  - numbered sections, with `### Reading the result` after key outputs;
  - `## Discussion` and `**Try it:**`.
- Prose not yet reviewed carries `<!-- DRAFT: Graeme to edit -->`.

## Running
```bash
# CPU smoke test (tiny panel, few steps; writes results/day2_full_smoke):
GD_SMOKE=1 GD_TRAIN=1 BOOTCAMP_ACCELERATOR=cpu jupyter nbconvert --to notebook --execute <nb>.ipynb --output /tmp/smoke.ipynb
# Unit tests:
GD_SMOKE=1 python -m pytest day2_neural/code_full/tests -q
```
See `docs/progress.md` for the GPU commands.
