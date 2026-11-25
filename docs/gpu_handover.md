# Handover: running `code_full` on the GPU box

For a coding agent (or person) picking up the Day 2 `code_full` work on the Linux GPU machine. The notebooks were built and smoke-tested on a laptop without a usable GPU. Your job is to run them at full size, fix whatever breaks, and report back.

## Read first
1. `day2_neural/code_full/AGENTS.md`: ground rules, style, the config chain. Follow it.
2. `docs/progress.md`: setup, the run commands, the GPU budget, known limitations and the timing log.
3. `docs/gfm_learnings.md`: the research the notebook choices come from. Context only.

## State at handover
- **Branch:** `graeme/step1-learnings`. All work stays on it.
- **Built:** 12 notebooks (00 → 10, plus 05b), each a jupytext `.py` with a paired `.ipynb`, and the helpers `gd_helpers.py`, `gd_price.py`, `gd_plots.py`.
- **Tested:** 21 unit tests pass. The full chain passes in smoke mode (`GD_SMOKE=1`: 150 series, 20 steps) in TRAIN mode, LOAD mode and as executed notebooks.
- **Results:** none at full size. Every `≈ __` placeholder in the prose and the timing log in `progress.md` waits on this run.
- **Environment:** uv is installed, and CUDA works under torch on this box.
- **Decisions already made:** don't reopen these.
  - Train on the ≈17,241 eligible series; score on the 2,000 ids in `results/eval_ids.csv`.
  - Every model choice is made on the validation window; the test window is reported once.
  - Elasticity ≈ −3.6 on this data is expected (cleaner data than the later research sets).
  - The CPI deflator is the public ABS series.
  - The `<!-- DRAFT: ... -->` tags stay until release.

## Do
1. Set up as in `docs/progress.md` → "GPU box setup": a fresh `.venv` in the repo, pinned requirements, CUDA check, tests.
2. Inside `tmux`, run `bash day2_neural/code_full/run_gpu.sh 00_recap_full 01_gfm_recipe`.
3. **Stop and report after 01.** Give minutes per fit (`results/day2_full/configs/01_recipe.json` → `minutes_per_fit`) and the projected total (≈200 fits × that number, from the budget table in `progress.md`). The user decides whether to trim before going further:
   - `N_TRIALS` in 05 goes first (about 90 fits);
   - then `GRID` in 02.
4. On the go-ahead, run the remaining notebooks in order with `run_gpu.sh`. Check each notebook before the next:
   - the executed notebook has no errors;
   - figures display;
   - the `.ipynb` is under ~3 MB;
   - the config it writes looks sane.
5. When the chain is done:
   - fill in the timing log in `progress.md` from `results/day2_full/gpu_timings.csv`;
   - write a short "Findings from the GPU run" section in `progress.md` with the headline numbers per notebook;
   - commit;
   - ask before pushing.

## Don't
- **Don't edit `day2_neural/code/`.** It is the original labs and is read-only. Never write to `results/day2/`.
- **Don't push** without asking. Commit locally as you go, with small, descriptive messages.
- **Don't commit** `models/`, `.claude/`, session files, `.env`, or anything with personal details. This is a public repo. No client names, internal system names, people's names or usernames in code, docs or commit messages.
- **Don't change a model choice to make a result look better.** If a number looks wrong, report it with the evidence.
- **Don't delete cached results without saying so.** That means `results/day2_full/runs/` or `tables/`. Runs are cached by name, so a changed upstream config means deleting every *later* notebook's runs and tables, then rerunning in order.

## If something breaks
- **Fix it in the `.py`,** then `jupytext --to ipynb --update <nb>.py`. Rerun that notebook with `run_gpu.sh`. TRAIN mode resumes from whatever is cached, so finished fits are not repeated.
- **CUDA out of memory:** lower `batch_size` in `gd.NBEATSX_DEFAULTS` (`gd_helpers.py`) or the notebook's params. `gd.scaled_max_steps` scales the step count from `batch_size`, so the visits per series stay the same. Record the change in `progress.md`.
- **StatsForecast worker errors (00, 08):** set `GD_N_JOBS=4`, or `1`, in the environment.
- **07 (Chronos-2):** downloads `amazon/chronos-2` from Hugging Face on first use, so the box needs internet access or a pre-filled HF cache. Fine-tuning stays off unless `GD_FM_FINETUNE=1`. Don't turn it on without asking.
- **A helper bug:** fix it in `gd_*.py`, add or adjust a test in `tests/test_gd.py`, and run `GD_SMOKE=1 python -m pytest day2_neural/code_full/tests -q`.
- **Changing an upstream config's settings** (not just fixing a crash): delete the cached runs and tables of every later notebook, then rerun in order. Say so in the commit message.

## Report back with
- The setup outcome: test results, `torch` version, GPU model and memory.
- Minutes per fit from 01, and the projected total.
- Per notebook:
  - did it finish;
  - wall-clock time;
  - the headline result (median RMSSE on the eval ids, elasticity at ±20% where relevant);
  - any warnings worth reading.
- Every code change, and why.
