# Day 2 slides: our copy, with the full-panel results

A copy of the v1 Day 2 deck (`day2_neural/v1/slides/` on `lm.day2-slides`, 25 Nov 2026), with
the lab and result slides rewritten around the full-panel notebooks in `../code_full/` and
their results. The v1 deck stays untouched on its branch; this copy is for picking and choosing.

What changed from the v1 deck:
- the agenda's notebook column points at `code_full` (`00`–`11`);
- every lab and "what we found" slide, the questions-answered slide and the take-home use our
  numbers: median RMSSE on the 2,000-item evaluation sample, trained on all 17,241 items, and
  all 17,238 items in notebook 11;
- one new slide, **Making the GFM stable · `05c`** (`img/res_lab05c.png`);
- the result charts `img/res_*.png` are drawn from our results by `make_result_figures.py`,
  which also writes the numbers it uses to `full_results.json`;
- the PDF and the 53-slide `day2_slides_full.md` are not in this copy (they carry v1 numbers);
  the theory slides and figures are the v1 deck's, unchanged.

**`day2_slides.pptx` is now edited by hand** (27 Nov 2026: slides 1–21 rewritten, 37 slides in
all), so it is the master copy. `day2_slides.md` is the earlier generated source and no longer
matches it. Don't rebuild the pptx from the markdown, or the hand edits are lost; write a
rebuilt deck to another file name instead.

Figures:

    .venv/bin/python day2_neural/slides_full/make_result_figures.py   # result charts
    .venv/bin/python day2_neural/slides_full/make_figures.py          # theory figures
    node day2_neural/slides_full/build_pptx.js day2_slides.md day2_slides_generated.pptx   # from this folder

`build_pptx.js` needs `pptxgenjs@3.12.0`, patched as its first comment says (write the
paragraph properties for the first run of a line only). A PDF needs `marp-cli` and a browser:
`marp day2_slides.md --pdf --allow-local-files`.
