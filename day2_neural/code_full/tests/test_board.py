"""Tests for gd.median_board's notebook filter. Run from the repo root:

    GD_SMOKE=1 .venv/bin/python -m pytest day2_neural/code_full/tests -q
"""

import os
import sys
from pathlib import Path

import pandas as pd

os.environ.setdefault("GD_SMOKE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gd_helpers as gd  # noqa: E402


def test_median_board_keeps_this_notebook_and_earlier(monkeypatch):
    labs = ["00 baselines", "05 selected", "05b synthetic", "07 foundation",
            "07b new foundation", "08 hierarchy", "10 ensembles"]
    board = pd.DataFrame({"model": [f"m{k}" for k in range(len(labs))], "lab": labs,
                          "median_rmsse": 0.5, "mean_rmsse": 1.0})
    monkeypatch.setattr(gd.bc, "leaderboard", lambda include_day1=False: board)
    assert set(gd.median_board("07")["lab"]) == set(labs[:4])
    assert set(gd.median_board("07b")["lab"]) == set(labs[:5])
    assert set(gd.median_board("05")["lab"]) == {"00 baselines", "05 selected"}
    assert len(gd.median_board()) == len(labs)
