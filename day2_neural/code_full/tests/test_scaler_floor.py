"""Tests for gd.floor_robust_scaler. Run from the repo root:

    GD_SMOKE=1 .venv/bin/python -m pytest day2_neural/code_full/tests -q
"""

import copy
import os
import sys
from pathlib import Path

import pytest
import torch

os.environ.setdefault("GD_SMOKE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gd_helpers as gd  # noqa: E402
from neuralforecast.common._scalers import TemporalNorm  # noqa: E402


def _windows():
    """[batch 1, 6 weeks, 2 channels]: units that move, and a price that barely does."""
    units = torch.tensor([10.0, 12, 9, 11, 10, 13])
    price = torch.tensor([1.000, 1.001, 1.000, 1.001, 1.000, 0.800])    # a cut in week 6
    x = torch.stack([units, price], dim=-1)[None]
    mask = torch.ones(1, 6, 1)
    mask[:, -1] = 0                                     # the scaler sees the first 5 weeks
    return x, mask


def test_floor_bounds_a_near_flat_covariate_and_leaves_the_target():
    x, mask = _windows()
    plain = TemporalNorm(scaler_type="robust", dim=1)
    floored = gd.floor_robust_scaler(TemporalNorm(scaler_type="robust", dim=1), floor=0.01)
    z_plain, z_floor = plain.transform(x, mask), floored.transform(x, mask)
    assert z_plain[0, -1, 1].abs() > 100                # the library: a 20% cut is huge
    assert z_floor[0, -1, 1].abs() == pytest.approx(0.2 / 0.01, rel=1e-3)
    torch.testing.assert_close(z_plain[..., 0], z_floor[..., 0])       # units: spread > floor
    back = floored.inverse_transform(z_floor, floored.x_shift, floored.x_scale)
    torch.testing.assert_close(back, x)


def test_floor_survives_a_deepcopy_and_needs_a_robust_scaler():
    norm = gd.floor_robust_scaler(TemporalNorm(scaler_type="robust", dim=1), floor=0.01)
    x, mask = _windows()
    torch.testing.assert_close(copy.deepcopy(norm).transform(x, mask), norm.transform(x, mask))
    with pytest.raises(ValueError):
        gd.floor_robust_scaler(TemporalNorm(scaler_type="standard", dim=1), floor=0.01)
