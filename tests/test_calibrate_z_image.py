"""Unit tests for the pure helpers in scripts/calibrate_z_image.py.

Pure-core (mflux-free): imports the calibration script (mlx + numpy + the
model-agnostic _kernel.gate only; mflux is imported lazily inside main()).
NOT added to conftest._MFLUX_FILES — runs in the pure-core lane.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from calibrate_z_image import fit_signal  # noqa: E402  (after sys.path setup)


def test_fit_signal_free_recovers_known_degree4_polynomial():
    true = [2.0, -0.5, 1.0, 3.0, 0.7]  # c4..c0
    xs = list(np.linspace(0.0, 1.0, 50))
    ys = [float(np.poly1d(true)(x)) for x in xs]
    out = fit_signal(xs, ys, fit_mode="free")
    assert out["fit_r_squared"] > 0.9999
    for got, exp in zip(out["coefficients_c4_to_c0"], true, strict=True):
        assert abs(got - exp) < 1e-3


def test_fit_signal_origin_forces_c0_zero():
    xs = list(np.linspace(0.05, 1.0, 40))
    ys = [float(0.8 * x + 0.3 * x**2) for x in xs]
    out = fit_signal(xs, ys, fit_mode="origin")
    assert out["coefficients_c4_to_c0"][-1] == 0.0  # c0 forced through the origin
    assert len(out["coefficients_c4_to_c0"]) == 5


def test_fit_signal_reports_curve_range_and_pairs():
    # >=5 points: a degree-4 fit on fewer is rank-deficient (RankWarning -> error
    # under the repo's filterwarnings=error). Real calibration has hundreds.
    xs = list(np.linspace(0.1, 0.4, 10))
    ys = list(np.linspace(0.0, 0.9, 10))
    out = fit_signal(xs, ys, fit_mode="free")
    assert out["x_min"] == pytest.approx(0.1)
    assert out["x_max"] == pytest.approx(0.4)
    assert out["y_min"] == pytest.approx(0.0)
    assert out["y_max"] == pytest.approx(0.9)
    assert out["n_pairs"] == 10


def test_fit_signal_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown fit_mode"):
        fit_signal([0.1, 0.2], [0.1, 0.2], fit_mode="bogus")
