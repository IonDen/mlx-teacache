"""gate_step rejects a calibrated_range given as (hi, lo).

Pure-core: mlx.core + the kernel, no weights, no mflux."""

import mlx.core as mx
import pytest

from mlx_teacache._kernel.cache import TeaCacheState
from mlx_teacache._kernel.gate import gate_step
from mlx_teacache.errors import TeaCacheValueError

_COEFFS = (-1841.022165607874, 848.4417137572868, -131.3554469956159, 8.179509586828413, 0.0)


def _gate(calibrated_range: tuple[float, float]) -> None:
    gate_step(
        TeaCacheState(),
        rel_l1_thresh=0.1,
        coefficients=_COEFFS,
        skip_first=1,
        skip_last=1,
        num_steps=10,
        step_idx=0,
        mod_in=mx.array([1.0]),
        calibrated_range=calibrated_range,
    )


def test_reversed_calibrated_range_raises() -> None:
    """Bug caught: (hi, lo) clamps every delta to hi, silently."""
    with pytest.raises(TeaCacheValueError, match="calibrated_range"):
        _gate((0.5, 0.1))


def test_ordered_and_degenerate_calibrated_ranges_are_accepted() -> None:
    """Bug caught: the check uses >= and rejects a lo == hi range, which is legal."""
    _gate((0.1, 0.5))
    _gate((0.3, 0.3))
