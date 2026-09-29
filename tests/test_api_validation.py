import math

import numpy as np
import pytest

from mlx_teacache import TeaCacheValueError, apply_teacache


class _NoModel:  # validation runs before dispatch, so no model is needed
    pass


@pytest.mark.parametrize("kw", ["skip_first_n_steps", "skip_last_n_steps"])
@pytest.mark.parametrize("bad", [math.nan, 1.5, None, True, "1", -1])
def test_skip_window_rejects_non_int(kw: str, bad: object) -> None:
    """Bug caught: NaN passes `< 0`, silently removing the forced window (last step skipped)."""
    with pytest.raises(TeaCacheValueError, match=kw):
        apply_teacache(_NoModel(), **{kw: bad})


@pytest.mark.parametrize("kw", ["skip_first_n_steps", "skip_last_n_steps"])
def test_skip_window_accepts_numpy_int(kw: str) -> None:
    """Bug caught: an isinstance(int) check rejects np.int64."""
    from mlx_teacache import IncompatibleModelError

    with pytest.raises(IncompatibleModelError):  # got past validation to dispatch
        apply_teacache(_NoModel(), **{kw: np.int64(1)})


@pytest.mark.parametrize("bad", [True, "0.2", math.nan, math.inf, -0.1, 1.1])
def test_threshold_rejects_non_real(bad: object) -> None:
    """Bug caught: True is accepted and stored as the threshold."""
    with pytest.raises(TeaCacheValueError, match="rel_l1_thresh"):
        apply_teacache(_NoModel(), rel_l1_thresh=bad)  # type: ignore[arg-type]
