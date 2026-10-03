"""check_step_window: the skip-window rule apply_teacache applies on a generation's first denoising step, callable
without a model. Rule: skip_first_n_steps + skip_last_n_steps must be less than the active denoising steps. With the
default window, 3 or fewer active steps skip nothing; 2 or fewer are refused."""

import numpy as np
import pytest

from mlx_teacache import InvalidStepWindowError, TeaCacheValueError, check_step_window


@pytest.mark.parametrize("active_num_steps", [3, 4, 50])
def test_default_window_accepts_three_or_more_active_steps(active_num_steps: int) -> None:
    """Bug: the boundary moved so a 3-step run is refused, though apply_teacache accepts it (it skips nothing: the
    one gated step between the forced first and last steps is the seed step)."""
    assert check_step_window(active_num_steps) is None


@pytest.mark.parametrize("active_num_steps", [2, 1, 0])
def test_default_window_refuses_two_or_fewer_active_steps(active_num_steps: int) -> None:
    """Bug: the boundary moved so a 2-step run, where both steps are forced to compute, is accepted."""
    with pytest.raises(InvalidStepWindowError) as caught:
        check_step_window(active_num_steps)
    error = caught.value
    assert (error.skip_first, error.skip_last, error.active_num_steps) == (1, 1, active_num_steps)


def test_the_count_is_accepted_by_its_keyword_name() -> None:
    """Bug: the first parameter is renamed (say back to active_steps), and a caller that passes it by keyword, as the
    mflux plugin does, gets a TypeError instead of the window check."""
    check_step_window(active_num_steps=3)
    with pytest.raises(InvalidStepWindowError):
        check_step_window(active_num_steps=2, nominal_num_inference_steps=20)


@pytest.mark.parametrize(
    ("active_num_steps", "first", "last", "too_short"),
    [(5, 2, 2, False), (4, 2, 2, True), (4, 3, 0, False), (3, 3, 0, True), (1, 0, 0, False), (0, 0, 0, True)],
)
def test_custom_window_counts_both_ends(
    active_num_steps: int, first: int, last: int, too_short: bool
) -> None:
    """Bug: only one end of the window is counted, or skip_last_n_steps is ignored."""
    if too_short:
        with pytest.raises(InvalidStepWindowError):
            check_step_window(active_num_steps, skip_first_n_steps=first, skip_last_n_steps=last)
    else:
        check_step_window(active_num_steps, skip_first_n_steps=first, skip_last_n_steps=last)


def test_nominal_schedule_reaches_the_message() -> None:
    """Bug: the nominal schedule is dropped, so an img2img user sees only '2 steps' and cannot tell why their
    20-step run was refused."""
    with pytest.raises(InvalidStepWindowError, match="active_num_steps=2, nominal_num_inference_steps=20"):
        check_step_window(2, nominal_num_inference_steps=20)


@pytest.mark.parametrize(
    ("kwargs", "name"),
    [
        ({"active_num_steps": -1}, "active_num_steps"),
        ({"active_num_steps": True}, "active_num_steps"),
        ({"active_num_steps": 3.0}, "active_num_steps"),
        ({"active_num_steps": 10, "skip_first_n_steps": -1}, "skip_first_n_steps"),
        ({"active_num_steps": 10, "skip_last_n_steps": None}, "skip_last_n_steps"),
    ],
)
def test_bad_counts_raise_a_value_error_naming_the_argument(kwargs: dict, name: str) -> None:
    """Bug: a bool, float, None or negative count is coerced (True == 1) or crashes with a bare TypeError."""
    with pytest.raises(TeaCacheValueError, match=name):
        check_step_window(**kwargs)


def test_numpy_integer_counts_are_accepted() -> None:
    """Bug: numpy integers, which a scheduler array yields, are rejected as 'not an int'."""
    check_step_window(np.int64(3), skip_first_n_steps=np.int64(1), skip_last_n_steps=np.int64(1))
    with pytest.raises(InvalidStepWindowError):
        check_step_window(np.int64(2))


def test_window_sizes_are_keyword_only() -> None:
    """Bug: the window sizes stay positional, so a caller who swaps `first` and `last` (or passes the nominal
    step count third) gets a silently different window instead of a TypeError."""
    with pytest.raises(TypeError):
        check_step_window(10, 1, 1)  # type: ignore[misc]
