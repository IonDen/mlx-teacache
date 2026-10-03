"""The skip-window rule: how many active denoising steps a TeaCache generation needs.

Pure and weight-free, so a caller can refuse a run that is too short before it loads a model."""

import operator
from typing import SupportsIndex

from mlx_teacache.errors import InvalidStepWindowError, TeaCacheValueError


def validate_step_count(name: str, value: object) -> int:
    """`value` as a non-negative int (numpy integers accepted); TeaCacheValueError naming `name` otherwise."""
    if isinstance(value, bool):
        raise TeaCacheValueError(f"{name} must be a non-negative int, got {value!r}")
    try:
        as_int = operator.index(value)  # type: ignore[arg-type]
    except TypeError:
        raise TeaCacheValueError(f"{name} must be a non-negative int, got {value!r}") from None
    if as_int < 0:
        raise TeaCacheValueError(f"{name} must be >= 0, got {as_int}")
    return as_int


def window_too_short(active_num_steps: int, skip_first_n_steps: int, skip_last_n_steps: int) -> bool:
    """True when the steps forced to compute at both ends leave no step for the gate to decide."""
    return skip_first_n_steps + skip_last_n_steps >= active_num_steps


def check_step_window(
    active_num_steps: SupportsIndex,
    *,
    skip_first_n_steps: SupportsIndex = 1,
    skip_last_n_steps: SupportsIndex = 1,
    nominal_num_inference_steps: int | None = None,
) -> None:
    """Raise InvalidStepWindowError when a generation with `active_num_steps` denoising steps
    is too short for TeaCache: `skip_first_n_steps + skip_last_n_steps` must be less than
    `active_num_steps`. With the default window, 3 or fewer active steps skip nothing; 2 or
    fewer are refused.

    `active_num_steps` is the number of denoising steps mflux will run: `num_inference_steps`
    for text-to-image, minus mflux's `init_time_step` for image-to-image. The defaults are
    `apply_teacache`'s, so `check_step_window(steps)` answers "will a default TeaCache run
    accept this?" without loading a model; `apply_teacache` applies the same rule on the
    first denoising step of each generation. The answer is "accept" for 3 active steps, but
    `apply_teacache` then warns that it cannot skip any step. Pass
    `nominal_num_inference_steps` to have the error message show the full schedule too. A bool, float, None or negative count raises
    TeaCacheValueError. Zero active steps raises too, which is stricter than `apply_teacache`,
    where a zero-step generation (image_strength=1.0) is a valid no-op."""
    active = validate_step_count("active_num_steps", active_num_steps)
    first = validate_step_count("skip_first_n_steps", skip_first_n_steps)
    last = validate_step_count("skip_last_n_steps", skip_last_n_steps)
    if window_too_short(active, first, last):
        raise InvalidStepWindowError(
            skip_first=first,
            skip_last=last,
            num_steps=active,
            nominal_num_inference_steps=nominal_num_inference_steps,
        )
