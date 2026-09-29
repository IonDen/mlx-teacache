"""Canonical home for the Provenance dataclass and coefficient utilities.

The per-variant coefficient tuples live in each variant's config.py.
src/mlx_teacache/coefficients.py re-exports Provenance and validate_custom
from here.
"""

import math
from dataclasses import dataclass
from typing import Literal

from mlx_teacache.errors import TeaCacheValueError


@dataclass(frozen=True)
class Provenance:
    source: Literal["builtin", "user"]
    revision: str | None = None
    calibration_dataset: str | None = None
    fit_metric: str | None = None
    fit_metric_value: float | None = None
    reference_url: str | None = None
    default_thresh: float | None = None

    @classmethod
    def for_user_supplied(cls) -> "Provenance":
        return cls(source="user")


def validate_custom(coeffs: object) -> tuple[float, float, float, float, float]:
    """Coerce a user-supplied coefficient sequence into a length-5 tuple of finite floats.

    Raises TeaCacheValueError (IS-A ValueError) with a helpful message on any failure."""
    try:
        items = list(coeffs)  # type: ignore[call-overload]
    except TypeError as e:
        raise TeaCacheValueError(
            f"coefficients must be a sequence of 5 floats, got {type(coeffs).__name__}"
        ) from e
    if len(items) != 5:
        raise TeaCacheValueError(f"coefficients must have length 5 (got {len(items)})")
    try:
        floats = tuple(float(x) for x in items)
    except (TypeError, ValueError) as e:
        raise TeaCacheValueError(f"coefficients must be convertible to float: {coeffs!r}") from e
    if not all(math.isfinite(x) for x in floats):
        raise TeaCacheValueError(f"coefficients must all be finite (no nan/inf): {floats!r}")
    return floats  # type: ignore[return-value]
