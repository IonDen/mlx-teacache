# src/mlx_teacache/coefficients.py
"""Per-variant polynomial coefficient registry with explicit provenance.

The polynomial maps relative-L1 distance of the modulated block-0 input to
the predicted relative-L1 distance of the transformer output. Coefficients are
ordered [c4, c3, c2, c1, c0] for poly_eval (highest-degree first, matching
numpy.polyval). FLUX.1 coefficients are vendored from ali-vilab/TeaCache
upstream (TeaCache4FLUX/teacache_flux.py); flux1-schnell reuses the same
set because upstream does not distinguish, and FLUX.1 dev/schnell share
the same transformer architecture. flux2-klein-4b coefficients are derived
in-repo via scripts/calibrate_flux2.py."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from mlx_teacache.errors import CalibrationError

# Vendored from https://github.com/ali-vilab/TeaCache/blob/main/TeaCache4FLUX/teacache_flux.py
# License: Apache-2.0. See NOTICE for attribution.
#
# Upstream source uses these values in numpy poly1d order (high-to-low):
#   coefficients = [4.98651651e+02, -2.83781631e+02,  5.58554382e+01,
#                   -3.82021401e+00,  2.64230861e-01]
#   rescale_func = np.poly1d(coefficients)
#
# Our `poly_eval` uses the same high-to-low convention (see gate.py). The
# earlier version of this constant had coefficients c3..c0 transcribed
# incorrectly (predicted distances ~10x too large), which prevented the
# default rel_l1_thresh=0.25 from ever skipping a step on FLUX.1-dev. Fixed
# 2026-05-15; verified via the test_image_quality_flux1.py SSIM gate that
# the cache now engages at the documented threshold.
_UPSTREAM_FLUX_COEFFS: tuple[float, float, float, float, float] = (
    498.651651244,
    -283.781631,
    55.8554382,
    -3.82021401,
    0.264230861,
)

# Derived in-repo by scripts/calibrate_flux2.py --variant klein-4b on 2026-05-15:
#   10 prompts × 8 steps × seed=42 on M1 Max 32GB, bf16, guidance=1.0,
#   512×512, vanilla forward (no caching). 70 consecutive-step pairs of
#   (rel_l1(mod_in_t, mod_in_{t-1}), rel_l1(body_out_t, body_out_{t-1})).
#   numpy.polyfit degree=4 → R² = 0.6530.
# See scripts/_calibration_flux2_klein_4b.json for the full report.
# Stored verbatim; do not hand-edit. New calibrations bump revision and minor version.
_FLUX2_KLEIN_4B_COEFFS: tuple[float, float, float, float, float] = (
    236.9190176127698,
    -201.47401360106662,
    66.91354236854073,
    -11.14796738073235,
    1.2674506310647067,
)


@dataclass(frozen=True)
class Provenance:
    source: Literal["builtin", "user"]
    revision: str | None = None
    calibration_dataset: str | None = None
    fit_metric: str | None = None
    fit_metric_value: float | None = None
    reference_url: str | None = None

    @classmethod
    def for_user_supplied(cls) -> Provenance:
        return cls(source="user")


_REGISTRY: dict[str, tuple[tuple[float, float, float, float, float], Provenance]] = {
    "flux1-dev": (
        _UPSTREAM_FLUX_COEFFS,
        Provenance(
            source="builtin",
            revision="upstream-flux-v1",
            calibration_dataset="upstream ali-vilab TeaCache (no in-repo calibration)",
            fit_metric=None,
            fit_metric_value=None,
            reference_url="https://github.com/ali-vilab/TeaCache/blob/main/TeaCache4FLUX/teacache_flux.py",
        ),
    ),
    "flux1-schnell": (
        _UPSTREAM_FLUX_COEFFS,
        Provenance(
            source="builtin",
            revision="upstream-flux-v1-shared",
            calibration_dataset="upstream ali-vilab TeaCache (FLUX architecture is shared between dev and schnell)",
            fit_metric=None,
            fit_metric_value=None,
            reference_url="https://github.com/ali-vilab/TeaCache/blob/main/TeaCache4FLUX/teacache_flux.py",
        ),
    ),
    "flux2-klein-4b": (
        _FLUX2_KLEIN_4B_COEFFS,
        Provenance(
            source="builtin",
            revision="in-repo-2026-05-15",
            calibration_dataset="10 prompts × 8 steps × seed=42, M1 Max 32GB, bf16, 512x512, guidance=1.0",
            fit_metric="numpy.polyfit R^2 on 70 consecutive-step (mod_in, body_out) rel-L1 pairs",
            fit_metric_value=0.6530168924992779,
            reference_url="https://github.com/IonDen/mlx-teacache/blob/main/scripts/calibrate_flux2.py",
        ),
    ),
}


def load_builtin(variant_id: str) -> tuple[tuple[float, float, float, float, float], Provenance]:
    """Return (coefficients, provenance) for a built-in variant.

    Raises CalibrationError if the variant is unknown or if the registry entry
    is malformed (a regression guard — should never fire in normal flow)."""
    if variant_id not in _REGISTRY:
        raise CalibrationError(
            variant_id=variant_id,
            reason=f"not in built-in registry; known: {sorted(_REGISTRY)}",
        )
    coeffs, prov = _REGISTRY[variant_id]
    if len(coeffs) != 5 or not all(math.isfinite(c) for c in coeffs):
        raise CalibrationError(variant_id=variant_id, reason=f"corrupt entry: coefficients {coeffs!r}")
    return coeffs, prov


def validate_custom(coeffs: object) -> tuple[float, float, float, float, float]:
    """Coerce a user-supplied coefficient sequence into a length-5 tuple of finite floats.

    Raises ValueError with a helpful message on any failure."""
    try:
        items = list(coeffs)  # type: ignore[call-overload]
    except TypeError as e:
        raise ValueError(f"coefficients must be a sequence of 5 floats, got {type(coeffs).__name__}") from e
    if len(items) != 5:
        raise ValueError(f"coefficients must have length 5 (got {len(items)})")
    try:
        floats = tuple(float(x) for x in items)
    except (TypeError, ValueError) as e:
        raise ValueError(f"coefficients must be convertible to float: {coeffs!r}") from e
    if not all(math.isfinite(x) for x in floats):
        raise ValueError(f"coefficients must all be finite (no nan/inf): {floats!r}")
    return floats  # type: ignore[return-value]
