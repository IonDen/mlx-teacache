"""Threshold, coefficient and provenance resolution for one patched model.

The caller's ``rel_l1_thresh`` always wins, a 0.0 included. Otherwise a variant's own
default applies; under ``"fallback-for-caller-coefficients"`` that default is tuned for
the bundled polynomial, so caller-supplied coefficients get the package fallback
instead, while ``"variant-default"`` keeps the default whatever the coefficients. A
variant without a default resolves to the package fallback. Caller-supplied
coefficients get a user provenance and no calibrated range; built-in ones carry the
variant's provenance object and range."""

from dataclasses import dataclass
from typing import Final, Literal

from mlx_teacache._kernel.coefficients import Provenance

PACKAGE_FALLBACK_THRESH: Final[float] = 0.20

ThreshPolicy = Literal["variant-default", "fallback-for-caller-coefficients"]

_Coefficients = tuple[float, float, float, float, float]


@dataclass(frozen=True, slots=True, kw_only=True)
class BuiltinFit:
    """A variant's bundled fit: polynomial, provenance, default threshold and calibrated range."""

    coefficients: _Coefficients
    provenance: Provenance
    default_thresh: float | None
    calibrated_range: tuple[float, float] | None


@dataclass(frozen=True, slots=True, kw_only=True)
class ResolvedRule:
    """The values a patched model gates with."""

    rel_l1_thresh: float
    coefficients: _Coefficients
    provenance: Provenance
    calibrated_range: tuple[float, float] | None


def resolve_thresh(
    *,
    caller: float | None,
    default: float | None,
    coefficients: _Coefficients | None,
    policy: ThreshPolicy,
) -> float:
    """The threshold to gate with; `coefficients` is the caller's (None means built-in)."""
    if caller is not None:
        return caller
    if default is not None and (policy == "variant-default" or coefficients is None):
        return default
    return PACKAGE_FALLBACK_THRESH


def resolve_rule(
    *,
    rel_l1_thresh: float | None,
    coefficients: _Coefficients | None,
    builtin: BuiltinFit,
    policy: ThreshPolicy,
) -> ResolvedRule:
    """Resolve the caller's arguments against a variant's built-in fit."""
    thresh = resolve_thresh(
        caller=rel_l1_thresh, default=builtin.default_thresh, coefficients=coefficients, policy=policy
    )
    if coefficients is not None:
        return ResolvedRule(
            rel_l1_thresh=thresh,
            coefficients=coefficients,
            provenance=Provenance.for_user_supplied(),
            calibrated_range=None,
        )
    return ResolvedRule(
        rel_l1_thresh=thresh,
        coefficients=builtin.coefficients,
        provenance=builtin.provenance,
        calibrated_range=builtin.calibrated_range,
    )
