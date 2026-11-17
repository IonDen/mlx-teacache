"""The threshold, coefficient and provenance rule each variant's apply() resolves:
the caller's threshold wins; otherwise the variant default, which a fit tuned for the
bundled polynomial gives up to the package fallback when the caller brings their own
coefficients (unless the variant keeps its default whatever the coefficients)."""

import pytest

from mlx_teacache._kernel.coefficients import Provenance
from mlx_teacache._kernel.resolve import (
    PACKAGE_FALLBACK_THRESH,
    BuiltinFit,
    ResolvedRule,
    ThreshPolicy,
    resolve_rule,
    resolve_thresh,
)

_CUSTOM = (0.0, 0.0, 0.0, 2.0, 0.0)
_BUILTIN_COEFFS = (1.0, -2.0, 3.0, 4.0, 0.0)
_BUILTIN_PROVENANCE = Provenance(source="builtin", revision="test-fit", default_thresh=0.17)
_FALLBACK: ThreshPolicy = "fallback-for-caller-coefficients"
_KEEP: ThreshPolicy = "variant-default"


def test_package_fallback_is_0_20() -> None:
    """bug caught: the package fallback drifts from 0.20."""
    assert PACKAGE_FALLBACK_THRESH == 0.20


@pytest.mark.parametrize(
    ("caller", "default", "coefficients", "policy", "expected"),
    [
        pytest.param(0.33, 0.17, None, _FALLBACK, 0.33, id="caller-wins-builtin"),
        pytest.param(0.33, 0.17, _CUSTOM, _FALLBACK, 0.33, id="caller-wins-custom"),
        pytest.param(0.33, 0.30, _CUSTOM, _KEEP, 0.33, id="caller-wins-variant-default"),
        pytest.param(0.0, 0.17, None, _FALLBACK, 0.0, id="caller-zero-is-given"),
        pytest.param(0.0, 0.17, _CUSTOM, _FALLBACK, 0.0, id="caller-zero-is-given-custom"),
        pytest.param(0.0, 0.30, None, _KEEP, 0.0, id="caller-zero-is-given-variant-default"),
        pytest.param(None, 0.17, None, _FALLBACK, 0.17, id="default-with-builtin"),
        pytest.param(None, 0.17, _CUSTOM, _FALLBACK, 0.20, id="fallback-with-custom"),
        pytest.param(None, 0.30, None, _KEEP, 0.30, id="variant-default-builtin"),
        pytest.param(None, 0.30, _CUSTOM, _KEEP, 0.30, id="variant-default-kept-with-custom"),
        pytest.param(None, None, None, _FALLBACK, 0.20, id="no-default-builtin"),
        pytest.param(None, None, _CUSTOM, _FALLBACK, 0.20, id="no-default-custom"),
        pytest.param(None, None, None, _KEEP, 0.20, id="no-default-variant-default"),
    ],
)
def test_resolve_thresh_table(
    caller: float | None,
    default: float | None,
    coefficients: tuple[float, float, float, float, float] | None,
    policy: ThreshPolicy,
    expected: float,
) -> None:
    """bugs caught: a falsy 0.0 read as "not given"; the default kept for caller coefficients
    under the fallback policy (or dropped under variant-default); a None default leaking out."""
    got = resolve_thresh(caller=caller, default=default, coefficients=coefficients, policy=policy)
    assert got == expected
    assert isinstance(got, float)


def _fit(*, calibrated_range: tuple[float, float] | None = (0.01, 0.4)) -> BuiltinFit:
    return BuiltinFit(
        coefficients=_BUILTIN_COEFFS,
        provenance=_BUILTIN_PROVENANCE,
        default_thresh=0.17,
        calibrated_range=calibrated_range,
    )


def test_resolve_rule_with_caller_coefficients_is_user_supplied_and_unclamped() -> None:
    """bug caught: the built-in clamp range (or provenance) leaks onto caller coefficients."""
    rule = resolve_rule(rel_l1_thresh=None, coefficients=_CUSTOM, builtin=_fit(), policy=_FALLBACK)
    assert rule == ResolvedRule(
        rel_l1_thresh=0.20,
        coefficients=_CUSTOM,
        provenance=Provenance(source="user"),
        calibrated_range=None,
    )


def test_resolve_rule_with_builtin_coefficients_keeps_the_fit() -> None:
    """bug caught: the built-in range is dropped, or a copied provenance replaces the variant's object."""
    fit = _fit()
    rule = resolve_rule(rel_l1_thresh=None, coefficients=None, builtin=fit, policy=_FALLBACK)
    assert rule.rel_l1_thresh == 0.17
    assert rule.coefficients == _BUILTIN_COEFFS
    assert rule.provenance is _BUILTIN_PROVENANCE
    assert rule.calibrated_range == (0.01, 0.4)


def test_resolve_rule_without_a_range_stays_unclamped() -> None:
    """bug caught: a variant with no calibrated range (FLUX.1) gets one invented."""
    rule = resolve_rule(
        rel_l1_thresh=None, coefficients=None, builtin=_fit(calibrated_range=None), policy=_KEEP
    )
    assert rule.calibrated_range is None


def test_resolve_rule_applies_the_policy_it_is_given() -> None:
    """bug caught: resolve_rule ignores `policy`, so a FLUX.1 variant loses its default for caller coefficients."""
    fit = _fit()
    kept = resolve_rule(rel_l1_thresh=None, coefficients=_CUSTOM, builtin=fit, policy=_KEEP)
    assert kept.rel_l1_thresh == 0.17
    given = resolve_rule(rel_l1_thresh=0.33, coefficients=_CUSTOM, builtin=fit, policy=_KEEP)
    assert given.rel_l1_thresh == 0.33
