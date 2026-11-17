"""Every variant's threshold rule, pinned per variant: the threshold a patched model gates with, the provenance it
reports and whether the built-in calibrated range applies, for each combination of a caller threshold and caller
coefficients. Pure-core: each variant's apply() runs on a duck-typed model, no mflux, no weights."""

import dataclasses
import importlib
from types import SimpleNamespace
from typing import Any

import pytest

from mlx_teacache.variants import _REGISTRY
from tests._fakes import FaithfulCallbackRegistry
from tests._variant_fakes import fake_for_variant

_CUSTOM = (1.0, 0.5, 0.25, 0.0, 0.0)
_FLUX1_ALIASES = {"flux1-dev": "dev", "flux1-schnell": "schnell", "flux1-krea-dev": "krea-dev"}

# variant id: (threshold with built-in coefficients, threshold with caller coefficients, the built-in calibrated
# range, which applies only with the built-in coefficients). The ranges are the min and max rel-L1 of each fit's
# calibration run; Klein base 9B reuses the base 4B fit.
_RULES: dict[str, tuple[float, float, tuple[float, float] | None]] = {
    "flux1-dev": (0.20, 0.20, None),
    "flux1-schnell": (0.20, 0.20, None),
    "flux1-krea-dev": (0.30, 0.30, None),
    "flux2-klein-base-4b": (0.17, 0.20, (0.028261402621865273, 0.21984468400478363)),
    "flux2-klein-base-9b": (0.17, 0.20, (0.028261402621865273, 0.21984468400478363)),
    "flux2-klein-4b": (0.20, 0.20, None),
    "flux2-klein-9b": (0.20, 0.20, None),
    "z-image-base": (0.12, 0.20, (0.027887196237753466, 0.25914120883567665)),
    "qwen-image": (0.30, 0.20, (0.04675976472922664, 0.587799896108221)),
    "qwen-image-2512": (0.25, 0.20, (0.08094493952514002, 0.4705715918534816)),
}


def _fake(variant_id: str) -> Any:
    if variant_id in _FLUX1_ALIASES:
        alias = _FLUX1_ALIASES[variant_id]
        return SimpleNamespace(
            model_config=SimpleNamespace(model_name="black-forest-labs/FLUX.1-" + alias, aliases=[alias]),
            transformer=SimpleNamespace(name="real-transformer"),
            callbacks=FaithfulCallbackRegistry(),
            generate_image=lambda **kw: "image",
        )
    return fake_for_variant(variant_id)


_ROWS = [
    pytest.param(
        variant_id,
        thresh,
        coeffs,
        id=f"{variant_id}-thresh={thresh}-coeffs={'caller' if coeffs else 'builtin'}",
    )
    for variant_id in _RULES
    for thresh in (None, 0.33)
    for coeffs in (None, _CUSTOM)
]


def test_the_table_covers_every_registered_variant() -> None:
    """bug caught: a new variant ships with no pinned threshold rule."""
    assert sorted(_RULES) == sorted(_REGISTRY)


@pytest.mark.parametrize(("variant_id", "thresh", "coeffs"), _ROWS)
def test_threshold_provenance_and_range_per_variant(
    variant_id: str, thresh: float | None, coeffs: tuple[float, float, float, float, float] | None
) -> None:
    """bug caught: moving a variant onto the shared resolver changes its threshold when the caller passes
    coefficients, lets the built-in range clamp a caller's polynomial, or wires another fit's range (or an
    edited one) into a variant."""
    builtin_thresh, caller_coeff_thresh, builtin_range = _RULES[variant_id]
    expected = 0.33 if thresh is not None else (builtin_thresh if coeffs is None else caller_coeff_thresh)
    flux = _fake(variant_id)
    handle = _REGISTRY[variant_id]["load_integration"]()(flux, rel_l1_thresh=thresh, coefficients=coeffs)
    try:
        assert handle.rel_l1_thresh == expected
        assert handle._session is not None
        assert handle._session.rel_l1_thresh == expected
        assert handle.provenance.source == ("builtin" if coeffs is None else "user")
        if coeffs is not None:
            assert handle.coefficients == _CUSTOM
            assert handle._session.coefficients == _CUSTOM
        else:
            # The handle reports the polynomial the session gates with, never the constructor's placeholder.
            assert handle.coefficients == handle._session.coefficients
            assert handle.coefficients != (0.0, 0.0, 0.0, 0.0, 0.0)
        assert handle._session.calibrated_range == (builtin_range if coeffs is None else None)
    finally:
        handle.restore()


@pytest.mark.parametrize(
    ("variant_id", "module", "caller_coeff_thresh"),
    [
        ("flux2-klein-4b", "mlx_teacache.variants.flux2_klein_4b.integration", 0.25),
        ("flux2-klein-9b", "mlx_teacache.variants.flux2_klein_9b.integration", 0.25),
        # The contrast: Klein base keeps its default for its own fit only.
        ("flux2-klein-base-4b", "mlx_teacache.variants.flux2_klein_base_4b.integration", 0.20),
    ],
)
def test_a_variant_default_reaches_caller_coefficients_on_the_distilled_klein(
    monkeypatch: pytest.MonkeyPatch, variant_id: str, module: str, caller_coeff_thresh: float
) -> None:
    """bug caught: the distilled Klein 4B or 9B resolves with the fallback-for-caller-coefficients policy instead
    of variant-default, so once it ships a default threshold, a caller's own coefficients gate at 0.20 instead of
    that default. DEFAULT_THRESH is None for both today, so the policy shows only with a default put in."""
    integration = importlib.import_module(module)
    monkeypatch.setattr(
        integration, "_BUILTIN", dataclasses.replace(integration._BUILTIN, default_thresh=0.25)
    )
    for coeffs, expected in ((None, 0.25), (_CUSTOM, caller_coeff_thresh)):
        handle = integration.apply(_fake(variant_id), coefficients=coeffs)
        try:
            assert handle.rel_l1_thresh == expected
            assert handle._session.rel_l1_thresh == expected
        finally:
            handle.restore()
