# tests/test_api_warnings.py
"""Apply-time TeaCacheNoBenefitWarning for the two distilled Klein variants.

flux2-klein-4b and flux2-klein-9b have DEFAULT_THRESH=None (see their
config.py) because the polynomial gate structurally does not engage on
their 4-8 step distilled schedules — apply_teacache warns once, at apply
time, so the caller learns this before running any generation.

This is a DIFFERENT mechanism from the per-generation TeaCacheNoBenefitWarning
covered in test_distilled_warning.py: that one fires from
GenerationContextCallback based on the active step window for THIS
generation (possible_skips == 0) and applies to any variant with a short
enough schedule. This one fires unconditionally at apply time, for exactly
the two variants whose gate never engages regardless of schedule length —
flux1-schnell (DEFAULT_THRESH=0.20) does NOT trigger it and keeps only the
per-generation warning.

Pure-core: both fakes are plain SimpleNamespaces (detect.matches keys on
model_config.aliases; apply() only touches callbacks/generate_image/transformer
or _predict). No mflux import, no weights — mirrors tests/test_z_image_apply.py.
"""

from types import SimpleNamespace

import pytest

from mlx_teacache import TeaCacheNoBenefitWarning, apply_teacache
from tests._fakes import FaithfulCallbackRegistry


def _fake_klein_distilled(alias: str) -> SimpleNamespace:
    """Duck-typed distilled FLUX.2 Klein (flux2-klein-4b / flux2-klein-9b)."""
    return SimpleNamespace(
        model_config=SimpleNamespace(aliases=[alias]),
        callbacks=FaithfulCallbackRegistry(),
        generate_image=lambda **kw: "image",
    )


def _fake_flux1_dev() -> SimpleNamespace:
    """Duck-typed FLUX.1 dev. apply() wraps flux.transformer in
    ProxyFlux1Transformer(inner=...), never calling anything on it at apply
    time, so a bare SimpleNamespace stand-in is sufficient."""
    return SimpleNamespace(
        model_config=SimpleNamespace(aliases=["dev"]),
        transformer=SimpleNamespace(),
        callbacks=FaithfulCallbackRegistry(),
        generate_image=lambda **kw: "image",
    )


@pytest.mark.parametrize("alias", ["flux2-klein-4b", "flux2-klein-9b"])
def test_apply_on_distilled_variant_warns_no_benefit(alias: str) -> None:
    flux = _fake_klein_distilled(alias)
    with pytest.warns(TeaCacheNoBenefitWarning, match="distilled"):
        handle = apply_teacache(flux)
    handle.restore()


def test_apply_on_engaged_variant_does_not_warn() -> None:
    import warnings

    flux = _fake_flux1_dev()
    with warnings.catch_warnings():
        warnings.simplefilter("error", TeaCacheNoBenefitWarning)
        handle = apply_teacache(flux)
    handle.restore()


# --- Registry contract backing the warning (Step 3's actual deliverable) ---


def test_registry_entries_expose_default_thresh() -> None:
    from mlx_teacache.variants import _REGISTRY

    for entry in _REGISTRY.values():
        assert "default_thresh" in entry
        assert entry["default_thresh"] is None or isinstance(entry["default_thresh"], float)


def test_only_distilled_kleins_have_no_default_thresh() -> None:
    from mlx_teacache.variants import _REGISTRY

    no_default = {vid for vid, entry in _REGISTRY.items() if entry["default_thresh"] is None}
    assert no_default == {"flux2-klein-4b", "flux2-klein-9b"}
