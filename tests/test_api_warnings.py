# tests/test_api_warnings.py
"""Apply-time TeaCacheNoBenefitWarning for the distilled variants.

flux1-schnell, flux2-klein-4b and flux2-klein-9b run distilled few-step
schedules on which the polynomial gate skips nothing; their config.py declares
META["non_distilled"] = False, and apply_teacache warns once, at apply time,
so the caller learns this before running any generation.

This is a DIFFERENT mechanism from the per-generation TeaCacheNoBenefitWarning
covered in test_distilled_warning.py: that one fires from
GenerationContextCallback based on the active step window for THIS
generation (possible_skips == 0) and applies to any variant with a short
enough schedule. This one fires at apply time, for exactly the distilled
variants, whenever the caller passes no coefficients of their own.

Pure-core: the fakes are plain SimpleNamespaces (detect.matches keys on
model_config.aliases; apply() only touches callbacks/generate_image/transformer
or _predict). No mflux import, no weights — mirrors tests/test_z_image_apply.py.
"""

import warnings
from types import SimpleNamespace

import pytest

from mlx_teacache import TeaCacheNoBenefitWarning, TeaCacheUncalibratedCheckpointWarning, apply_teacache
from tests._fakes import FaithfulCallbackRegistry

# The distilled few-step variants (META["non_distilled"] is False); every other registered variant is non-distilled.
_DISTILLED = frozenset({"flux1-schnell", "flux2-klein-4b", "flux2-klein-9b"})

_ALL_VARIANTS = (
    "flux1-dev",
    "flux1-krea-dev",
    "flux1-schnell",
    "flux2-klein-4b",
    "flux2-klein-9b",
    "flux2-klein-base-4b",
    "flux2-klein-base-9b",
    "qwen-image",
    "qwen-image-2512",
    "z-image-base",
)


def _fake_klein_distilled(alias: str) -> SimpleNamespace:
    """Duck-typed distilled FLUX.2 Klein (flux2-klein-4b / flux2-klein-9b)."""
    return SimpleNamespace(
        model_config=SimpleNamespace(aliases=[alias]),
        callbacks=FaithfulCallbackRegistry(),
        generate_image=lambda **kw: "image",
    )


def _fake_flux1(alias: str) -> SimpleNamespace:
    """Duck-typed FLUX.1 (dev or schnell). apply() wraps flux.transformer in
    ProxyFlux1Transformer(inner=...), never calling anything on it at apply
    time, so a bare SimpleNamespace stand-in is sufficient."""
    return SimpleNamespace(
        model_config=SimpleNamespace(aliases=[alias]),
        transformer=SimpleNamespace(),
        callbacks=FaithfulCallbackRegistry(),
        generate_image=lambda **kw: "image",
    )


class _StopAtIntegration(RuntimeError):
    """Raised by the stubbed integration loader so only apply_teacache's pre-dispatch path runs."""


def _only_variant_matches(monkeypatch: pytest.MonkeyPatch, variant_id: str) -> object:
    """Make `variant_id` the registry's only match for a sentinel model and stop apply_teacache at its integration
    loader, so every variant's apply-time warning check runs without mflux or a variant-shaped fake."""
    from mlx_teacache.variants import _REGISTRY

    sentinel = object()

    def _boom() -> None:
        raise _StopAtIntegration

    for vid, entry in _REGISTRY.items():
        monkeypatch.setitem(entry, "matches", lambda flux, _vid=vid: _vid == variant_id and flux is sentinel)
    monkeypatch.setitem(_REGISTRY[variant_id], "load_integration", _boom)
    return sentinel


def test_registry_holds_the_ten_variants_the_distilled_test_covers() -> None:
    """Bug: a new variant registers without this file's parametrization learning whether it is distilled."""
    from mlx_teacache.variants import _REGISTRY

    assert set(_REGISTRY) == set(_ALL_VARIANTS)


@pytest.mark.parametrize("variant_id", _ALL_VARIANTS)
def test_no_benefit_warning_follows_non_distilled(monkeypatch: pytest.MonkeyPatch, variant_id: str) -> None:
    """Bug: the warning keys on `default_thresh is None` again, so flux1-schnell (default 0.20) applies silently,
    or it fires on a non-distilled variant whose gate does skip steps."""
    flux = _only_variant_matches(monkeypatch, variant_id)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(_StopAtIntegration):
            apply_teacache(flux)
    no_benefit = [w for w in caught if issubclass(w.category, TeaCacheNoBenefitWarning)]
    assert len(no_benefit) == (1 if variant_id in _DISTILLED else 0)


@pytest.mark.parametrize("alias", ["schnell", "flux2-klein-4b", "flux2-klein-9b"])
def test_no_benefit_message_makes_no_compile_claim(alias: str) -> None:
    """Bug: the warning credits a wall-clock gain to bypassing mx.compile, which FLUX.1's proxy (schnell) does not
    bypass; the message must only say that step-skipping gives no speedup."""
    variant_id = {"schnell": "flux1-schnell"}.get(alias, alias)
    flux = _fake_flux1(alias) if alias == "schnell" else _fake_klein_distilled(alias)
    with pytest.warns(TeaCacheNoBenefitWarning) as caught:
        handle = apply_teacache(flux)
    handle.restore()
    (warning,) = [w for w in caught if issubclass(w.category, TeaCacheNoBenefitWarning)]
    message = str(warning.message)
    assert message == (
        f"variant '{variant_id}' runs a distilled few-step schedule where the polynomial gate skips nothing; "
        "apply_teacache adds per-step overhead and gives no speedup from step-skipping"
    )
    assert "compile" not in message


@pytest.mark.parametrize("alias", ["flux2-klein-4b", "flux2-klein-9b"])
def test_apply_on_distilled_variant_warns_no_benefit(alias: str) -> None:
    flux = _fake_klein_distilled(alias)
    with pytest.warns(TeaCacheNoBenefitWarning, match="distilled"):
        handle = apply_teacache(flux)
    handle.restore()


_SILENCE_UNCALIBRATED = 'warnings.filterwarnings("ignore", category=TeaCacheUncalibratedCheckpointWarning)'


def test_uncalibrated_checkpoint_message_names_the_filter_call() -> None:
    """Bug: the message says "filter this category" without the call that does it, or wraps the call in markdown
    backticks that a terminal prints literally, or drifts from the call the warning class documents."""
    flux = _fake_flux1("dev")
    flux.model_config.model_name = "someone/finetune"
    flux.model_config.base_model = None
    with pytest.warns(TeaCacheUncalibratedCheckpointWarning) as caught:
        handle = apply_teacache(flux)
    handle.restore()
    (warning,) = [w for w in caught if issubclass(w.category, TeaCacheUncalibratedCheckpointWarning)]
    message = str(warning.message)
    assert f"silence it with {_SILENCE_UNCALIBRATED}" in message
    assert "filter this category" not in message
    assert "`" not in message
    assert _SILENCE_UNCALIBRATED in (TeaCacheUncalibratedCheckpointWarning.__doc__ or "")


def test_apply_on_engaged_variant_does_not_warn() -> None:
    flux = _fake_flux1("dev")
    with warnings.catch_warnings():
        warnings.simplefilter("error", TeaCacheNoBenefitWarning)
        handle = apply_teacache(flux)
    handle.restore()


# --- Registry contract backing the warning ---


def test_registry_entries_expose_default_thresh() -> None:
    from mlx_teacache.variants import _REGISTRY

    for entry in _REGISTRY.values():
        assert "default_thresh" in entry
        assert entry["default_thresh"] is None or isinstance(entry["default_thresh"], float)
