# tests/test_detect.py
"""Variant detection via the registry dispatcher.

Tests the cross-variant dispatch contract: given a fake flux object, the
registry walk in apply_teacache picks the correct variant_id (or raises
IncompatibleModelError on no-match). This is the successor to the old
identify_variant tests removed in T17.

Detection surface tested here is intentionally coarser than the per-variant
detect tests in tests/variants/*/test_detect.py — those cover each
variant's matches() in isolation; this file covers the full dispatch chain
across all registered variants.
"""

from __future__ import annotations

import pytest

from mlx_teacache.errors import IncompatibleModelError
from mlx_teacache.variants import _REGISTRY


def _dispatch(flux: object) -> str:
    """Replicate the registry walk from apply_teacache without triggering
    load_integration (which would require mflux weights). Returns the
    variant_id of the first matching entry, or raises IncompatibleModelError."""
    for variant_id, entry in _REGISTRY.items():
        if entry["matches"](flux):
            return variant_id
    model_config = getattr(flux, "model_config", None)
    model_name = getattr(model_config, "model_name", None)
    raise IncompatibleModelError(
        actual_type=type(flux).__name__,
        actual_model_name=model_name,
        supported=sorted(_REGISTRY.keys()),
    )


class _FakeModelConfig:
    def __init__(self, alias: str) -> None:
        self.model_name = f"fake/{alias}"
        self.aliases = [alias]


class _FakeFlux1:
    def __init__(self, alias: str) -> None:
        self.model_config = _FakeModelConfig(alias)


class _FakeFlux2Klein:
    def __init__(self, alias: str) -> None:
        self.model_config = _FakeModelConfig(alias)


def test_identify_flux1_dev():
    assert _dispatch(_FakeFlux1("dev")) == "flux1-dev"


def test_identify_flux1_schnell():
    assert _dispatch(_FakeFlux1("schnell")) == "flux1-schnell"


def test_identify_flux2_klein_4b():
    assert _dispatch(_FakeFlux2Klein("flux2-klein-4b")) == "flux2-klein-4b"


def test_identify_flux2_klein_9b():
    assert _dispatch(_FakeFlux2Klein("flux2-klein-9b")) == "flux2-klein-9b"


def test_identify_flux2_klein_base_4b():
    assert _dispatch(_FakeFlux2Klein("flux2-klein-base-4b")) == "flux2-klein-base-4b"


def test_identify_flux2_klein_base_9b():
    """v0.5.0 added klein-base-9b as a supported variant."""
    assert _dispatch(_FakeFlux2Klein("flux2-klein-base-9b")) == "flux2-klein-base-9b"


def test_unknown_flux1_model_raises():
    with pytest.raises(IncompatibleModelError) as exc:
        _dispatch(_FakeFlux1("something-weird"))
    assert "something-weird" in str(exc.value)
    assert "flux1-dev" in str(exc.value)


def test_completely_unknown_type_rejected():
    class Other: ...

    with pytest.raises(IncompatibleModelError):
        _dispatch(Other())
