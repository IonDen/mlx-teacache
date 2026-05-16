# tests/test_detect.py
"""Variant detection. Maps an mflux model instance to a variant_id string.
Uses (class name, model_config.aliases) so we can distinguish:
- Flux1 + dev   ⇒ flux1-dev
- Flux1 + schnell ⇒ flux1-schnell
- Flux2Klein + flux2_klein_4b ⇒ flux2-klein-4b
- Flux2Klein + flux2_klein_9b ⇒ flux2-klein-9b
Rejects everything else (Klein base-4b, base-9b, unknown Flux1 aliases,
non-Flux types) with IncompatibleModelError."""

import pytest

from mlx_teacache.errors import IncompatibleModelError
from mlx_teacache.integrations.mflux.detect import identify_variant


class _FakeModelConfig:
    def __init__(self, alias: str) -> None:
        # Match mflux 0.17 shape: aliases holds short names; model_name holds
        # an HF-style path. Embed the alias in model_name so error-message
        # assertions can still find it.
        self.model_name = f"fake/{alias}"
        self.aliases = [alias]


class _FakeFlux1:
    def __init__(self, alias: str) -> None:
        self.model_config = _FakeModelConfig(alias)


class _FakeFlux2Klein:
    def __init__(self, alias: str) -> None:
        self.model_config = _FakeModelConfig(alias)


@pytest.fixture(autouse=True)
def _patch_mflux_types(monkeypatch):
    """Make the detect module recognize our fake classes as the mflux ones."""
    import mlx_teacache.integrations.mflux.detect as detect

    monkeypatch.setattr(detect, "_Flux1Type", _FakeFlux1)
    monkeypatch.setattr(detect, "_Flux2KleinType", _FakeFlux2Klein)


def test_identify_flux1_dev():
    assert identify_variant(_FakeFlux1("dev")) == "flux1-dev"


def test_identify_flux1_schnell():
    assert identify_variant(_FakeFlux1("schnell")) == "flux1-schnell"


def test_identify_flux2_klein_4b():
    assert identify_variant(_FakeFlux2Klein("flux2-klein-4b")) == "flux2-klein-4b"


def test_unknown_flux1_model_raises():
    with pytest.raises(IncompatibleModelError) as exc:
        identify_variant(_FakeFlux1("something-weird"))
    assert "something-weird" in str(exc.value)
    assert "flux1-dev" in str(exc.value)


def test_identify_flux2_klein_9b():
    assert identify_variant(_FakeFlux2Klein("flux2-klein-9b")) == "flux2-klein-9b"


def test_flux2_klein_base_4b_rejected():
    with pytest.raises(IncompatibleModelError):
        identify_variant(_FakeFlux2Klein("flux2-klein-base-4b"))


def test_flux2_klein_base_9b_rejected():
    with pytest.raises(IncompatibleModelError):
        identify_variant(_FakeFlux2Klein("flux2-klein-base-9b"))


def test_completely_unknown_type_rejected():
    class Other: ...

    with pytest.raises(IncompatibleModelError):
        identify_variant(Other())
