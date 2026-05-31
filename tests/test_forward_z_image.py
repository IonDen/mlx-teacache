"""Pure-core tests for the z-image-base variant (mflux-free).

Covers detect.matches (base vs turbo disambiguation), config.META + the
calibrated coefficients shape, and registry auto-discovery. NONE of these
import mflux — they exercise the eagerly-imported config.py / detect.py only,
so the file stays in the pure-core lane (NOT in conftest._MFLUX_FILES).

The forward/integration math (which DOES need mflux + weights) is covered by
tests/test_parity_z_image.py under the mflux marker.
"""

from __future__ import annotations

from types import SimpleNamespace


def _fake_flux(aliases: list[str]) -> object:
    return SimpleNamespace(model_config=SimpleNamespace(aliases=aliases))


# --- detect.matches -------------------------------------------------------


def test_detect_matches_base_aliases():
    from mlx_teacache.variants.z_image_base.detect import matches

    assert matches(_fake_flux(["z-image", "zimage"])) is True


def test_detect_excludes_turbo():
    from mlx_teacache.variants.z_image_base.detect import matches

    # Turbo's aliases do NOT contain the bare "z-image"/"zimage" as elements,
    # so element-membership correctly excludes it (turbo is not a variant).
    assert matches(_fake_flux(["z-image-turbo", "zimage-turbo"])) is False


def test_detect_missing_model_config():
    from mlx_teacache.variants.z_image_base.detect import matches

    assert matches(SimpleNamespace()) is False
    assert matches(SimpleNamespace(model_config=None)) is False


def test_detect_empty_aliases():
    from mlx_teacache.variants.z_image_base.detect import matches

    assert matches(_fake_flux([])) is False
    assert matches(SimpleNamespace(model_config=SimpleNamespace(aliases=None))) is False


# --- config.META + coefficients ------------------------------------------


def test_config_meta_identifies_variant():
    from mlx_teacache.variants.z_image_base import config

    assert config.META["variant_id"] == "z-image-base"
    assert config.META["non_distilled"] is True
    assert config.META["memory_cap_hint_gb"] == 22  # findings 2026-05-31


def test_config_coefficients_are_signal_b_origin_fit():
    from mlx_teacache.variants.z_image_base import config

    coeffs = config.COEFFICIENTS
    assert len(coeffs) == 5
    assert all(isinstance(c, float) for c in coeffs)
    # origin-constrained fit ⇒ c0 == 0.0 (poly(0) = 0)
    assert coeffs[-1] == 0.0
    # Signal B leading coefficient (from scripts/_calibration_z_image.json).
    assert abs(coeffs[0] - (-898.9907628349583)) < 1e-6


def test_config_default_thresh_is_float():
    from mlx_teacache.variants.z_image_base import config

    assert isinstance(config.DEFAULT_THRESH, float)
    assert 0.0 < config.DEFAULT_THRESH < 1.0


# --- registry auto-discovery ----------------------------------------------


def test_registry_registers_z_image_base():
    from mlx_teacache.variants import _REGISTRY

    assert "z-image-base" in _REGISTRY
    entry = _REGISTRY["z-image-base"]
    assert entry["META"]["variant_id"] == "z-image-base"
    # detect is wired through the registry and base-only.
    assert entry["matches"](_fake_flux(["z-image", "zimage"])) is True
    assert entry["matches"](_fake_flux(["z-image-turbo"])) is False
