"""Registry walks variants/ at import time. config.py + detect.py load
eagerly. integration.py is lazy (loaded on first dispatch). The walker
must be mflux-free at import time."""

from __future__ import annotations

from typing import Any

import pytest

from mlx_teacache.errors import CalibrationError


def test_registry_is_a_mapping() -> None:
    from mlx_teacache.variants import _REGISTRY

    assert isinstance(_REGISTRY, dict)


def test_registry_entries_have_required_shape() -> None:
    from mlx_teacache.variants import _REGISTRY

    for entry in _REGISTRY.values():
        assert "META" in entry
        assert "matches" in entry
        assert "load_integration" in entry
        assert callable(entry["matches"])
        assert callable(entry["load_integration"])


def test_registry_keys_match_meta_variant_ids() -> None:
    from mlx_teacache.variants import _REGISTRY

    for variant_id, entry in _REGISTRY.items():
        assert entry["META"]["variant_id"] == variant_id


def _fake_entry(variant_id: str) -> Any:
    return {
        "META": {"variant_id": variant_id},
        "matches": lambda _flux: False,
        "load_integration": lambda: lambda *a, **k: None,
        "default_thresh": None,
    }


def test_duplicate_variant_id_raises_naming_both_subpackages() -> None:
    """Bug caught: a copied subpackage silently overwrites the original registry entry."""
    from mlx_teacache.variants import _register

    reg: dict[str, Any] = {}
    sources: dict[str, str] = {}
    _register(reg, sources, "flux1-dev", _fake_entry("flux1-dev"), subname="flux1_dev")
    with pytest.raises(CalibrationError, match="flux1_dev.*flux1_dev_copy"):
        _register(reg, sources, "flux1-dev", _fake_entry("flux1-dev"), subname="flux1_dev_copy")
    assert list(reg) == ["flux1-dev"]
    assert sources == {"flux1-dev": "flux1_dev"}


def test_registry_holds_exactly_the_nine_shipped_variant_ids() -> None:
    """Bug caught: a variant silently dropped or shadowed by the registry walk."""
    from mlx_teacache.variants import _REGISTRY

    assert sorted(_REGISTRY) == [
        "flux1-dev",
        "flux1-krea-dev",
        "flux1-schnell",
        "flux2-klein-4b",
        "flux2-klein-9b",
        "flux2-klein-base-4b",
        "flux2-klein-base-9b",
        "qwen-image",
        "z-image-base",
    ]
