"""Registry walks variants/ at import time. config.py + detect.py load
eagerly. integration.py is lazy (loaded on first dispatch). The walker
must be mflux-free at import time."""

from __future__ import annotations


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
