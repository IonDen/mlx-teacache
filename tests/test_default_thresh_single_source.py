"""Every variant's Provenance records the same default threshold the registry resolves."""

import pytest

from mlx_teacache import TeaCacheNoBenefitWarning, apply_teacache
from mlx_teacache.variants import _REGISTRY
from tests._variant_fakes import fake_for_variant

pytestmark = pytest.mark.mflux


@pytest.mark.parametrize("variant_id", sorted(_REGISTRY))
def test_provenance_default_thresh_equals_registry_default(variant_id: str) -> None:
    """Bug caught: Provenance.default_thresh None for dev/schnell/krea; literal 0.17 drifting from DEFAULT_THRESH."""
    fake = fake_for_variant(variant_id)
    if _REGISTRY[variant_id]["default_thresh"] is None:
        with pytest.warns(TeaCacheNoBenefitWarning):
            handle = apply_teacache(fake)
    else:
        handle = apply_teacache(fake)
    try:
        assert handle.variant_id == variant_id
        assert handle.provenance.default_thresh == _REGISTRY[variant_id]["default_thresh"]
    finally:
        handle.restore()


@pytest.mark.parametrize(
    ("variant_id", "expected"),
    [
        ("flux1-dev", 0.20),
        ("flux1-schnell", 0.20),
        ("flux1-krea-dev", 0.30),
        ("flux2-klein-base-4b", 0.17),
        ("flux2-klein-base-9b", 0.17),
        ("z-image-base", 0.12),
        ("qwen-image", 0.30),
        ("flux2-klein-4b", None),
        ("flux2-klein-9b", None),
    ],
)
def test_provenance_default_thresh_literal_values(variant_id: str, expected: float | None) -> None:
    """Bug caught: registry and provenance agree on a wrong value (both edited, or both None)."""
    fake = fake_for_variant(variant_id)
    if expected is None:
        with pytest.warns(TeaCacheNoBenefitWarning):
            handle = apply_teacache(fake)
    else:
        handle = apply_teacache(fake)
    try:
        assert handle.provenance.default_thresh == expected
    finally:
        handle.restore()
