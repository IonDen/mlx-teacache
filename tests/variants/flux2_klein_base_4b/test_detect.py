from __future__ import annotations


class _FC:
    def __init__(self, aliases: list[str]) -> None:
        self.aliases = aliases
        self.model_name = "fake/flux2-klein-base-4b"


class _FakeFlux2:
    def __init__(self, aliases: list[str]) -> None:
        self.model_config = _FC(aliases)


def test_meta_variant_id() -> None:
    from mlx_teacache.variants.flux2_klein_base_4b.config import META
    assert META["variant_id"] == "flux2-klein-base-4b"
    assert META["non_distilled"] is True


def test_coefficients_match_v05_registry() -> None:
    """Audit F4 guard: the variant's COEFFICIENTS must equal the v0.5.x
    registry entry. This catches transcription errors before the legacy
    registry is removed in Task 18."""
    from mlx_teacache.coefficients import _FLUX2_KLEIN_BASE_4B_COEFFS
    from mlx_teacache.variants.flux2_klein_base_4b.config import COEFFICIENTS
    assert COEFFICIENTS == _FLUX2_KLEIN_BASE_4B_COEFFS


def test_default_thresh_is_017() -> None:
    from mlx_teacache.variants.flux2_klein_base_4b.config import DEFAULT_THRESH
    assert DEFAULT_THRESH == 0.17


def test_matches_alias() -> None:
    from mlx_teacache.variants.flux2_klein_base_4b.detect import matches
    assert matches(_FakeFlux2(["flux2-klein-base-4b"])) is True


def test_does_not_match_dev_alias() -> None:
    from mlx_teacache.variants.flux2_klein_base_4b.detect import matches
    assert matches(_FakeFlux2(["dev"])) is False
