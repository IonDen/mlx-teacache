"""flux2-klein-9b variant detection + config."""
from __future__ import annotations


class _FC:
    def __init__(self, aliases: list[str]) -> None:
        self.aliases = aliases
        self.model_name = "fake/x"


class _FakeFlux2:
    def __init__(self, aliases: list[str]) -> None:
        self.model_config = _FC(aliases)


def test_meta_variant_id() -> None:
    from mlx_teacache.variants.flux2_klein_9b.config import META
    assert META["variant_id"] == "flux2-klein-9b"
    assert META["non_distilled"] is False
    assert META["recipes"]["default"]["num_inference_steps"] == 8
    assert META["recipes"]["default"]["guidance"] == 1.0



def test_default_thresh_is_none() -> None:
    """Distilled 8-step gate doesn't engage; package fallback used."""
    from mlx_teacache.variants.flux2_klein_9b.config import DEFAULT_THRESH
    assert DEFAULT_THRESH is None


def test_matches_alias() -> None:
    from mlx_teacache.variants.flux2_klein_9b.detect import matches
    assert matches(_FakeFlux2(["flux2-klein-9b"])) is True


def test_does_not_match_base_9b_alias() -> None:
    from mlx_teacache.variants.flux2_klein_9b.detect import matches
    assert matches(_FakeFlux2(["flux2-klein-base-9b"])) is False
