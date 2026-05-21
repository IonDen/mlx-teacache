from __future__ import annotations


class _FC:
    def __init__(self, aliases: list[str]) -> None:
        self.aliases = aliases
        self.model_name = "fake/flux1-dev"


class _FakeFlux1:
    def __init__(self, aliases: list[str]) -> None:
        self.model_config = _FC(aliases)


def test_meta_variant_id() -> None:
    from mlx_teacache.variants.flux1_dev.config import META

    assert META["variant_id"] == "flux1-dev"
    assert META["non_distilled"] is True


def test_matches_dev_alias() -> None:
    from mlx_teacache.variants.flux1_dev.detect import matches

    assert matches(_FakeFlux1(["dev"])) is True


def test_does_not_match_schnell() -> None:
    from mlx_teacache.variants.flux1_dev.detect import matches

    assert matches(_FakeFlux1(["schnell"])) is False
