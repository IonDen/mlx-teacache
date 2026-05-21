from __future__ import annotations


def test_meta() -> None:
    from mlx_teacache.variants.flux1_schnell.config import META

    assert META["variant_id"] == "flux1-schnell"
    assert META["non_distilled"] is False
    assert META["recipes"]["default"]["num_inference_steps"] == 4


def test_coefficients_shared_with_dev() -> None:
    from mlx_teacache.variants.flux1_dev.config import COEFFICIENTS as DEV
    from mlx_teacache.variants.flux1_schnell.config import COEFFICIENTS as SCHNELL

    assert SCHNELL is DEV


def test_matches_schnell() -> None:
    from mlx_teacache.variants.flux1_schnell.detect import matches

    class _FC:
        def __init__(self, a: list[str]) -> None:
            self.aliases = a
            self.model_name = "fake/x"

    class _FakeFlux1:
        def __init__(self, a: list[str]) -> None:
            self.model_config = _FC(a)

    assert matches(_FakeFlux1(["schnell"])) is True
    assert matches(_FakeFlux1(["dev"])) is False
