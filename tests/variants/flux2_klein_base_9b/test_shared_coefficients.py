from __future__ import annotations


def test_klein_base_9b_reuses_base_4b_coefficients() -> None:
    """v0.5.0 validated this reuse (SSIM 0.986). The test catches any
    accidental drift from the intentional reuse."""
    from mlx_teacache.variants.flux2_klein_base_4b.config import COEFFICIENTS as BASE_4B
    from mlx_teacache.variants.flux2_klein_base_9b.config import COEFFICIENTS as BASE_9B

    assert BASE_9B is BASE_4B
