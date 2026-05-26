"""Real-weight smoke test for flux1-dev integration. Marked slow+network
because Flux1.from_name("dev") downloads the gated black-forest-labs/FLUX.1-dev
weights from HuggingFace — CI does not have HF auth for that repo. Run
locally with `pytest -m "slow and network" tests/variants/flux1_dev/` after
accepting the model license."""

import pytest


@pytest.mark.slow
@pytest.mark.network
def test_apply_returns_handle_and_restores_pristine() -> None:
    pytest.importorskip("mflux")
    from mflux.models.flux.variants.txt2img.flux import Flux1

    flux = Flux1.from_name("dev", quantize=4)
    flux.freeze()
    transformer_before = flux.transformer

    from mlx_teacache.handle import TeaCacheHandle
    from mlx_teacache.variants.flux1_dev.integration import apply

    handle = apply(flux)
    assert isinstance(handle, TeaCacheHandle)
    # transformer was wrapped during apply
    assert flux.transformer is not transformer_before
    handle.restore()
    # original transformer is back
    assert flux.transformer is transformer_before
