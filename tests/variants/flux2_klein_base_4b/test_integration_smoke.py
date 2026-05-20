"""Real-weight smoke test for flux2-klein-base-4b integration. Skipped
without mflux installed or if Flux2Klein.from_name constructor is not
available in the installed mflux version (constructor signature varies
across versions; T20 validation covers the full load path)."""
from __future__ import annotations

import pytest


@pytest.mark.skip("smoke deferred until mflux 0.17 Flux2Klein constructor pinned — Flux2Klein has no from_name; T20 validation covers the real load path")
def test_apply_returns_handle_and_restores_pristine() -> None:
    pytest.importorskip("mflux")
    try:
        from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
    except ImportError:
        pytest.skip("Flux2Klein not found at expected mflux path")

    flux = Flux2Klein.from_name("klein-base-4b", quantize=4)  # type: ignore[attr-defined]
    flux.freeze()
    original_predict = vars(flux).get("_predict")  # None before apply — not an instance attr

    from mlx_teacache.handle import TeaCacheHandle
    from mlx_teacache.variants.flux2_klein_base_4b.integration import apply

    handle = apply(flux)
    assert isinstance(handle, TeaCacheHandle)
    # _predict was set as an instance attribute during apply
    assert "_predict" in vars(flux)
    handle.restore()
    # _predict was deleted on restore (was not an instance attr before apply)
    assert "_predict" not in vars(flux)
    # original_predict was None (not an instance attr) — confirm we got back to pristine
    assert vars(flux).get("_predict") == original_predict
