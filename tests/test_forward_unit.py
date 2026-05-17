"""Pure-shape unit tests for the new CFG forward. Real-model parity is in
tests/test_parity_flux2.py."""


def test_flux2_cfg_forward_with_gate_is_importable():
    """v0.4.1 contract: forward.py exposes flux2_cfg_forward_with_gate."""
    from mlx_teacache.integrations.mflux.forward import flux2_cfg_forward_with_gate

    assert callable(flux2_cfg_forward_with_gate)
