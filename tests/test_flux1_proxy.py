# tests/test_flux1_proxy.py
"""Compatibility tests for ProxyFlux1Transformer. Use a tiny synthetic
nn.Module as a stand-in for mflux's Transformer — we're testing the proxy's
own surface here, not mflux's transformer logic. Deep parity tests against
a real Flux1 live in tests/test_parity_flux1.py (Task 25)."""

import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

pytestmark = pytest.mark.parity


class _TinyInnerTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.x_embedder = nn.Linear(8, 8, bias=False)
        self.norm_out = nn.LayerNorm(8)

    def __call__(self, hidden_states, **kwargs):
        return self.norm_out(self.x_embedder(hidden_states))


@pytest.fixture
def proxy_and_inner():
    from mlx_teacache.integrations.mflux.flux1 import ProxyFlux1Transformer

    inner = _TinyInnerTransformer()
    handle = object()  # not exercised in these compat tests
    proxy = ProxyFlux1Transformer(inner, handle)
    return proxy, inner


def test_parameters_method_returns_inner_params(proxy_and_inner):
    proxy, inner = proxy_and_inner
    proxy_params = dict(tree_flatten(proxy.parameters()))
    inner_params = dict(tree_flatten(inner.parameters()))
    assert set(proxy_params) == set(inner_params)


def test_trainable_parameters_method(proxy_and_inner):
    proxy, inner = proxy_and_inner
    pt = dict(tree_flatten(proxy.trainable_parameters()))
    it = dict(tree_flatten(inner.trainable_parameters()))
    assert set(pt) == set(it)


def test_freeze_delegates(proxy_and_inner):
    proxy, inner = proxy_and_inner
    proxy.freeze()
    pt = dict(tree_flatten(proxy.trainable_parameters()))
    # Frozen => no trainable parameters reachable from the inner
    assert len(pt) == 0


def test_attribute_passthrough(proxy_and_inner):
    proxy, inner = proxy_and_inner
    assert proxy.x_embedder is inner.x_embedder
    assert proxy.norm_out is inner.norm_out


def test_parent_traversal_misses_inner_documented_limitation():
    """Negative test: assert the documented limitation holds, so it can't
    silently regress. Parent-level traversal of a holder that uses the proxy
    as a child should NOT include the proxy's inner params (because _inner
    is filtered by MLX's valid_parameter_filter)."""
    from mlx_teacache.integrations.mflux.flux1 import ProxyFlux1Transformer

    class Holder(nn.Module):
        def __init__(self, transformer):
            super().__init__()
            self.transformer = transformer

    inner = _TinyInnerTransformer()
    proxy = ProxyFlux1Transformer(inner, object())
    holder = Holder(proxy)

    parent_keys = set(dict(tree_flatten(holder.parameters())).keys())
    # The proxy's _inner is under "_inner" key, which MLX excludes. So
    # holder.parameters() may not include x_embedder.weight. This is the
    # documented limitation — users needing parent traversal must restore().
    # We assert the limitation rather than fight MLX's filter.
    inner_param_keys = {k for k in parent_keys if "x_embedder" in k}
    assert len(inner_param_keys) == 0, (
        "Documented limitation broke: parent traversal now includes inner. "
        "Either MLX changed or the proxy was redesigned. Update docs."
    )
