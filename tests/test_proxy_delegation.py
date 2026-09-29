"""Module-level operations on the TeaCache proxies must reach the wrapped transformer."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_teacache.variants.flux1_dev.integration import ProxyFlux1Transformer
from mlx_teacache.variants.qwen_image.integration import ProxyQwenTransformer

pytestmark = pytest.mark.mflux


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4)


class _Tiny64(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(64, 64)


@pytest.fixture(params=[ProxyFlux1Transformer, ProxyQwenTransformer], ids=["flux1", "qwen"])
def proxy_cls(request):
    return request.param


def test_load_weights_reaches_inner(proxy_cls) -> None:
    """Bug caught: update() walks the empty proxy dict; load_weights silently loads nothing."""
    inner = _Tiny()
    proxy = proxy_cls(inner, None)
    new_w = mx.full((4, 4), 7.0)
    proxy.load_weights([("proj.weight", new_w), ("proj.bias", mx.zeros((4,)))])
    assert mx.array_equal(inner.proj.weight, new_w).item()


def test_load_weights_strict_missing_key_raises(proxy_cls) -> None:
    """Bug caught: delegation swallows strict=True errors."""
    proxy = proxy_cls(_Tiny(), None)
    with pytest.raises(ValueError):
        proxy.load_weights([("proj.weight", mx.zeros((4, 4)))], strict=True)


def test_load_weights_non_strict_missing_key_loads_present_ones(proxy_cls) -> None:
    """Bug caught: strict is hard-wired True in the delegation, so a partial load raises."""
    inner = _Tiny()
    proxy = proxy_cls(inner, None)
    proxy.load_weights([("proj.weight", mx.full((4, 4), 3.0))], strict=False)
    assert inner.proj.weight[0, 0].item() == 3.0


def test_set_dtype_reaches_inner(proxy_cls) -> None:
    """Bug caught: set_dtype casts the empty proxy tree, leaving inner weights in the old dtype."""
    inner = _Tiny()
    proxy_cls(inner, None).set_dtype(mx.float16)
    assert inner.proj.weight.dtype == mx.float16


def test_quantize_of_parent_keeps_proxy_installed(proxy_cls) -> None:
    """Bug caught: quantizing the owning model replaces or drops the proxy installed on it."""
    proxy = proxy_cls(_Tiny64(), None)
    holder = _Holder(proxy)
    nn.quantize(holder, group_size=32, bits=4)
    assert holder.transformer is proxy


def test_quantize_directly_on_proxy(proxy_cls) -> None:
    """Bug caught: quantizing the proxy object itself is a silent no-op."""
    inner = _Tiny64()
    proxy = proxy_cls(inner, None)
    nn.quantize(proxy, group_size=32, bits=4)
    assert isinstance(inner.proj, nn.QuantizedLinear)
    assert isinstance(proxy, proxy_cls)


def test_eval_reaches_inner(proxy_cls) -> None:
    """Bug caught: children() is empty on the proxy, so eval() never flips inner layers."""
    inner = _Tiny()
    proxy_cls(inner, None).eval()
    assert inner.proj.training is False


def test_children_and_leaf_modules_terminate_and_show_inner(proxy_cls) -> None:
    """Bug caught: delegation recurses forever or exposes the proxy instead of the inner tree."""
    inner = _Tiny()
    proxy = proxy_cls(inner, None)
    assert proxy.children()["proj"] is inner.proj
    assert proxy.leaf_modules()["proj"] is inner.proj


class _Holder(nn.Module):
    def __init__(self, proxy) -> None:
        super().__init__()
        self.transformer = proxy


def test_update_forwards_strict_flag(proxy_cls) -> None:
    """Bug caught: update() drops strict, so an unknown key is rejected even with strict=False."""
    proxy = proxy_cls(_Tiny(), None)
    with pytest.raises(ValueError):
        proxy.update({"nope": {"weight": mx.zeros((4, 4))}}, strict=True)
    proxy.update({"nope": {"weight": mx.zeros((4, 4))}}, strict=False)


def test_update_modules_replaces_inner_child_and_forwards_strict(proxy_cls) -> None:
    """Bug caught: update_modules edits the empty proxy dict or drops strict."""
    inner = _Tiny()
    proxy = proxy_cls(inner, None)
    new = nn.Linear(4, 4)
    proxy.update_modules({"proj": new})
    assert inner.proj is new
    with pytest.raises(ValueError):
        proxy.update_modules({"nope": nn.Linear(4, 4)}, strict=True)
    proxy.update_modules({"nope": nn.Linear(4, 4)}, strict=False)
