"""The two shared patch installers, through every registered variant: install order, the one
`wrap_generate_image` seam, rollback on a failed install, and the session the public handle carries.
Pure-core: each variant's apply() runs on a duck-typed model, no mflux, no weights."""

import inspect
from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import pytest

from mlx_teacache import api
from mlx_teacache._kernel.coefficients import Provenance
from mlx_teacache._kernel.resolve import ResolvedRule
from mlx_teacache.integrations.mflux import lifecycle
from mlx_teacache.variants import _REGISTRY
from tests._fakes import FaithfulCallbackRegistry
from tests._variant_fakes import fake_for_variant

_PROXY_IDS = ("flux1-dev", "flux1-schnell", "flux1-krea-dev", "qwen-image", "qwen-image-2512")
_PREDICT_IDS = (
    "flux2-klein-4b",
    "flux2-klein-9b",
    "flux2-klein-base-4b",
    "flux2-klein-base-9b",
    "z-image-base",
)
_ALL_IDS = _PROXY_IDS + _PREDICT_IDS
_FLUX1_ALIASES = {"flux1-dev": "dev", "flux1-schnell": "schnell", "flux1-krea-dev": "krea-dev"}
_LISTS = ("before_loop", "in_loop", "after_loop", "interrupt")


def _fake(variant_id: str) -> Any:
    if variant_id in _FLUX1_ALIASES:
        alias = _FLUX1_ALIASES[variant_id]
        return SimpleNamespace(
            model_config=SimpleNamespace(model_name="black-forest-labs/FLUX.1-" + alias, aliases=[alias]),
            transformer=SimpleNamespace(name="real-transformer"),
            callbacks=FaithfulCallbackRegistry(),
            generate_image=lambda **kw: "image",
        )
    return fake_for_variant(variant_id)


def _apply(variant_id: str) -> Any:
    return _REGISTRY[variant_id]["load_integration"]()


def _lists(flux: Any) -> dict[str, list[Any]]:
    return {name: list(getattr(flux.callbacks, name)) for name in _LISTS}


def _forward_session(flux: Any, variant_id: str) -> Any:
    """The session object the installed forward reads and writes."""
    if variant_id in _PROXY_IDS:
        return flux.transformer._handle
    return inspect.getclosurevars(flux._predict).nonlocals["handle"]


def _raise_wrap(*_a: Any, **_k: Any) -> None:
    raise RuntimeError("wrap boom")


def test_the_two_strategies_cover_the_registry() -> None:
    """bug caught: a new variant is registered and none of the install checks below run on it."""
    assert sorted(_ALL_IDS) == sorted(_REGISTRY)


@pytest.mark.parametrize("variant_id", _PROXY_IDS)
def test_proxy_install_swaps_the_transformer_before_registering(variant_id: str) -> None:
    """bug caught: registering before the swap changes what a failed registration has to undo."""
    flux = _fake(variant_id)
    original = flux.transformer
    seen: list[Any] = []
    real_register = flux.callbacks.register

    def _spy(cb: Any) -> None:
        seen.append(flux.transformer)
        real_register(cb)

    flux.callbacks.register = _spy
    handle = _apply(variant_id)(flux)
    try:
        assert len(seen) == 1
        assert seen[0] is not original
        assert seen[0] is flux.transformer
    finally:
        handle.restore()
    assert flux.transformer is original


@pytest.mark.parametrize("variant_id", _PREDICT_IDS)
def test_predict_install_assigns_predict_after_wrapping(
    variant_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """bug caught: `_predict` assigned before the wrap, so a failed wrap leaves it patched; or the variant binds
    `wrap_generate_image` by name and the lifecycle seam never sees the call."""
    apply = _apply(variant_id)  # import the integration before patching the seam
    flux = _fake(variant_id)
    seen: list[bool] = []
    real_wrap = lifecycle.wrap_generate_image

    def _spy(f: Any, session: Any) -> None:
        seen.append("_predict" in vars(f))
        real_wrap(f, session)

    monkeypatch.setattr(lifecycle, "wrap_generate_image", _spy)
    handle = apply(flux)
    try:
        assert seen == [False]
        assert "_predict" in vars(flux)
    finally:
        handle.restore()
    assert "_predict" not in vars(flux)


@pytest.mark.parametrize("variant_id", _ALL_IDS)
def test_wrap_failure_rolls_back_every_variant_through_the_one_seam(
    variant_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """bug caught: a variant still binds `wrap_generate_image` by name, so the one seam misses it; or the
    rollback leaves the callback, the proxy or `_predict` behind."""
    apply = _apply(variant_id)  # import the integration before patching the seam
    flux = _fake(variant_id)
    original_transformer = flux.transformer
    original_generate_image = flux.generate_image
    before = _lists(flux)
    monkeypatch.setattr(lifecycle, "wrap_generate_image", _raise_wrap)
    with pytest.raises(RuntimeError, match="wrap boom"):
        apply(flux)
    assert _lists(flux) == before, "callback left registered"
    assert flux.transformer is original_transformer, "proxy transformer left installed"
    assert "_predict" not in vars(flux), "_predict left patched"
    assert flux.generate_image is original_generate_image, "generate_image left wrapped"


@pytest.mark.parametrize("variant_id", _ALL_IDS)
def test_handle_carries_the_session_and_its_callback(variant_id: str) -> None:
    """bug caught: the handle's callback is not the one registered for its session, so the generate_image
    wrapper's registration check and restore() look at different callbacks."""
    flux = _fake(variant_id)
    handle = _apply(variant_id)(flux)
    try:
        assert handle._session is not None
        assert handle._session._callback_instance is handle._callback_instance
        assert handle._callback_instance in flux.callbacks.before_loop
        assert handle.stats is handle._session._state.stats
    finally:
        handle.restore()


@pytest.mark.parametrize("variant_id", _ALL_IDS)
def test_release_hook_reaches_the_cache_the_forward_writes(variant_id: str) -> None:
    """bug caught: a variant leaves `handle._session` unset (or set to another session), so restore() keeps the
    forward's cached arrays alive."""
    flux = _fake(variant_id)
    handle = _apply(variant_id)(flux)
    try:
        forward = _forward_session(flux, variant_id)
        assert handle._session is forward
        cache = forward._state.cache
        cache.previous_mod_input = mx.zeros((4,))
        cache.cached_residual = mx.zeros((4,))
        cache.cached_residual_neg = mx.zeros((4,))
        api._release_cache_arrays(handle)
        assert cache.previous_mod_input is None
        assert cache.cached_residual is None
        assert cache.cached_residual_neg is None
    finally:
        handle.restore()


_RULE = ResolvedRule(
    rel_l1_thresh=0.33,
    coefficients=(1.0, 0.5, 0.25, 0.0, 0.0),
    provenance=Provenance.for_user_supplied(),
    calibrated_range=(0.1, 0.4),
)


class _Proxy:
    def __init__(self, inner: Any, session: Any) -> None:
        self._inner = inner
        self._handle = session


def _assert_session_follows_the_rule(handle: Any, session: Any) -> None:
    assert handle._session is session
    assert handle.rel_l1_thresh == 0.33
    assert handle.coefficients == (1.0, 0.5, 0.25, 0.0, 0.0)
    assert handle.provenance is _RULE.provenance
    assert session.rel_l1_thresh == 0.33
    assert session.coefficients == (1.0, 0.5, 0.25, 0.0, 0.0)
    assert session.calibrated_range == (0.1, 0.4)
    assert (session.skip_first_n_steps, session.skip_last_n_steps) == (2, 3)


def test_proxy_installer_builds_the_session_from_the_rule() -> None:
    """bug caught: the installer drops the calibrated range, swaps the two skip counts, or hands the proxy a
    different session than the handle's."""
    from mlx_teacache.integrations.mflux.patch import install_proxy_patch

    flux = _fake("flux1-dev")
    original = flux.transformer
    seen: list[tuple[Any, Any]] = []

    def _make_proxy(inner: Any, session: Any) -> _Proxy:
        seen.append((inner, session))
        return _Proxy(inner, session)

    handle = install_proxy_patch(
        flux, make_proxy=_make_proxy, rule=_RULE, skip_first_n_steps=2, skip_last_n_steps=3
    )
    try:
        assert len(seen) == 1
        assert seen[0][0] is original
        assert flux.transformer._handle is seen[0][1]
        _assert_session_follows_the_rule(handle, seen[0][1])
    finally:
        handle.restore()
    assert flux.transformer is original


def test_predict_installer_builds_the_session_from_the_rule() -> None:
    """bug caught: the installer drops the calibrated range, swaps the two skip counts, or builds the factory on a
    different session than the handle's."""
    from mlx_teacache.integrations.mflux.patch import install_predict_patch

    flux = _fake("z-image-base")
    seen: list[Any] = []

    def _factory(_transformer: Any) -> Any:
        return None

    def _make_factory(session: Any) -> Any:
        seen.append(session)
        return _factory

    handle = install_predict_patch(
        flux, make_factory=_make_factory, rule=_RULE, skip_first_n_steps=2, skip_last_n_steps=3
    )
    try:
        assert len(seen) == 1
        assert vars(flux)["_predict"] is _factory
        _assert_session_follows_the_rule(handle, seen[0])
    finally:
        handle.restore()
    assert "_predict" not in vars(flux)
