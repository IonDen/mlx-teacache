# tests/test_api_dispatch.py
"""apply_teacache dispatches via _REGISTRY and preserves the v0.5.x
public signature (audit F3)."""

import inspect

import pytest

from mlx_teacache import apply_teacache
from mlx_teacache.errors import IncompatibleModelError


def test_apply_teacache_signature_is_frozen() -> None:
    """Bug caught: a new kwarg, a dropped `*`, or a changed default passes a subset check."""
    params = inspect.signature(apply_teacache).parameters
    assert [(n, p.kind, p.default) for n, p in params.items()] == [
        ("flux", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
        ("rel_l1_thresh", inspect.Parameter.KEYWORD_ONLY, None),
        ("coefficients", inspect.Parameter.KEYWORD_ONLY, None),
        ("skip_first_n_steps", inspect.Parameter.KEYWORD_ONLY, 1),
        ("skip_last_n_steps", inspect.Parameter.KEYWORD_ONLY, 1),
    ]


def test_skip_first_n_steps_default_is_1() -> None:
    sig = inspect.signature(apply_teacache)
    assert sig.parameters["skip_first_n_steps"].default == 1


def test_skip_last_n_steps_default_is_1() -> None:
    sig = inspect.signature(apply_teacache)
    assert sig.parameters["skip_last_n_steps"].default == 1


def test_coefficients_default_is_none() -> None:
    sig = inspect.signature(apply_teacache)
    assert sig.parameters["coefficients"].default is None


class _FC:
    def __init__(self, a: list[str]) -> None:
        self.aliases = a
        self.model_name = "fake/x"


class _FakeFlux1:
    def __init__(self, a: list[str]) -> None:
        self.model_config = _FC(a)


def test_unknown_variant_raises() -> None:
    with pytest.raises(IncompatibleModelError) as exc:
        apply_teacache(_FakeFlux1(["bogus"]))
    msg = str(exc.value)
    assert "flux1-dev" in msg
