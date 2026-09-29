"""Tests for the shared helpers in tests/conftest.py (mflux-free: the variant
integration loader is stubbed, so only the pre-dispatch path of
`apply_teacache` runs)."""

from types import SimpleNamespace

import pytest

from mlx_teacache import apply_teacache
from tests.conftest import expect_distilled_warning


class _StopAtIntegration(RuntimeError):
    """Raised by the stubbed loader so apply_teacache stops after its warning check."""


def _fake_distilled_klein_4b() -> SimpleNamespace:
    return SimpleNamespace(model_config=SimpleNamespace(aliases=["flux2-klein-4b"]))


@pytest.fixture
def stubbed_klein_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    from mlx_teacache.variants import _REGISTRY

    def _boom() -> None:
        raise _StopAtIntegration

    monkeypatch.setitem(_REGISTRY["flux2-klein-4b"], "load_integration", _boom)


def test_expect_distilled_warning_expects_nothing_when_coefficients_given(
    stubbed_klein_loader,
) -> None:
    """Bug caught: helper demands the warning api.py deliberately suppresses for caller coefficients."""
    coeffs = (1.0, 0.0, 0.0, 0.0, 0.0)
    with (
        expect_distilled_warning("flux2-klein-4b", coefficients=coeffs),
        pytest.raises(_StopAtIntegration),
    ):
        apply_teacache(_fake_distilled_klein_4b(), coefficients=coeffs)


def test_expect_distilled_warning_still_requires_warning_on_builtin_coefficients(
    stubbed_klein_loader,
) -> None:
    """Bug caught: helper became a no-op and no longer fails when the distilled warning is missing."""
    with (
        pytest.raises(pytest.fail.Exception, match="TeaCacheNoBenefitWarning"),
        expect_distilled_warning("flux2-klein-4b"),
    ):
        pass  # no apply call: the required warning never fires
    with expect_distilled_warning("flux2-klein-4b"), pytest.raises(_StopAtIntegration):
        apply_teacache(_fake_distilled_klein_4b())
