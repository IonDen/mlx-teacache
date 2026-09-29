"""apply() hands a variant's CALIBRATED_RANGE to the gate only when the gate runs that variant's own
coefficients, and every forward passes it on (backlog 0094). Pure-core: duck-typed fake flux, no weights."""

import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests._fakes import FaithfulCallbackRegistry

# Literals from the committed calibration JSONs (x_min, x_max of the fit each variant ships).
_BASE_4B = (0.028261402621865273, 0.21984468400478363)
_Z_IMAGE_B = (0.027887196237753466, 0.25914120883567665)
_QWEN_A = (0.04675976472922664, 0.587799896108221)

_RANGED = [
    ("flux2_klein_base_4b", _BASE_4B),
    ("flux2_klein_base_9b", _BASE_4B),
    ("z_image_base", _Z_IMAGE_B),
    ("qwen_image", _QWEN_A),
]
_SRC = Path(__file__).resolve().parents[1] / "src" / "mlx_teacache"


def _fake_flux() -> SimpleNamespace:
    return SimpleNamespace(
        transformer=SimpleNamespace(name="real-transformer"),
        callbacks=FaithfulCallbackRegistry(),
        generate_image=lambda **kw: "image",
    )


def _gate_handle(flux: SimpleNamespace) -> Any:
    """The internal handle the forward reads, reached through the lifecycle callback apply() registered."""
    return next(cb for cb in flux.callbacks.before_loop if hasattr(cb, "_handle"))._handle


@pytest.mark.parametrize(("variant", "expected"), _RANGED, ids=[v for v, _ in _RANGED])
def test_the_variants_own_coefficients_run_with_its_calibrated_range(
    variant: str, expected: tuple[float, float]
) -> None:
    """Bug: apply() never sets the range (the clamp is dead code in production) or sets another variant's."""
    apply = importlib.import_module(f"mlx_teacache.variants.{variant}.integration").apply
    flux = _fake_flux()
    handle = apply(flux, rel_l1_thresh=0.25)
    assert _gate_handle(flux).calibrated_range == expected
    handle.restore()


@pytest.mark.parametrize(("variant", "expected"), _RANGED, ids=[v for v, _ in _RANGED])
def test_caller_coefficients_run_without_a_range(variant: str, expected: tuple[float, float]) -> None:
    """Bug: a caller's own polynomial gets clamped to a range it was never fitted on."""
    apply = importlib.import_module(f"mlx_teacache.variants.{variant}.integration").apply
    flux = _fake_flux()
    handle = apply(flux, rel_l1_thresh=0.25, coefficients=(0.0, 0.0, 0.0, 1.0, 0.0))
    assert _gate_handle(flux).calibrated_range is None
    handle.restore()


def test_distilled_klein_runs_without_a_range() -> None:
    """Bug: the range leaks through the shared FLUX.1-dev bridge into a variant that declares none."""
    apply = importlib.import_module("mlx_teacache.variants.flux2_klein_4b.integration").apply
    flux = _fake_flux()
    handle = apply(flux, rel_l1_thresh=0.25)
    assert _gate_handle(flux).calibrated_range is None
    handle.restore()


def test_every_gate_step_call_site_passes_the_handles_range() -> None:
    """Bug: a forward (FLUX.1, FLUX.2 plain/CFG, Z-Image plain/CFG, Qwen) calls gate_step without the range,
    or with a constant instead of the handle's, so that path silently extrapolates the fit again. The forwards
    only run with real weights, so each call is checked in the source: every gate_step(...) must pass
    calibrated_range=handle.calibrated_range."""
    import ast

    calls = []
    for path in sorted((_SRC / "variants").glob("*/integration.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "gate_step":
                kw = {k.arg: k.value for k in node.keywords}
                value = kw.get("calibrated_range")
                wired = (
                    isinstance(value, ast.Attribute)
                    and value.attr == "calibrated_range"
                    and isinstance(value.value, ast.Name)
                    and value.value.id == "handle"
                )
                calls.append((f"{path.parent.name}:{node.lineno}", wired))
    assert len(calls) == 6, calls
    assert all(wired for _, wired in calls), [site for site, wired in calls if not wired]
