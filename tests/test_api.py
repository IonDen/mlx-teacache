# tests/test_api.py
"""End-to-end tests for apply_teacache using a synthetic flux model that
mimics enough of the Flux1 surface to exercise the patching/restore cycle.
Real-model parity is in tests/test_parity_*.py."""

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_teacache import (
    AlreadyPatchedError,
    IncompatibleModelError,
    apply_teacache,
)
from mlx_teacache.errors import TeaCacheDisabledWarning, TeaCacheValueError
from tests._fakes import FaithfulCallbackRegistry


class _FakeTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.x_embedder = nn.Linear(4, 8, bias=False)

    def __call__(self, **kwargs):
        return mx.zeros((1, 8))


def _make_fake_flux1(alias="dev"):
    """Build a fake that detect.identify_variant will accept as Flux1.

    mflux 0.17+ requires `model_config.aliases` to contain the short name
    ("dev" or "schnell") — `model_name` alone is the HF repo path and is
    ambiguous with controlnet/upscaler variants."""
    from mflux.models.flux.variants.txt2img.flux import Flux1

    flux = Flux1.__new__(Flux1)
    flux.model_config = SimpleNamespace(
        model_name="black-forest-labs/FLUX.1-" + alias,
        aliases=[alias],
    )
    flux.transformer = _FakeTransformer()
    flux.callbacks = FaithfulCallbackRegistry()
    flux.generate_image = lambda **kw: "image"
    return flux


def test_apply_and_restore_roundtrip():
    flux = _make_fake_flux1()
    original_transformer = flux.transformer
    original_generate = flux.generate_image
    handle = apply_teacache(flux, rel_l1_thresh=0.25)
    assert handle.variant_id == "flux1-dev"
    assert handle.rel_l1_thresh == 0.25
    assert flux.transformer is not original_transformer
    assert flux.generate_image is not original_generate
    assert flux._teacache_handle is handle
    assert handle._callback_instance in flux.callbacks.before_loop
    handle.restore()
    assert flux.transformer is original_transformer
    # generate_image: was an instance attr ⇒ should be restored to original
    assert flux.generate_image is original_generate
    assert getattr(flux, "_teacache_handle", None) is None
    assert handle._callback_instance not in flux.callbacks.before_loop


def test_double_apply_raises():
    flux = _make_fake_flux1()
    apply_teacache(flux, rel_l1_thresh=0.25)
    with pytest.raises(AlreadyPatchedError):
        apply_teacache(flux, rel_l1_thresh=0.4)


def test_double_apply_raises_already_patched_flux1():
    flux = _make_fake_flux1()
    h = apply_teacache(flux)
    try:
        with pytest.raises(AlreadyPatchedError):
            apply_teacache(flux)
    finally:
        h.restore()


def test_re_apply_after_restore_succeeds():
    flux = _make_fake_flux1()
    h1 = apply_teacache(flux, rel_l1_thresh=0.25)
    h1.restore()
    h2 = apply_teacache(flux, rel_l1_thresh=0.4)
    assert h2.rel_l1_thresh == 0.4
    h2.restore()


def test_failed_restore_blocks_reapply_until_retry_succeeds():
    flux = _make_fake_flux1()
    handle = apply_teacache(flux, rel_l1_thresh=0.25)
    callback = handle._callback_instance
    attempts = 0

    def _fail_once():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient rollback failure")

    handle._patch.rollbacks.append(_fail_once)

    with pytest.raises(RuntimeError, match="transient rollback failure"):
        handle.restore()
    assert flux._teacache_handle is handle
    assert callback not in flux.callbacks.before_loop
    with pytest.raises(AlreadyPatchedError):
        apply_teacache(flux)

    handle.restore()
    assert getattr(flux, "_teacache_handle", None) is None
    reapplied = apply_teacache(flux, rel_l1_thresh=0.25)
    reapplied.restore()


def test_context_manager_restores():
    flux = _make_fake_flux1()
    original_transformer = flux.transformer
    with apply_teacache(flux) as h:
        assert flux.transformer is not original_transformer
        assert h.variant_id == "flux1-dev"
    assert flux.transformer is original_transformer


def test_invalid_threshold_raises():
    flux = _make_fake_flux1()
    with pytest.raises(ValueError, match="rel_l1_thresh"):
        apply_teacache(flux, rel_l1_thresh=1.5)


def test_invalid_skip_negative_raises():
    flux = _make_fake_flux1()
    with pytest.raises(ValueError, match="skip_first"):
        apply_teacache(flux, skip_first_n_steps=-1)


def test_invalid_coefficients_length_raises():
    flux = _make_fake_flux1()
    with pytest.raises(ValueError, match="length 5"):
        apply_teacache(flux, coefficients=[1.0, 2.0])


def test_unsupported_model_raises_incompatible():
    class Other: ...

    other = Other()
    with pytest.raises(IncompatibleModelError):
        apply_teacache(other)


def test_stats_initially_empty():
    flux = _make_fake_flux1()
    h = apply_teacache(flux)
    assert h.stats.total_steps_seen == 0
    assert h.stats.generations == 0
    assert h.stats.speedup_estimate == 1.0
    h.restore()


def test_zero_threshold_emits_disable_warning():
    flux = _make_fake_flux1()
    with pytest.warns(TeaCacheDisabledWarning, match="disables"):
        h = apply_teacache(flux, rel_l1_thresh=0.0)
    h.restore()


def test_apply_teacache_rejects_nonfinite_coefficients_at_call_time():
    flux = _make_fake_flux1()
    with pytest.raises(TeaCacheValueError, match="finite"):
        apply_teacache(flux, coefficients=(1.0, float("nan"), 0.0, 0.0, 0.0))


def test_apply_teacache_coerces_list_coefficients_to_tuple():
    flux = _make_fake_flux1()
    h = apply_teacache(flux, coefficients=[1.0, -0.5, 0.1, 0.0, 0.0])
    try:
        assert h.coefficients == (1.0, -0.5, 0.1, 0.0, 0.0)
        assert isinstance(h.coefficients, tuple)
        assert h.provenance.source == "user"
    finally:
        h.restore()


@pytest.mark.parametrize(
    "alias,patch_module",
    [
        # dev calls wrap_generate_image via the module object (deliberately
        # monkeypatchable) — patch the lifecycle module itself.
        ("dev", "mlx_teacache.integrations.mflux.lifecycle"),
        # schnell calls wrap_generate_image via its module-level imported name —
        # patch the VARIANT module's copy; patching lifecycle would not
        # intercept the call and the test would become an always-green no-op.
        ("schnell", "mlx_teacache.variants.flux1_schnell.integration"),
    ],
)
def test_apply_rollback_on_failure_flux1(alias, patch_module, monkeypatch):
    """Per audit medium #3: if wrap_generate_image raises mid-apply, the
    transactional guard must roll back fully — no leftover callback, no proxy
    transformer, no sentinel.

    dev   → patches the lifecycle module (dev calls via the module object).
    schnell → patches the variant module's imported-name binding.
    """
    import importlib

    flux = _make_fake_flux1(alias)
    before = list(flux.callbacks.before_loop)
    original_transformer = flux.transformer
    target = importlib.import_module(patch_module)
    monkeypatch.setattr(
        target,
        "wrap_generate_image",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    with pytest.raises(RuntimeError):
        apply_teacache(flux)
    # Pristine-state asserts — schnell is the RED signal (dangling state IS the bug):
    assert flux.callbacks.before_loop == before, "callback left registered"
    assert flux.transformer is original_transformer, "proxy transformer left installed"
    assert "_teacache_handle" not in vars(flux), "sentinel left behind"


@pytest.mark.parametrize("alias", ["dev", "schnell"])
def test_apply_rollback_on_register_failure_flux1(alias, monkeypatch):
    flux = _make_fake_flux1(alias)
    original_transformer = flux.transformer
    before = {
        name: list(getattr(flux.callbacks, name))
        for name in ("before_loop", "in_loop", "after_loop", "interrupt")
    }

    def _partially_register_then_boom(callback):
        flux.callbacks.before_loop.append(callback)
        raise RuntimeError("register boom")

    monkeypatch.setattr(flux.callbacks, "register", _partially_register_then_boom)
    with pytest.raises(RuntimeError, match="register boom"):
        apply_teacache(flux)
    assert flux.transformer is original_transformer, "proxy transformer left installed"
    assert "_teacache_handle" not in vars(flux), "sentinel left behind"
    for name, expected in before.items():
        assert getattr(flux.callbacks, name) == expected, f"callback left in {name}"


@pytest.mark.parity
def test_apply_teacache_accepts_flux2_klein_9b():
    """Smoke: apply_teacache returns a handle with the right variant_id on Klein 9B.
    Catches api.py regressions in the variant_id Literal or the FLUX.2 _predict guard."""
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

    from mlx_teacache import apply_teacache
    from tests.conftest import expect_distilled_warning

    flux = Flux2Klein(quantize=4, model_config=ModelConfig.flux2_klein_9b())
    flux.freeze()
    with expect_distilled_warning("flux2-klein-9b"):
        handle = apply_teacache(flux)
    try:
        assert handle.variant_id == "flux2-klein-9b"
        assert len(handle.coefficients) == 5
        assert handle.provenance.source == "builtin"
    finally:
        handle.restore()


@pytest.mark.parity
def test_apply_teacache_accepts_flux2_klein_base_4b():
    """Smoke: apply_teacache returns a handle with the right variant_id on Klein base-4B.
    Catches api.py regressions in the variant_id Literal or the FLUX.2 _predict guard."""
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

    from mlx_teacache import apply_teacache

    flux = Flux2Klein(quantize=4, model_config=ModelConfig.flux2_klein_base_4b())
    flux.freeze()
    handle = apply_teacache(flux)
    try:
        assert handle.variant_id == "flux2-klein-base-4b"
        assert len(handle.coefficients) == 5
        assert handle.provenance.source == "builtin"
    finally:
        handle.restore()


@pytest.mark.parity
def test_apply_teacache_uses_per_variant_default_for_klein_base_4b():
    """When no rel_l1_thresh is passed, base-4b's per-variant default (0.17) is applied."""
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

    from mlx_teacache import apply_teacache

    flux = Flux2Klein(quantize=4, model_config=ModelConfig.flux2_klein_base_4b())
    flux.freeze()
    handle = apply_teacache(flux)  # no rel_l1_thresh — should use variant default
    try:
        assert handle.rel_l1_thresh == 0.17
    finally:
        handle.restore()


@pytest.mark.parity
def test_apply_teacache_explicit_thresh_overrides_per_variant_default():
    """Explicit rel_l1_thresh wins over per-variant default."""
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

    from mlx_teacache import apply_teacache

    flux = Flux2Klein(quantize=4, model_config=ModelConfig.flux2_klein_base_4b())
    flux.freeze()
    handle = apply_teacache(flux, rel_l1_thresh=0.05)
    try:
        assert handle.rel_l1_thresh == 0.05
    finally:
        handle.restore()


@pytest.mark.parity
def test_apply_teacache_user_coefficients_skip_per_variant_default():
    """User-supplied coefficients on base-4b fall back to the package default 0.20,
    NOT the per-variant 0.17 (which was tuned for the bundled polynomial)."""
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

    from mlx_teacache import apply_teacache

    flux = Flux2Klein(quantize=4, model_config=ModelConfig.flux2_klein_base_4b())
    flux.freeze()
    custom_coeffs = (1.0, -0.5, 0.1, 0.0, 0.0)
    handle = apply_teacache(flux, coefficients=custom_coeffs)
    try:
        assert handle.rel_l1_thresh == 0.20
        assert handle.provenance.source == "user"
    finally:
        handle.restore()


@pytest.mark.parity
def test_apply_teacache_cfg_records_cfg_was_active_klein_base_4b():
    """v0.4.1: at guidance > 1.0, the gated CFG forward fires; the staging
    buffer's cfg_was_active flag flips True. GenerationStats.cfg_was_active
    then propagates from staging at finalize time."""
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

    from mlx_teacache import apply_teacache

    flux = Flux2Klein(quantize=4, model_config=ModelConfig.flux2_klein_base_4b())
    flux.freeze()
    handle = apply_teacache(flux)
    try:
        flux.generate_image(
            prompt="a red apple",
            seed=42,
            num_inference_steps=4,
            height=512,
            width=512,
            guidance=4.0,
        )
        assert handle.stats.last_generation is not None
        assert handle.stats.last_generation.cfg_was_active is True
        kinds = {d.decision for d in handle.stats.last_generation.decisions}
        assert "cfg-fallback" not in kinds
    finally:
        handle.restore()


@pytest.mark.parity
def test_invalid_skip_window_raises_under_cfg_klein_base_4b():
    """v0.4.1 behavior change: an all-CFG generation with skip_first + skip_last
    >= num_inference_steps used to silently run vanilla in v0.4.0. In v0.4.1 the
    CFG path is gated, so the lazy skip-window validation fires and raises."""
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

    from mlx_teacache import InvalidStepWindowError, apply_teacache

    flux = Flux2Klein(quantize=4, model_config=ModelConfig.flux2_klein_base_4b())
    flux.freeze()
    handle = apply_teacache(flux, skip_first_n_steps=2, skip_last_n_steps=2)
    try:
        with pytest.raises(InvalidStepWindowError):
            flux.generate_image(
                prompt="a red apple",
                seed=42,
                num_inference_steps=4,
                height=512,
                width=512,
                guidance=4.0,
            )
    finally:
        handle.restore()


@pytest.mark.parity
@pytest.mark.parametrize(
    "variant_id,patch_module,make_flux",
    [
        (
            "flux2-klein-4b",
            "mlx_teacache.variants.flux2_klein_4b.integration",
            lambda: __import__(
                "mflux.models.flux2.variants.txt2img.flux2_klein", fromlist=["Flux2Klein"]
            ).Flux2Klein(
                quantize=4,
                model_config=__import__(
                    "mflux.models.common.config.model_config", fromlist=["ModelConfig"]
                ).ModelConfig.flux2_klein_4b(),
            ),
        ),
        (
            "flux2-klein-9b",
            "mlx_teacache.variants.flux2_klein_9b.integration",
            lambda: __import__(
                "mflux.models.flux2.variants.txt2img.flux2_klein", fromlist=["Flux2Klein"]
            ).Flux2Klein(
                quantize=4,
                model_config=__import__(
                    "mflux.models.common.config.model_config", fromlist=["ModelConfig"]
                ).ModelConfig.flux2_klein_9b(),
            ),
        ),
        (
            "flux2-klein-base-4b",
            "mlx_teacache.variants.flux2_klein_base_4b.integration",
            lambda: __import__(
                "mflux.models.flux2.variants.txt2img.flux2_klein", fromlist=["Flux2Klein"]
            ).Flux2Klein(
                quantize=4,
                model_config=__import__(
                    "mflux.models.common.config.model_config", fromlist=["ModelConfig"]
                ).ModelConfig.flux2_klein_base_4b(),
            ),
        ),
        (
            "flux2-klein-base-9b",
            "mlx_teacache.variants.flux2_klein_base_9b.integration",
            lambda: __import__(
                "mflux.models.flux2.variants.txt2img.flux2_klein", fromlist=["Flux2Klein"]
            ).Flux2Klein(
                quantize=4,
                model_config=__import__(
                    "mflux.models.common.config.model_config", fromlist=["ModelConfig"]
                ).ModelConfig.flux2_klein_base_9b(),
            ),
        ),
        (
            "z-image-base",
            "mlx_teacache.variants.z_image_base.integration",
            lambda: __import__("mflux.models.z_image.variants.z_image", fromlist=["ZImage"]).ZImage(
                quantize=8,
                model_config=__import__(
                    "mflux.models.common.config.model_config", fromlist=["ModelConfig"]
                ).ModelConfig.z_image(),
            ),
        ),
    ],
)
def test_apply_rollback_on_failure_flux2(variant_id: str, patch_module: str, make_flux, monkeypatch) -> None:
    """Per audit medium #3: if wrap_generate_image raises mid-apply on a FLUX.2/Z-Image
    variant, the transactional guard must restore pristine state — no dangling _predict,
    no leftover callback, no instance-level generate_image.

    Each variant patches ITS OWN module's local wrap_generate_image binding.

    Distilled Kleins (flux2-klein-4b/9b) warn TeaCacheNoBenefitWarning at
    apply time, before load_integration()/wrap_generate_image ever run — under
    filterwarnings=error that warning-as-exception would preempt the
    monkeypatched RuntimeError, so it must be expected via
    expect_distilled_warning inside the pytest.raises(RuntimeError) block
    (pytest.warns resets the filter to "always" for its scope, letting
    execution continue to the real failure)."""
    import importlib

    from tests.conftest import expect_distilled_warning

    flux = make_flux()
    flux.freeze()
    before_callback_count = len(flux.callbacks.before_loop)
    target = importlib.import_module(patch_module)
    monkeypatch.setattr(
        target,
        "wrap_generate_image",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    with pytest.raises(RuntimeError), expect_distilled_warning(variant_id):
        apply_teacache(flux)
    assert len(flux.callbacks.before_loop) == before_callback_count, "callback left registered"
    assert "_predict" not in vars(flux), "_predict instance attr left behind"
    assert "generate_image" not in vars(flux), "generate_image instance attr left behind"
    assert "_teacache_handle" not in vars(flux), "sentinel left behind"


@pytest.mark.parity
@pytest.mark.parametrize(
    "variant_id,make_flux",
    [
        (
            "flux1-dev",
            lambda: __import__("mflux.models.flux.variants.txt2img.flux", fromlist=["Flux1"]).Flux1.from_name(
                "dev", quantize=4
            ),
        ),
        (
            "flux1-schnell",
            lambda: __import__("mflux.models.flux.variants.txt2img.flux", fromlist=["Flux1"]).Flux1.from_name(
                "schnell", quantize=4
            ),
        ),
        (
            "flux2-klein-4b",
            lambda: __import__(
                "mflux.models.flux2.variants.txt2img.flux2_klein", fromlist=["Flux2Klein"]
            ).Flux2Klein(
                quantize=4,
                model_config=__import__(
                    "mflux.models.common.config.model_config", fromlist=["ModelConfig"]
                ).ModelConfig.flux2_klein_4b(),
            ),
        ),
        (
            "flux2-klein-9b",
            lambda: __import__(
                "mflux.models.flux2.variants.txt2img.flux2_klein", fromlist=["Flux2Klein"]
            ).Flux2Klein(
                quantize=4,
                model_config=__import__(
                    "mflux.models.common.config.model_config", fromlist=["ModelConfig"]
                ).ModelConfig.flux2_klein_9b(),
            ),
        ),
        (
            "flux2-klein-base-4b",
            lambda: __import__(
                "mflux.models.flux2.variants.txt2img.flux2_klein", fromlist=["Flux2Klein"]
            ).Flux2Klein(
                quantize=4,
                model_config=__import__(
                    "mflux.models.common.config.model_config", fromlist=["ModelConfig"]
                ).ModelConfig.flux2_klein_base_4b(),
            ),
        ),
        (
            "flux2-klein-base-9b",
            lambda: __import__(
                "mflux.models.flux2.variants.txt2img.flux2_klein", fromlist=["Flux2Klein"]
            ).Flux2Klein(
                quantize=4,
                model_config=__import__(
                    "mflux.models.common.config.model_config", fromlist=["ModelConfig"]
                ).ModelConfig.flux2_klein_base_9b(),
            ),
        ),
        (
            "z-image-base",
            lambda: __import__("mflux.models.z_image.variants.z_image", fromlist=["ZImage"]).ZImage(
                quantize=8,
                model_config=__import__(
                    "mflux.models.common.config.model_config", fromlist=["ModelConfig"]
                ).ModelConfig.z_image(),
            ),
        ),
    ],
)
def test_user_coefficients_provenance_all_variants(variant_id: str, make_flux) -> None:
    """All 7 variants: custom coefficients yield handle.provenance.source == 'user'.

    User-supplied coefficients don't change the apply-time no-benefit warning:
    it's keyed on the registry's default_thresh (a per-variant fact), not on
    what the caller passes, so distilled Kleins still warn here too."""
    from tests.conftest import expect_distilled_warning

    flux = make_flux()
    flux.freeze()
    custom_coeffs = (1.0, -0.5, 0.1, 0.0, 0.0)
    with expect_distilled_warning(variant_id):
        handle = apply_teacache(flux, coefficients=custom_coeffs)
    try:
        assert handle.variant_id == variant_id
        assert handle.provenance.source == "user"
        assert handle.coefficients == custom_coeffs
    finally:
        handle.restore()


# --- at-apply distilled warning vs custom coefficients --------------------------


class _StopAtIntegration(RuntimeError):
    """Raised by the stubbed integration loader so the test observes only the
    pre-dispatch path in apply_teacache (warning check) without importing mflux."""


def _fake_distilled_klein_4b():
    return SimpleNamespace(model_config=SimpleNamespace(aliases=["flux2-klein-4b"]))


def _stub_loader(monkeypatch):
    from mlx_teacache.variants import _REGISTRY

    def _boom():
        raise _StopAtIntegration

    monkeypatch.setitem(_REGISTRY["flux2-klein-4b"], "load_integration", _boom)


def test_distilled_variant_warns_at_apply_with_builtin_coefficients(monkeypatch):
    """Fast pin of the at-apply trigger: a distilled Klein on builtin
    coefficients warns before the integration is even loaded."""
    from mlx_teacache.errors import TeaCacheNoBenefitWarning

    _stub_loader(monkeypatch)
    with pytest.warns(TeaCacheNoBenefitWarning), pytest.raises(_StopAtIntegration):
        apply_teacache(_fake_distilled_klein_4b())


def test_distilled_variant_does_not_warn_when_caller_supplies_coefficients(monkeypatch):
    """RED if the at-apply warning fires regardless of `coefficients`. The
    warning text says the *builtin* polynomial does not engage on a few-step
    schedule; a caller passing their own tuple is deliberately experimenting
    and, under filterwarnings=error, would be unable to call apply at all."""
    import warnings

    from mlx_teacache.errors import TeaCacheNoBenefitWarning

    _stub_loader(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("error", TeaCacheNoBenefitWarning)
        with pytest.raises(_StopAtIntegration):
            apply_teacache(_fake_distilled_klein_4b(), coefficients=(0.1, 0.2, 0.3, 0.4, 0.5))
