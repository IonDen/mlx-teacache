"""restore() refuses, with RestoreRefusedError and without undoing anything, while a generation still has steps to run
and when flux.generate_image or flux._predict was replaced after apply_teacache.

The pipelines are the equivalence oracle's small surrogates, one per patch strategy (FLUX.1 proxy, FLUX.2 predict,
Z-Image predict, Qwen-Image proxy). They run mflux's callback order: before-loop callbacks, the steps (a per-step
`step_hook` after each one), after-loop callbacks. The surrogates never dispatch `callbacks.in_loop`, so the before-loop
restore() that docs/stability.md documents also runs on `_GuardedLoopPipeline`, which does, with the predict
variants' in-loop guard registered.
"""

import pickle
from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import pytest

from mlx_teacache import RestoreRefusedError, apply_teacache
from tests import _equivalence_oracle as oracle
from tests._fakes import FaithfulCallbackRegistry
from tests._variant_fakes import fake_for_variant

pytestmark = pytest.mark.mflux

REPRESENTATIVES = ("flux1-dev", "flux2-klein-base-4b", "z-image-base", "qwen-image")
PREDICT_REPRESENTATIVES = ("flux2-klein-base-4b", "z-image-base")


def _patched(variant_id: str, *, cfg: bool = False) -> tuple[Any, Any, Any]:
    flux = oracle._build(variant_id, cfg=cfg)
    original_transformer = flux.transformer
    handle = apply_teacache(flux, rel_l1_thresh=0.25, coefficients=oracle._CONST_COEFFICIENTS)
    return flux, handle, original_transformer


def _assert_pristine(flux: Any, original_transformer: Any) -> None:
    assert flux.transformer is original_transformer
    assert "generate_image" not in vars(flux)  # the surrogates define it on the class
    assert "_predict" not in vars(flux)
    assert "_teacache_handle" not in vars(flux)
    assert list(flux.callbacks.before_loop) == []
    assert list(flux.callbacks.in_loop) == []  # the predict variants' guard is an in-loop callback
    assert list(flux.callbacks.after_loop) == []


def _patch_snapshot(flux: Any) -> dict[str, Any]:
    """What apply_teacache installed: a refusal must leave every one of these in place."""
    return {
        "transformer": flux.transformer,
        "generate_image": vars(flux).get("generate_image"),
        "_predict": vars(flux).get("_predict"),
        "_teacache_handle": vars(flux).get("_teacache_handle"),
        "before_loop": list(flux.callbacks.before_loop),
        "after_loop": list(flux.callbacks.after_loop),
    }


def _unregister(flux: Any, callback: Any) -> None:
    """Remove the test's own callback by identity, so whatever else is left registered stays visible."""
    for attribute in ("before_loop", "after_loop"):
        kept = [item for item in getattr(flux.callbacks, attribute) if item is not callback]
        setattr(flux.callbacks, attribute, kept)


class _RestoreInCallback:
    def __init__(
        self,
        handle: Any,
        *,
        before_loop: bool = False,
        evict_transformer_of: Any = None,
        snapshot_of: Any = None,
    ) -> None:
        self._handle = handle
        self._before_loop = before_loop
        self._evict = evict_transformer_of
        self._snapshot_of = snapshot_of
        self.error: BaseException | None = None
        self.patch_after_error: dict[str, Any] | None = None
        self.calls = 0

    def _restore(self) -> None:
        self.calls += 1
        if self._evict is not None:
            self._evict.transformer = None  # mflux's memory saver, before the decode
        try:
            self._handle.restore()
        except BaseException as exc:  # recorded so the generation goes on
            self.error = exc
            if self._snapshot_of is not None:
                self.patch_after_error = _patch_snapshot(self._snapshot_of)

    def call_before_loop(self, *a: Any, **k: Any) -> None:
        if self._before_loop:
            self._restore()

    def call_after_loop(self, *a: Any, **k: Any) -> None:
        if not self._before_loop:
            self._restore()


_NO_CFG_IN_THE_FAKE = pytest.mark.skip(
    reason="FLUX.1 dev runs no CFG pass (guidance is distilled into the model); its surrogate has none to turn on"
)
_QWEN_ALWAYS_CFG = pytest.mark.skip(
    reason="the Qwen-Image surrogate runs the positive and negative pass on every step, so cfg=False covers CFG"
)
_IN_LOOP_CASES = [pytest.param(v, False, id=f"{v}-no-cfg") for v in REPRESENTATIVES] + [
    pytest.param("flux1-dev", True, id="flux1-dev-cfg", marks=_NO_CFG_IN_THE_FAKE),
    pytest.param("flux2-klein-base-4b", True, id="flux2-klein-base-4b-cfg"),
    pytest.param("z-image-base", True, id="z-image-base-cfg"),
    pytest.param("qwen-image", True, id="qwen-image-cfg", marks=_QWEN_ALWAYS_CFG),
]


@pytest.mark.parametrize(("variant_id", "cfg"), _IN_LOOP_CASES)
def test_restore_from_an_in_loop_callback_is_refused(variant_id: str, cfg: bool) -> None:
    """Bug: the refusal leaves the handle half torn down, or one strategy keeps the silent behaviour (FLUX.1 and
    Qwen-Image ran the rest without TeaCache and recorded nothing; FLUX.2 and Z-Image raised StatsFrozenError).
    Under CFG, bug: the step count advances once per branch instead of once per step, so the refusal reports the
    wrong number of steps left."""
    flux, handle, original_transformer = _patched(variant_id, cfg=cfg)
    installed = _patch_snapshot(flux)
    seen: dict[str, Any] = {}

    def restore_after_step_3(t: int) -> None:
        if t == 3:
            try:
                handle.restore()
            except RestoreRefusedError as exc:
                seen["error"] = exc
                seen["patch"] = _patch_snapshot(flux)

    flux.step_hook = restore_after_step_3
    flux.generate_image(request=oracle._Request(seed=1))
    del flux.step_hook

    error = seen["error"]
    assert error.reason == "generation-in-flight"
    assert error.attribute is None
    assert str(error) == (
        "restore() was called while a generation still had 6 of 10 steps to run; call it after generate_image "
        "returns or from an after-loop callback. The handle is unchanged."
    )
    assert seen["patch"] == installed
    # The generation completed with TeaCache still active and its stats committed.
    assert handle.stats.generations == 1
    assert len(handle.stats.last_generation.decisions) == 10
    assert handle.stats.skipped_count >= 1  # TeaCache kept skipping after the refusal
    handle.restore()
    _assert_pristine(flux, original_transformer)


@pytest.mark.parametrize("variant_id", REPRESENTATIVES)
def test_restore_from_a_before_loop_callback_is_refused(variant_id: str) -> None:
    """Bug: the check reads `<=` or only looks at in-loop steps, so a restore() before step 0 tears down mid-run; or
    the refusal undoes part of the patch before raising, so the run goes on without TeaCache."""
    flux, handle, _ = _patched(variant_id)
    restorer = _RestoreInCallback(handle, before_loop=True, snapshot_of=flux)
    flux.callbacks.register(restorer)  # after TeaCache's: its before-loop has set the step count
    installed = _patch_snapshot(flux)
    flux.generate_image(request=oracle._Request(seed=1))
    assert isinstance(restorer.error, RestoreRefusedError)
    assert "still had 10 of 10 steps" in str(restorer.error)
    assert restorer.patch_after_error == installed
    assert handle.stats.generations == 1
    assert handle.stats.skipped_count >= 1  # TeaCache stayed engaged for the run


@pytest.mark.parametrize("registered", ["after-apply", "before-apply"])
@pytest.mark.parametrize("variant_id", REPRESENTATIVES)
def test_restore_from_an_after_loop_callback_is_allowed(variant_id: str, registered: str) -> None:
    """Bug: the 0.12.1 "free TeaCache before decode" pattern breaks. Registered before apply, the callback runs while
    TeaCache's own after-loop has not cleared the step count yet, with every step done (a `<=` check refuses it)."""
    flux = oracle._build(variant_id, cfg=False)
    original_transformer = flux.transformer
    early = None
    if registered == "before-apply":
        early = _RestoreInCallback(None)
        flux.callbacks.register(early)
    handle = apply_teacache(flux, rel_l1_thresh=0.25, coefficients=oracle._CONST_COEFFICIENTS)
    restorer = early or _RestoreInCallback(handle)
    restorer._handle = handle
    if early is None:
        flux.callbacks.register(restorer)
    latents = flux.generate_image(request=oracle._Request(seed=1))
    assert latents.shape == flux.latent_shape
    assert restorer.calls == 1
    assert restorer.error is None
    _unregister(flux, restorer)  # a TeaCache lifecycle callback left registered makes the next check fail
    _assert_pristine(flux, original_transformer)


@pytest.mark.parametrize("variant_id", REPRESENTATIVES)
def test_restore_after_memory_saver_eviction_is_not_refused(variant_id: str) -> None:
    """Bug: a check reads the evicted transformer slot (or the proxy behind it) and refuses or raises."""
    flux = oracle._build(variant_id, cfg=False)
    restorer = _RestoreInCallback(None, evict_transformer_of=flux)
    flux.callbacks.register(restorer)  # before apply, so it runs before TeaCache's after-loop
    handle = apply_teacache(flux, rel_l1_thresh=0.25, coefficients=oracle._CONST_COEFFICIENTS)
    restorer._handle = handle
    flux.generate_image(request=oracle._Request(seed=1))
    assert restorer.error is None
    assert flux.transformer is None  # the slot stays empty, as with plain mflux
    assert "generate_image" not in vars(flux)
    assert "_teacache_handle" not in vars(flux)


@pytest.mark.parametrize("variant_id", REPRESENTATIVES)
def test_zero_step_generation_does_not_block_restore(variant_id: str) -> None:
    """image_strength 1.0: mflux runs no step. Bug: the check refuses because the step count is 0, not None."""
    flux, handle, original_transformer = _patched(variant_id)
    restorer = _RestoreInCallback(handle, before_loop=True)
    flux.callbacks.register(restorer)
    flux.generate_image(request=oracle._Request(seed=1, init_time_step=oracle._STEPS))
    assert restorer.calls == 1
    assert restorer.error is None
    _unregister(flux, restorer)  # a TeaCache lifecycle callback left registered makes the next check fail
    _assert_pristine(flux, original_transformer)


@pytest.mark.parametrize("variant_id", REPRESENTATIVES)
def test_restore_from_a_before_loop_callback_registered_before_apply_is_allowed(variant_id: str) -> None:
    """Registered before apply, the callback's before-loop runs ahead of TeaCache's, so no step count is set yet and
    restore() goes through. Bug: the generation that is already running then raises StatsFrozenError when it commits
    its stats, or leaves the model half patched."""
    flux = oracle._build(variant_id, cfg=False)
    original_transformer = flux.transformer
    restorer = _RestoreInCallback(None, before_loop=True)
    flux.callbacks.register(restorer)
    handle = apply_teacache(flux, rel_l1_thresh=0.25, coefficients=oracle._CONST_COEFFICIENTS)
    restorer._handle = handle
    latents = flux.generate_image(request=oracle._Request(seed=1))
    assert latents.shape == flux.latent_shape
    assert restorer.calls == 1
    assert restorer.error is None
    _unregister(flux, restorer)
    _assert_pristine(flux, original_transformer)


class _GuardedLoopPipeline:
    """A predict-strategy pipeline whose generate_image runs mflux's loop with the in-loop callbacks: before-loop
    callbacks, `predict = self._predict(self.transformer)`, per step one predict call followed by the in-loop
    callbacks (the list is read again at each step, as mflux does), then the after-loop callbacks.

    `_predict` and `generate_image` live on the class, as on mflux's ZImage and Flux2Klein. The class-level
    `_predict` is the pre-apply one: it returns a working step function that counts its calls, so a run that
    TeaCache no longer patches completes."""

    def __init__(self, variant_id: str) -> None:
        self.model_config = fake_for_variant(variant_id).model_config  # what the variant detector reads
        self.transformer = SimpleNamespace(name="transformer")
        self.callbacks = FaithfulCallbackRegistry()
        self.unpatched_steps = 0
        self.in_loop_rounds = 0

    def _predict(self, transformer: Any, **_kwargs: Any) -> Any:
        def step(**_kw: Any) -> None:
            self.unpatched_steps += 1  # stands in for one call of mflux's own transformer

        return step

    def generate_image(self, seed: int = 0, prompt: str = "", num_inference_steps: int = 4) -> str:
        config = SimpleNamespace(num_inference_steps=num_inference_steps, init_time_step=0)
        latents = mx.zeros((1, 4, 4))
        for cb in self.callbacks.before_loop:
            cb.call_before_loop(seed=seed, prompt=prompt, latents=latents, config=config)
        predict = self._predict(self.transformer)
        for t in range(num_inference_steps):
            predict(t=t)
            for cb in self.callbacks.in_loop:
                cb.call_in_loop(
                    t=t, seed=seed, prompt=prompt, latents=latents, config=config, time_steps=None
                )
            self.in_loop_rounds += 1
        del predict
        for cb in self.callbacks.after_loop:
            cb.call_after_loop(seed=seed, prompt=prompt, latents=latents, config=config)
        return "image"


@pytest.mark.parametrize("variant_id", PREDICT_REPRESENTATIVES)
def test_documented_before_loop_restore_runs_the_generation_without_the_guard(variant_id: str) -> None:
    """docs/stability.md: a callback registered before apply that calls restore() just before the loop is not
    refused, raises nothing, and that generation runs without TeaCache. Bug: restore() leaves the predict guard in
    `callbacks.in_loop`, so the run that already lost TeaCache's `_predict` is stopped at its first step with
    IncompatibleModelError (and the guard stays registered on the restored model)."""
    flux = _GuardedLoopPipeline(variant_id)
    original_transformer = flux.transformer
    restorer = _RestoreInCallback(None, before_loop=True)
    flux.callbacks.register(restorer)
    handle = apply_teacache(flux, rel_l1_thresh=0.25, coefficients=oracle._CONST_COEFFICIENTS)
    restorer._handle = handle
    assert "_predict" in vars(flux)  # TeaCache's factory is installed until the restore
    assert len(flux.callbacks.in_loop) == 1  # and its guard is in play: the in-loop list holds it

    assert flux.generate_image(seed=1) == "image"

    assert restorer.calls == 1
    assert restorer.error is None
    assert flux.unpatched_steps == 4  # every step ran on the pre-apply _predict
    assert flux.in_loop_rounds == 4
    _unregister(flux, restorer)
    _assert_pristine(flux, original_transformer)


def _wrap_over(flux: Any, attribute: str) -> tuple[Any, Any]:
    installed = vars(flux)[attribute]

    def user_wrapper(*args: Any, **kwargs: Any) -> Any:
        return installed(*args, **kwargs)

    setattr(flux, attribute, user_wrapper)
    return installed, user_wrapper


@pytest.mark.parametrize("variant_id", REPRESENTATIVES)
def test_wrapper_over_generate_image_is_refused_then_retryable(variant_id: str) -> None:
    """Bug: restore drops the later wrapper (0.17 overwrote it silently), or undoes part of the patch first."""
    flux, handle, original_transformer = _patched(variant_id)
    installed, user_wrapper = _wrap_over(flux, "generate_image")
    before = _patch_snapshot(flux)
    with pytest.raises(RestoreRefusedError) as info:
        handle.restore()
    assert info.value.reason == "patched-over"
    assert info.value.attribute == "generate_image"
    assert str(info.value) == (
        "flux.generate_image was replaced after apply_teacache; restore() would remove that replacement. Undo it "
        "first (restore in reverse order of patching), then call restore() again. The handle is unchanged."
    )
    assert _patch_snapshot(flux) == before
    assert vars(flux)["generate_image"] is user_wrapper
    assert flux._teacache_handle is handle
    flux.generate_image = installed
    handle.restore()
    _assert_pristine(flux, original_transformer)


@pytest.mark.parametrize("variant_id", PREDICT_REPRESENTATIVES)
def test_wrapper_over_predict_is_refused(variant_id: str) -> None:
    """Bug: restore puts back the pre-apply _predict over a wrapper the caller installed after apply."""
    flux, handle, original_transformer = _patched(variant_id)
    installed, user_wrapper = _wrap_over(flux, "_predict")
    before = _patch_snapshot(flux)
    with pytest.raises(RestoreRefusedError) as info:
        handle.restore()
    assert info.value.reason == "patched-over"
    assert info.value.attribute == "_predict"
    assert str(info.value).startswith("flux._predict was replaced after apply_teacache;")
    assert _patch_snapshot(flux) == before
    assert vars(flux)["_predict"] is user_wrapper
    flux._predict = installed
    handle.restore()
    _assert_pristine(flux, original_transformer)


@pytest.mark.parametrize(
    ("variant_id", "attribute"),
    [(v, "generate_image") for v in REPRESENTATIVES] + [(v, "_predict") for v in PREDICT_REPRESENTATIVES],
)
def test_deleted_slot_is_not_refused(variant_id: str, attribute: str) -> None:
    """An empty slot drops nothing. Bug: the check treats a missing attribute as a replacement and refuses."""
    flux, handle, original_transformer = _patched(variant_id)
    delattr(flux, attribute)
    handle.restore()
    _assert_pristine(flux, original_transformer)


@pytest.mark.parametrize(
    ("variant_id", "attribute"),
    [(v, "generate_image") for v in REPRESENTATIVES] + [(v, "_predict") for v in PREDICT_REPRESENTATIVES],
)
def test_deleted_slot_over_an_instance_attribute_is_not_refused(variant_id: str, attribute: str) -> None:
    """The pre-apply value was an instance attribute (a caller's wrapper installed before apply); the slot is then
    deleted after apply. Bug: the empty slot is called a replacement and restore() refuses."""
    flux = oracle._build(variant_id, cfg=False)
    pre_apply = getattr(flux, attribute)
    setattr(flux, attribute, pre_apply)
    handle = apply_teacache(flux, rel_l1_thresh=0.25, coefficients=oracle._CONST_COEFFICIENTS)
    delattr(flux, attribute)
    handle.restore()
    assert vars(flux)[attribute] is pre_apply
    assert "_teacache_handle" not in vars(flux)


@pytest.mark.parametrize(
    "error",
    [
        RestoreRefusedError(reason="generation-in-flight", remaining_steps=6, active_num_steps=10),
        RestoreRefusedError(reason="patched-over", attribute="_predict"),
    ],
    ids=["generation-in-flight", "patched-over"],
)
def test_refusal_round_trips_through_pickle(error: RestoreRefusedError) -> None:
    """Bug: the keyword-only constructor breaks unpickling in a worker, or the reason and attribute are lost."""
    error.add_note("raised in a worker")
    clone = pickle.loads(pickle.dumps(error))
    assert type(clone) is RestoreRefusedError
    assert str(clone) == str(error)
    assert (clone.reason, clone.attribute) == (error.reason, error.attribute)
    assert (clone.remaining_steps, clone.active_num_steps) == (error.remaining_steps, error.active_num_steps)
    assert clone.__notes__ == ["raised in a worker"]


@pytest.mark.parametrize("variant_id", REPRESENTATIVES)
def test_restore_retries_after_a_failed_teardown(variant_id: str) -> None:
    """The first restore() fails after its rollbacks put the pre-apply generate_image (and _predict) back as instance
    attributes. Bug: the retry sees the original in the slot, calls it a replacement and refuses forever."""
    flux = oracle._build(variant_id, cfg=False)
    original_transformer = flux.transformer
    pre_apply = {"generate_image": flux.generate_image}
    if variant_id in PREDICT_REPRESENTATIVES:
        pre_apply["_predict"] = flux._predict
    for name, value in pre_apply.items():
        setattr(flux, name, value)  # instance attributes before apply, as a caller's own wrapper would be
    handle = apply_teacache(flux, rel_l1_thresh=0.25, coefficients=oracle._CONST_COEFFICIENTS)
    attempts = 0

    def fail_once() -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient rollback failure")

    handle._patch.rollbacks.append(fail_once)
    with pytest.raises(RuntimeError, match="transient rollback failure"):
        handle.restore()
    handle.restore()
    assert {name: vars(flux).get(name) for name in pre_apply} == pre_apply
    assert flux.transformer is original_transformer
    assert "_teacache_handle" not in vars(flux)
