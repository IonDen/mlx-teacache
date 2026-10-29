"""A TeaCache-patched pipeline refuses a generate_image call that also turns on mflux's own step cache.

Both skip denoising steps. Run together, TeaCache's step count falls behind the schedule, its last-step guard never
fires and the run's stats are dropped. On mflux 0.23.0 and 0.24.0 the step cache exists only on models mlx-teacache
does not patch (Qwen-Image 2.1, and on 0.24.0 also FIBO 2 and MOSS-Image), so these tests use FLUX.1 fakes whose
generate_image takes the step-cache arguments (as named parameters, or through **kwargs) or does not take them at all.
"""

import importlib
import inspect
import pickle
import sys
from typing import Any

import pytest

from mlx_teacache import apply_teacache
from mlx_teacache.errors import IncompatibleModelError
from tests._variant_fakes import make_flux1_fake

pytestmark = pytest.mark.mflux

_STEPS = 6
_STEP_CACHE_NAMES = frozenset({"step_cache_ratio", "teacache_ratio", "use_step_cache"})


def _recording_fake() -> tuple[Any, list[dict[str, Any]]]:
    """A FLUX.1 fake whose generate_image records each entry (its keyword arguments) before running."""
    flux = make_flux1_fake()
    inner = flux.generate_image
    entries: list[dict[str, Any]] = []

    def generate_image(*args: Any, **kwargs: Any) -> Any:
        entries.append(dict(kwargs))
        return inner(*args, **kwargs)

    flux.generate_image = generate_image
    return flux, entries


def _named_params_fake(default_ratio: float | None = None) -> tuple[Any, list[dict[str, Any]]]:
    """A FLUX.1 fake whose generate_image names the step-cache parameters and has no **kwargs, the shape of mflux's
    Qwen-Image 2.1 (`step_cache_ratio` is a positional slot there). `default_ratio` is step_cache_ratio's default."""
    flux = make_flux1_fake()
    inner = flux.generate_image
    entries: list[dict[str, Any]] = []

    def generate_image(
        seed: int = 0,
        prompt: str = "",
        num_inference_steps: int = 4,
        step_cache_ratio: float | None = default_ratio,
        teacache_ratio: float | None = None,
        use_step_cache: bool = False,
    ) -> Any:
        entries.append({"seed": seed, "step_cache_ratio": step_cache_ratio, "use_step_cache": use_step_cache})
        return inner(seed=seed, prompt=prompt, num_inference_steps=num_inference_steps)

    flux.generate_image = generate_image
    return flux, entries


def _no_step_cache_fake() -> Any:
    """A FLUX.1 fake whose generate_image takes neither the step-cache parameters nor **kwargs, the shape of every
    generate_image mlx-teacache patches on mflux 0.23.0 and 0.24.0."""
    flux = make_flux1_fake()
    inner = flux.generate_image

    def generate_image(seed: int = 0, prompt: str = "", num_inference_steps: int = 4) -> Any:
        return inner(seed=seed, prompt=prompt, num_inference_steps=num_inference_steps)

    flux.generate_image = generate_image
    return flux


@pytest.mark.parametrize(
    ("name", "value"),
    [("step_cache_ratio", 0.5), ("teacache_ratio", 0.5), ("use_step_cache", True)],
)
def test_a_step_cache_argument_is_refused(name: str, value: object) -> None:
    """Bug caught: the wrapper does not check this argument, so mflux's step cache and TeaCache both skip steps."""
    flux, _ = _recording_fake()
    apply_teacache(flux, rel_l1_thresh=0.2)
    with pytest.raises(IncompatibleModelError, match=name):
        flux.generate_image(seed=0, prompt="p", num_inference_steps=_STEPS, **{name: value})


@pytest.mark.parametrize(
    ("name", "value"),
    [("step_cache_ratio", 0.5), ("teacache_ratio", 0.5), ("use_step_cache", True)],
)
def test_a_refused_call_never_enters_the_original_generate_image(name: str, value: object) -> None:
    """Bug caught: the check runs after original() (or inside a callback), so the generation starts before the
    refusal."""
    flux, entries = _recording_fake()
    apply_teacache(flux, rel_l1_thresh=0.2)
    with pytest.raises(IncompatibleModelError):
        flux.generate_image(seed=0, prompt="p", num_inference_steps=_STEPS, **{name: value})
    assert entries == []


def test_the_refusal_message_says_both_skip_steps_and_how_to_pick_one() -> None:
    """Bug caught: the error carries IncompatibleModelError's default text, which tells the user of a supported
    model that the model is unsupported and gives no way out."""
    flux, _ = _recording_fake()
    apply_teacache(flux, rel_l1_thresh=0.2)
    with pytest.raises(IncompatibleModelError) as excinfo:
        flux.generate_image(seed=0, prompt="p", num_inference_steps=_STEPS, step_cache_ratio=0.5)
    message = str(excinfo.value)
    assert "Unsupported model" not in message
    assert "step_cache_ratio=0.5" in message
    assert "both skip steps" in message
    assert "handle.restore()" in message


def _refusal() -> IncompatibleModelError:
    flux, _ = _recording_fake()
    apply_teacache(flux, rel_l1_thresh=0.2)
    with pytest.raises(IncompatibleModelError) as excinfo:
        flux.generate_image(seed=0, prompt="p", num_inference_steps=_STEPS, step_cache_ratio=0.5)
    return excinfo.value


def test_a_pickled_refusal_keeps_its_type_message_and_requested_argument() -> None:
    """Bug caught: the error is rebuilt from IncompatibleModelError's keywords alone, so a pickled copy (raised in a
    worker process) reverts to "Unsupported model" and loses the step-cache argument."""
    err = _refusal()
    clone = pickle.loads(pickle.dumps(err))
    assert type(clone) is type(err)
    assert isinstance(clone, IncompatibleModelError)
    assert str(clone) == str(err)
    assert "Unsupported model" not in str(clone)
    assert clone.requested == "step_cache_ratio=0.5"  # type: ignore[attr-defined]
    assert (clone.actual_type, clone.actual_model_name) == ("Flux1", "black-forest-labs/FLUX.1-dev")


@pytest.mark.parametrize(
    ("name", "value"),
    [("step_cache_ratio", None), ("teacache_ratio", None), ("use_step_cache", False)],
)
def test_an_off_step_cache_argument_runs_normally(name: str, value: object) -> None:
    """Bug caught: the check tests presence (`name in kwargs`) or `is not None` for use_step_cache, so an explicit
    off value is refused."""
    flux, entries = _recording_fake()
    handle = apply_teacache(flux, rel_l1_thresh=0.2)
    image = flux.generate_image(seed=0, prompt="p", num_inference_steps=_STEPS, **{name: value})
    assert image == "image"
    assert entries == [{"seed": 0, "prompt": "p", "num_inference_steps": _STEPS, name: value}]
    assert handle.stats.generations == 1


def test_a_normal_call_after_a_refusal_runs_and_commits_its_stats() -> None:
    """Bug caught: the refusal leaves the handle disabled or with stale per-run state, so the next plain call is
    refused, raises, or commits nothing."""
    flux, entries = _recording_fake()
    handle = apply_teacache(flux, rel_l1_thresh=0.2)
    with pytest.raises(IncompatibleModelError):
        flux.generate_image(seed=0, prompt="p", num_inference_steps=_STEPS, step_cache_ratio=0.5)
    assert handle.stats.generations == 0
    assert flux.generate_image(seed=1, prompt="p", num_inference_steps=_STEPS) == "image"
    assert len(entries) == 1
    assert handle.stats.generations == 1
    last = handle.stats.last_generation
    assert last is not None
    assert len(last.decisions) == _STEPS


def test_restore_after_a_refusal_passes_the_step_cache_argument_to_the_unpatched_generate_image() -> None:
    """Bug caught: the refusal is stored on the model rather than checked in the wrapper, so it survives
    handle.restore() and the unpatched generate_image still refuses the argument. The fake ignores the argument,
    so this proves only that the call reaches mflux's own method with the argument intact."""
    flux, entries = _recording_fake()
    handle = apply_teacache(flux, rel_l1_thresh=0.2)
    with pytest.raises(IncompatibleModelError):
        flux.generate_image(seed=0, prompt="p", num_inference_steps=_STEPS, step_cache_ratio=0.5)
    handle.restore()
    image = flux.generate_image(seed=0, prompt="p", num_inference_steps=_STEPS, step_cache_ratio=0.5)
    assert image == "image"
    assert entries == [{"seed": 0, "prompt": "p", "num_inference_steps": _STEPS, "step_cache_ratio": 0.5}]


def test_both_ratios_are_named_in_the_message() -> None:
    """Bug caught: the description stops at the first argument found, so a call passing both the ratio and its
    alias names only one of them."""
    flux, _ = _recording_fake()
    apply_teacache(flux, rel_l1_thresh=0.2)
    with pytest.raises(IncompatibleModelError) as excinfo:
        flux.generate_image(
            seed=0, prompt="p", num_inference_steps=_STEPS, step_cache_ratio=0.5, teacache_ratio=0.25
        )
    assert "(step_cache_ratio=0.5, teacache_ratio=0.25)" in str(excinfo.value)


def test_the_refusal_names_the_patched_variant_in_supported() -> None:
    """Bug caught: `supported` is read from the internal handle, which has no variant id, so it is always []."""
    assert _refusal().supported == ["flux1-dev"]


def test_a_positionally_passed_step_cache_ratio_is_refused() -> None:
    """Bug caught: only keyword names are checked, so `step_cache_ratio` in its positional slot (the 11th on
    mflux's Qwen-Image 2.1) turns the step cache on under TeaCache."""
    flux, entries = _named_params_fake()
    apply_teacache(flux, rel_l1_thresh=0.2)
    with pytest.raises(IncompatibleModelError, match=r"step_cache_ratio=0\.5"):
        flux.generate_image(0, "p", _STEPS, 0.5)
    assert entries == []


def test_a_named_parameter_left_off_runs_normally() -> None:
    """Bug caught: the bound check tests that the parameter exists, or reads use_step_cache's default False as
    on, so every call on a model that has the step cache is refused."""
    flux, entries = _named_params_fake()
    handle = apply_teacache(flux, rel_l1_thresh=0.2)
    assert flux.generate_image(0, "p", _STEPS) == "image"
    assert entries == [{"seed": 0, "step_cache_ratio": None, "use_step_cache": False}]
    assert handle.stats.generations == 1


def test_a_step_cache_that_is_on_by_default_is_refused_with_how_to_turn_it_off() -> None:
    """Bug caught: defaults are not applied before the check, so an mflux whose step cache is on by default runs it
    under TeaCache; or the message only says to drop an argument the caller never passed."""
    flux, entries = _named_params_fake(default_ratio=0.25)
    apply_teacache(flux, rel_l1_thresh=0.2)
    with pytest.raises(IncompatibleModelError) as excinfo:
        flux.generate_image(seed=0, prompt="p", num_inference_steps=_STEPS)
    message = str(excinfo.value)
    assert "(step_cache_ratio=0.25 by default)" in message
    assert "None for the ratio" in message
    assert "False for use_step_cache" in message
    assert entries == []


def test_a_model_without_the_step_cache_parameters_raises_its_own_type_error() -> None:
    """Bug caught: the guard refuses a keyword the patched generate_image does not take, replacing mflux's
    TypeError with advice to restore() for a step cache that model does not have (every patched model on mflux
    0.23.0 and 0.24.0)."""
    flux = _no_step_cache_fake()
    apply_teacache(flux, rel_l1_thresh=0.2)
    with pytest.raises(TypeError, match="unexpected keyword argument 'step_cache_ratio'"):
        flux.generate_image(seed=0, prompt="p", num_inference_steps=_STEPS, step_cache_ratio=0.5)


def test_an_unreadable_signature_falls_back_to_the_keyword_check() -> None:
    """Bug caught: a generate_image whose signature cannot be read makes apply_teacache fail, or turns the guard
    off, instead of checking the keyword names."""
    flux, entries = _recording_fake()
    flux.generate_image.__signature__ = "not a signature"  # inspect.signature refuses it
    with pytest.raises((TypeError, ValueError)):
        inspect.signature(flux.generate_image)
    apply_teacache(flux, rel_l1_thresh=0.2)
    with pytest.raises(IncompatibleModelError, match=r"step_cache_ratio=0\.5"):
        flux.generate_image(seed=0, prompt="p", num_inference_steps=_STEPS, step_cache_ratio=0.5)
    assert entries == []


def test_the_refusal_runs_the_parent_constructor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bug caught: the subclass skips IncompatibleModelError.__init__, so an attribute the parent sets there (now or
    in a later release) is missing on the refusal."""
    parent_init = IncompatibleModelError.__init__

    def init_with_a_new_attribute(self: IncompatibleModelError, **kwargs: Any) -> None:
        parent_init(self, **kwargs)
        self.added_later = "set by the parent"  # type: ignore[attr-defined]

    monkeypatch.setattr(IncompatibleModelError, "__init__", init_with_a_new_attribute)
    err = _refusal()
    assert err.added_later == "set by the parent"  # type: ignore[attr-defined]
    assert str(err).startswith("This call turns on mflux's step cache (step_cache_ratio=0.5).")


# --- the premise: no pipeline class mlx-teacache patches takes a step-cache argument ---------------------------

# Import path of each mflux pipeline class a registered variant may patch (the union of the variants' _ALLOWED).
_PATCHED_CLASSES = {
    "Flux1": "mflux.models.flux.variants.txt2img.flux",
    "Flux2Klein": "mflux.models.flux2.variants.txt2img.flux2_klein",
    "QwenImage": "mflux.models.qwen.variants.txt2img.qwen_image",
    "ZImage": "mflux.models.z_image.variants.z_image",
}


def _step_cache_parameters(cls: type) -> set[str]:
    return set(inspect.signature(cls.generate_image).parameters) & _STEP_CACHE_NAMES  # type: ignore[attr-defined]


def _allowed_pipeline_class_names() -> set[str]:
    from mlx_teacache.variants import _REGISTRY

    names: set[str] = set()
    for entry in _REGISTRY.values():
        names |= sys.modules[entry["matches_config"].__module__]._ALLOWED
    return names


def test_the_class_table_covers_every_class_a_variant_patches() -> None:
    """Bug caught: a variant starts patching a new pipeline class and the premise check below never looks at it."""
    assert _allowed_pipeline_class_names() == set(_PATCHED_CLASSES)


@pytest.mark.parametrize("class_name", sorted(_PATCHED_CLASSES))
def test_no_patched_pipeline_takes_a_step_cache_argument(class_name: str) -> None:
    """A failure means mflux added its step cache to a model mlx-teacache patches. The wrapper then refuses those
    arguments (it is dormant while no patched generate_image takes them); decide whether to keep refusing or to
    integrate with mflux's step cache before the release that supports that mflux."""
    cls = getattr(importlib.import_module(_PATCHED_CLASSES[class_name]), class_name)
    assert _step_cache_parameters(cls) == set()


def test_the_premise_check_sees_qwen_image_21s_step_cache() -> None:
    """Bug caught: the premise check reads the wrong signature or misspells the names, so it passes on a class that
    does take the step cache. Qwen-Image 2.1, which mlx-teacache does not patch, is the one that does."""
    txt2img = pytest.importorskip("mflux.models.qwen21.variants.txt2img.qwen_image_21")
    edit = pytest.importorskip("mflux.models.qwen21.variants.edit.qwen_image_21_edit")
    assert _step_cache_parameters(txt2img.QwenImage21) == {"step_cache_ratio", "teacache_ratio"}
    assert _step_cache_parameters(edit.QwenImage21Edit) == {"use_step_cache"}
