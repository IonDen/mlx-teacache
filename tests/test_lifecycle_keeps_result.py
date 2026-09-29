# tests/test_lifecycle_keeps_result.py
"""The generate_image wrapper must hand back the finished image even when TeaCache
cannot commit the generation's stats."""

import pytest

from mlx_teacache import apply_teacache
from tests._variant_fakes import make_flux1_fake

pytestmark = pytest.mark.mflux


class _RestoreAfterLoop:
    def __init__(self, handle):
        self._handle = handle

    def call_after_loop(self, *a, **k):
        self._handle.restore()


def test_restore_from_after_loop_callback_returns_the_image() -> None:
    """Bug caught: restore() freezes stats; the wrapper's finally raises StatsFrozenError and drops the result."""
    flux = make_flux1_fake()
    handle = apply_teacache(flux, rel_l1_thresh=0.2)
    flux.callbacks.after_loop.append(_RestoreAfterLoop(handle))
    image = flux.generate_image(seed=0, prompt="p", num_inference_steps=6)
    assert image == "image"
    assert handle.stats.generations == 0  # nothing committed after restore, by design


def test_second_generation_after_restore_runs_vanilla() -> None:
    """Bug caught: a leftover wrapper or callback still touches the frozen stats on the next run."""
    flux = make_flux1_fake()
    handle = apply_teacache(flux, rel_l1_thresh=0.2)
    flux.callbacks.after_loop.append(_RestoreAfterLoop(handle))
    flux.generate_image(seed=0, prompt="p", num_inference_steps=6)
    assert "generate_image" in vars(flux)  # the pre-patch instance attribute is back
    assert flux.generate_image.__name__ == "generate_image"  # not TeaCache's `wrapped`
    assert flux.generate_image(seed=1, prompt="p", num_inference_steps=6) == "image"
    assert handle.stats.generations == 0


class _RecordAfterLoop:
    def __init__(self) -> None:
        self.calls = 0

    def call_after_loop(self, *a, **k):
        self.calls += 1


def test_restore_from_after_loop_callback_keeps_later_callbacks_running() -> None:
    """Bug caught: restore() deletes TeaCache's callback from the after-loop list
    mflux is iterating, which shifts the list under the loop and skips the next
    user callback."""
    flux = make_flux1_fake()
    handle = apply_teacache(flux, rel_l1_thresh=0.2)
    later = _RecordAfterLoop()
    flux.callbacks.register(_RestoreAfterLoop(handle))
    flux.callbacks.register(later)
    image = flux.generate_image(seed=0, prompt="p", num_inference_steps=6)
    assert image == "image"
    assert later.calls == 1
    assert len(flux.callbacks.after_loop) == 2  # TeaCache's own callback is gone


def test_apply_again_after_restore_from_a_callback_registers_and_commits() -> None:
    """Bug caught: after restore() rebinds the callback lists, a second
    apply_teacache registers somewhere the wrapper's presence check or the loop
    does not see, so the next generation raises or commits nothing."""
    flux = make_flux1_fake()
    first = apply_teacache(flux, rel_l1_thresh=0.2)
    flux.callbacks.register(_RestoreAfterLoop(first))
    flux.generate_image(seed=0, prompt="p", num_inference_steps=6)
    flux.callbacks.after_loop.clear()  # drop the one-shot restorer
    second = apply_teacache(flux, rel_l1_thresh=0.2)
    assert flux.generate_image(seed=1, prompt="p", num_inference_steps=6) == "image"
    assert second.stats.generations == 1


def test_step_count_mismatch_warns_and_returns_the_image() -> None:
    """Bug caught: InternalStateError from finalize replaces the returned image."""
    flux = make_flux1_fake()
    handle = apply_teacache(flux, rel_l1_thresh=0.2)
    with pytest.warns(RuntimeWarning, match="expected 6 step decisions, got 7"):
        image = flux.generate_image(seed=0, prompt="p", num_inference_steps=6, _extra_transformer_calls=1)
    assert image == "image"
    assert handle.stats.generations == 0
