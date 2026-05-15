# tests/test_distilled_warning.py
"""TeaCacheNoBenefitWarning semantics.

The warning fires once per handle when possible_skips == 0 — i.e., when the
current configuration produces no opportunity to skip any step. Specifically
suppressed for FLUX.2 all-CFG (different no-benefit mode, tracked separately)
and when the configuration is about to raise InvalidStepWindowError.
"""

import warnings
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from mlx_teacache import TeaCacheNoBenefitWarning
from mlx_teacache.cache import TeaCacheState
from mlx_teacache.integrations.mflux.lifecycle import (
    GenerationContext,
    GenerationContextCallback,
)
from mlx_teacache.stats import TeaCacheStats


@dataclass
class _FakeHandleState:
    cache: TeaCacheState = field(default_factory=TeaCacheState)
    stats: TeaCacheStats = field(default_factory=TeaCacheStats)
    no_benefit_warned: bool = False


@dataclass
class _FakeHandle:
    variant_id: str = "flux1-schnell"
    skip_first_n_steps: int = 1
    skip_last_n_steps: int = 1
    _gen_ctx: GenerationContext = field(default_factory=GenerationContext)
    _state: _FakeHandleState = field(default_factory=_FakeHandleState)


def _config(num_inference_steps: int, *, image_strength: float | None = None,
            guidance: float = 1.0) -> SimpleNamespace:
    if image_strength is None or image_strength <= 0.0:
        init_time_step = 0
    else:
        init_time_step = max(1, int(num_inference_steps * image_strength))
    return SimpleNamespace(
        num_inference_steps=num_inference_steps,
        init_time_step=init_time_step,
        image_strength=image_strength,
        guidance=guidance,
    )


# --- Fires / doesn't fire ---


def test_does_not_fire_at_schnell_default_eligible_is_two():
    """FLUX.1 schnell default: 4 steps, skip_first=1, skip_last=1
    → eligible=2 → possible_skips=1. Should NOT fire."""
    handle = _FakeHandle()
    cb = GenerationContextCallback(handle)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cb.call_before_loop(seed=1, prompt="hi", latents=None,
                            config=_config(4))
    matched = [w for w in caught if issubclass(w.category, TeaCacheNoBenefitWarning)]
    assert matched == [], f"unexpected warning at eligible=2: {[str(w.message) for w in matched]}"


def test_fires_when_eligible_is_one():
    """active=3, skip_first=1, skip_last=1 → eligible=1 → possible_skips=0.
    Should fire."""
    handle = _FakeHandle(skip_first_n_steps=1, skip_last_n_steps=1)
    cb = GenerationContextCallback(handle)
    with pytest.warns(TeaCacheNoBenefitWarning, match=r"eligible"):
        cb.call_before_loop(seed=1, prompt="hi", latents=None,
                            config=_config(3))


def test_does_not_fire_when_eligible_is_two():
    """eligible=2 → possible_skips=1. Caching can engage technically; no warning."""
    handle = _FakeHandle(skip_first_n_steps=1, skip_last_n_steps=1)
    cb = GenerationContextCallback(handle)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cb.call_before_loop(seed=1, prompt="hi", latents=None,
                            config=_config(4))
    matched = [w for w in caught if issubclass(w.category, TeaCacheNoBenefitWarning)]
    assert matched == []


def test_does_not_fire_when_window_is_invalid():
    """skip_first + skip_last >= active_num_steps → InvalidStepWindowError
    will fire elsewhere. Don't add a duplicate warning."""
    handle = _FakeHandle(skip_first_n_steps=3, skip_last_n_steps=3)
    cb = GenerationContextCallback(handle)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cb.call_before_loop(seed=1, prompt="hi", latents=None,
                            config=_config(4))
    matched = [w for w in caught if issubclass(w.category, TeaCacheNoBenefitWarning)]
    assert matched == [], "no-benefit warning should be suppressed when window is invalid"


def test_does_not_fire_under_flux2_all_cfg():
    """FLUX.2 with guidance > 1.0 routes every step through the CFG-fallback
    path; the schedule-shape warning is out of scope for that case."""
    handle = _FakeHandle(variant_id="flux2-klein-4b")
    cb = GenerationContextCallback(handle)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # Eligible would be 1 (would normally fire), but variant + guidance trigger suppression.
        cb.call_before_loop(seed=1, prompt="hi", latents=None,
                            config=_config(3, guidance=3.5))
    matched = [w for w in caught if issubclass(w.category, TeaCacheNoBenefitWarning)]
    assert matched == []


def test_does_not_fire_for_active_zero():
    """image_strength=1.0 yields active=0; valid no-op generation, no warning."""
    handle = _FakeHandle()
    cb = GenerationContextCallback(handle)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cb.call_before_loop(seed=1, prompt="hi", latents=None,
                            config=_config(25, image_strength=1.0))
    matched = [w for w in caught if issubclass(w.category, TeaCacheNoBenefitWarning)]
    assert matched == []


# --- Cadence ---


def test_fires_once_per_handle():
    """Second call under the same handle stays quiet even when the
    config still has no possible skips."""
    handle = _FakeHandle(skip_first_n_steps=1, skip_last_n_steps=1)
    cb = GenerationContextCallback(handle)
    bad_cfg = _config(3)  # eligible=1 → possible_skips=0

    # First call: warning fires.
    with pytest.warns(TeaCacheNoBenefitWarning):
        cb.call_before_loop(seed=1, prompt="hi", latents=None, config=bad_cfg)
    assert handle._state.no_benefit_warned is True

    # Second call: no warning.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cb.call_before_loop(seed=1, prompt="hi", latents=None, config=bad_cfg)
    matched = [w for w in caught if issubclass(w.category, TeaCacheNoBenefitWarning)]
    assert matched == []


def test_fires_again_after_new_handle():
    """A fresh handle has no_benefit_warned=False and emits again."""
    bad_cfg = _config(3)

    # First handle.
    h1 = _FakeHandle(skip_first_n_steps=1, skip_last_n_steps=1)
    cb1 = GenerationContextCallback(h1)
    with pytest.warns(TeaCacheNoBenefitWarning):
        cb1.call_before_loop(seed=1, prompt="hi", latents=None, config=bad_cfg)

    # Fresh handle.
    h2 = _FakeHandle(skip_first_n_steps=1, skip_last_n_steps=1)
    cb2 = GenerationContextCallback(h2)
    with pytest.warns(TeaCacheNoBenefitWarning):
        cb2.call_before_loop(seed=1, prompt="hi", latents=None, config=bad_cfg)


# --- Suppression ---


def test_filterwarnings_ignore_suppresses_emission():
    """Standard warnings.filterwarnings('ignore', category=...) silences the warning."""
    handle = _FakeHandle(skip_first_n_steps=1, skip_last_n_steps=1)
    cb = GenerationContextCallback(handle)
    bad_cfg = _config(3)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=TeaCacheNoBenefitWarning)
        # Should produce zero warnings:
        cb.call_before_loop(seed=1, prompt="hi", latents=None, config=bad_cfg)
    # Flag still flips even when filter suppresses display.
    assert handle._state.no_benefit_warned is True


# --- Message content ---


def test_message_contains_numeric_values():
    handle = _FakeHandle(skip_first_n_steps=1, skip_last_n_steps=1)
    cb = GenerationContextCallback(handle)
    bad_cfg = _config(3)

    with pytest.warns(TeaCacheNoBenefitWarning) as caught:
        cb.call_before_loop(seed=1, prompt="hi", latents=None, config=bad_cfg)
    message = str(caught[0].message)
    assert "active_num_steps=3" in message
    assert "skip_first_n_steps=1" in message
    assert "skip_last_n_steps=1" in message
    assert "possible skip" in message.lower()
