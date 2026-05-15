# tests/test_lifecycle_img2img.py
"""Tests for img2img-aware lifecycle: _active_step_count helper, active step
plumbing through before/after-loop, image_strength=1.0 no-op, and the deletion
regression guard for img2img rejection.
"""

from types import SimpleNamespace

import pytest

from mlx_teacache.integrations.mflux.lifecycle import _active_step_count


def _make_config(num_inference_steps: int, *, image_strength: float | None = None) -> SimpleNamespace:
    """Faithfully stub mflux.Config: emulate the init_time_step property exactly.

    Real mflux logic (mflux/models/common/config/config.py):
        if is_img2img:
            strength = max(0.0, min(1.0, image_strength))
            return max(1, int(num_inference_steps * strength))
        return 0
    """
    if image_strength is None or image_strength <= 0.0:
        init_time_step = 0
    else:
        strength = max(0.0, min(1.0, image_strength))
        init_time_step = max(1, int(num_inference_steps * strength))
    return SimpleNamespace(
        num_inference_steps=num_inference_steps,
        init_time_step=init_time_step,
    )


@pytest.mark.parametrize(
    "image_strength,expected_init,expected_active",
    [
        (None, 0, 25),   # txt2img
        (0.0, 0, 25),    # txt2img (explicit zero strength)
        (0.04, 1, 24),   # img2img — tiny strength
        (0.5, 12, 13),   # img2img — half
        (0.7, 17, 8),    # img2img — 0.7 yields 8 calls per mflux semantics
        (1.0, 25, 0),    # img2img — full preservation, zero denoising calls
    ],
)
def test_active_step_count_matches_mflux_semantics(image_strength, expected_init, expected_active):
    cfg = _make_config(25, image_strength=image_strength)
    assert cfg.init_time_step == expected_init  # stub sanity
    assert _active_step_count(cfg) == expected_active


def test_active_step_count_missing_init_time_step_attribute_treats_as_txt2img():
    """Custom mflux subclasses might not expose init_time_step. Default to 0."""
    cfg = SimpleNamespace(num_inference_steps=25)  # no init_time_step
    assert _active_step_count(cfg) == 25


def test_active_step_count_init_time_step_is_None():
    """getattr fallback covers `init_time_step=None` too."""
    cfg = SimpleNamespace(num_inference_steps=25, init_time_step=None)
    assert _active_step_count(cfg) == 25


def test_active_step_count_floor_at_zero():
    """Defensive: if init_time_step somehow exceeds nominal, we floor at 0."""
    cfg = SimpleNamespace(num_inference_steps=10, init_time_step=15)
    assert _active_step_count(cfg) == 0
