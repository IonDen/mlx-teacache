"""FLUX.1 img2img: when the lifecycle-set active step count is unavailable
(the defensive fallback path in flux1_forward_with_gate), skip-window validation
must fall back to the ACTIVE window (num_inference_steps - init_time_step), not
the nominal schedule. This pins the margin case where the two disagree, i.e. an
img2img run that starts late enough that the skip window is invalid against the
active count but looks fine against the nominal count (backlog 0048 #1).

Pure-core: skip-window validation runs before any `inner.*` call, so no real
transformer / mflux is needed — a sentinel inner proves validation fired first.
"""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_teacache.errors import InvalidStepWindowError
from mlx_teacache.variants.flux1_dev.integration import _InternalHandle, flux1_forward_with_gate


class _SentinelInner:
    """Any attribute access means the forward proceeded PAST window validation —
    which, for this img2img margin case, means it used the wrong (nominal) count
    and failed to reject the invalid window."""

    def __getattr__(self, name):
        raise AssertionError(
            f"forward reached inner.{name}: window validation did not fire, so it "
            "used the nominal schedule instead of the active img2img window"
        )


def _handle() -> _InternalHandle:
    return _InternalHandle(
        rel_l1_thresh=0.25,
        coefficients=(1.0, 0.0, 0.0, 0.0, 0.0),
        skip_first_n_steps=1,
        skip_last_n_steps=1,
    )


def test_img2img_fallback_validates_against_active_window() -> None:
    handle = _handle()
    # Defensive path: lifecycle did not set active_num_steps (stays None).
    assert handle._gen_ctx.active_num_steps is None
    # img2img: 10 nominal steps starting at step 8 -> only 2 ACTIVE denoising steps.
    config = SimpleNamespace(num_inference_steps=10, init_time_step=8)
    # skip_first + skip_last = 2 >= 2 active (invalid), but 2 < 10 nominal (looks fine).
    with pytest.raises(InvalidStepWindowError):
        flux1_forward_with_gate(
            _SentinelInner(),
            handle,
            t=8,
            config=config,
            hidden_states=mx.ones((1, 16, 64)),
            prompt_embeds=mx.ones((1, 8, 64)),
            pooled_prompt_embeds=mx.ones((1, 64)),
        )
