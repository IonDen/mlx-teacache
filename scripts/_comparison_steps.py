"""Per-step timing around the taef preview callback.

mflux calls ``ctx.in_loop(t, latents)`` BEFORE its own ``mx.eval(latents)`` (flux.py, flux2_klein.py, z_image.py,
qwen_image.py in 0.20), so a plain timestamp in a callback fires before step t's compute. The "pre" stamper
evaluates the latents first; registered before the preview and a "post" stamper after it, post[t] - pre[t] is the
preview cost and pre[t] - post[t-1] is step t's compute, in both conditions alike. mflux's registry keeps
registration order (callback_registry.py).
"""

import statistics
import time
from collections.abc import Callable, Sequence
from typing import Any


def _mx_eval(latents: Any) -> None:
    import mlx.core as mx

    mx.eval(latents)


class StepStamper:
    def __init__(
        self,
        *,
        phase: str,
        clock: Callable[[], float] = time.perf_counter,
        eval_fn: Callable[[Any], None] | None = None,
    ) -> None:
        if phase not in ("pre", "post"):
            raise ValueError(f"phase must be 'pre' or 'post', got {phase!r}")
        self.phase = phase
        self._clock = clock
        self._eval = (eval_fn or _mx_eval) if phase == "pre" else None
        self.stamps: list[float] = []

    def call_in_loop(
        self, t: int, seed: int, prompt: str, latents: Any, config: Any, time_steps: Any
    ) -> None:
        if self._eval is not None:
            self._eval(latents)
        self.stamps.append(self._clock())


def register_stamped_preview(
    register: Callable[[Any], None], preview: Any
) -> tuple[StepStamper, StepStamper]:
    pre, post = StepStamper(phase="pre"), StepStamper(phase="post")
    register(pre)
    register(preview)
    register(post)
    return pre, post


def split_steps(
    gen_start: float, pre: Sequence[float], post: Sequence[float]
) -> tuple[list[float], list[float]]:
    if len(pre) != len(post):
        raise ValueError(f"stamp length mismatch: pre={len(pre)} post={len(post)}")
    compute: list[float] = []
    preview: list[float] = []
    previous_end = gen_start
    for before, after in zip(pre, post, strict=True):
        if not (previous_end <= before <= after):
            raise ValueError(f"stamps out of order: {previous_end} -> {before} -> {after}")
        compute.append(before - previous_end)
        preview.append(after - before)
        previous_end = after
    return compute, preview


def decision_kinds(decisions: Sequence[Any], steps: int) -> list[str]:
    ordered = sorted(decisions, key=lambda d: d.step_idx)
    indices = [d.step_idx for d in ordered]
    if indices != list(range(steps)):
        raise ValueError(f"expected decisions for steps 0..{steps - 1}, got {indices}")
    return ["skipped" if d.decision == "skipped" else "computed" for d in ordered]


def steady_state_speedup(a_compute: Sequence[float], b_compute: Sequence[float]) -> float:
    if len(a_compute) != len(b_compute) or len(a_compute) < 2:
        raise ValueError("steady-state speedup needs two equal-length runs of at least two steps")
    return sum(a_compute[1:]) / sum(b_compute[1:])


def preview_subtracted_speedup(
    *, a_wall: float, a_preview: Sequence[float], b_wall: float, b_preview: Sequence[float]
) -> float:
    return (a_wall - sum(a_preview)) / (b_wall - sum(b_preview))


def medians_by_kind(compute: Sequence[float], kinds: Sequence[str]) -> dict[str, float | None]:
    if len(compute) != len(kinds):
        raise ValueError("compute and kinds differ in length")
    out: dict[str, float | None] = {}
    for kind in ("computed", "skipped"):
        values = [c for i, (c, k) in enumerate(zip(compute, kinds, strict=True)) if i > 0 and k == kind]
        out[kind] = statistics.median(values) if values else None
    return out
