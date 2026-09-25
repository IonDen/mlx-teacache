"""Step stamps around the taef preview, reproducing mflux's in_loop-before-eval order (pure-core lane)."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import _comparison_steps as cs  # noqa: E402


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _LazyLatents:
    """An unevaluated mx.array: evaluating it runs the step's compute (advances the clock)."""

    def __init__(self, clock: _Clock, cost: float) -> None:
        self.clock, self.cost, self.done = clock, cost, False

    def evaluate(self) -> None:
        if not self.done:
            self.clock.now += self.cost
            self.done = True


def _loop(callbacks: list, clock: _Clock, compute: list[float]) -> None:
    for t, cost in enumerate(compute):
        latents = _LazyLatents(clock, cost)  # mflux: the scheduler step builds a lazy graph
        for cb in callbacks:  # mflux: ctx.in_loop(t, latents) BEFORE its own mx.eval(latents)
            cb.call_in_loop(t=t, seed=42, prompt="p", latents=latents, config=None, time_steps=None)
        latents.evaluate()


def _preview(clock: _Clock, preview: list[float]) -> SimpleNamespace:
    def call_in_loop(t, seed, prompt, latents, config, time_steps) -> None:  # noqa: ANN001
        latents.evaluate()  # decoding forces the latents, like LivePreviewCallback
        clock.now += preview[t]

    return SimpleNamespace(call_in_loop=call_in_loop)


def test_compute_and_preview_are_attributed_to_their_own_step() -> None:
    """Bug: the pre stamper does not evaluate the latents, so step t's compute lands in the preview gap."""
    clock = _Clock()
    compute, preview = [5.0, 3.0, 1.0, 3.0], [0.2, 0.3, 0.2, 0.25]
    registered: list = []
    pre, post = cs.register_stamped_preview(
        registered.append, _preview(clock, preview), clock=clock, eval_fn=lambda lat: lat.evaluate()
    )
    _loop(registered, clock, compute)
    got_compute, got_preview = cs.split_steps(0.0, pre.stamps, post.stamps)
    assert got_compute == pytest.approx(compute)
    assert got_preview == pytest.approx(preview)


def test_register_stamped_preview_orders_pre_preview_post() -> None:
    """Bug: the stampers are registered on the wrong side of the preview."""
    preview = SimpleNamespace(call_in_loop=lambda **kw: None)
    registered: list = []
    pre, post = cs.register_stamped_preview(registered.append, preview)
    assert registered == [pre, preview, post] and pre.phase == "pre" and post.phase == "post"


def test_default_register_stamped_preview_evaluates_only_before_the_preview(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bug: the pre stamper stops evaluating the latents (e.g. its eval_fn wiring regresses to None), which
    would silently push step t's compute cost into the preview gap without any test catching it. Spies on
    the module-level ``_mx_eval`` default rather than reaching into the stampers' private ``_eval``
    attribute, so this keeps working if the implementation stops storing a bound method there."""
    calls: list[object] = []
    monkeypatch.setattr(cs, "_mx_eval", calls.append)
    preview = SimpleNamespace(call_in_loop=lambda **kw: None)
    pre, post = cs.register_stamped_preview(lambda cb: None, preview)

    latents = object()
    pre.call_in_loop(t=0, seed=42, prompt="p", latents=latents, config=None, time_steps=None)
    post.call_in_loop(t=0, seed=42, prompt="p", latents=latents, config=None, time_steps=None)

    assert calls == [latents]  # pre evaluated the latents exactly once; post never touched _mx_eval


def test_split_steps_rejects_length_mismatch_and_time_going_backwards() -> None:
    """Bug: a lost stamp or clock skew yields plausible but wrong per-step numbers."""
    with pytest.raises(ValueError, match="length"):
        cs.split_steps(0.0, [1.0, 2.0], [1.5])
    with pytest.raises(ValueError, match="order"):
        cs.split_steps(0.0, [1.0, 0.5], [1.2, 0.6])


def _d(i: int, kind: str) -> SimpleNamespace:
    return SimpleNamespace(step_idx=i, decision=kind)


def test_decision_kinds_sorts_by_step_idx_and_maps_every_non_skip_to_computed() -> None:
    """Bug: decisions taken in arrival order, or a forced/numerical-miss step shown as skipped."""
    decisions = [_d(2, "skipped"), _d(0, "forced"), _d(1, "numerical-miss")]
    assert cs.decision_kinds(decisions, 3) == ["computed", "computed", "skipped"]


def test_decision_kinds_rejects_gaps_and_wrong_counts() -> None:
    """Bug: a missing step shifts every sheet label after it."""
    with pytest.raises(ValueError):
        cs.decision_kinds([_d(0, "computed"), _d(2, "computed")], 2)
    with pytest.raises(ValueError):
        cs.decision_kinds([_d(0, "computed")], 2)


def test_steady_state_speedup_excludes_step_zero() -> None:
    """Bug: A's compile trace in step 0 is counted as TeaCache speedup."""
    assert cs.steady_state_speedup([35.0, 4.0, 4.0], [5.0, 4.0, 2.0]) == pytest.approx(8.0 / 6.0)


def test_preview_subtracted_speedup() -> None:
    """Bug: preview time subtracted from one side only."""
    got = cs.preview_subtracted_speedup(a_wall=110.0, a_preview=[5.0, 5.0], b_wall=90.0, b_preview=[5.0, 5.0])
    assert got == pytest.approx(100.0 / 80.0)


def test_medians_by_kind_excludes_step_zero_and_reports_none_without_skips() -> None:
    """Bug: step 0 (compile trace) pollutes the computed median, or 'no skips' becomes 0.0."""
    compute = [9.0, 4.0, 1.0, 5.0, 1.2]
    kinds = ["computed", "computed", "skipped", "computed", "skipped"]
    assert cs.medians_by_kind(compute, kinds) == {"computed": 4.5, "skipped": pytest.approx(1.1)}
    assert cs.medians_by_kind([9.0, 4.0], ["computed", "computed"]) == {"computed": 4.0, "skipped": None}
