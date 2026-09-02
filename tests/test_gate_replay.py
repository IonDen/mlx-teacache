"""Replay the shipped gate over a committed calibration trace and pin the skip
count and longest streak it produces at the shipped default threshold.

Pure-core: mlx.core plus the kernel; no weights, no mflux, milliseconds. The
trace is the per-step consecutive-delta rel_l1 series the polynomial was fit
on, so under consecutive-delta anchoring this replay reproduces the real-weights
bench exactly (v0.10.0 qwen bench: 33 skips, streak 4, same per-step pattern).

The bug this exists to catch: a change in what the gate measures. Anchoring
on the last *computed* step (the 0.9.x behaviour) replays to 27-28 skips with a
streak of 2 on the same trace, well outside the band below. That shift went
unnoticed for weeks because the only guard was a parity band wide enough to
admit both; this test would have gone red the day the anchoring changed.
"""

import json
from pathlib import Path

import mlx.core as mx

from mlx_teacache._kernel.cache import TeaCacheState
from mlx_teacache._kernel.gate import gate_step
from mlx_teacache.variants.qwen_image.config import COEFFICIENTS, DEFAULT_THRESH

_REPO_ROOT = Path(__file__).resolve().parent.parent
_QWEN_TRACE = _REPO_ROOT / "scripts" / "_calibration_qwen.json"
_NUM_STEPS = 50
_NUM_FIT_PROMPTS = 7

# Measured by scripts/bench_speedup.py --variant qwen --three-way --reps 3 on
# 2026-09-01, identical in all three reps (_artifacts/v0.10.0_bench_qwen_image.json).
_BENCH_PATTERN = "CCCSCSSCSSCSSSCSSSSCSSSSCSSSCSSSCSSSCSSCSSCSSCSCSC"


def _prompt_traces() -> list[list[float]]:
    xs = json.loads(_QWEN_TRACE.read_text())["signals"]["A"]["x_values"]
    per = len(xs) // _NUM_FIT_PROMPTS
    return [xs[i * per : (i + 1) * per] for i in range(_NUM_FIT_PROMPTS)]


def _replay(trace: list[float], thresh: float) -> tuple[str, int]:
    """Drive the real gate with a scalar series whose consecutive rel_l1 equals
    the trace, simulating the integration's residual write on every cached step."""
    state = TeaCacheState()
    mod = mx.array([1.0])
    pattern: list[str] = []
    streak = best = 0
    for step in range(_NUM_STEPS):
        if step > 0:
            mod = mod * (1.0 + trace[step - 1])
        dec = gate_step(
            state,
            rel_l1_thresh=thresh,
            coefficients=COEFFICIENTS,
            skip_first=1,
            skip_last=1,
            num_steps=_NUM_STEPS,
            step_idx=step,
            mod_in=mod,
        )
        if dec.should_update_cache:
            state.cached_residual = mx.zeros((1,))
        skipped = not dec.should_compute
        pattern.append("S" if skipped else "C")
        streak = streak + 1 if skipped else 0
        best = max(best, streak)
    return "".join(pattern), best


def test_qwen_replay_at_default_matches_the_measured_bench():
    """RED if the gate's anchoring or accumulation semantics change: 0.9.x
    anchoring gives 27-28 skips / streak 2 on this trace; the shipped gate gives
    32-33 / 3-4 and reproduces the bench's exact pattern on several prompts."""
    results = [_replay(t, DEFAULT_THRESH) for t in _prompt_traces()]
    for pattern, streak in results:
        skips = pattern.count("S")
        assert 32 <= skips <= 33, f"skips {skips} outside the measured 32-33 band: {pattern}"
        assert 3 <= streak <= 4, f"streak {streak} outside the measured 3-4 band: {pattern}"
    assert any(p == _BENCH_PATTERN for p, _ in results), (
        "no calibration prompt reproduces the bench's measured per-step pattern"
    )


def test_qwen_replay_streak_stays_under_the_cap_at_default():
    """RED if a coefficient or threshold change pushes the default operating
    point onto the runaway cap, which the docs say never engages at a default."""
    from mlx_teacache._kernel.gate import MAX_CONSECUTIVE_SKIPS

    worst = max(streak for _, streak in (_replay(t, DEFAULT_THRESH) for t in _prompt_traces()))
    assert worst < MAX_CONSECUTIVE_SKIPS, f"replayed streak {worst} reached the cap"
