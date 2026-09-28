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


# --- FLUX.2 [klein] base 4B: a real out-of-domain trace (backlog 0094) ------------------------------------------

# Consecutive-delta rel_l1 the gate measured on steps 2..48 of a real run: klein-base-4b, q8, 864x1152, 50 steps,
# guidance 4.0, seed 42, the comparison-page tennis prompt (bench_comparison.py --quality-probe K1t08trace, mflux
# 0.18.0 / MLX 0.31.2, 2026-09-27). Steps 2, 6 and 7 sit far below the calibrated minimum 0.0283.
_BASE_4B_REAL_DELTAS = [
    0.0034982485977523574,
    0.06031291097290305,
    0.04270178358726343,
    0.02419897612563297,
    0.00669994565553993,
    0.0032478764796644503,
    0.015051099722946043,
    0.023963179965826226,
    0.029155384668014244,
    0.038466841092107254,
    0.04376499338587171,
    0.04652267773939855,
    0.049595660850532755,
    0.10049775968864925,
    0.05289975709043216,
    0.0521894678745624,
    0.09697673595912798,
    0.04949576098500734,
    0.09101430066383265,
    0.045950293425028925,
    0.08390853471788967,
    0.07791773861376637,
    0.07738864782660876,
    0.0733772354979847,
    0.06604350056871569,
    0.06285653430698007,
    0.08294085647927421,
    0.05542977419405649,
    0.07511500609249892,
    0.06537854039177436,
    0.061450448518569095,
    0.07660473330631054,
    0.06623462874832724,
    0.0633270935018087,
    0.054946259346157954,
    0.06863546639178497,
    0.06854170170548804,
    0.06488383455197172,
    0.07045037291365668,
    0.06127275030413559,
    0.08193544935391277,
    0.08257663275756147,
    0.08088001386431126,
    0.07672106670433294,
    0.10627643776033534,
    0.0803376596747848,
    0.09093298632567949,
]
# Measured on hardware with the pre-0094 gate: at 0.08 (K1t08, and again K1t08trace) and at the 0.17 default (K1).
_BASE_4B_MEASURED_AT_008 = "CCSCCCSSCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"
_BASE_4B_MEASURED_AT_017 = "CCSCSCSSCSCSCCCSCCSCSCCCCCCCCCCCCCCCCCCCCCCCCCSCSC"


def _replay_base_4b(thresh: float, calibrated_range: tuple[float, float] | None) -> str:
    from mlx_teacache.variants.flux2_klein_base_4b.config import COEFFICIENTS as BASE_4B

    state = TeaCacheState()
    mod = mx.array([1.0])
    pattern = []
    # Step 1 only seeds and step 49 is forced (neither delta is measured), so any value stands in for them.
    deltas = [0.1, *_BASE_4B_REAL_DELTAS, 0.1]
    for step in range(_NUM_STEPS):
        if step > 0:
            mod = mod * (1.0 + deltas[step - 1])
        dec = gate_step(
            state,
            rel_l1_thresh=thresh,
            coefficients=BASE_4B,
            skip_first=1,
            skip_last=1,
            num_steps=_NUM_STEPS,
            step_idx=step,
            mod_in=mod,
            calibrated_range=calibrated_range,
        )
        if dec.should_update_cache:
            state.cached_residual = mx.zeros((1,))
        pattern.append("C" if dec.should_compute else "S")
    return "".join(pattern)


def test_base_4b_replay_without_the_range_reproduces_both_measured_runs():
    """Fidelity of the replay on this trace. RED if the gate's accumulation changes, or if the replay stops
    being a faithful stand-in for the real forward (then the clamped pins below prove nothing)."""
    assert _replay_base_4b(0.08, None) == _BASE_4B_MEASURED_AT_008
    assert _replay_base_4b(0.17, None) == _BASE_4B_MEASURED_AT_017


def test_base_4b_calibrated_range_stops_the_threshold_independent_skips():
    """RED on the 0094 bug: without the variant's calibrated range the 0.0035 / 0.0067 / 0.0033 deltas are
    skipped at 0.08 and 0.12 alike. With it, no step is skipped at either threshold, and the 0.17 default
    loses exactly one skip (the steps 6-7 double skip becomes single)."""
    from mlx_teacache.variants.flux2_klein_base_4b.config import CALIBRATED_RANGE

    assert _replay_base_4b(0.08, CALIBRATED_RANGE) == "C" * _NUM_STEPS
    assert _replay_base_4b(0.12, CALIBRATED_RANGE) == "C" * _NUM_STEPS
    assert _replay_base_4b(0.17, CALIBRATED_RANGE) == "CCSCSCSCSCSCCCCSCCSCSCCCCCCCCCCCCCCCCCCCCCCCCCSCSC"


def _replay_with(trace: list[float], coefficients, thresh: float, calibrated_range) -> str:
    state = TeaCacheState()
    mod = mx.array([1.0])
    pattern = []
    deltas = [*trace, 0.1]  # the forced last step's delta is never measured
    for step in range(_NUM_STEPS):
        if step > 0:
            mod = mod * (1.0 + deltas[step - 1])
        dec = gate_step(
            state,
            rel_l1_thresh=thresh,
            coefficients=coefficients,
            skip_first=1,
            skip_last=1,
            num_steps=_NUM_STEPS,
            step_idx=step,
            mod_in=mod,
            calibrated_range=calibrated_range,
        )
        if dec.should_update_cache:
            state.cached_residual = mx.zeros((1,))
        pattern.append("C" if dec.should_compute else "S")
    return "".join(pattern)


def test_qwen_calibrated_range_leaves_the_measured_bench_pattern_unchanged():
    """RED if qwen's CALIBRATED_RANGE is narrower than its fit's data: the bench recipe is the calibration
    recipe, so the clamp must not move the shipped 3.0x operating point (32-33 skips, the bench pattern)."""
    from mlx_teacache.variants.qwen_image.config import CALIBRATED_RANGE

    patterns = [_replay_with(t, COEFFICIENTS, DEFAULT_THRESH, CALIBRATED_RANGE) for t in _prompt_traces()]
    assert all(32 <= p.count("S") <= 33 for p in patterns), patterns
    assert _BENCH_PATTERN in patterns


# Measured by scripts/bench_speedup.py --variant z-image --three-way --reps 3 (identical in all three reps,
# _artifacts/v0.10.0_bench_z_image.json); the bench recipe is Z-Image's calibration recipe.
_Z_IMAGE_BENCH_PATTERN = "CCCCCCCCCCSCSCSCSCSCSCSCSCSCSCSCSCSCSCSCCCCCCCCCCC"


def test_z_image_replay_with_its_calibrated_range_matches_the_committed_bench():
    """RED if z-image's CALIBRATED_RANGE is too narrow (or taken from signal A instead of the shipped signal B
    fit) and changes decisions at the recipe the published speedup was measured on."""
    from mlx_teacache.variants.z_image_base.config import CALIBRATED_RANGE
    from mlx_teacache.variants.z_image_base.config import COEFFICIENTS as Z_IMAGE
    from mlx_teacache.variants.z_image_base.config import DEFAULT_THRESH as Z_IMAGE_THRESH

    data = json.loads((_REPO_ROOT / "scripts" / "_calibration_z_image.json").read_text())
    xs = data["signals"]["B"]["x_values"]
    first_prompt = xs[: len(xs) // data["n_fit_prompts"]]
    assert _replay_with(first_prompt, Z_IMAGE, Z_IMAGE_THRESH, CALIBRATED_RANGE) == _Z_IMAGE_BENCH_PATTERN
