"""Pin the decision rule: replay every committed rel-L1 trace through the shipped gate and compare the
per-step decisions with the golden lists in tests/fixtures/decision_rule_golden.json.

The golden file is tied to DECISION_RULE_REVISION. A change to gate_step decisions, to default-threshold
resolution, to MAX_CONSECUTIVE_SKIPS or to the skip-window defaults turns these tests red until the revision
is bumped and the golden file is re-recorded in the same change:

    HF_HUB_OFFLINE=1 uv run --no-sync python -m tests.test_decision_rule_golden

Scope: the gate and the cap, driven by recorded traces. The rel-L1 values a real model produces depend on
mflux and MLX and are not pinned here. Pure-core: mlx.core plus the kernel, and duck-typed models for the
threshold a None default resolves to; no weights, no mflux, milliseconds.

One character per step: C computed, F forced (the skip window), S skipped, N numerical miss.
"""

import importlib
import inspect
import json
import math
import re
import warnings
from pathlib import Path
from typing import Any

import mlx.core as mx
import pytest

import mlx_teacache._kernel.gate as gate
from mlx_teacache._kernel.cache import TeaCacheState
from mlx_teacache._kernel.gate import DECISION_RULE_REVISION, gate_step
from mlx_teacache.api import apply_teacache
from mlx_teacache.errors import TeaCacheNoBenefitWarning
from mlx_teacache.variants import _REGISTRY
from tests._variant_fakes import fake_for_variant
from tests.test_gate_replay import (
    _BASE_4B_BENCH_RECIPE_DELTAS,
    _BASE_4B_REAL_DELTAS,
    _BENCH_PATTERN,
    _NUM_STEPS,
    _QWEN_2512_TRACE,
    _prompt_traces,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_GOLDEN = Path(__file__).resolve().parent / "fixtures" / "decision_rule_golden.json"

_KIND_CHAR = {"computed": "C", "forced": "F", "skipped": "S", "numerical-miss": "N"}

# Thresholds at which a high-threshold replay is tried for the cap case, lowest first.
_CAP_CANDIDATES = (0.5, 0.6, 0.8, 1.0)

# The numerical-miss case: qwen-image prompt 0 at its default threshold with one non-finite step, placed inside
# a run of skips (step 16 skips on the finite trace).
_MISS_STEP = 16
_MISS_KEY = f"qwen-image/prompt0-nonfinite-step{_MISS_STEP}/thresh=0.30"


def _variant(subpackage: str) -> tuple[Any, Any]:
    """(coefficients, calibrated_range) of a variant, read the way apply() passes them."""
    config = importlib.import_module(f"mlx_teacache.variants.{subpackage}.config")
    return config.COEFFICIENTS, getattr(config, "CALIBRATED_RANGE", None)


def _kinds(trace: list[float], coefficients: Any, thresh: float, calibrated_range: Any) -> str:
    """The real gate driven by a consecutive-delta trace, as test_gate_replay._replay_with drives it: the
    forced last step's delta is never measured (a trailing 0.1 stands in), and every step that asks for a
    cache update gets a residual. A non-finite delta makes that one step's modulated input non-finite; the
    next step continues from the last finite input."""
    state = TeaCacheState()
    mod = mx.array([1.0])
    kinds = []
    deltas = [*trace, 0.1]
    for step in range(_NUM_STEPS):
        mod_in = mod
        if step > 0:
            mod_in = mod * (1.0 + deltas[step - 1])
            if math.isfinite(deltas[step - 1]):
                mod = mod_in
        dec = gate_step(
            state,
            rel_l1_thresh=thresh,
            coefficients=coefficients,
            skip_first=1,
            skip_last=1,
            num_steps=_NUM_STEPS,
            step_idx=step,
            mod_in=mod_in,
            calibrated_range=calibrated_range,
        )
        if dec.should_update_cache:
            state.cached_residual = mx.zeros((1,))
        kinds.append(_KIND_CHAR[dec.kind])
    return "".join(kinds)


def _z_image_signal_b_traces() -> list[list[float]]:
    data = json.loads((_REPO_ROOT / "scripts" / "_calibration_z_image.json").read_text())
    xs = data["signals"]["B"]["x_values"]
    per = len(xs) // data["n_fit_prompts"]
    return [xs[i * per : (i + 1) * per] for i in range(data["n_fit_prompts"])]


def _cap_thresh() -> float:
    """The lowest candidate threshold at which qwen-image prompt 0 replays to a run of exactly
    MAX_CONSECUTIVE_SKIPS skips (the cap cannot let a run grow past it)."""
    coefficients, calibrated_range = _variant("qwen_image")
    trace = _prompt_traces()[0]
    for thresh in _CAP_CANDIDATES:
        if "S" * gate.MAX_CONSECUTIVE_SKIPS in _kinds(trace, coefficients, thresh, calibrated_range):
            return thresh
    raise AssertionError(
        f"no candidate threshold in {_CAP_CANDIDATES} reaches the cap on qwen-image prompt 0"
    )


def _replay_all() -> dict[str, str]:
    sequences: dict[str, str] = {}

    def add(
        variant_id: str, trace_name: str, thresh: float, trace: list[float], variant: tuple[Any, Any]
    ) -> None:
        sequences[f"{variant_id}/{trace_name}/thresh={thresh:.2f}"] = _kinds(
            trace, variant[0], thresh, variant[1]
        )

    qwen = _variant("qwen_image")
    for n, trace in enumerate(_prompt_traces()):
        add("qwen-image", f"prompt{n}", 0.30, trace, qwen)
    qwen_2512 = _variant("qwen_image_2512")
    for n, trace in enumerate(_prompt_traces(_QWEN_2512_TRACE)):
        add("qwen-image-2512", f"prompt{n}", 0.25, trace, qwen_2512)
    # Step 1 only seeds, so its delta is never priced; 0.1 stands in for it.
    base_4b = _variant("flux2_klein_base_4b")
    add("flux2-klein-base-4b", "real-864x1152", 0.17, [0.1, *_BASE_4B_REAL_DELTAS], base_4b)
    add("flux2-klein-base-4b", "bench-512", 0.17, [0.1, *_BASE_4B_BENCH_RECIPE_DELTAS], base_4b)
    z_image = _variant("z_image_base")
    for n, trace in enumerate(_z_image_signal_b_traces()):
        add("z-image-base", f"prompt{n}", 0.12, trace, z_image)
    add("qwen-image", "prompt0", _cap_thresh(), _prompt_traces()[0], qwen)
    # Step s reads trace[s - 1], so a non-finite trace[_MISS_STEP - 1] makes step _MISS_STEP non-finite.
    miss_trace = list(_prompt_traces()[0])
    miss_trace[_MISS_STEP - 1] = math.nan
    add("qwen-image", f"prompt0-nonfinite-step{_MISS_STEP}", 0.30, miss_trace, qwen)
    return sequences


def _resolved_thresh(variant_id: str) -> float:
    """The threshold apply_teacache resolves when the caller passes neither rel_l1_thresh nor coefficients, read
    from the handle of a weight-free model the variant's detector matches."""
    with warnings.catch_warnings():
        # The no-benefit notice for distilled variants is not what this reads.
        warnings.filterwarnings("ignore", category=TeaCacheNoBenefitWarning)
        handle = apply_teacache(fake_for_variant(variant_id))
    try:
        assert handle.variant_id == variant_id, (variant_id, handle.variant_id)
        return handle.rel_l1_thresh
    finally:
        handle.restore()


def _rule_parameters() -> dict[str, Any]:
    params = inspect.signature(apply_teacache).parameters
    return {
        "max_consecutive_skips": gate.MAX_CONSECUTIVE_SKIPS,
        "window_defaults": [params["skip_first_n_steps"].default, params["skip_last_n_steps"].default],
        "default_thresh": {
            variant_id: entry["default_thresh"] for variant_id, entry in sorted(_REGISTRY.items())
        },
        # A None default (the distilled Klein variants) still resolves to a threshold; pin which one.
        "default_thresh_none_resolves_to": {
            variant_id: _resolved_thresh(variant_id)
            for variant_id, entry in sorted(_REGISTRY.items())
            if entry["default_thresh"] is None
        },
    }


def _golden() -> dict[str, Any]:
    golden: dict[str, Any] = json.loads(_GOLDEN.read_text())
    return golden


def _diff(expected: dict[str, str], actual: dict[str, str]) -> str:
    lines = []
    for key in sorted(expected.keys() | actual.keys()):
        want, got = expected.get(key), actual.get(key)
        if want != got:
            lines.append(f"{key}\n  golden  {want}\n  replay  {got}")
    return "\n".join(lines)


def test_golden_revision_is_the_current_revision() -> None:
    """Bug: DECISION_RULE_REVISION is bumped without re-recording the golden lists."""
    assert _golden()["decision_rule_revision"] == DECISION_RULE_REVISION


def test_every_committed_trace_replays_to_its_golden_sequence() -> None:
    """Bug: a gate or cap change moves a decision on a committed trace without a revision bump. Covers the
    gate and the cap on recorded traces, not the rel-L1 values a real model produces (those depend on mflux
    and MLX)."""
    golden = _golden()["sequences"]
    replayed = _replay_all()
    assert replayed == golden, "decisions moved off the golden lists:\n" + _diff(golden, replayed)


def test_rule_parameters_match_the_golden() -> None:
    """Bug: a variant's default threshold, the threshold a None default resolves to (the distilled Klein
    fallback), the runaway cap or a skip-window default changes without a revision bump."""
    assert _rule_parameters() == _golden()["rule"]


def test_the_golden_set_exercises_a_numerical_miss() -> None:
    """Bug: the pin cannot see a change to the numerical-miss path, because no golden sequence holds an N. The
    miss sequence must also read as the gate documents it: nothing before the non-finite step moves, that step
    is the only miss, and the step after it computes (the residual cached before the miss was dropped, so the
    gate re-seeds instead of skipping on it)."""
    golden = _golden()["sequences"]
    with_miss = [k for k, v in golden.items() if "N" in v]
    assert with_miss == [_MISS_KEY]
    seq = golden[_MISS_KEY]
    finite = golden["qwen-image/prompt0/thresh=0.30"]
    assert seq.count("N") == 1
    assert seq.index("N") == _MISS_STEP
    assert finite[_MISS_STEP] == "S"
    assert seq[:_MISS_STEP] == finite[:_MISS_STEP]
    assert seq[_MISS_STEP + 1] == "C"


def test_the_golden_set_exercises_the_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bug: the pin cannot see a cap change, because no golden sequence reaches the cap."""
    golden = _golden()["sequences"]
    cap_keys = [
        k for k in golden if k.startswith("qwen-image/prompt0/") and k != "qwen-image/prompt0/thresh=0.30"
    ]
    assert len(cap_keys) == 1, cap_keys
    (cap_key,) = cap_keys
    assert "S" * 8 in golden[cap_key]
    assert "S" * 9 not in golden[cap_key]

    thresh = float(cap_key.rsplit("=", 1)[1])
    coefficients, calibrated_range = _variant("qwen_image")
    monkeypatch.setattr(gate, "MAX_CONSECUTIVE_SKIPS", 7)
    assert _kinds(_prompt_traces()[0], coefficients, thresh, calibrated_range) != golden[cap_key]


def test_golden_sequences_agree_with_the_hardware_patterns() -> None:
    """Bug: the golden lists were recorded from a broken gate. With forced and numerical-miss steps read as
    computed, one qwen-image prompt reproduces the v0.10.0 bench pattern and the Klein base 4B bench-recipe
    trace reproduces the v0.12.0 bench's measured pattern."""
    golden = _golden()["sequences"]

    def as_bench(kinds: str) -> str:
        return kinds.replace("F", "C").replace("N", "C")

    qwen_at_default = [
        as_bench(v) for k, v in golden.items() if re.fullmatch(r"qwen-image/prompt\d/thresh=0\.30", k)
    ]
    assert len(qwen_at_default) == 7
    assert _BENCH_PATTERN in qwen_at_default

    report = json.loads((_REPO_ROOT / "_artifacts" / "v0.12.0_bench_klein_base_4b.json").read_text())
    assert as_bench(golden["flux2-klein-base-4b/bench-512/thresh=0.17"]) == report["skip_patterns"][0]


def _record() -> None:
    """Write the golden file from the current rule. Run only together with a DECISION_RULE_REVISION bump."""
    golden = {
        "decision_rule_revision": DECISION_RULE_REVISION,
        "rule": _rule_parameters(),
        "sequences": _replay_all(),
    }
    _GOLDEN.write_text(json.dumps(golden, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    _record()
