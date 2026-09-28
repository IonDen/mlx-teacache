"""gate_step's calibrated_range: the polynomial is evaluated only inside the rel_l1 range it was fitted on.

The FLUX.2 [klein] base 4B fit is origin-constrained (p(0) = 0), so below its calibrated minimum (0.0283) it
predicts almost no change, although every output change measured during calibration was at least 0.107. On a
real 864x1152 / 50-step run the gate skipped steps whose delta was 0.0035 at every threshold down to 0.08.
Clamping the delta into the calibrated range before the polynomial removes that extrapolation at both ends.

Pure-core: mlx.core + the kernel, no weights, no mflux."""

import mlx.core as mx
import pytest

from mlx_teacache._kernel.cache import TeaCacheState
from mlx_teacache._kernel.gate import GateDecision, gate_step

# FLUX.2 [klein] base 4B coefficients and calibrated x-range, from
# scripts/_calibration_flux2_klein_base_4b.json (coefficients_c4_to_c0, x_min, x_max).
_COEFFS = (-1841.022165607874, 848.4417137572868, -131.3554469956159, 8.179509586828413, 0.0)
_RANGE = (0.028261402621865273, 0.21984468400478363)


def _gate_on_delta(x: float, *, thresh: float, calibrated_range: tuple[float, float] | None) -> GateDecision:
    """Step 0 forced (anchors), step 1 seeds the residual, step 2 is gated on a consecutive delta of x."""
    state = TeaCacheState()

    def step(idx: int, mod: float) -> GateDecision:
        return gate_step(
            state,
            rel_l1_thresh=thresh,
            coefficients=_COEFFS,
            skip_first=1,
            skip_last=1,
            num_steps=10,
            step_idx=idx,
            mod_in=mx.array([mod]),
            calibrated_range=calibrated_range,
        )

    step(0, 1.0)
    seed = step(1, 1.0)
    assert seed.should_update_cache
    state.cached_residual = mx.zeros((1,))
    return step(2, 1.0 + x)


def test_a_delta_below_the_range_is_priced_at_the_calibrated_minimum() -> None:
    """Bug: the clamp is missing or clamps the wrong end, so a 0.0035 delta is priced at p(0.0035) = 0.027 and
    skipped at threshold 0.08 instead of at p(0.0283) = 0.144 and computed."""
    unclamped = _gate_on_delta(0.0035, thresh=0.08, calibrated_range=None)
    assert unclamped.kind == "skipped"
    assert unclamped.predicted_distance == pytest.approx(0.02706, abs=1e-4)

    clamped = _gate_on_delta(0.0035, thresh=0.08, calibrated_range=_RANGE)
    assert clamped.kind == "computed"
    assert clamped.predicted_distance == pytest.approx(0.14423, abs=1e-4)


def test_a_delta_above_the_range_is_priced_at_the_calibrated_maximum() -> None:
    """Bug: only the low end is clamped, so a 0.30 delta lands on the fit's negative lobe (p = -1.37, clamped to
    0) and is skipped at threshold 0.12 instead of priced at p(0.2198) = 0.164 and computed."""
    unclamped = _gate_on_delta(0.30, thresh=0.12, calibrated_range=None)
    assert unclamped.kind == "skipped"
    assert unclamped.predicted_distance == 0.0

    clamped = _gate_on_delta(0.30, thresh=0.12, calibrated_range=_RANGE)
    assert clamped.kind == "computed"
    assert clamped.predicted_distance == pytest.approx(0.16414, abs=1e-4)


def test_a_delta_inside_the_range_is_priced_exactly_as_without_it() -> None:
    """Bug: the range is applied to every delta (e.g. always evaluating at a bound), changing in-domain
    decisions the shipped defaults were tuned on."""
    plain = _gate_on_delta(0.06, thresh=0.17, calibrated_range=None)
    ranged = _gate_on_delta(0.06, thresh=0.17, calibrated_range=_RANGE)
    assert ranged.predicted_distance == pytest.approx(0.17729, abs=1e-4)
    assert ranged.predicted_distance == plain.predicted_distance
    assert ranged.kind == plain.kind == "computed"


def test_the_decision_reports_the_measured_delta_not_the_clamped_one() -> None:
    """Bug: stats record the clamped x, hiding that the run left the calibrated domain."""
    clamped = _gate_on_delta(0.0035, thresh=0.08, calibrated_range=_RANGE)
    assert clamped.rel_l1 == pytest.approx(0.0035, rel=1e-4)
