"""Qwen-Image integration: its provenance, ORIGINAL_CALIBRATION and apply(), which runs the
shared Qwen-Image apply (integrations/mflux/forward_qwen.py) with this checkpoint's calibration.

mflux is never imported at module level (the registry loads this only after detect.matches() wins).
"""

from typing import Any

from mlx_teacache._kernel.coefficients import Provenance
from mlx_teacache.handle import TeaCacheHandle
from mlx_teacache.integrations._session import InternalHandle
from mlx_teacache.integrations.mflux.forward_qwen import ProxyQwenTransformer as ProxyQwenTransformer
from mlx_teacache.integrations.mflux.forward_qwen import QwenCalibration as QwenCalibration
from mlx_teacache.integrations.mflux.forward_qwen import _Prelude as _Prelude
from mlx_teacache.integrations.mflux.forward_qwen import _qwen_prelude as _qwen_prelude
from mlx_teacache.integrations.mflux.forward_qwen import _qwen_run_body as _qwen_run_body
from mlx_teacache.integrations.mflux.forward_qwen import _qwen_signal_a as _qwen_signal_a
from mlx_teacache.integrations.mflux.forward_qwen import _qwen_tail as _qwen_tail
from mlx_teacache.integrations.mflux.forward_qwen import _step_decision_from_gate as _step_decision_from_gate
from mlx_teacache.integrations.mflux.forward_qwen import apply_qwen as apply_qwen
from mlx_teacache.integrations.mflux.forward_qwen import qwen_forward_with_gate as qwen_forward_with_gate

from .config import CALIBRATED_RANGE, COEFFICIENTS, DEFAULT_THRESH, META

# The shared session type under the name the tests still import from here.
_InternalHandle = InternalHandle


_PROVENANCE = Provenance(
    source="builtin",
    revision="in-repo-2026-06-17-origin-signalA-768-50",
    calibration_dataset=(
        "10 prompts (7 fit / 3 held-out) x 50 steps x seed=42, M1 Max 32GB, q4, "
        "768x768, guidance=4.0 (CFG), origin-constrained polyfit, chunked per-prompt"
    ),
    fit_metric=(
        "constrained-LSQ R^2 on consecutive-step Signal-A (modulated block-0 input, "
        "worst-branch body_out) rel-L1 pairs (poly(0)=0)"
    ),
    fit_metric_value=0.8490,
    reference_url="https://github.com/IonDen/mlx-teacache/blob/main/scripts/calibrate_qwen.py",
    default_thresh=DEFAULT_THRESH,
)


ORIGINAL_CALIBRATION = QwenCalibration(
    variant_id="qwen-image",
    coefficients=COEFFICIENTS,
    calibrated_range=CALIBRATED_RANGE,
    default_thresh=DEFAULT_THRESH,
    provenance=_PROVENANCE,
    hf_model_id=META["hf_model_id"],
)


def apply(
    flux: Any,
    *,
    rel_l1_thresh: float | None = None,
    coefficients: tuple[float, float, float, float, float] | None = None,
    skip_first_n_steps: int = 1,
    skip_last_n_steps: int = 1,
) -> TeaCacheHandle:
    """Qwen-Image apply. Proxies flux.transformer; lifecycle-callback driven."""
    return apply_qwen(
        flux,
        ORIGINAL_CALIBRATION,
        rel_l1_thresh=rel_l1_thresh,
        coefficients=coefficients,
        skip_first_n_steps=skip_first_n_steps,
        skip_last_n_steps=skip_last_n_steps,
    )
