# src/mlx_teacache/integrations/_session.py
"""The internal session every integration shares, independent of the runner.

`InternalHandle` is the object the forwards and the lifecycle hooks read and write during a generation: the gate
state, the staged stats, the generation context and the CFG branch pairer. It is never returned to the user; the
public `TeaCacheHandle` keeps a reference to it so `restore()` can release the cached arrays.

Imports only `_kernel`. Of the three runner-hook slots, `_generate_image_was_instance_attr` and
`_original_generate_image` are written by the runner's lifecycle module; `_callback_instance` is written by its patch
module.
"""

from dataclasses import dataclass, field
from typing import Any

from mlx_teacache._kernel.cache import TeaCacheState
from mlx_teacache._kernel.stats import TeaCacheStats


@dataclass
class GenerationContext:
    token: int = 0  # incremented in call_before_loop
    active_num_steps: int | None = None  # set in call_before_loop, cleared by wrapper
    consumed_at_token: int | None = None  # set when FLUX.2 predict closure consumes


@dataclass(frozen=True)
class PendingFinalize:
    """Set by call_after_loop; consumed by the generate_image wrapper after
    original() returns naturally. Typed (rather than dict) so strict mypy
    can verify the field shapes at every callsite."""

    num_inference_steps: int
    cfg_was_active: bool


@dataclass
class InternalHandleState:
    cache: TeaCacheState = field(default_factory=TeaCacheState)
    stats: TeaCacheStats = field(default_factory=TeaCacheStats)
    no_benefit_warned: bool = False


@dataclass
class CfgBranchPairer:
    """CFG two-call branch-pairing bookkeeping for runners that call the transformer once per branch.

    Qwen-Image's generate_image calls the transformer twice per denoising step, positive (prompt) then negative
    (negative prompt), and combines outside the transformer. The proxy forward therefore fires once per branch. This
    object threads the per-step shared gate decision and the branch parity across the two calls, resetting when a new
    generation starts (detected by the GenerationContext token, bumped in call_before_loop).

    Reset-on-token-change self-heals a KeyboardInterrupt landing between the positive and negative calls (parity left
    mid-pair): the next generation bumps the token and resets to positive.
    """

    branch_idx: int = 0  # 0 = positive (decide); 1 = negative (reuse)
    last_seen_token: int | None = None
    shared_decision: Any = None  # GateDecision stored on the positive call

    def on_generation_token(self, token: int) -> None:
        """Reset branch parity + shared decision when a new generation starts."""
        if token != self.last_seen_token:
            self.last_seen_token = token
            self.branch_idx = 0
            self.shared_decision = None

    def is_positive(self) -> bool:
        return self.branch_idx == 0

    def advance(self) -> None:
        """Flip parity after a branch call completes.

        Precondition: called EXACTLY ONCE per branch forward call (once on the
        positive call, once on the negative). The integration forward upholds
        this; a double-advance before the negative branch is consumed would
        silently desync parity. Behaviorally pinned by the two-calls-per-step
        orchestration tests in tests/test_qwen_branch_pairing.py.
        """
        self.branch_idx ^= 1


class InternalHandle:
    """The per-apply session the forwards and the lifecycle hooks share. Not returned to the user."""

    def __init__(
        self,
        *,
        rel_l1_thresh: float,
        coefficients: tuple[float, float, float, float, float],
        skip_first_n_steps: int,
        skip_last_n_steps: int,
        calibrated_range: tuple[float, float] | None = None,
    ) -> None:
        self._state = InternalHandleState()
        self._gen_ctx = GenerationContext()
        # Read only by forwards that call the transformer once per CFG branch (Qwen-Image).
        self._pairer = CfgBranchPairer()
        self.rel_l1_thresh = rel_l1_thresh
        self.coefficients = coefficients
        self.skip_first_n_steps = skip_first_n_steps
        self.skip_last_n_steps = skip_last_n_steps
        # (min, max) rel_l1 the coefficients were fitted on; None = evaluate the raw delta.
        self.calibrated_range = calibrated_range
        # Fields set by wrap_generate_image
        self._generate_image_was_instance_attr: bool = False
        self._original_generate_image: Any = None
        self._pending_finalize: PendingFinalize | None = None
        # Field set by GenerationContextCallback registration
        self._callback_instance: Any = None

    def release_arrays(self) -> None:
        """Drop the cached body-sized arrays, keeping the counters."""
        self._state.cache.release_arrays()
