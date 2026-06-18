"""Pure CFG two-call branch-pairing bookkeeping for Qwen-Image. mflux-free.

Qwen's generate_image calls the transformer twice per denoising step — positive
(prompt) then negative (negative prompt) — and combines OUTSIDE the transformer
(QwenImage.compute_guided_noise). The proxy forward therefore fires once per
branch. This object threads the per-step shared gate decision and the branch
parity across the two calls, resetting when a new generation starts (detected by
the lifecycle GenerationContext token, bumped in call_before_loop).

Reset-on-token-change self-heals a KeyboardInterrupt landing between the positive
and negative calls (parity left mid-pair): the next generation bumps the token
and resets to positive.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class CfgBranchPairer:
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
