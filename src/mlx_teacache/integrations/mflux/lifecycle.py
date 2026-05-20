# src/mlx_teacache/integrations/mflux/lifecycle.py
"""Lifecycle helpers for both FLUX.1 and FLUX.2:

1. _GenerationContextCallback — registered on flux.callbacks. Implements all
   three protocols (BeforeLoopCallback, AfterLoopCallback, InterruptCallback)
   with the exact signatures mflux 0.17.5 uses.

2. wrap_generate_image — replaces flux.generate_image with a try/finally
   wrapper that clears handle._gen_ctx and discards/commits in-progress stats
   based on completion status (per spec §4.5 + §5.5 v2.5).

Both signatures match mflux/callbacks/callback.py exactly. Extra **kwargs are
accepted for forward-compat with future mflux releases that add new keyword
arguments (e.g., kontext_image).

v0.4.1 changes:
- Dropped the flux2_cfg_fallback warning-suppression block in call_before_loop.
  The CFG path is now gated (Task 4/5), so skipping is achievable even at
  guidance > 1.0; the old suppression was masking legitimate no-benefit warnings.
- call_after_loop now reads cfg_was_active from
  _staging.cfg_was_active (set by the predict closure on first CFG branch entry)
  instead of the obsolete cfg_fallback > 0 derivation."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any


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


def _active_step_count(config: Any) -> int:
    """Number of denoising calls mflux will actually run for this generation.

    For txt2img: equals `config.num_inference_steps`. For img2img: equals
    `num_inference_steps - init_time_step` per mflux's Config.time_steps
    property (`range(init_time_step, num_inference_steps)`). We do NOT consume
    `config.time_steps` directly because that property constructs a tqdm
    instance mflux later reuses for progress timing — touching it would
    fork the iterator.

    Returns 0 when no denoising will happen (e.g., image_strength=1.0).
    """
    nominal = int(config.num_inference_steps)
    init_step = int(getattr(config, "init_time_step", 0) or 0)
    return max(0, nominal - init_step)


class GenerationContextCallback:
    """Single callback class for both variants.

    - call_before_loop: captures active_num_steps, resets cache, bumps token.
    - call_after_loop: marks PendingFinalize (does NOT commit stats — the
      wrapper commits after original() returns naturally).
    - call_interrupt: no-op for stats (would violate len(decisions) == num_steps
      invariant); the generate_image wrapper's try/finally clears context.
    """

    def __init__(self, handle: Any) -> None:
        self._handle = handle

    def call_before_loop(
        self,
        seed: int,
        prompt: str,
        latents: Any,
        config: Any,
        canny_image: Any | None = None,
        depth_image: Any | None = None,
        **_extra: Any,
    ) -> None:
        # Bump generation token. FLUX.2 predict closure uses this to detect a
        # fresh, unconsumed context. FLUX.1 increments but doesn't read it.
        active_num_steps = _active_step_count(config)
        self._handle._gen_ctx.token += 1
        self._handle._gen_ctx.active_num_steps = active_num_steps
        self._handle._gen_ctx.consumed_at_token = None
        # Lifecycle is the single owner of cache reset (was: scattered across
        # forward.py for FLUX.1 and flux2.py for FLUX.2). Using active_num_steps
        # (not nominal) makes the cache invariants consistent under img2img.
        self._handle._state.cache.reset_for_new_generation(num_steps=active_num_steps)

        # --- Distilled-step / short-window no-benefit warning ---
        # Suppression: the configuration is going to raise InvalidStepWindowError
        # at lazy validation time — the error covers the case; a duplicate
        # warning is noise.
        window_invalid = (
            active_num_steps > 0
            and self._handle.skip_first_n_steps + self._handle.skip_last_n_steps >= active_num_steps
        )

        if active_num_steps == 0:
            # image_strength=1.0 → mflux runs zero denoising calls. Valid no-op.
            return
        if window_invalid:
            return

        eligible = active_num_steps - self._handle.skip_first_n_steps - self._handle.skip_last_n_steps
        possible_skips = max(0, eligible - 1)  # need ≥1 seed step + ≥1 candidate step

        if possible_skips == 0 and not self._handle._state.no_benefit_warned:
            from mlx_teacache.errors import TeaCacheNoBenefitWarning

            warnings.warn(
                f"TeaCache: only {eligible} eligible step(s) for caching with "
                f"active_num_steps={active_num_steps}, "
                f"skip_first_n_steps={self._handle.skip_first_n_steps}, "
                f"skip_last_n_steps={self._handle.skip_last_n_steps} "
                f"({possible_skips} possible skip(s)). The generation will run at "
                f"vanilla speed. Increase num_inference_steps or reduce "
                f"skip_first/skip_last to benefit from TeaCache.",
                category=TeaCacheNoBenefitWarning,
                stacklevel=2,
            )
            self._handle._state.no_benefit_warned = True

    def call_after_loop(
        self,
        seed: int,
        prompt: str,
        latents: Any,
        config: Any,
        **_extra: Any,
    ) -> None:
        # Mark "loop completed cleanly" — we do NOT finalize stats here because
        # another user-registered AfterLoopCallback could still raise after us.
        # If we finalized eagerly, public counters would be committed for a
        # generation that ends up raising. Instead the generate_image wrapper
        # finalizes after original() returns naturally.
        active_num_steps = self._handle._gen_ctx.active_num_steps
        if active_num_steps is None:
            # Defensive: before_loop should have set this. Recompute rather
            # than skipping finalization, so a missing setup doesn't silently
            # discard stats.
            active_num_steps = _active_step_count(config)

        # v0.4.1+: cfg_was_active is set by the predict closure on first CFG branch entry.
        # The old cfg_fallback>0 derivation is obsolete because production no longer
        # records "cfg-fallback" decisions.
        self._handle._pending_finalize = PendingFinalize(
            num_inference_steps=active_num_steps,
            cfg_was_active=self._handle._state.stats._staging.cfg_was_active,
        )
        # Clear gen-ctx fields only after capture so a subsequent before_loop
        # sees None.
        self._handle._gen_ctx.active_num_steps = None
        self._handle._gen_ctx.consumed_at_token = None

    def call_interrupt(
        self,
        t: int,
        seed: int,
        prompt: str,
        latents: Any,
        config: Any,
        time_steps: Any,
        **_extra: Any,
    ) -> None:
        # KeyboardInterrupt: do NOT finalize stats. A partial GenerationStats
        # with fewer than num_inference_steps decisions would violate the
        # invariant len(decisions) == num_inference_steps. The generate_image
        # wrapper's try/finally clears _gen_ctx regardless. Partial stats are
        # simply discarded.
        return None


def wrap_generate_image(flux: Any, handle: Any) -> None:
    """Replace flux.generate_image with a try/finally wrapper that:
    - Verifies our lifecycle callback is still registered (per audit medium #4).
    - On natural completion: finalizes staged stats via _pending_finalize.
    - On any other exit: discards staged stats so failed runs leave no trace.
    - Always clears _gen_ctx so context can't leak across runs.

    Records whether generate_image was an instance attribute pre-patch so
    restore() can do a pristine unpatch."""
    handle._generate_image_was_instance_attr = "generate_image" in vars(flux)
    if handle._generate_image_was_instance_attr:
        handle._original_generate_image = flux.generate_image
    else:
        handle._original_generate_image = None

    original = flux.generate_image  # bound regardless of source

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        # Per audit medium #4: verify our lifecycle callback is still registered
        # BEFORE the generation runs. If the user replaced or cleared
        # flux.callbacks after apply_teacache(), we must fail loudly rather than
        # silently disable img2img rejection / stats finalization.
        cb = getattr(handle, "_callback_instance", None)
        registry = getattr(flux, "callbacks", None)
        if cb is not None and not _callback_present_by_identity(registry, cb):
            from mlx_teacache.errors import MissingGenerationContextError

            raise MissingGenerationContextError(
                "TeaCache's lifecycle callback is no longer registered on "
                "flux.callbacks. This usually means flux.callbacks was "
                "replaced or cleared after apply_teacache(). Call "
                "handle.restore() and apply_teacache() again."
            )

        # Clear any leftover pending finalize from a previous run (defensive).
        handle._pending_finalize = None
        completed = False
        try:
            result = original(*args, **kwargs)
            completed = True
            return result
        finally:
            handle._gen_ctx.active_num_steps = None
            handle._gen_ctx.consumed_at_token = None
            if completed and handle._pending_finalize is not None:
                pf: PendingFinalize = handle._pending_finalize
                handle._state.stats.finalize_last_generation(
                    num_inference_steps=pf.num_inference_steps,
                    cfg_was_active=pf.cfg_was_active,
                )
            else:
                handle._state.stats.discard_current_generation()
            handle._pending_finalize = None

    flux.generate_image = wrapped


def _remove_callback_by_identity(registry: Any, target: Any) -> bool:
    """Walk every callback list on the registry and remove `target` by identity.
    Returns True iff at least one removal succeeded. mflux 0.17 stores the
    actual lists on `before_loop` / `in_loop` / `after_loop` / `interrupt`;
    the suffixed names (`*_callbacks`) are methods returning those same lists.
    We try the real list names first, then the suffixed names (for backward
    compat with existing fake-registry test fixtures), then generic fallbacks."""
    removed_any = False
    for attr in (
        "before_loop",
        "in_loop",
        "after_loop",
        "interrupt",
        "before_loop_callbacks",
        "in_loop_callbacks",
        "after_loop_callbacks",
        "interrupt_callbacks",
        "_callbacks",
        "callbacks",
    ):
        lst = getattr(registry, attr, None)
        if isinstance(lst, list):
            for i in range(len(lst) - 1, -1, -1):
                if lst[i] is target:
                    del lst[i]
                    removed_any = True
    return removed_any


def _callback_present_by_identity(registry: Any, target: Any) -> bool:
    """Return True iff target is registered (by identity) on any of the standard
    callback lists. mflux 0.17's CallbackRegistry stores lists on `before_loop`
    etc.; the suffixed names are methods. Check real names first, then the
    suffixed names (for fake-registry test fixtures), then generic fallbacks."""
    if registry is None:
        return False
    for attr in (
        "before_loop",
        "in_loop",
        "after_loop",
        "interrupt",
        "before_loop_callbacks",
        "in_loop_callbacks",
        "after_loop_callbacks",
        "interrupt_callbacks",
        "_callbacks",
        "callbacks",
    ):
        lst = getattr(registry, attr, None)
        if isinstance(lst, list) and any(item is target for item in lst):
            return True
    return False
