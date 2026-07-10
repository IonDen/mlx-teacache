"""Variant-agnostic context-manager handle.

`apply_teacache(flux)` returns a `TeaCacheHandle`. Variants build the
handle in their `apply()` and pass a `VariantPatch` describing how to
undo their mutations + unsubscribe their callbacks. The handle does NOT
own stats finalization — that's the mflux lifecycle wrapper's job (see
src/mlx_teacache/integrations/mflux/lifecycle.py::wrap_generate_image).
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from mlx_teacache._kernel.coefficients import Provenance
from mlx_teacache._kernel.stats import TeaCacheStats


@dataclass
class VariantPatch:
    """Teardown contract returned by each variant's apply().

    rollbacks: undo callables in install order (handle runs them in reverse).
    finalizers: callables that run after rollbacks (e.g., callback unsubscribe).
    on_restored: callables that run only after every teardown action succeeds.

    Stats commit/discard is NOT in this list. The mflux lifecycle owns that.
    """

    rollbacks: list[Callable[[], None]] = field(default_factory=list)
    finalizers: list[Callable[[], None]] = field(default_factory=list)
    on_restored: list[Callable[[], None]] = field(default_factory=list)


class TeaCacheHandle:
    """Context-manager handle returned by apply_teacache."""

    def __init__(
        self,
        *,
        patch: VariantPatch,
        stats: TeaCacheStats,
        provenance: Provenance,
        rel_l1_thresh: float,
    ) -> None:
        self._patch = patch
        self.stats = stats
        self.provenance = provenance
        self.rel_l1_thresh = rel_l1_thresh
        self._torn_down = False

    def __enter__(self) -> "TeaCacheHandle":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.restore()

    def restore(self) -> None:
        if self._torn_down:
            return
        errors: list[BaseException] = []
        for rollback in reversed(self._patch.rollbacks):
            try:
                rollback()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
        for finalize in self._patch.finalizers:
            try:
                finalize()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
        if errors:
            raise errors[0]
        for on_success in self._patch.on_restored:
            on_success()
        if hasattr(self.stats, "_freeze"):
            self.stats._freeze()
        self._torn_down = True
