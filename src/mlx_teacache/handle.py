"""Variant-agnostic context-manager handle.

`apply_teacache(flux)` returns a `TeaCacheHandle`. Variants build the
handle in their `apply()` and pass a `VariantPatch` describing how to
undo their mutations + unsubscribe their callbacks. The handle does NOT
own stats finalization — that's the mflux lifecycle wrapper's job (see
src/mlx_teacache/integrations/mflux/lifecycle.py::wrap_generate_image).
"""

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from mlx_teacache._kernel import decision_rule as _decision_rule
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


_READ_ONLY_FROM_1_0 = "The handle is read-only from 1.0."
_WRITE_MESSAGES = {
    "rel_l1_thresh": (
        "Setting TeaCacheHandle.rel_l1_thresh is deprecated and does not change what the gate skips; "
        f"pass rel_l1_thresh= to apply_teacache. {_READ_ONLY_FROM_1_0}"
    ),
    "stats": (
        "Setting TeaCacheHandle.stats is deprecated; replacing it detaches the handle from what the "
        f"generation records. {_READ_ONLY_FROM_1_0}"
    ),
    "provenance": (
        "Setting TeaCacheHandle.provenance is deprecated; replacing it detaches the handle from what the "
        f"generation records. {_READ_ONLY_FROM_1_0}"
    ),
    "coefficients": (
        "Setting TeaCacheHandle.coefficients is deprecated and does not change what the gate uses; "
        f"pass coefficients= to apply_teacache. {_READ_ONLY_FROM_1_0}"
    ),
}


def _warn_handle_write(name: str) -> None:
    """Warn that writing handle attribute `name` is deprecated, pointing at the assignment (helper -> setter ->
    caller). Setters call it before they store, so an error filter leaves the old value in place."""
    warnings.warn(_WRITE_MESSAGES[name], DeprecationWarning, stacklevel=3)


class TeaCacheHandle:
    """Context-manager handle returned by apply_teacache.

    Besides `stats`, `provenance` and `rel_l1_thresh`, it records what was patched and how: `variant_id`,
    `coefficients`, `skip_first_n_steps`, `skip_last_n_steps`, `decision_rule_revision` and `mflux_version` (None
    when mflux's package metadata is missing). These stay readable after restore()."""

    def __init__(
        self,
        *,
        patch: VariantPatch,
        stats: TeaCacheStats,
        provenance: Provenance,
        rel_l1_thresh: float,
    ) -> None:
        self._patch = patch
        self._stats = stats
        self._provenance = provenance
        self._rel_l1_thresh = rel_l1_thresh
        # Placeholders until apply_teacache and the variant integration record the real values.
        self._variant_id: str = "unknown"
        self._coefficients: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)
        self._skip_first_n_steps: int = 1
        self._skip_last_n_steps: int = 1
        self._decision_rule_revision: int = _decision_rule.DECISION_RULE_REVISION
        self._mflux_version: str | None = None
        self._torn_down = False

    @property
    def stats(self) -> TeaCacheStats:
        """Step decisions and per-generation totals recorded while the patch is installed."""
        return self._stats

    @stats.setter
    def stats(self, value: TeaCacheStats) -> None:
        _warn_handle_write("stats")
        self._stats = value

    @property
    def provenance(self) -> Provenance:
        """Where the coefficients and the default threshold came from."""
        return self._provenance

    @provenance.setter
    def provenance(self, value: Provenance) -> None:
        _warn_handle_write("provenance")
        self._provenance = value

    @property
    def rel_l1_thresh(self) -> float:
        """The effective threshold the gate uses."""
        return self._rel_l1_thresh

    @rel_l1_thresh.setter
    def rel_l1_thresh(self, value: float) -> None:
        _warn_handle_write("rel_l1_thresh")
        self._rel_l1_thresh = value

    @property
    def variant_id(self) -> str:
        """The registry id of the matched variant ("unknown" on a handle built without apply_teacache)."""
        return self._variant_id

    @variant_id.setter
    def variant_id(self, value: str) -> None:
        # Silent on purpose: mflux-teacache's test fake builds a handle and then writes variant_id, under
        # warnings-as-errors. The write starts warning once that fake has moved to mlx_teacache.testing.
        self._variant_id = value

    @property
    def coefficients(self) -> tuple[float, float, float, float, float]:
        """The five polynomial coefficients the gate uses (built-in or caller-supplied)."""
        return self._coefficients

    @coefficients.setter
    def coefficients(self, value: tuple[float, float, float, float, float]) -> None:
        _warn_handle_write("coefficients")
        self._coefficients = value

    @property
    def skip_first_n_steps(self) -> int:
        """Leading steps that always compute."""
        return self._skip_first_n_steps

    @property
    def skip_last_n_steps(self) -> int:
        """Trailing steps that always compute."""
        return self._skip_last_n_steps

    @property
    def decision_rule_revision(self) -> int:
        """Revision of the rule that turns a step's inputs into a compute or skip decision."""
        return self._decision_rule_revision

    @property
    def mflux_version(self) -> str | None:
        """The installed mflux version when the patch was applied, or None without mflux package metadata."""
        return self._mflux_version

    def __enter__(self) -> "TeaCacheHandle":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.restore()

    def restore(self) -> None:
        """Undo every patch apply_teacache installed; a second call does nothing.

        If mflux's memory saver dropped the transformer after the denoising loop, the transformer slot stays empty
        (as with plain mflux) and the rest of the patch is still undone. On a teardown error the first error is
        re-raised and the handle can be restored again."""
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
        self._stats._freeze()
        self._torn_down = True
