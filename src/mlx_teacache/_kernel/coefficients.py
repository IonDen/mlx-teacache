"""Canonical home for the Provenance dataclass (extracted in v0.6.0).

The coefficient _REGISTRY and per-variant tuples remain in
src/mlx_teacache/coefficients.py for the duration of Phase A — they
move to per-variant config.py files in Phase C, with the legacy
registry deleted in Task 18.

See docs/superpowers/specs/2026-05-19-per-variant-cores-design.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Provenance:
    source: Literal["builtin", "user"]
    revision: str | None = None
    calibration_dataset: str | None = None
    fit_metric: str | None = None
    fit_metric_value: float | None = None
    reference_url: str | None = None
    default_thresh: float | None = None

    @classmethod
    def for_user_supplied(cls) -> Provenance:
        return cls(source="user")
