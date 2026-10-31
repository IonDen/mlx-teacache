"""Deprecated. Import Provenance from mlx_teacache instead; validate_custom has no public
replacement. The module is removed in 2.0.

Every attribute access warns with a DeprecationWarning at the caller's line.
"""

from typing import TYPE_CHECKING

from mlx_teacache._deprecation import shim_getattr

if TYPE_CHECKING:
    from mlx_teacache._kernel.coefficients import Provenance, validate_custom

__all__ = ["Provenance", "validate_custom"]

_SOURCES = dict.fromkeys(__all__, "mlx_teacache._kernel.coefficients")


def __getattr__(name: str) -> object:
    return shim_getattr(__name__, name, _SOURCES)


def __dir__() -> list[str]:
    return sorted(__all__)
