"""Compatibility shim. Re-exports Provenance and validate_custom from _kernel.coefficients.

The per-variant coefficient tuples live in
src/mlx_teacache/variants/<name>/config.py.
"""

from mlx_teacache._kernel.coefficients import Provenance, validate_custom

__all__ = ["Provenance", "validate_custom"]
