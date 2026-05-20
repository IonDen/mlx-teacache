"""Compatibility shim. Re-exports Provenance and validate_custom from _kernel.coefficients.

The v0.5.x _REGISTRY and per-variant coefficient tuples moved to
src/mlx_teacache/variants/<name>/config.py in Task 18 (v0.6.0).
"""
from mlx_teacache._kernel.coefficients import Provenance, validate_custom

__all__ = ["Provenance", "validate_custom"]
