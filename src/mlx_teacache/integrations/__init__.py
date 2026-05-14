# src/mlx_teacache/integrations/__init__.py
"""Integration layer. Each subpackage (currently only `mflux`) provides
runtime hooks for a specific upstream framework. The package root
(`mlx_teacache`) does NOT import any submodule of `integrations` at import
time — mflux imports are deferred until `apply_teacache` is actually called."""
