"""Drift guard for the mflux forwards this library copies.

Three variant integrations re-walk the vanilla transformer body step by step
(FLUX.2 Klein, Z-Image, Qwen-Image) and FLUX.1 delegates to mflux's own block
helpers; all four also depend on the shape of the generation loop that calls
them. A change upstream to any of those bodies is invisible off the real-weights
parity lane, so this test pins an AST fingerprint of each one per mflux version.

On red: the installed mflux is either unknown here (add it only after diffing
each listed function against the copy it feeds and re-running the parity lane)
or a known version's function no longer matches (the copy must be re-verified).
This is a heads-up, not proof of breakage."""

import importlib
from importlib.metadata import version

import pytest

from tests._mflux_surface import ast_fingerprint

_TARGETS: list[tuple[str, str, str, str]] = [
    # (label, module, class, member)  — member is looked up via the class __dict__ so staticmethods resolve
    (
        "flux1.Transformer.__call__",
        "mflux.models.flux.model.flux_transformer.transformer",
        "Transformer",
        "__call__",
    ),
    (
        "flux2.Flux2Transformer.__call__",
        "mflux.models.flux2.model.flux2_transformer.transformer",
        "Flux2Transformer",
        "__call__",
    ),
    (
        "z_image.ZImageTransformer.__call__",
        "mflux.models.z_image.model.z_image_transformer.transformer",
        "ZImageTransformer",
        "__call__",
    ),
    (
        "qwen.QwenTransformer.__call__",
        "mflux.models.qwen.model.qwen_transformer.qwen_transformer",
        "QwenTransformer",
        "__call__",
    ),
    ("flux1.Flux1.generate_image", "mflux.models.flux.variants.txt2img.flux", "Flux1", "generate_image"),
    (
        "flux2.Flux2Klein.generate_image",
        "mflux.models.flux2.variants.txt2img.flux2_klein",
        "Flux2Klein",
        "generate_image",
    ),
    (
        "flux2.Flux2Klein._predict",
        "mflux.models.flux2.variants.txt2img.flux2_klein",
        "Flux2Klein",
        "_predict",
    ),
    ("z_image.ZImage.generate_image", "mflux.models.z_image.variants.z_image", "ZImage", "generate_image"),
    ("z_image.ZImage._predict", "mflux.models.z_image.variants.z_image", "ZImage", "_predict"),
    (
        "qwen.QwenImage.generate_image",
        "mflux.models.qwen.variants.txt2img.qwen_image",
        "QwenImage",
        "generate_image",
    ),
]

# Filled per version after the copies were verified against it. Keys are exact
# `importlib.metadata.version("mflux")` strings.
KNOWN: dict[str, dict[str, str]] = {
    # Verified 2026-09-05 against the copies in v0.10.1 (the real-weights parity lane
    # passed on 0.18.0 for FLUX.1, FLUX.2 Klein, Z-Image and Qwen-Image; 0.18.1 differs
    # from 0.18.0 only in the two `_predict` factories, which the eager closures replace;
    # 0.17.5, the floor, differs in the FLUX.2 transformer forward, which gained its
    # KV-cache path in 0.18.0 — a path the copy does not take because kv_cache is None).
    "0.17.5": {
        "flux1.Transformer.__call__": "5d91b45c47115222",
        "flux2.Flux2Transformer.__call__": "2de85c3795cb9a5c",
        "z_image.ZImageTransformer.__call__": "2d492d0912e9e066",
        "qwen.QwenTransformer.__call__": "6600c3d8f0a874dc",
        "flux1.Flux1.generate_image": "0307a4e6de9d9a5e",
        "flux2.Flux2Klein.generate_image": "e52a3fd4d37a783f",
        "flux2.Flux2Klein._predict": "1eb65d3778f18f8e",
        "z_image.ZImage.generate_image": "945ff36f66596563",
        "z_image.ZImage._predict": "807bb08910d0b839",
        "qwen.QwenImage.generate_image": "2f04c1c09ce2512b",
    },
    "0.18.0": {
        "flux1.Transformer.__call__": "5d91b45c47115222",
        "flux2.Flux2Transformer.__call__": "77c738101d62cb9e",
        "z_image.ZImageTransformer.__call__": "2d492d0912e9e066",
        "qwen.QwenTransformer.__call__": "6600c3d8f0a874dc",
        "flux1.Flux1.generate_image": "0307a4e6de9d9a5e",
        "flux2.Flux2Klein.generate_image": "e52a3fd4d37a783f",
        "flux2.Flux2Klein._predict": "f4344a22c1c41345",
        "z_image.ZImage.generate_image": "945ff36f66596563",
        "z_image.ZImage._predict": "f25b760f51447925",
        "qwen.QwenImage.generate_image": "2f04c1c09ce2512b",
    },
    "0.18.1": {
        "flux1.Transformer.__call__": "5d91b45c47115222",
        "flux2.Flux2Transformer.__call__": "77c738101d62cb9e",
        "z_image.ZImageTransformer.__call__": "2d492d0912e9e066",
        "qwen.QwenTransformer.__call__": "6600c3d8f0a874dc",
        "flux1.Flux1.generate_image": "0307a4e6de9d9a5e",
        "flux2.Flux2Klein.generate_image": "e52a3fd4d37a783f",
        "flux2.Flux2Klein._predict": "1eb65d3778f18f8e",
        "z_image.ZImage.generate_image": "945ff36f66596563",
        "z_image.ZImage._predict": "807bb08910d0b839",
        "qwen.QwenImage.generate_image": "2f04c1c09ce2512b",
    },
}


def _member(module: str, cls_name: str, member: str):  # noqa: ANN202
    cls = getattr(importlib.import_module(module), cls_name)
    raw = cls.__dict__[member]
    return raw.__func__ if isinstance(raw, staticmethod | classmethod) else raw


def _installed_fingerprints() -> dict[str, str]:
    return {label: ast_fingerprint(_member(m, c, f)) for label, m, c, f in _TARGETS}


def test_installed_mflux_version_has_verified_forward_fingerprints() -> None:
    installed = version("mflux")
    assert installed in KNOWN, (
        f"mflux {installed} is not a version whose forwards were verified against the copies in "
        f"src/mlx_teacache/variants/*/integration.py; current fingerprints: {_installed_fingerprints()}"
    )


@pytest.mark.parametrize("label", [t[0] for t in _TARGETS])
def test_forward_fingerprint_matches_the_verified_one(label: str) -> None:
    installed = version("mflux")
    if installed not in KNOWN:
        pytest.fail(
            f"mflux {installed} unknown; see test_installed_mflux_version_has_verified_forward_fingerprints"
        )
    fresh = _installed_fingerprints()[label]
    assert fresh == KNOWN[installed][label], (
        f"{label} changed in mflux {installed} (was {KNOWN[installed][label]}, now {fresh}); diff it against the "
        f"copy that feeds on it and re-verify before updating the pin"
    )
