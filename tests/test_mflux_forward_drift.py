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
    # Digests come from fingerprint_function_node over the wheel sources and are the
    # same on CPython 3.10 through 3.14 (interpreter-dependent AST fields are dropped).
    # 0.17.5, the floor: the FLUX.2 transformer forward differs from 0.18.0, which
    # gained its KV-cache path there; the copy does not take it (kv_cache is None).
    "0.17.5": {
        "flux1.Transformer.__call__": "06ef78be1cd4e97c",
        "flux2.Flux2Transformer.__call__": "3c93bbfb5aaaf31f",
        "z_image.ZImageTransformer.__call__": "779a9ad06bfc2a65",
        "qwen.QwenTransformer.__call__": "b12184fbe7e98fe8",
        "flux1.Flux1.generate_image": "09830d48dfa71077",
        "flux2.Flux2Klein.generate_image": "e5b748fe91a48d44",
        "flux2.Flux2Klein._predict": "bffa1bd25b24cacd",
        "z_image.ZImage.generate_image": "97e2e2a3d9808ac5",
        "z_image.ZImage._predict": "016c64c92ceefbdf",
        "qwen.QwenImage.generate_image": "59ba0f1448730c80",
    },
    # Verified 2026-09-05 on real weights (FLUX.1, FLUX.2 Klein, Z-Image, Qwen-Image).
    "0.18.0": {
        "flux1.Transformer.__call__": "06ef78be1cd4e97c",
        "flux2.Flux2Transformer.__call__": "214c37be79a602b4",
        "z_image.ZImageTransformer.__call__": "779a9ad06bfc2a65",
        "qwen.QwenTransformer.__call__": "b12184fbe7e98fe8",
        "flux1.Flux1.generate_image": "09830d48dfa71077",
        "flux2.Flux2Klein.generate_image": "e5b748fe91a48d44",
        "flux2.Flux2Klein._predict": "bffa1bd25b24cacd",
        "z_image.ZImage.generate_image": "97e2e2a3d9808ac5",
        "z_image.ZImage._predict": "016c64c92ceefbdf",
        "qwen.QwenImage.generate_image": "59ba0f1448730c80",
    },
    # 0.18.1 is identical to 0.18.0 on all ten targets.
    "0.18.1": {
        "flux1.Transformer.__call__": "06ef78be1cd4e97c",
        "flux2.Flux2Transformer.__call__": "214c37be79a602b4",
        "z_image.ZImageTransformer.__call__": "779a9ad06bfc2a65",
        "qwen.QwenTransformer.__call__": "b12184fbe7e98fe8",
        "flux1.Flux1.generate_image": "09830d48dfa71077",
        "flux2.Flux2Klein.generate_image": "e5b748fe91a48d44",
        "flux2.Flux2Klein._predict": "bffa1bd25b24cacd",
        "z_image.ZImage.generate_image": "97e2e2a3d9808ac5",
        "z_image.ZImage._predict": "016c64c92ceefbdf",
        "qwen.QwenImage.generate_image": "59ba0f1448730c80",
    },
    # Verified 2026-09-06 on real weights (FLUX.1-dev, FLUX.1-schnell, FLUX.1 Krea
    # [dev], the four FLUX.2 Klein variants, Z-Image; mlx 0.32.2). What moved since
    # 0.18.x, all off by default on the copied path: ZImageTransformer.__call__
    # gained an optional controlnet_block_samples kwarg (a per-layer add when
    # given, None here); every generate_image gained a bake_lora flag and a
    # PiD-decoder branch after the loop. Both _predict factories are unchanged
    # since 0.17.5, M1/M2 eager special case included. Qwen-Image's loop is
    # unchanged in shape, but 0.19's qwen-image alias loads Qwen/Qwen-Image-2512,
    # which the coefficients were not calibrated on (see the variant page).
    "0.19.1": {
        "flux1.Transformer.__call__": "06ef78be1cd4e97c",
        "flux2.Flux2Transformer.__call__": "214c37be79a602b4",
        "z_image.ZImageTransformer.__call__": "da0a464f9e29b64b",
        "qwen.QwenTransformer.__call__": "b12184fbe7e98fe8",
        "flux1.Flux1.generate_image": "879f66c3de7a0b52",
        "flux2.Flux2Klein.generate_image": "8cee664523fcb6ed",
        "flux2.Flux2Klein._predict": "bffa1bd25b24cacd",
        "z_image.ZImage.generate_image": "bd70fe34ad24c416",
        "z_image.ZImage._predict": "016c64c92ceefbdf",
        "qwen.QwenImage.generate_image": "9e6846e3f5ed09bb",
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
        # The summary test above already fails once for an unknown version; ten
        # more failures would only repeat it.
        pytest.skip(
            f"mflux {installed} unknown; see test_installed_mflux_version_has_verified_forward_fingerprints"
        )
    fresh = _installed_fingerprints()[label]
    assert fresh == KNOWN[installed][label], (
        f"{label} changed in mflux {installed} (was {KNOWN[installed][label]}, now {fresh}); diff it against the "
        f"copy that feeds on it and re-verify before updating the pin"
    )
