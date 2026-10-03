"""Drift guard for the mflux forwards this library copies.

Three variant integrations re-walk the vanilla transformer body step by step
(FLUX.2 Klein, Z-Image, Qwen-Image) and FLUX.1 delegates to mflux's own block
helpers; all four also depend on the shape of the generation loop that calls
them. A change upstream to any of those bodies is invisible off the real-weights
parity lane, so this test pins an AST fingerprint of each one per mflux version.

On red: either a fingerprinted function moved (diff it against the copy that feeds on
it, re-verify on real weights, then record a row for that version), or the installed
mflux is an unrecorded release older than the newest row. A release newer than every
row is held to the newest row and passes while none of these functions moved. This is
a heads-up, not proof of breakage."""

import importlib
from importlib.metadata import version

from tests._mflux_surface import ast_fingerprint, drift_failures

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
    (
        "flux2.Flux2TransformerBlock.__call__",
        "mflux.models.flux2.model.flux2_transformer.transformer_block",
        "Flux2TransformerBlock",
        "__call__",
    ),
    (
        "flux2.Flux2Modulation.__call__",
        "mflux.models.flux2.model.flux2_transformer.modulation",
        "Flux2Modulation",
        "__call__",
    ),
    (
        "qwen.QwenTransformerBlock.__call__",
        "mflux.models.qwen.model.qwen_transformer.qwen_transformer_block",
        "QwenTransformerBlock",
        "__call__",
    ),
    (
        "qwen.QwenTransformerBlock._modulate",
        "mflux.models.qwen.model.qwen_transformer.qwen_transformer_block",
        "QwenTransformerBlock",
        "_modulate",
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
    # Flux2TransformerBlock, Flux2Modulation and QwenTransformerBlock (its forward and the
    # static `_modulate` that `_qwen_signal_a` calls directly) are pinned because the
    # gate-signal extractors (`_flux2_extract_mod_input`, `_qwen_signal_a`) re-implement
    # their first lines and unpack the modulation tuple by position.
    # 0.17.5, the floor: the FLUX.2 transformer forward differs from 0.18.0, which
    # gained its KV-cache path there; the copy does not take it (kv_cache is None). The
    # FLUX.2 block forward differs too (same KV-cache change); modulation and Qwen block match.
    "0.17.5": {
        "flux1.Transformer.__call__": "06ef78be1cd4e97c",
        "flux2.Flux2Transformer.__call__": "3c93bbfb5aaaf31f",
        "z_image.ZImageTransformer.__call__": "779a9ad06bfc2a65",
        "qwen.QwenTransformer.__call__": "b12184fbe7e98fe8",
        "flux2.Flux2TransformerBlock.__call__": "358009e3df2458c7",
        "flux2.Flux2Modulation.__call__": "634d78cbbe8dd7a5",
        "qwen.QwenTransformerBlock.__call__": "581446ca347138fd",
        "qwen.QwenTransformerBlock._modulate": "b84461ff22355cb1",
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
        "flux2.Flux2TransformerBlock.__call__": "ede07d21519550be",
        "flux2.Flux2Modulation.__call__": "634d78cbbe8dd7a5",
        "qwen.QwenTransformerBlock.__call__": "581446ca347138fd",
        "qwen.QwenTransformerBlock._modulate": "b84461ff22355cb1",
        "flux1.Flux1.generate_image": "09830d48dfa71077",
        "flux2.Flux2Klein.generate_image": "e5b748fe91a48d44",
        "flux2.Flux2Klein._predict": "bffa1bd25b24cacd",
        "z_image.ZImage.generate_image": "97e2e2a3d9808ac5",
        "z_image.ZImage._predict": "016c64c92ceefbdf",
        "qwen.QwenImage.generate_image": "59ba0f1448730c80",
    },
    # 0.18.1 is identical to 0.18.0 on all fourteen targets.
    "0.18.1": {
        "flux1.Transformer.__call__": "06ef78be1cd4e97c",
        "flux2.Flux2Transformer.__call__": "214c37be79a602b4",
        "z_image.ZImageTransformer.__call__": "779a9ad06bfc2a65",
        "qwen.QwenTransformer.__call__": "b12184fbe7e98fe8",
        "flux2.Flux2TransformerBlock.__call__": "ede07d21519550be",
        "flux2.Flux2Modulation.__call__": "634d78cbbe8dd7a5",
        "qwen.QwenTransformerBlock.__call__": "581446ca347138fd",
        "qwen.QwenTransformerBlock._modulate": "b84461ff22355cb1",
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
        "flux2.Flux2TransformerBlock.__call__": "ede07d21519550be",
        "flux2.Flux2Modulation.__call__": "634d78cbbe8dd7a5",
        "qwen.QwenTransformerBlock.__call__": "581446ca347138fd",
        "qwen.QwenTransformerBlock._modulate": "b84461ff22355cb1",
        "flux1.Flux1.generate_image": "879f66c3de7a0b52",
        "flux2.Flux2Klein.generate_image": "8cee664523fcb6ed",
        "flux2.Flux2Klein._predict": "bffa1bd25b24cacd",
        "z_image.ZImage.generate_image": "bd70fe34ad24c416",
        "z_image.ZImage._predict": "016c64c92ceefbdf",
        "qwen.QwenImage.generate_image": "9e6846e3f5ed09bb",
    },
    # 0.19.2: Flux2Transformer.__call__ and ZImageTransformer.__call__ gained optional
    # gradient checkpointing (each block wrapped in nn.utils.checkpoint only when
    # getattr(self, "gradient_checkpointing", False) is set by a training adapter). Off at
    # inference, so the block walk the eager copies reproduce is unchanged.
    "0.19.2": {
        "flux1.Transformer.__call__": "06ef78be1cd4e97c",
        "flux2.Flux2Transformer.__call__": "ba3f261b0a4bb536",
        "z_image.ZImageTransformer.__call__": "179208f45cc524e1",
        "qwen.QwenTransformer.__call__": "b12184fbe7e98fe8",
        "flux2.Flux2TransformerBlock.__call__": "ede07d21519550be",
        "flux2.Flux2Modulation.__call__": "634d78cbbe8dd7a5",
        "qwen.QwenTransformerBlock.__call__": "581446ca347138fd",
        "qwen.QwenTransformerBlock._modulate": "b84461ff22355cb1",
        "flux1.Flux1.generate_image": "879f66c3de7a0b52",
        "flux2.Flux2Klein.generate_image": "8cee664523fcb6ed",
        "flux2.Flux2Klein._predict": "bffa1bd25b24cacd",
        "z_image.ZImage.generate_image": "bd70fe34ad24c416",
        "z_image.ZImage._predict": "016c64c92ceefbdf",
        "qwen.QwenImage.generate_image": "9e6846e3f5ed09bb",
    },
    # 0.20.0, verified 2026-09-24 on real weights (FLUX.1-dev, FLUX.1-schnell's step-window test,
    # FLUX.1 Krea [dev], the four FLUX.2 Klein variants incl. CFG and image quality, Z-Image,
    # the mlx-taef live-preview composition; mlx 0.32.2). Qwen-Image not re-run: 0.19+/0.20
    # `qwen-image` is Qwen-Image-2512, still uncalibrated and warned about.
    # Only Flux2Klein.generate_image moved: the VAE decode after the loop now passes
    # tiling_config, which the wrapper passes through untouched. Also new in 0.20, outside
    # these fourteen targets: RopeEmbedder.__init__ evaluates Z-Image's rotary tables (the lazy
    # tables the Z-Image parity fixture materialises by hand), and call_in_loop can receive a
    # `denoised` argument, but only when a subscriber declares it; ours do not.
    "0.20.0": {
        "flux1.Transformer.__call__": "06ef78be1cd4e97c",
        "flux2.Flux2Transformer.__call__": "ba3f261b0a4bb536",
        "z_image.ZImageTransformer.__call__": "179208f45cc524e1",
        "qwen.QwenTransformer.__call__": "b12184fbe7e98fe8",
        "flux2.Flux2TransformerBlock.__call__": "ede07d21519550be",
        "flux2.Flux2Modulation.__call__": "634d78cbbe8dd7a5",
        "qwen.QwenTransformerBlock.__call__": "581446ca347138fd",
        "qwen.QwenTransformerBlock._modulate": "b84461ff22355cb1",
        "flux1.Flux1.generate_image": "879f66c3de7a0b52",
        "flux2.Flux2Klein.generate_image": "80a99e48781d55be",
        "flux2.Flux2Klein._predict": "bffa1bd25b24cacd",
        "z_image.ZImage.generate_image": "bd70fe34ad24c416",
        "z_image.ZImage._predict": "016c64c92ceefbdf",
        "qwen.QwenImage.generate_image": "9e6846e3f5ed09bb",
    },
    # 0.21.0, fingerprinted 2026-10-03 from the PyPI wheel (identical to mflux main at f63b4f0,
    # the release commit). Only the two generate_image loops moved: Z-Image and FLUX.2 Klein now
    # `del predict` right before `ctx.after_loop(latents)` so `--low-ram` can free the transformer
    # before the decode (mflux PR 802). Z-Image's generate_image also changed its return annotation to
    # GeneratedImage; that is not part of the fingerprint, and wrap_generate_image passes the value
    # through. The deleted object is the per-generation closure the
    # TeaCache `_predict` replacement returns, and nothing else in the patch holds the transformer
    # (tests/test_predict_closure_release.py), so the copied path is unchanged.
    "0.21.0": {
        "flux1.Transformer.__call__": "06ef78be1cd4e97c",
        "flux2.Flux2Transformer.__call__": "ba3f261b0a4bb536",
        "z_image.ZImageTransformer.__call__": "179208f45cc524e1",
        "qwen.QwenTransformer.__call__": "b12184fbe7e98fe8",
        "flux2.Flux2TransformerBlock.__call__": "ede07d21519550be",
        "flux2.Flux2Modulation.__call__": "634d78cbbe8dd7a5",
        "qwen.QwenTransformerBlock.__call__": "581446ca347138fd",
        "qwen.QwenTransformerBlock._modulate": "b84461ff22355cb1",
        "flux1.Flux1.generate_image": "879f66c3de7a0b52",
        "flux2.Flux2Klein.generate_image": "76e7cf2460d5e121",
        "flux2.Flux2Klein._predict": "bffa1bd25b24cacd",
        "z_image.ZImage.generate_image": "305359bd4490aea8",
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


def test_installed_mflux_forwards_match_their_verified_fingerprints() -> None:
    """Bug: a copied mflux function changes upstream and the integration keeps running a stale copy."""
    failures = drift_failures(version("mflux"), _installed_fingerprints(), KNOWN)
    assert failures == [], "\n".join(failures)
