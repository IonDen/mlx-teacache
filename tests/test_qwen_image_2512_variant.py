"""Qwen-Image-2512 is its own variant with its own calibration. mflux 0.19+ gives every Qwen-Image text-to-image
config the 2512 aliases, including the original checkpoint loaded by its repo id, so the two Qwen variants are told
apart by the checkpoint the config names: 2512 aliases mean 2512 unless model_name is the original Qwen/Qwen-Image.
Weight-free: SimpleNamespace configs, the real QwenImage class, duck-typed fakes. Mflux lane via tests/_lanes.py."""

import importlib
from types import SimpleNamespace

import pytest

from mlx_teacache import apply_teacache, match_variant
from mlx_teacache.errors import IncompatibleModelError, TeaCacheUncalibratedCheckpointWarning
from tests._fakes import FaithfulCallbackRegistry
from tests._variant_fakes import fake_for_variant

_ALIASES_2512 = ["qwen-image", "qwen", "qwen-image-2512", "qwen-2512"]  # mflux 0.19+ qwen-image config
_ALIASES_018 = ["qwen-image", "qwen"]  # mflux 0.18 qwen-image config

# Signal A of scripts/_calibration_qwen_2512.json (the plan's recorded values, not read back from the config).
_COEFFS_2512 = (-13.200421839759947, 4.851407025105924, 2.4891133197469846, 0.2453630032662089, 0.0)
_RANGE_2512 = (0.08094493952514002, 0.4705715918534816)
_COEFFS_ORIGINAL = (-12.954226906135869, 8.883805167578382, -0.9363839862290331, 1.4538816050570036, 0.0)


def _qwen_image_class() -> type:
    from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage

    return QwenImage


def _config(
    aliases: list[str],
    model_name: str | None,
    base_model: str | None = None,
    transformer_overrides: dict[str, bool] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        aliases=aliases,
        model_name=model_name,
        base_model=base_model,
        transformer_overrides=transformer_overrides or {},
    )


def _loaded(model_config: SimpleNamespace, transformer: object | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=model_config,
        transformer=transformer or SimpleNamespace(name="real-qwen-transformer"),
        callbacks=FaithfulCallbackRegistry(),
        generate_image=lambda **kw: "image",
    )


def _gate_handle(flux: SimpleNamespace) -> object:
    return next(cb for cb in flux.callbacks.before_loop if hasattr(cb, "_handle"))._handle


def test_mflux_019_qwen_image_alias_picks_the_2512_variant() -> None:
    """Bug: the 2512 config (what `--model qwen-image` loads on mflux 0.19+) still matches the original variant and
    runs the original checkpoint's fit, or the new variant reports itself uncalibrated on its own checkpoint, or it
    lists names a launcher would offer for it that mflux resolves to the original's aliases."""
    info = match_variant(_config(_ALIASES_2512, "Qwen/Qwen-Image-2512"), _qwen_image_class())
    assert info is not None
    assert (info.variant_id, info.calibrated, info.default_thresh) == ("qwen-image-2512", True, 0.25)
    assert info.model_names == ("qwen-image-2512", "qwen-2512")


def test_mflux_018_qwen_image_alias_keeps_the_original_variant() -> None:
    """Bug: the 2512 detector keys on the shared qwen-image alias and steals the original checkpoint on mflux 0.18."""
    info = match_variant(_config(_ALIASES_018, "Qwen/Qwen-Image"), _qwen_image_class())
    assert info is not None
    assert (info.variant_id, info.calibrated) == ("qwen-image", True)


def test_original_checkpoint_by_repo_id_on_mflux_019_stays_on_the_original_variant() -> None:
    """Bug: the split keys on aliases alone. mflux 0.19+ resolves `--model Qwen/Qwen-Image` as a mirror of the
    qwen-image config (all four 2512 aliases, base_model Qwen/Qwen-Image-2512), so an alias-only rule hands the
    original checkpoint the 2512 fit and, through base_model, calls it calibrated, with no warning."""
    config = _config(_ALIASES_2512, "Qwen/Qwen-Image", base_model="Qwen/Qwen-Image-2512")
    info = match_variant(config, _qwen_image_class())
    assert info is not None
    assert (info.variant_id, info.calibrated) == ("qwen-image", True)
    # filterwarnings = error: an uncalibrated-checkpoint warning here fails the test.
    handle = apply_teacache(_loaded(config))
    try:
        assert handle.variant_id == "qwen-image"
        assert handle.coefficients == _COEFFS_ORIGINAL
    finally:
        handle.restore()


def test_original_checkpoint_by_a_differently_cased_repo_id_stays_on_the_original_variant() -> None:
    """Bug: the original checkpoint is identified by a case-sensitive match. Hugging Face resolves repo ids
    case-insensitively and mflux 0.23 accepts `--model qwen/qwen-image` (model_name 'qwen/qwen-image', the 2512
    aliases, base_model Qwen/Qwen-Image-2512), so an exact match hands the original weights the 2512 fit, reported
    calibrated, with no warning."""
    config = _config(_ALIASES_2512, "qwen/qwen-image", base_model="Qwen/Qwen-Image-2512")
    info = match_variant(config, _qwen_image_class())
    assert info is not None
    assert (info.variant_id, info.calibrated) == ("qwen-image", True)
    # filterwarnings = error: an uncalibrated-checkpoint warning here fails the test.
    handle = apply_teacache(_loaded(config))
    try:
        assert handle.variant_id == "qwen-image"
        assert handle.coefficients == _COEFFS_ORIGINAL
    finally:
        handle.restore()


def test_prequantized_2512_copy_counts_as_calibrated() -> None:
    """Bug: a pre-quantized 2512 copy (model_name = the copy, base_model = the 2512 repo, as mflux records it for
    `--base-model qwen-image`) is sent to the original variant or reported uncalibrated."""
    config = _config(
        _ALIASES_2512, "mflux-community/qwen-image-2512-mflux-q4", base_model="Qwen/Qwen-Image-2512"
    )
    info = match_variant(config, _qwen_image_class())
    assert info is not None
    assert (info.variant_id, info.calibrated) == ("qwen-image-2512", True)


def test_edit_2511_and_qwen_21_match_neither_qwen_variant() -> None:
    """Bug: the 2512 detector drops the zero_cond_t refusal (Edit-2511's config) or keys on a name Qwen-Image 2.1
    shares, so a model the gated forward does not implement is reported as supported."""
    edit_2511 = _config(_ALIASES_2512, "Qwen/Qwen-Image-2512", transformer_overrides={"zero_cond_t": True})
    qwen_21 = _config(["qwen-image-2.1", "qwen-2.1"], "Qwen/Qwen-Image-2.1")
    assert match_variant(edit_2511, _qwen_image_class()) is None
    assert match_variant(qwen_21, _qwen_image_class()) is None


def test_a_2512_model_whose_built_transformer_has_zero_cond_t_is_refused() -> None:
    """Bug: the 2512 detector's post-load matches() reads only the config, so a transformer built (or swapped in)
    with zero_cond_t is patched with a forward that does not implement it."""
    flux = _loaded(
        _config(_ALIASES_2512, "Qwen/Qwen-Image-2512"), transformer=SimpleNamespace(zero_cond_t=True)
    )
    with pytest.raises(IncompatibleModelError):
        apply_teacache(flux)


_CONFIGS = [
    (_config(_ALIASES_2512, "Qwen/Qwen-Image-2512"), "qwen-image-2512"),
    (_config(_ALIASES_2512, None), "qwen-image-2512"),
    (_config(_ALIASES_2512, "someone/qwen-finetune"), "qwen-image-2512"),
    (
        _config(_ALIASES_2512, "mflux-community/qwen-image-2512-mflux-q4", "Qwen/Qwen-Image-2512"),
        "qwen-image-2512",
    ),
    (_config(["qwen-image-2512"], "Qwen/Qwen-Image-2512"), "qwen-image-2512"),
    (_config(_ALIASES_2512, "Qwen/Qwen-Image", "Qwen/Qwen-Image-2512"), "qwen-image"),
    (_config(_ALIASES_2512, "qwen/qwen-image", "Qwen/Qwen-Image-2512"), "qwen-image"),
    (_config(_ALIASES_018, "Qwen/Qwen-Image"), "qwen-image"),
    (_config(_ALIASES_018, None), "qwen-image"),
    (_config(_ALIASES_018, "someone/qwen-finetune"), "qwen-image"),
    (_config(_ALIASES_2512, "Qwen/Qwen-Image-2512", transformer_overrides={"zero_cond_t": True}), None),
    (_config(_ALIASES_018, "Qwen/Qwen-Image", transformer_overrides={"zero_cond_t": True}), None),
    (_config(["qwen-image-2.1", "qwen-2.1"], "Qwen/Qwen-Image-2.1"), None),
]


@pytest.mark.parametrize(("config", "expected"), _CONFIGS)
def test_exactly_one_qwen_variant_matches_each_config(config: SimpleNamespace, expected: str | None) -> None:
    """Bug: the two Qwen detectors drift apart, so a config matches both (registry order silently picks one) or
    neither (a supported model is refused)."""
    cls = _qwen_image_class()
    matched = [
        variant_id
        for variant_id, subname in (("qwen-image", "qwen_image"), ("qwen-image-2512", "qwen_image_2512"))
        if importlib.import_module(f"mlx_teacache.variants.{subname}.detect").matches_config(config, cls)
    ]
    assert matched == ([] if expected is None else [expected])
    info = match_variant(config, cls)
    assert (None if info is None else info.variant_id) == expected


def test_2512_apply_uses_its_own_calibration() -> None:
    """Bug: the 2512 variant delegates to the original calibration (its coefficients or range), or warns on its own
    checkpoint."""
    flux = fake_for_variant("qwen-image-2512")
    handle = apply_teacache(flux)  # filterwarnings = error: any warning fails the test.
    try:
        assert handle.variant_id == "qwen-image-2512"
        assert handle.coefficients == _COEFFS_2512
        assert handle.rel_l1_thresh == 0.25
        assert handle.provenance.revision == "in-repo-2026-10-20-origin-signalA-768-50-2512"
        assert _gate_handle(flux).calibrated_range == _RANGE_2512
    finally:
        handle.restore()


def test_caller_coefficients_on_2512_get_user_provenance_and_no_clamp() -> None:
    """Bug: the 2512 range clamps a caller's own polynomial, or a caller fit is labelled with the built-in
    provenance."""
    flux = fake_for_variant("qwen-image-2512")
    handle = apply_teacache(flux, coefficients=(0.5, 1.5, 2.5, 3.5, 0.0))
    try:
        assert handle.provenance.source == "user"
        assert _gate_handle(flux).calibrated_range is None
    finally:
        handle.restore()


def test_an_unknown_qwen_image_checkpoint_still_warns_on_the_original_variant() -> None:
    """Bug: the split drops the uncalibrated-checkpoint warning for a finetune that only carries the qwen-image
    alias."""
    flux = _loaded(_config(["qwen-image"], "someone/qwen-finetune"))
    with pytest.warns(TeaCacheUncalibratedCheckpointWarning, match="someone/qwen-finetune"):
        handle = apply_teacache(flux)
    try:
        assert handle.variant_id == "qwen-image"
    finally:
        handle.restore()


def test_an_unknown_2512_checkpoint_warns_on_the_2512_variant_at_the_callers_line() -> None:
    """Bug: the 2512 apply adds a wrapper frame around apply_qwen (the warning then points into the package instead of
    the caller's line), or names the original variant and checkpoint."""
    flux = _loaded(_config(_ALIASES_2512, "someone/qwen-2512-finetune"))
    with pytest.warns(TeaCacheUncalibratedCheckpointWarning) as record:
        handle = apply_teacache(flux)
    try:
        message = str(record[0].message)
        assert "'qwen-image-2512'" in message and "'Qwen/Qwen-Image-2512'" in message
        assert record[0].filename == __file__
    finally:
        handle.restore()
