"""match_variant: the variant apply_teacache would pick, decided from an mflux ModelConfig and the pipeline class
before any weights load."""

import subprocess
import sys
import types

import pytest

from mlx_teacache import TeaCacheValueError, VariantInfo, match_variant
from mlx_teacache.variants import _REGISTRY


def _mflux_class(name: str) -> type:
    return type(name, (), {"__module__": f"mflux.models.fake.{name.lower()}"})


def _config(*aliases: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(aliases=list(aliases), model_name=None)


def _apply_dispatch(model_config: object, cls: type) -> str | None:
    """apply_teacache's detection (api.py): the first registry entry whose matches(flux) is True."""
    flux = cls()
    flux.model_config = model_config
    return next((vid for vid, entry in _REGISTRY.items() if entry["matches"](flux)), None)


def _info(variant_id: str, display_name: str, default_thresh: float | None, *names: str) -> VariantInfo:
    # model_name is None in _config, so every variant, Qwen-Image included, reports calibrated=True here.
    return VariantInfo(
        variant_id=variant_id,
        display_name=display_name,
        default_thresh=default_thresh,
        model_names=names,
        calibrated=True,
    )


@pytest.mark.parametrize(
    ("cls_name", "aliases", "expected"),
    [
        ("Flux1", ("dev",), _info("flux1-dev", "FLUX.1 dev", 0.20, "dev")),
        ("Flux1", ("krea-dev", "dev-krea"), _info("flux1-krea-dev", "FLUX.1 Krea [dev]", 0.30, "krea-dev")),
        ("Flux1", ("schnell",), _info("flux1-schnell", "FLUX.1 schnell", 0.20, "schnell")),
        (
            "Flux2Klein",
            ("flux2-klein-4b",),
            _info("flux2-klein-4b", "FLUX.2 Klein 4B", None, "flux2-klein-4b"),
        ),
        (
            "Flux2Klein",
            ("flux2-klein-9b",),
            _info("flux2-klein-9b", "FLUX.2 Klein 9B", None, "flux2-klein-9b"),
        ),
        (
            "Flux2Klein",
            ("flux2-klein-base-4b",),
            _info("flux2-klein-base-4b", "FLUX.2 Klein base 4B", 0.17, "flux2-klein-base-4b"),
        ),
        (
            "Flux2Klein",
            ("flux2-klein-base-9b",),
            _info("flux2-klein-base-9b", "FLUX.2 Klein base 9B", 0.17, "flux2-klein-base-9b"),
        ),
        ("QwenImage", ("qwen-image", "qwen"), _info("qwen-image", "Qwen-Image", 0.30, "qwen-image", "qwen")),
        ("ZImage", ("z-image", "zimage"), _info("z-image-base", "Z-Image base", 0.12, "z-image", "zimage")),
    ],
)
def test_supported_model_reports_its_variant_info(cls_name: str, aliases: tuple[str, ...], expected) -> None:
    """Bug: VariantInfo filled from the wrong field (display name = id, the 0.20 fallback for a distilled model,
    another variant's model names, calibrated False for a config that names no checkpoint)."""
    assert match_variant(_config(*aliases), _mflux_class(cls_name)) == expected


@pytest.mark.parametrize(
    ("model_name", "base_model", "calibrated"),
    [
        ("Qwen/Qwen-Image", None, True),
        ("Qwen/Qwen-Image-2512", None, False),
        ("/models/qwen-image-q4-mirror", "Qwen/Qwen-Image", True),
        ("someorg/qwen-finetune", "Qwen/Qwen-Image-2512", False),
    ],
)
def test_qwen_calibrated_follows_the_checkpoint_the_config_names(
    model_name: str, base_model: str | None, calibrated: bool
) -> None:
    """Bug: calibrated is hard-wired True (or read from the wrong META key), so a launcher cannot warn before loading
    Qwen-Image-2512, which mflux 0.19+ gives for the qwen-image alias, though apply_teacache warns after loading."""
    config = types.SimpleNamespace(
        aliases=["qwen-image", "qwen"], model_name=model_name, base_model=base_model
    )
    info = match_variant(config, _mflux_class("QwenImage"))
    assert info is not None and info.variant_id == "qwen-image"
    assert info.calibrated is calibrated


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [({"zero_cond_t": True}, None), ({"zero_cond_t": False}, "qwen-image"), ({}, "qwen-image")],
    ids=["zero_cond_t-on", "zero_cond_t-off", "no-overrides"],
)
def test_qwen_config_with_zero_cond_t_is_refused_like_apply_teacache_refuses_it(
    overrides: dict[str, bool], expected: str | None
) -> None:
    """Bug: the zero_cond_t refusal (mflux 0.22's Qwen-Image-Edit-2511 transformer, which TeaCache's Qwen forward
    does not implement) is checked only after loading, so match_variant says yes to a config apply_teacache then
    refuses; or the check reads the key's presence instead of its value and refuses text-to-image configs."""
    config = types.SimpleNamespace(
        aliases=["qwen-image", "qwen"], model_name=None, transformer_overrides=overrides
    )
    cls = _mflux_class("QwenImage")
    info = match_variant(config, cls)
    assert (None if info is None else info.variant_id) == expected == _apply_dispatch(config, cls)


def test_a_variant_without_a_checkpoint_check_is_always_calibrated() -> None:
    """Bug: calibrated is computed for every variant by comparing META's hf_model_id with model_name, so the library
    reports a Z-Image finetune as uncalibrated; for Z-Image that warning belongs
    to the caller (the mflux plugin warns on a custom checkpoint itself)."""
    config = types.SimpleNamespace(aliases=["z-image", "zimage"], model_name="someorg/zimage-finetune")
    info = match_variant(config, _mflux_class("ZImage"))
    assert info is not None and info.calibrated is True


def test_variant_info_is_keyword_only_and_immutable() -> None:
    """Bug: VariantInfo accepts positional arguments (a field inserted later, such as calibrated, silently shifts
    every positional caller) or can be mutated after match_variant returns it."""
    with pytest.raises(TypeError):
        VariantInfo("z-image-base", "Z-Image base", 0.12, ("z-image",), True)  # type: ignore[misc]
    info = _info("z-image-base", "Z-Image base", 0.12, "z-image")
    with pytest.raises(AttributeError):
        info.calibrated = False  # type: ignore[misc]


_GRID_CLASS_NAMES = [
    "Flux1",
    "Flux2Klein",
    "ZImage",
    "QwenImage",
    "Flux1Concept",
    "Flux1InContextDev",
    "Flux1Redux",
    "Flux2KleinEdit",
    "QwenImageEdit",
]
_GRID_CONFIGS = [
    _config("dev"),
    _config("krea-dev", "dev-krea"),
    _config("schnell"),
    _config("flux2-klein-4b"),
    _config("flux2-klein-9b"),
    _config("flux2-klein-base-4b"),
    _config("flux2-klein-base-9b"),
    _config("qwen-image", "qwen"),
    _config("z-image", "zimage"),
    _config("z-image-turbo", "zimage-turbo"),
    _config("dev-kontext"),
    _config(),
    None,
    types.SimpleNamespace(model_name="no-aliases"),
]


def test_same_answer_as_apply_teacaches_detection_for_every_class_and_config() -> None:
    """Bug: match_variant drops the pipeline-class guard or reads configs its own way, so it says yes to a pipeline
    apply_teacache refuses (or no to one it patches)."""
    classes = [_mflux_class(n) for n in _GRID_CLASS_NAMES] + [
        type("MyWrapper", (), {"__module__": "my_project.x"})
    ]
    mismatches, reached = [], set()
    for cls in classes:
        for cfg in _GRID_CONFIGS:
            info = match_variant(cfg, cls)
            got = None if info is None else info.variant_id
            want = _apply_dispatch(cfg, cls)
            reached.add(want)
            if got != want:
                mismatches.append((cls.__name__, getattr(cfg, "aliases", None), got, want))
    assert mismatches == []
    assert reached == set(_REGISTRY) | {None}


@pytest.mark.parametrize(
    ("cls", "aliases", "first"),
    [
        (_mflux_class("Flux1"), ("dev", "schnell"), "flux1-dev"),
        (type("MyModel", (), {"__module__": "my_project.models"}), ("z-image", "qwen"), "qwen-image"),
    ],
)
def test_a_config_two_detectors_accept_resolves_in_registry_order(
    cls: type, aliases: tuple[str, ...], first: str
) -> None:
    """Bug: match_variant walks the variants in a different order from apply_teacache, so a config two detectors
    accept gets one answer before loading and another after."""
    info = match_variant(_config(*aliases), cls)
    assert info is not None
    assert info.variant_id == first == _apply_dispatch(_config(*aliases), cls)


def test_an_instance_instead_of_a_class_is_a_clear_error() -> None:
    """Bug: passing the loaded model (or any instance) fails with an AttributeError deep inside detection."""
    with pytest.raises(TeaCacheValueError, match="must be a class"):
        match_variant(_config("z-image"), _mflux_class("ZImage")())


def test_support_helpers_import_no_mflux() -> None:
    """Bug: the support helpers (or anything they import) pull in mflux, so a launcher asking before load pays for
    mflux's import, or fails without it. Runs in the normal environment, mflux installed or not, and checks that
    nothing named mflux was imported after both calls."""
    code = (
        "import sys, types\n"
        "from mlx_teacache import check_step_window, match_variant\n"
        "cls = type('ZImage', (), {'__module__': 'mflux.models.z_image.variants.z_image'})\n"
        "info = match_variant(types.SimpleNamespace(aliases=['z-image']), cls)\n"
        "check_step_window(50)\n"
        "assert 'mflux' not in sys.modules, 'mflux was imported'\n"
        "assert not [m for m in sys.modules if m.startswith('mflux.')], 'an mflux submodule was imported'\n"
        "print(info.variant_id)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "z-image-base"


def test_config_matches_rules() -> None:
    """Bug: config_matches drops one of its guards: accepts a foreign mflux class, crashes on a missing model config,
    or matches on something other than the config's aliases."""
    from mlx_teacache.variants._pipeline_class import config_matches

    allowed = frozenset({"ZImage"})
    names = ("z-image", "zimage")
    assert config_matches(_config("z-image"), _mflux_class("ZImage"), allowed, names) is True
    assert config_matches(_config("z-image"), _mflux_class("ZImageEdit"), allowed, names) is False
    assert config_matches(None, _mflux_class("ZImage"), allowed, names) is False
    assert config_matches(_config("dev"), _mflux_class("ZImage"), allowed, names) is False
