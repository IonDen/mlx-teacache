"""The page loads the ORIGINAL Qwen-Image on mflux 0.19+, where every qwen-image alias means 2512."""

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import _comparison_models as models  # noqa: E402


def test_original_qwen_config_names_the_calibrated_checkpoint() -> None:
    """Bug: the loader uses ModelConfig.qwen_image(), which is 2512 on mflux 0.19+ (fails in the 0.20 CI job)."""
    config = models.qwen_original_config()
    assert config.model_name == "Qwen/Qwen-Image" and "qwen-image" in config.aliases


def test_original_config_counts_as_calibrated_and_2512_does_not() -> None:
    """Bug: the integration's calibration check inverted, or it reads aliases (shared by 2512)."""
    from mlx_teacache.variants.qwen_image.config import META
    from mlx_teacache.variants.qwen_image.detect import is_calibrated_checkpoint

    assert is_calibrated_checkpoint(models.qwen_original_config(), META["hf_model_id"])
    assert not is_calibrated_checkpoint(
        SimpleNamespace(model_name="Qwen/Qwen-Image-2512", base_model=None), META["hf_model_id"]
    )


def test_apply_on_the_original_config_emits_no_uncalibrated_warning() -> None:
    """Bug: the page's Qwen row would carry TeaCacheUncalibratedCheckpointWarning (filterwarnings=error)."""
    from mlx_teacache.variants.qwen_image.integration import apply
    from tests.test_qwen_checkpoint_warning import _fake_flux

    config = models.qwen_original_config()
    flux = _fake_flux(config.model_name, config.base_model)
    apply(flux, rel_l1_thresh=0.25).restore()


def test_overridden_encoder_methods_take_the_keys_the_harness_uses() -> None:
    """Bug: mflux renames a keyword, so the harness override would never be hit (cache miss at run time)."""
    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
    from mflux.models.z_image.variants.z_image import ZImage

    for fn in (Flux2Klein._encode_prompt_pair, ZImage._encode_prompts):
        params = set(inspect.signature(fn).parameters) - {"self"}
        assert params == {"prompt", "negative_prompt", "guidance"}, (fn.__qualname__, params)


def test_report_header_records_mflux_own_predict_compile_decision(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bug: the report header doesn't record mflux's own mx.compile decision for FLUX.2/Z-Image predict, so
    a reader can't separate mlx-teacache's mx.compile-avoidance speedup from mflux's baseline behavior --
    or the flag is only ever exercised on whatever this machine's own chip happens to be."""
    import bench_comparison as bc
    from mflux.utils.apple_silicon import AppleSiliconUtil

    monkeypatch.setattr(AppleSiliconUtil, "is_m1_or_m2", classmethod(lambda cls: True))
    assert bc._report_header()["mflux_compiles_on_this_chip"] is False

    monkeypatch.setattr(AppleSiliconUtil, "is_m1_or_m2", classmethod(lambda cls: False))
    assert bc._report_header()["mflux_compiles_on_this_chip"] is True


def test_report_header_machine_os_is_the_macos_marketing_version_not_the_kernel_name() -> None:
    """Bug: the "os" field reads platform.uname().system ("Darwin") instead of platform.mac_ver(), so a
    reader sees the kernel name ("Darwin 27.0.0") instead of the macOS version ("macOS 27.0")."""
    import bench_comparison as bc

    header = bc._report_header()
    assert "Darwin" not in header["hardware"]["os"]
    assert header["hardware"]["os"].startswith("macOS ")
