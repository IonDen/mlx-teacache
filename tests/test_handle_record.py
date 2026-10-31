"""The handle as a read-only record: what apply_teacache patched, with what, and under which rules.

Pure-core: the FLUX.1 dev model is a duck-typed fake (same shape as test_untested_mflux_warning), and the FLUX.2,
Qwen-Image and Z-Image models are the duck-typed fakes from tests._variant_fakes, so nothing here imports mflux or
loads weights. Expected values are literals from the apply_teacache call or the release record.
"""

import warnings
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest

import mlx_teacache._mflux_versions as versions
import mlx_teacache.variants
from mlx_teacache import IncompatibleModelError, Provenance, TeaCacheHandle, TeaCacheStats, apply_teacache
from mlx_teacache.handle import VariantPatch
from mlx_teacache.variants.flux1_dev import config as flux1_dev_config
from mlx_teacache.variants.flux2_klein_base_4b import config as flux2_klein_base_4b_config
from mlx_teacache.variants.qwen_image import config as qwen_image_config
from mlx_teacache.variants.z_image_base import config as z_image_base_config
from tests._fakes import FaithfulCallbackRegistry
from tests._variant_fakes import fake_for_variant

_GETTER_ONLY = [
    "skip_first_n_steps",
    "skip_last_n_steps",
    "decision_rule_revision",
    "mflux_version",
]
_DEPRECATED_WRITES = ["rel_l1_thresh", "stats", "provenance", "coefficients"]


def _fake_flux1_dev() -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(aliases=["dev"]),
        transformer=SimpleNamespace(),
        callbacks=FaithfulCallbackRegistry(),
        generate_image=lambda **kw: "image",
    )


@pytest.fixture
def mflux_version(monkeypatch: pytest.MonkeyPatch) -> Callable[[str | None], list[int]]:
    """Pretend an mflux version is installed (None: no mflux metadata). Returns a call log: one entry per read.
    The autouse conftest fixture marks the untested-mflux warning as shown, so warn_if_untested_mflux reads no
    version and every logged read is apply_teacache's own."""

    def _set(value: str | None) -> list[int]:
        calls: list[int] = []

        def _installed() -> str | None:
            calls.append(1)
            return value

        monkeypatch.setattr(versions, "installed_mflux_version", _installed)
        return calls

    return _set


def _new_value(name: str) -> object:
    return {
        "rel_l1_thresh": 0.5,
        "stats": TeaCacheStats(),
        "provenance": Provenance.for_user_supplied(),
        "coefficients": (0.5, 0.4, 0.3, 0.2, 0.1),
    }[name]


def test_apply_records_variant_window_coefficients_revision_and_mflux_version(mflux_version) -> None:
    """Bug: the dispatcher forgets one fact and the property returns its default."""
    mflux_version("0.24.0")
    h = apply_teacache(
        _fake_flux1_dev(),
        coefficients=(0.1, 0.2, 0.3, 0.4, 0.5),
        skip_first_n_steps=2,
        skip_last_n_steps=3,
    )
    try:
        assert h.variant_id == "flux1-dev"
        assert h.coefficients == (0.1, 0.2, 0.3, 0.4, 0.5)
        assert (h.skip_first_n_steps, h.skip_last_n_steps) == (2, 3)
        assert h.decision_rule_revision == 1
        assert h.mflux_version == "0.24.0"
    finally:
        h.restore()


def test_builtin_coefficients_are_recorded(mflux_version) -> None:
    """Bug: the integration stops writing _coefficients and the handle reports the zero placeholder."""
    mflux_version("0.24.0")
    with apply_teacache(_fake_flux1_dev()) as h:
        assert h.coefficients == flux1_dev_config.COEFFICIENTS
        assert h.coefficients != (0.0, 0.0, 0.0, 0.0, 0.0)


def _fake_model(variant_id: str) -> SimpleNamespace:
    """A duck-typed model the variant's detector matches; nothing here imports mflux."""
    if variant_id == "flux1-dev":
        return _fake_flux1_dev()
    return fake_for_variant(variant_id)


@pytest.mark.parametrize(
    ("variant_id", "builtin"),
    [
        ("flux1-dev", flux1_dev_config.COEFFICIENTS),
        ("flux2-klein-base-4b", flux2_klein_base_4b_config.COEFFICIENTS),
        ("qwen-image", qwen_image_config.COEFFICIENTS),
        ("z-image-base", z_image_base_config.COEFFICIENTS),
    ],
    ids=["flux1-dev", "flux2-klein-base-4b", "qwen-image", "z-image-base"],
)
def test_each_family_records_builtin_coefficients(
    mflux_version, variant_id: str, builtin: tuple[float, ...]
) -> None:
    """Bug: a family's integration stops writing the handle's coefficients and it reports the zero placeholder."""
    mflux_version("0.24.0")
    with apply_teacache(_fake_model(variant_id)) as h:
        assert h.variant_id == variant_id
        assert h.coefficients == builtin
        assert h.coefficients != (0.0, 0.0, 0.0, 0.0, 0.0)


def test_every_integration_that_builds_a_handle_writes_its_coefficients() -> None:
    """Bug: a variant (or a new one) leaves the handle's coefficients at the placeholder.

    A text check, the floor for the variants without a behavioural check above (FLUX.1 schnell and Krea, distilled
    FLUX.2 Klein, Klein base 9B, Qwen-Image-2512)."""
    variants_dir = Path(mlx_teacache.variants.__file__).parent
    builders = [
        path
        for path in sorted(variants_dir.glob("*/integration.py"))
        if "TeaCacheHandle(" in path.read_text(encoding="utf-8")
    ]
    assert len(builders) >= 9
    missing = [
        str(path.relative_to(variants_dir))
        for path in builders
        if "handle._coefficients = " not in path.read_text(encoding="utf-8")
    ]
    assert missing == []


@pytest.mark.parametrize("variant_id", ["flux1-dev", "flux2-klein-base-4b", "qwen-image", "z-image-base"])
def test_each_family_records_caller_supplied_coefficients(mflux_version, variant_id: str) -> None:
    """Bug: a family's integration records its built-in tuple even when the caller passed coefficients=."""
    mflux_version("0.24.0")
    with apply_teacache(_fake_model(variant_id), coefficients=(0.1, 0.2, 0.3, 0.4, 0.5)) as h:
        assert h.variant_id == variant_id
        assert h.coefficients == (0.1, 0.2, 0.3, 0.4, 0.5)


def test_mflux_version_is_none_without_mflux_metadata(mflux_version) -> None:
    """Bug: a crash or a made-up string when mflux's package metadata is missing."""
    mflux_version(None)
    with apply_teacache(_fake_flux1_dev()) as h:
        assert h.mflux_version is None


def test_mflux_version_is_not_read_for_an_unsupported_model(mflux_version) -> None:
    """Bug: the metadata is read before a variant matched, so a refused model still pays for (or fails on) it."""
    calls = mflux_version("0.24.0")
    turbo = SimpleNamespace(
        model_config=SimpleNamespace(aliases=["z-image-turbo"]),
        transformer=SimpleNamespace(),
        callbacks=FaithfulCallbackRegistry(),
        generate_image=lambda **kw: "image",
    )
    with pytest.raises(IncompatibleModelError):
        apply_teacache(turbo)
    assert calls == []


@pytest.mark.parametrize("name", _GETTER_ONLY)
def test_new_properties_refuse_assignment(mflux_version, name: str) -> None:
    """Bug: a setter added to a fact that must stay read-only."""
    mflux_version("0.24.0")
    with apply_teacache(_fake_flux1_dev()) as h:
        before = getattr(h, name)
        with pytest.raises(AttributeError):
            setattr(h, name, before)


@pytest.mark.parametrize("name", _DEPRECATED_WRITES)
def test_writing_a_deprecated_attribute_warns_at_the_caller_and_sets(mflux_version, name: str) -> None:
    """Bug: the warning points into handle.py, so a caller's filter cannot target it, or the write is dropped."""
    mflux_version("0.24.0")
    new = _new_value(name)
    with apply_teacache(_fake_flux1_dev()) as h:
        with pytest.warns(DeprecationWarning, match=name) as rec:
            setattr(h, name, new)
        assert len(rec) == 1
        assert "deprecated" in str(rec[0].message)
        assert Path(rec[0].filename).resolve() == Path(__file__).resolve()
        assert getattr(h, name) is new


@pytest.mark.parametrize("name", _DEPRECATED_WRITES)
def test_writing_under_an_error_filter_leaves_the_old_value(mflux_version, name: str) -> None:
    """Bug: set-then-warn, so a caller whose filter turns the warning into an error still gets the write."""
    mflux_version("0.24.0")
    with apply_teacache(_fake_flux1_dev(), rel_l1_thresh=0.25) as h:
        before = getattr(h, name)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(DeprecationWarning, match=name):
                setattr(h, name, _new_value(name))
        assert getattr(h, name) is before


def test_writing_coefficients_says_the_gate_is_unchanged(mflux_version) -> None:
    """Bug: the notice reads as if the write retuned the gate, which keeps the coefficients it was built with."""
    mflux_version("0.24.0")
    with apply_teacache(_fake_flux1_dev()) as h:
        with pytest.warns(DeprecationWarning) as rec:
            h.coefficients = (0.5, 0.4, 0.3, 0.2, 0.1)
        message = str(rec[0].message)
        assert "does not change what the gate uses" in message
        assert "read-only from 1.0" in message


def test_writing_variant_id_is_silent_in_this_release(mflux_version) -> None:
    """Bug: a warning here turns mflux-teacache's warnings-as-errors suite red against this release (its test
    fake writes handle.variant_id)."""
    mflux_version("0.24.0")
    with apply_teacache(_fake_flux1_dev()) as h:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            h.variant_id = "x"
        assert h.variant_id == "x"


def test_properties_survive_restore(mflux_version) -> None:
    """Bug: a property tied to live patch state, so it changes or raises once the model is restored."""
    mflux_version("0.24.0")
    h = apply_teacache(
        _fake_flux1_dev(),
        rel_l1_thresh=0.25,
        coefficients=(0.1, 0.2, 0.3, 0.4, 0.5),
        skip_first_n_steps=2,
        skip_last_n_steps=3,
    )
    names = ["variant_id", *_DEPRECATED_WRITES, *_GETTER_ONLY]
    before = {name: getattr(h, name) for name in names}
    h.restore()
    after = {name: getattr(h, name) for name in names}
    assert after == before
    assert after["coefficients"] == (0.1, 0.2, 0.3, 0.4, 0.5)
    assert (after["skip_first_n_steps"], after["skip_last_n_steps"]) == (2, 3)
    assert after["mflux_version"] == "0.24.0"


def test_a_constructor_built_handle_reports_placeholders() -> None:
    """Bug: a fact is set only by apply_teacache, so a handle built with the constructor alone (as mflux-teacache's
    test fake builds one) raises AttributeError on a property it reads."""
    h = TeaCacheHandle(
        patch=VariantPatch(),
        stats=TeaCacheStats(),
        provenance=Provenance.for_user_supplied(),
        rel_l1_thresh=0.25,
    )
    assert h.variant_id == "unknown"
    assert h.coefficients == (0.0, 0.0, 0.0, 0.0, 0.0)
    assert (h.skip_first_n_steps, h.skip_last_n_steps) == (1, 1)
    assert h.decision_rule_revision == 1
    assert h.mflux_version is None
    assert h.rel_l1_thresh == 0.25
    h.restore()
