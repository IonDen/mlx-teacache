"""flux1-krea-dev detection: Krea's aliases are ["krea-dev", "dev-krea"], neither
equal to "dev", so the two FLUX.1 detectors stay disjoint in both directions."""

from types import SimpleNamespace

from mlx_teacache.variants.flux1_dev import detect as dev_detect
from mlx_teacache.variants.flux1_krea_dev import detect as krea_detect


def _stub(*aliases: str):
    return SimpleNamespace(model_config=SimpleNamespace(aliases=list(aliases)))


def test_matches_krea_aliases() -> None:
    # bug caught: matching on a substring of "dev" instead of the krea alias
    assert krea_detect.matches(_stub("krea-dev", "dev-krea")) is True


def test_does_not_match_dev_or_schnell() -> None:
    assert krea_detect.matches(_stub("dev")) is False
    assert krea_detect.matches(_stub("schnell")) is False


def test_dev_detector_does_not_match_krea() -> None:
    # bug caught: widening flux1_dev's detector to a substring test
    assert dev_detect.matches(_stub("krea-dev", "dev-krea")) is False


def test_rejects_objects_without_model_config() -> None:
    assert krea_detect.matches(object()) is False
