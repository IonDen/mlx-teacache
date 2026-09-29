# tests/test_variant_package_exports.py
"""Every variant subpackage re-exports its ``META`` and ``matches``.

``from mlx_teacache.variants.<id> import META, matches`` has worked since the
variant registry shipped, so it is part of the released surface. The re-export
must also stay mflux-free: the registry imports each package's ``config`` and
``detect`` (and therefore the package ``__init__``) in the pure-core install.
"""

import importlib
import pkgutil
import subprocess
import sys

import pytest

import mlx_teacache.variants as variants_pkg

_SUBPACKAGES = [
    "flux1_dev",
    "flux1_krea_dev",
    "flux1_schnell",
    "flux2_klein_4b",
    "flux2_klein_9b",
    "flux2_klein_base_4b",
    "flux2_klein_base_9b",
    "qwen_image",
    "z_image_base",
]


def test_the_subpackage_list_is_complete() -> None:
    """Bug caught: a new variant subpackage ships without being covered by the
    re-export checks below."""
    found = sorted(name for _, name, ispkg in pkgutil.iter_modules(variants_pkg.__path__) if ispkg)
    assert found == _SUBPACKAGES


@pytest.mark.parametrize("subname", _SUBPACKAGES)
def test_package_re_exports_config_meta_and_detect_matches(subname: str) -> None:
    """Bug caught: a variant ``__init__`` drops (or re-points) the ``META`` /
    ``matches`` re-export, breaking ``from mlx_teacache.variants.<id> import META``."""
    pkg = importlib.import_module(f"mlx_teacache.variants.{subname}")
    config = importlib.import_module(f"mlx_teacache.variants.{subname}.config")
    detect = importlib.import_module(f"mlx_teacache.variants.{subname}.detect")
    assert pkg.META is config.META
    assert pkg.matches is detect.matches
    assert pkg.__all__ == ["META", "matches"]
    assert pkg.__doc__ is not None
    assert pkg.__doc__.strip() != ""
    assert "\n" not in pkg.__doc__.strip()


def test_package_re_exports_import_without_mflux() -> None:
    """Bug caught: a variant ``__init__`` re-export pulls mflux (e.g. imports
    ``integration``), so ``import mlx_teacache`` fails without the extra."""
    code = (
        "import sys; sys.modules['mflux'] = None\n"
        "import importlib\n"
        f"for name in {_SUBPACKAGES!r}:\n"
        "    pkg = importlib.import_module('mlx_teacache.variants.' + name)\n"
        "    assert pkg.META['variant_id'] and callable(pkg.matches), name\n"
        "assert not any(m == 'mflux' or m.startswith('mflux.') for m in sys.modules if sys.modules[m] is not None)\n"
        "print('OK')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK"
