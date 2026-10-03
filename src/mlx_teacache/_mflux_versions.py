"""The mflux releases this package has been verified on, and how a newer one is recognised.

Imports no mflux: the installed version comes from package metadata, so this module is
safe on an install without the [mflux] extra."""

import re
import warnings
from importlib.metadata import PackageNotFoundError, version

from mlx_teacache.errors import TeaCacheUntestedMfluxWarning

# The newest mflux release whose copied functions were fingerprinted and checked on real
# weights. The [mflux] extra allows two minor versions past it.
NEWEST_VERIFIED_MFLUX = "0.21.0"

_LEADING_RELEASE = re.compile(r"\d+(?:\.\d+)*")

_warned = False


def release_tuple(text: str) -> tuple[int, ...] | None:
    """The numeric release part of a version string, padded to three fields.

    "0.21.0", "0.21", "0.21.0rc1" and "0.21.0.dev3+gabc" all give (0, 21, 0). None when the
    string does not start with a number."""
    match = _LEADING_RELEASE.match(text.strip())
    if match is None:
        return None
    parts = tuple(int(part) for part in match.group(0).split("."))
    return parts + (0,) * (3 - len(parts))


def is_newer_than_verified(installed: str | None, newest_verified: str = NEWEST_VERIFIED_MFLUX) -> bool:
    """True when `installed` is a later release than `newest_verified`. False for None or a
    version string with no release number."""
    if installed is None:
        return False
    mine = release_tuple(installed)
    verified = release_tuple(newest_verified)
    if mine is None or verified is None:
        return False
    return mine > verified


def installed_mflux_version() -> str | None:
    """The installed mflux distribution's version, or None when mflux is not installed."""
    try:
        return version("mflux")
    except PackageNotFoundError:
        return None


def warn_if_untested_mflux() -> None:
    """Warn once per process when the installed mflux is newer than NEWEST_VERIFIED_MFLUX.

    Called by apply_teacache after a variant matched; the warning points at its caller.
    Never refuses: TeaCache still applies."""
    global _warned
    if _warned:
        return
    installed = installed_mflux_version()
    if not is_newer_than_verified(installed):
        return
    _warned = True
    warnings.warn(
        f"mflux {installed} is newer than {NEWEST_VERIFIED_MFLUX}, the newest mflux release this "
        "version of mlx-teacache was verified on. TeaCache still applies; if a generation fails or "
        "looks wrong, please report it with both versions at https://github.com/IonDen/mlx-teacache/issues",
        TeaCacheUntestedMfluxWarning,
        stacklevel=3,
    )
