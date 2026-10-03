"""Pure-core lane classification: which test files need mflux.

Entries are paths relative to the `tests/` directory, POSIX-style. A root-level
test is listed by its bare name; a test under `tests/variants/` is mflux-free
unless it is listed here by its relative path.
"""

from collections.abc import Iterable


def build_allowlist(entries: Iterable[str]) -> frozenset[str]:
    items = list(entries)
    duplicates = sorted({e for e in items if items.count(e) > 1})
    if duplicates:
        raise ValueError(f"duplicate mflux allowlist entries: {duplicates}")
    return frozenset(items)


MFLUX_FILES = build_allowlist(
    [
        "test_lifecycle.py",
        "test_forward_flux1.py",
        "test_forward_flux2.py",
        "test_forward_z_image_fake.py",  # imports the z-image-base integration module
        "test_cfg_branch_independence.py",  # calls flux2_cfg_forward_with_gate which lazily imports mflux
        "test_api.py",
        "test_parity_flux1.py",
        "test_parity_flux2.py",
        "test_parity_z_image.py",
        "test_parity_qwen.py",
        "test_parity_krea.py",
        "test_image_quality_flux1.py",
        "test_image_quality_flux2.py",
        "test_detect.py",  # imports mflux types for variant detection
        "test_mflux_contract_smoke.py",
        "test_mflux_forward_drift.py",  # fingerprints the real mflux forwards
        "test_mflux_drift_newer_release.py",  # runs the real drift test against the installed mflux
        "test_comparison_qwen_config.py",
        "test_untested_mflux_warning.py",  # applies a duck-typed FLUX.1 dev, which lazily imports the integration
        "test_predict_closure_release.py",  # FLUX.2 cases call a forward that imports mflux inside the function
        "variants/flux1_dev/test_integration_smoke.py",  # imports mflux and the variant integration
    ]
)


def is_mflux_file(rel_path: str) -> bool:
    """True when the test file at `rel_path` (relative to tests/, POSIX) needs mflux."""
    return rel_path in MFLUX_FILES
