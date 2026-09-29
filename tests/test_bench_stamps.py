"""Report stamps written by the bench scripts: MLX version, code version, repo-relative paths.

``_bench_telemetry`` holds the pure helpers; ``bench_speedup`` and
``bench_klein_base_vs_distilled`` call them when they assemble a report.
No test here runs a real ``git`` process or a bench.
"""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import _bench_telemetry as bt  # noqa: E402
import bench_klein_base_vs_distilled as bk  # noqa: E402
import bench_speedup as bs  # noqa: E402


class _FakeGit:
    """Stand-in for ``subprocess.run`` that records the call and returns a canned answer."""

    def __init__(self, stdout: str = "", error: BaseException | None = None) -> None:
        self.stdout = stdout
        self.error = error
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(self, argv: list[str], **kwargs: Any) -> Any:
        self.calls.append((argv, kwargs))
        if self.error is not None:
            raise self.error
        return SimpleNamespace(stdout=self.stdout, returncode=0)


# --- mlx_version -----------------------------------------------------------


def test_mlx_version_is_the_imported_mlx_core_version() -> None:
    """Bug: stamping the dist-metadata name or a literal instead of the MLX that ran the bench."""
    assert bt.mlx_version() == mx.__version__


def test_speedup_hardware_stamp_carries_mlx_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bug: dropping ``mlx_version`` from the hardware dict makes cross-release bench numbers uncomparable."""
    monkeypatch.setattr(bs, "_git_revision", lambda cwd: {"git_commit": None, "git_dirty": None})
    monkeypatch.setattr(bs, "_mlx_teacache_version", lambda: "x")
    assert bs._detect_hardware(quantize=4)["mlx_version"] == mx.__version__


def test_klein_hardware_stamp_carries_mlx_core_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bug: the Klein report reading the ``mlx`` dist metadata instead of ``mlx.core.__version__``."""
    monkeypatch.setattr(bk, "_mlx_teacache_version", lambda: "x")
    monkeypatch.setattr("importlib.metadata.version", lambda name: "0.0.0-from-metadata")
    hw = bk._detect_hardware(quantize=4, machine_label="m", ram_gb=32)
    assert hw["mlx_version"] == mx.__version__


# --- teacache_version ------------------------------------------------------


def test_teacache_version_is_git_describe_output(tmp_path: Path) -> None:
    """Bug: reporting the install-time dist version (stale editable metadata) instead of the checkout."""
    run = _FakeGit(stdout="v0.12.0-3-gabc1234-dirty\n")
    assert bt.teacache_version(tmp_path, fallback="0.11.2.dev1", run=run) == "v0.12.0-3-gabc1234-dirty"
    argv, kwargs = run.calls[0]
    assert argv == ["git", "-C", str(tmp_path), "describe", "--tags", "--dirty", "--always"]
    assert kwargs["timeout"] > 0


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("git"),
        subprocess.CalledProcessError(128, "git"),
        subprocess.TimeoutExpired("git", 5),
    ],
)
def test_teacache_version_falls_back_when_git_fails(tmp_path: Path, error: BaseException) -> None:
    """Bug: a missing git or a non-checkout directory crashing the bench instead of using the dist version."""
    assert bt.teacache_version(tmp_path, fallback="0.12.0", run=_FakeGit(error=error)) == "0.12.0"


def test_teacache_version_falls_back_on_empty_output(tmp_path: Path) -> None:
    """Bug: an empty describe line stamped as the version."""
    assert bt.teacache_version(tmp_path, fallback="0.12.0", run=_FakeGit(stdout="\n")) == "0.12.0"


# --- repo_relative ---------------------------------------------------------


def test_repo_relative_strips_the_checkout_prefix(tmp_path: Path) -> None:
    """Bug: an absolute /Users/... path written into a report that ships in the sdist."""
    inside = tmp_path / "tests" / "_artifacts" / "bench_images" / "flux1-dev"
    assert bt.repo_relative(inside, tmp_path) == "tests/_artifacts/bench_images/flux1-dev"


def test_repo_relative_outside_the_repo_keeps_only_the_name(tmp_path: Path) -> None:
    """Bug: a directory outside the checkout leaking its absolute location."""
    root = tmp_path / "repo"
    root.mkdir()
    assert bt.repo_relative(tmp_path / "elsewhere" / "images", root) == "images"


def test_speedup_report_image_dir_field_is_not_absolute() -> None:
    """Bug: ``bench_images_dir`` written with ``str(bench_dir)`` (absolute) again."""
    value = bs._images_dir_field(bs.REPO_ROOT / "tests" / "_artifacts" / "bench_images" / "flux1-dev")
    assert not Path(value).is_absolute()
    assert value == "tests/_artifacts/bench_images/flux1-dev"


def test_klein_report_images_dir_field_is_not_absolute() -> None:
    """Bug: the Klein report's ``images_dir`` written as an absolute path."""
    images = bk.REPO_ROOT / "tests" / "_artifacts" / "klein_images" / "4b"
    report = bk._build_report(
        size="4b",
        quantize=4,
        reps=1,
        conditions=[],
        loaded={},
        ssim={},
        images_dir=images,
        hardware={},
        height=512,
        width=512,
    )
    assert not Path(report["images_dir"]).is_absolute()
    assert report["images_dir"] == "tests/_artifacts/klein_images/4b"
