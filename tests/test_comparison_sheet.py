"""Contact sheet: every preview frame, in order, labelled, skipped steps bordered."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import _comparison_sheet as sh  # noqa: E402

COLORS = [(200, 30, 30), (30, 200, 30), (30, 30, 200), (200, 200, 30), (30, 200, 200), (200, 30, 200)]


def _frames(tmp_path: Path, n: int, size: tuple[int, int] = (96, 128)) -> list[Path]:
    paths = []
    for i in range(n):
        p = tmp_path / f"step_step{i:02d}.png"
        Image.new("RGB", size, COLORS[i % len(COLORS)]).save(p)
        paths.append(p)
    return paths


def _tile_size(thumb_w: int, thumb_h: int) -> tuple[int, int]:
    return thumb_w + 2 * sh.TILE_PAD, thumb_h + sh.LABEL_H + 2 * sh.TILE_PAD


def test_grid_geometry(tmp_path: Path) -> None:
    """Bug: rows/cols or padding computed wrong, so tiles overlap or the sheet is cropped."""
    sheet = sh.build_contact_sheet(_frames(tmp_path, 12), ["computed"] * 12, cols=5, thumb_width=48)
    tile_w, tile_h = _tile_size(48, 64)
    assert sheet.size == (5 * tile_w, 3 * tile_h)


def test_each_frame_lands_in_its_own_tile_in_order(tmp_path: Path) -> None:
    """Bug: every frame pasted into tile 0, or frames placed out of order."""
    n = 6
    sheet = sh.build_contact_sheet(_frames(tmp_path, n), ["computed"] * n, cols=4, thumb_width=48).convert(
        "RGB"
    )
    tile_w, tile_h = _tile_size(48, 64)
    for i in range(n):
        cx = (i % 4) * tile_w + sh.TILE_PAD + 24
        cy = (i // 4) * tile_h + sh.TILE_PAD + 32
        assert sheet.getpixel((cx, cy)) == COLORS[i]


def test_skipped_tiles_get_the_border_and_computed_do_not(tmp_path: Path) -> None:
    """Bug: skipped steps are not marked, or every tile is."""
    sheet = sh.build_contact_sheet(
        _frames(tmp_path, 2), ["computed", "skipped"], cols=2, thumb_width=48
    ).convert("RGB")
    tile_w, _ = _tile_size(48, 64)
    assert sheet.getpixel((tile_w + 1, 1)) == sh.SKIPPED_BORDER
    assert sheet.getpixel((1, 1)) != sh.SKIPPED_BORDER


def test_count_mismatch_and_empty_input_raise(tmp_path: Path) -> None:
    """Bug: a missing frame silently shifts every label after it."""
    with pytest.raises(ValueError, match="3 frames"):
        sh.build_contact_sheet(_frames(tmp_path, 3), ["computed"] * 2)
    with pytest.raises(ValueError):
        sh.build_contact_sheet([], [])


def test_frame_paths_sort_numerically_and_ignore_other_files(tmp_path: Path) -> None:
    """Bug: lexical order (step10 before step2) or a stray final.png counted as a frame."""
    for i in (10, 2, 0):
        Image.new("RGB", (4, 4)).save(tmp_path / f"step_step{i:02d}.png")
    Image.new("RGB", (4, 4)).save(tmp_path / "final.png")
    assert [p.name for p in sh.frame_paths(tmp_path)] == [
        "step_step00.png",
        "step_step02.png",
        "step_step10.png",
    ]


def test_a_frame_of_another_size_raises(tmp_path: Path) -> None:
    """Bug: a frame with a different size is silently stretched instead of rejected."""
    frames = _frames(tmp_path, 2)  # (96, 128)
    wrong_size = tmp_path / "step_step02.png"
    Image.new("RGB", (128, 96), COLORS[2]).save(wrong_size)  # different dimensions
    frames.append(wrong_size)
    with pytest.raises(ValueError, match="expected"):
        sh.build_contact_sheet(frames, ["computed"] * 3, cols=3, thumb_width=48)


def test_unknown_step_kind_raises(tmp_path: Path) -> None:
    """Bug: an unknown step kind silently renders as computed instead of being rejected."""
    frames = _frames(tmp_path, 2)
    with pytest.raises(ValueError, match="skiped"):
        sh.build_contact_sheet(frames, ["computed", "skiped"], cols=2, thumb_width=48)
