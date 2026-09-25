"""Contact sheet of the per-step taef previews: one tile per step, labelled, skipped steps bordered.

mlx-taef numbered mode writes ``<stem>_step{NN}<suffix>`` (``save_to=frames/"step.png"`` → ``step_step00.png``).
"""

import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

SKIPPED_BORDER = (228, 87, 46)
BACKGROUND = (250, 250, 250)
TEXT = (30, 30, 30)
TILE_PAD = 6
LABEL_H = 22
THUMB_WIDTH = 128
_STEP_RE = re.compile(r"_step(\d+)\.png$")


def frame_paths(frames_dir: Path) -> list[Path]:
    found = [(int(m.group(1)), p) for p in frames_dir.glob("*.png") if (m := _STEP_RE.search(p.name))]
    return [p for _, p in sorted(found)]


def build_contact_sheet(
    frames: Sequence[Path], kinds: Sequence[str], *, cols: int = 10, thumb_width: int = THUMB_WIDTH
) -> Any:
    from PIL import Image, ImageDraw

    if not frames:
        raise ValueError("no frames to lay out")
    if len(frames) != len(kinds):
        raise ValueError(f"{len(frames)} frames but {len(kinds)} step kinds")
    # Validate all kinds are known
    valid_kinds = {"computed", "skipped"}
    for kind in kinds:
        if kind not in valid_kinds:
            raise ValueError(f"unknown step kind: {kind}")
    with Image.open(frames[0]) as first:
        first_size = (first.width, first.height)
        thumb_h = round(first.height * thumb_width / first.width)
    # Validate all frame sizes match the first frame
    for path in frames[1:]:
        with Image.open(path) as frame:
            if (frame.width, frame.height) != first_size:
                w, h = frame.width, frame.height
                fw, fh = first_size
                raise ValueError(f"{path.name} is {w}x{h}, expected {fw}x{fh} like the first frame")
    tile_w, tile_h = thumb_width + 2 * TILE_PAD, thumb_h + LABEL_H + 2 * TILE_PAD
    rows = math.ceil(len(frames) / cols)
    sheet = Image.new("RGB", (cols * tile_w, rows * tile_h), BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    for i, (path, kind) in enumerate(zip(frames, kinds, strict=True)):
        x, y = (i % cols) * tile_w, (i // cols) * tile_h
        if kind == "skipped":
            draw.rectangle([x, y, x + tile_w - 1, y + tile_h - 1], fill=SKIPPED_BORDER)
            draw.rectangle(
                [x + TILE_PAD - 2, y + TILE_PAD - 2, x + tile_w - TILE_PAD + 1, y + tile_h - TILE_PAD + 1],
                fill=BACKGROUND,
            )
        with Image.open(path) as frame:
            sheet.paste(frame.convert("RGB").resize((thumb_width, thumb_h)), (x + TILE_PAD, y + TILE_PAD))
        label = f"step {i + 1}" + (" · skipped" if kind == "skipped" else "")
        colour = SKIPPED_BORDER if kind == "skipped" else TEXT
        draw.text((x + TILE_PAD, y + TILE_PAD + thumb_h + 4), label, fill=colour)
    return sheet
