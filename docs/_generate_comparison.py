"""Render the number blocks of COMPARISON.md and docs/comparison/<slug>.md from _artifacts/comparison/report.json.

Prose outside the markers is hand-written. Blocks sit between
``<!-- COMPARISON:<key> START -->`` and ``<!-- COMPARISON:<key> END -->``.

    python docs/_generate_comparison.py --write   # update the pages in place
    python docs/_generate_comparison.py --check   # exit 1 if a page differs from the report
"""

import argparse
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
REPORT = REPO / "_artifacts" / "comparison" / "report.json"
MAIN_PAGE = REPO / "COMPARISON.md"
SUB_DIR = REPO / "docs" / "comparison"
GIB = 1024**3
_MARKER = re.compile(
    r"<!-- COMPARISON:(?P<key>[\w:.-]+) START -->\n.*?<!-- COMPARISON:(?P=key) END -->", re.DOTALL
)


def _s(x: float) -> str:
    return f"{x:.1f} s"


def _g(b: float) -> str:
    return f"{b / GIB:.1f} GiB"


def _x(x: float) -> str:
    return f"{x:.2f}×"


def _summary(slug: str, v: dict[str, Any]) -> str:
    a, b = v["a"], v["b"]
    return "\n".join(
        [
            "|  | A: TeaCache off | B: TeaCache on |",
            "|---|---|---|",
            f"| Image | ![A: TeaCache off]({v['images']['a']}) | ![B: TeaCache on]({v['images']['b']}) |",
            f"| Generation | {_s(a['generation_seconds'])} · peak {_g(a['mlx_peak_generation_bytes'])} | "
            f"{_s(b['generation_seconds'])} · peak {_g(b['mlx_peak_generation_bytes'])} · "
            f"{b['skipped']} of {v['steps']} steps skipped |",
            "",
            f"On this run: {_x(v['speedup_wall'])} faster ({_x(v['speedup_preview_subtracted'])} with preview decoding "
            f"left out) · SSIM {v['ssim']:.2f} · multi-run measurement: [bench report]({v['bench_report']}) · "
            f"[More details →](docs/comparison/{slug}.md)",
        ]
    )


def _sheets(slug: str, v: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"**A: TeaCache off**, every step in order\n\n![Steps, TeaCache off](../../{v['images']['steps_a']})",
            "",
            f"**B: TeaCache on**, skipped steps framed in orange\n\n![Steps, TeaCache on](../../{v['images']['steps_b']})",
        ]
    )


def _release_base(version: str) -> str:
    """The release a rendered page should credit. An editable-install dev version such as
    "0.11.2.dev14+gbcaac6f1c.d20260925" names the *next*, unreleased patch, so the base to show is the
    previous one (0.11.2.devN -> 0.11.1). A plain release version is shown as recorded."""
    base = version.split(".dev")[0]
    if base == version:
        return version
    major, minor, patch = (int(part) for part in base.split("."))
    return f"{major}.{minor}.{patch - 1}"


def library_line(provenance: dict[str, Any]) -> str:
    version = _release_base(str(provenance.get("mlx_teacache_version", "")))
    sha = provenance.get("git_sha_b") or provenance.get("git_sha_a")
    return f"mlx-teacache {version} with this branch's changes (commit `{sha}`)"


def _details(slug: str, v: dict[str, Any], seed: int) -> str:
    a, b, p = v["a"], v["b"], v["provenance"]

    def med(c: dict[str, Any], kind: str) -> str:
        m = c["step_medians"].get(kind)
        return _s(m) if m is not None else "—"

    rows = [
        ("Model load (weights evaluated)", _s(a["load_seconds"]), _s(b["load_seconds"])),
        ("Prompt encoding", _s(a["encode_seconds"]), _s(b["encode_seconds"])),
        ("Generation, wall clock", _s(a["generation_seconds"]), _s(b["generation_seconds"])),
        ("Median computed step", med(a, "computed"), med(b, "computed")),
        ("Median skipped step", "—", med(b, "skipped")),
        ("Preview decoding, total", _s(a["preview_seconds_total"]), _s(b["preview_seconds_total"])),
        ("Steps computed / skipped", f"{v['steps']} / 0", f"{b['computed']} / {b['skipped']}"),
        ("Skip pattern (S = skipped)", "—", f"`{b['skip_pattern']}`"),
        ("Threshold (rel_l1)", "—", f"{b['rel_l1_thresh']:.2f}"),
        (
            "MLX peak: load / encode / generation",
            f"{_g(a['mlx_peak_load_bytes'])} / {_g(a['mlx_peak_encode_bytes'])} / {_g(a['mlx_peak_generation_bytes'])}",
            f"{_g(b['mlx_peak_load_bytes'])} / {_g(b['mlx_peak_encode_bytes'])} / {_g(b['mlx_peak_generation_bytes'])}",
        ),
        ("MLX active + cache, peak", _g(a["peak_resident_bytes"]), _g(b["peak_resident_bytes"])),
        ("Process footprint (macOS), peak", _g(a["peak_footprint_bytes"]), _g(b["peak_footprint_bytes"])),
    ]
    table = ["|  | A: TeaCache off | B: TeaCache on |", "|---|---|---|"] + [
        f"| {r} | {x} | {y} |" for r, x, y in rows
    ]
    footer = (
        f"\n\nSpeedup on this run: {_x(v['speedup_wall'])} wall clock, {_x(v['speedup_preview_subtracted'])} with "
        f"preview decoding left out, {_x(v['speedup_steady'])} per step after the first. SSIM of B against A: "
        f"{v['ssim']:.3f}, measured on the lossless outputs.\n\nRecipe: {v['steps']} steps, guidance {v['guidance']}, "
        f"q{v['quantize']}, {v['width']}×{v['height']}, seed {seed}"
        f"{', text encoder freed once the prompt is encoded' if v['free_encoders'] else ''}. Checkpoint "
        f"`{v['checkpoint']}`; preview decoder `{v['decoder']}`. {library_line(p)}, "
        f"mflux {p.get('version_mflux')}, MLX {p.get('version_mlx')}, mlx-taef {p.get('version_mlx_taef')}. "
        f"Multi-run measurement of this model: [bench report](../../{v['bench_report']})."
    )
    return "\n".join(table) + footer


def _machine(report: dict[str, Any]) -> str:
    h = report["hardware"]
    return f"{h['chip']}, {h['ram_gb']} GB unified memory, {h['os']}, Python {h['python']}."


def render_blocks(report: dict[str, Any]) -> dict[str, str]:
    blocks = {"machine": _machine(report)}
    seed = report["seed"]
    for slug, v in report["variants"].items():
        blocks[f"{slug}:summary"] = _summary(slug, v)
        blocks[f"{slug}:sheets"] = _sheets(slug, v)
        blocks[f"{slug}:details"] = _details(slug, v, seed)
    return blocks


def page_blocks(slug: str | None, slugs: Sequence[str]) -> set[str]:
    if slug is None:
        return {"machine"} | {f"{s}:summary" for s in slugs}
    return {f"{slug}:sheets", f"{slug}:details"}


def splice(text: str, blocks: dict[str, str], *, required: set[str]) -> str:
    present = {m.group("key") for m in _MARKER.finditer(text)}
    missing = required - present
    if missing:
        raise ValueError(f"missing markers: {sorted(missing)}")
    orphans = present - set(blocks)
    if orphans:
        raise ValueError(f"orphan markers with no block: {sorted(orphans)}")

    def _replace(m: re.Match[str]) -> str:
        key = m.group("key")
        return f"<!-- COMPARISON:{key} START -->\n{blocks[key]}\n<!-- COMPARISON:{key} END -->"

    return _MARKER.sub(_replace, text)


def _pages(report: dict[str, Any]) -> list[tuple[Path, set[str]]]:
    slugs = list(report["variants"])
    return [(MAIN_PAGE, page_blocks(None, slugs))] + [
        (SUB_DIR / f"{s}.md", page_blocks(s, slugs)) for s in slugs
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = ap.parse_args()
    report = json.loads(REPORT.read_text())
    blocks = render_blocks(report)
    drift = []
    for page, required in _pages(report):
        original = page.read_text()
        updated = splice(original, blocks, required=required)
        if updated != original:
            drift.append(page)
            if args.write:
                page.write_text(updated)
    if args.check and drift:
        print("out of date: " + ", ".join(str(p.relative_to(REPO)) for p in drift))
        sys.exit(1)


if __name__ == "__main__":
    main()
