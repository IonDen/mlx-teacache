"""COMPARISON.md harness (schema 2): one tennis scene, A = TeaCache off, B = on, per-step taef previews.

Run from the py3.12 scratch venv (mflux 0.20):

    python scripts/bench_comparison.py --probe --only klein-base-9b          # memory probe (writes no chunk)
    python scripts/bench_comparison.py --only flux1-dev --max-workers 1       # one worker = one condition
    python scripts/bench_comparison.py --only flux1-dev --finalize            # SSIM + contact sheets + report
    python scripts/bench_comparison.py --only flux1-dev --smoke               # 2 steps at 256x256, throwaway

One worker subprocess per (slug, condition), one generation each. Each worker loads the model in stages and evaluates
the weights and the prompt embeddings before its timers (mflux loads lazily), stamps every step before and after the
taef preview callback, and samples memory per phase. A chunk counts only when its recipe stamp matches and its final
PNG and every preview frame exist. The v1 _artifacts/comparison_report.json is frozen (a published paper links it);
this writes _artifacts/comparison/report.json. The mlx-teacache version and git sha are provenance, not part of the
recipe stamp (an editable install changes them on every commit).
"""

import json
from pathlib import Path
from typing import Any, cast

from _comparison_recipes import Recipe
from _comparison_sheet import frame_paths
from _comparison_steps import medians_by_kind, preview_subtracted_speedup, steady_state_speedup

REPO = Path(__file__).resolve().parent.parent
REPORT_PATH = REPO / "_artifacts" / "comparison" / "report.json"
CHUNKS_DIR = REPO / "tests" / "_artifacts" / "comparison_chunks_v2"
RAW_ROOT = REPO / "tests" / "_artifacts" / "comparison_raw"
PROBE_DIR = REPO / "tests" / "_artifacts" / "comparison_probe"
SMOKE_ROOT = REPO / "tests" / "_artifacts" / "comparison_smoke"
CONDITIONS: tuple[str, ...] = ("a", "b")
WORKER_RESULT_SENTINEL = "::BENCH_RESULT::"
GIB = 1024**3


def chunk_path(chunks_dir: Path, slug: str, condition: str) -> Path:
    return chunks_dir / slug / f"{condition}.json"


def raw_dir_for(raw_root: Path, slug: str) -> Path:
    return raw_root / slug


def frames_dir_for(raw_root: Path, slug: str, condition: str) -> Path:
    return raw_root / slug / "frames" / condition


def chunk_is_complete(chunks_dir: Path, raw_root: Path, slug: str, condition: str, steps: int) -> bool:
    if not chunk_path(chunks_dir, slug, condition).exists():
        return False
    if not (raw_dir_for(raw_root, slug) / f"{condition}.png").exists():
        return False
    frames = frames_dir_for(raw_root, slug, condition)
    return frames.is_dir() and len(frame_paths(frames)) == steps


def check_chunk_stamp(chunk: dict[str, Any], expected: dict[str, object], path: Path) -> None:
    hint = f"move {path.parent} and the matching raw outputs to the Trash for a fresh measurement"
    stamp = chunk.get("stamp")
    if not isinstance(stamp, dict) or not stamp:
        raise SystemExit(f"{path}: no recipe stamp; {hint}")
    differing = sorted(k for k in set(stamp) | set(expected) if stamp.get(k) != expected.get(k))
    if differing:
        raise SystemExit(f"{path}: measured under a different recipe ({', '.join(differing)}); {hint}")


def plan_conditions(
    chunks_dir: Path, raw_root: Path, slug: str, steps: int, expected: dict[str, object], budget: int
) -> list[str]:
    pending: list[str] = []
    for condition in CONDITIONS:
        if chunk_is_complete(chunks_dir, raw_root, slug, condition, steps):
            path = chunk_path(chunks_dir, slug, condition)
            check_chunk_stamp(json.loads(path.read_text()), expected, path)
        else:
            pending.append(condition)
    return pending if budget < 0 else pending[:budget]


def load_pair(
    chunks_dir: Path, raw_root: Path, slug: str, steps: int, expected: dict[str, object]
) -> tuple[dict[str, Any], dict[str, Any]]:
    pending = [c for c in CONDITIONS if not chunk_is_complete(chunks_dir, raw_root, slug, c, steps)]
    if pending:
        raise SystemExit(f"{slug}: chunks pending {pending}")
    pair = []
    for condition in CONDITIONS:
        path = chunk_path(chunks_dir, slug, condition)
        chunk = cast(dict[str, Any], json.loads(path.read_text()))
        check_chunk_stamp(chunk, expected, path)
        pair.append(chunk)
    a, b = pair
    if (a["width"], a["height"]) != (b["width"], b["height"]):
        raise SystemExit(
            f"{slug}: A and B differ in size ({a['width']}x{a['height']} vs {b['width']}x{b['height']})"
        )
    return a, b


def parse_worker_line(stdout: str) -> dict[str, Any] | None:
    found: dict[str, Any] | None = None
    for line in stdout.splitlines():
        if line.startswith(WORKER_RESULT_SENTINEL):
            payload = cast(dict[str, Any], json.loads(line[len(WORKER_RESULT_SENTINEL) :]))
            if "aborted" in payload:
                return payload
            found = payload
    return found


_SUMMARY_KEYS = (
    "load_seconds",
    "encode_seconds",
    "generation_seconds",
    "mlx_peak_load_bytes",
    "mlx_peak_encode_bytes",
    "mlx_peak_generation_bytes",
    "frames",
    "released_encoders",
    "git_sha",
    "mlx_teacache_version",
)
_B_KEYS = ("rel_l1_thresh", "skipped", "computed", "max_consecutive_skips", "skip_pattern", "decision_kinds")


def condition_summary(result: dict[str, Any], *, is_b: bool) -> dict[str, Any]:
    out = {k: result[k] for k in _SUMMARY_KEYS}
    memory = result["memory"]
    out.update(
        peak_resident_bytes=memory["peak_resident_bytes"],
        peak_footprint_bytes=memory["peak_footprint_bytes"],
        min_host_free_pct=memory["min_host_free_pct"],
        memory_phases=memory.get("phases", {}),
    )
    out["compute_seconds"] = list(result["compute_seconds"])
    out["preview_seconds"] = list(result["preview_seconds"])
    out["preview_seconds_total"] = sum(result["preview_seconds"])
    kinds = result["decision_kinds"] if is_b else ["computed"] * len(result["compute_seconds"])
    out["step_medians"] = medians_by_kind(result["compute_seconds"], kinds)
    if is_b:
        out.update({k: result[k] for k in _B_KEYS})
    return out


def assemble_entry(
    recipe: Recipe, a: dict[str, Any], b: dict[str, Any], *, ssim: float, provenance: dict[str, str]
) -> dict[str, Any]:
    if len(a["compute_seconds"]) != len(b["compute_seconds"]):
        raise ValueError(f"{recipe.slug}: A and B recorded different step counts")
    base = f"_artifacts/comparison/{recipe.slug}"
    return {
        "variant_id": recipe.variant_id,
        "display_name": recipe.display_name,
        "checkpoint": recipe.checkpoint,
        "decoder": recipe.decoder,
        "bench_report": recipe.bench_report,
        "steps": recipe.steps,
        "guidance": recipe.guidance,
        "quantize": recipe.quantize,
        "width": a["width"],
        "height": a["height"],
        "free_encoders": recipe.free_encoders,
        "prompt_sha256": a["stamp"]["prompt_sha256"],
        "a": condition_summary(a, is_b=False),
        "b": condition_summary(b, is_b=True),
        "speedup_wall": a["generation_seconds"] / b["generation_seconds"],
        "speedup_steady": steady_state_speedup(a["compute_seconds"], b["compute_seconds"]),
        "speedup_preview_subtracted": preview_subtracted_speedup(
            a_wall=a["generation_seconds"],
            a_preview=a["preview_seconds"],
            b_wall=b["generation_seconds"],
            b_preview=b["preview_seconds"],
        ),
        "ssim": ssim,
        "provenance": dict(provenance),
        "images": {
            "a": f"{base}/a.jpg",
            "b": f"{base}/b.jpg",
            "steps_a": f"{base}/steps-a.jpg",
            "steps_b": f"{base}/steps-b.jpg",
        },
    }


def merge_entry(
    report: dict[str, Any], slug: str, entry: dict[str, Any], *, generated_at: str
) -> dict[str, Any]:
    out = dict(report)
    out["generated_at"] = generated_at
    out["variants"] = {**report.get("variants", {}), slug: entry}
    return out
