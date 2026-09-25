"""COMPARISON.md harness (schema 2): one tennis scene, A = TeaCache off, B = on, per-step taef previews.

Run from the py3.12 scratch venv (mflux 0.20):

    python scripts/bench_comparison.py --probe --only klein-base-9b          # memory probe (writes no chunk)
    python scripts/bench_comparison.py --only flux1-dev --max-workers 1       # one worker = one condition
    python scripts/bench_comparison.py --only flux1-dev --finalize            # SSIM + contact sheets + report
    python scripts/bench_comparison.py --only flux1-dev --finalize --export-jpg  # also writes the showcase JPGs
    python scripts/bench_comparison.py --only flux1-dev --smoke               # 4 steps at 256x256, throwaway

One worker subprocess per (slug, condition), one generation each. Each worker loads the model in stages and evaluates
the weights and the prompt embeddings before its timers (mflux loads lazily), stamps every step before and after the
taef preview callback, and samples memory per phase. A chunk counts only when its recipe stamp matches and its final
PNG and every preview frame exist. The v1 _artifacts/comparison_report.json is frozen (a published paper links it);
this writes _artifacts/comparison/report.json. The mlx-teacache version and git sha are provenance, not part of the
recipe stamp (an editable install changes them on every commit).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import warnings
from collections.abc import Callable
from datetime import datetime, timezone
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


def reset_condition_outputs(
    raw_root: Path, slug: str, condition: str, *, trash: Path, tag: str
) -> list[Path]:
    """Move a previous attempt's frames and final PNG to the Trash (rule F) before a worker writes new ones."""
    moved: list[Path] = []
    final = raw_dir_for(raw_root, slug) / f"{condition}.png"
    frames = frames_dir_for(raw_root, slug, condition)
    for path, label in ((final, f"{slug}-{condition}-final"), (frames, f"{slug}-{condition}-frames")):
        if path.exists():
            dest = trash / f"comparison-{label}-{tag}{path.suffix}"
            shutil.move(str(path), dest)
            moved.append(dest)
    return moved


def retire_chunk(path: Path, *, trash: Path, tag: str) -> Path | None:
    """Move a stale chunk JSON out of the way before its worker respawns (rule F: Trash, never rm).

    Without this, a re-run that fails after ``image.save`` can leave an old chunk paired with a fresh image
    and fresh preview frames, so the next successful worker's chunk gets counted complete against stale
    provenance. Returns the Trash destination, or None if there was nothing to move."""
    if not path.exists():
        return None
    trash.mkdir(parents=True, exist_ok=True)
    dest = trash / f"comparison-chunk-{path.parent.name}-{path.stem}-{tag}{path.suffix}"
    shutil.move(str(path), dest)
    return dest


def soft_cap_gb(wired_cap_gb: float, cache_gb: float, working_set_bytes: int) -> float:
    """The advisory soft memory cap: wired_cap_gb + 1 GiB headroom, but never above what the device's
    working set leaves after the cache pool. ``wired_cap_gb + 1`` alone can sit above the working set
    (Z-Image: wired 24 -> soft 25, above a 24.96 GiB working set)."""
    return min(wired_cap_gb + 1, working_set_bytes / GIB - cache_gb)


def positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return value


def _now_tag() -> str:
    """Second resolution alone lets two retires in the same run (a chunk and its stale .aborted.json
    marker, or two fast test retries) collide on one Trash destination; microseconds make every tag unique."""
    return datetime.now().strftime("%Y-%m-%d-%H%M%S-%f")


def _versions() -> dict[str, str]:
    from importlib.metadata import version

    return {"mflux": version("mflux"), "mlx": version("mlx"), "mlx_taef": version("mlx-taef")}


def _git_sha() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True, check=False
    )
    return out.stdout.strip() or "unknown"


def _load_probes(slug: str) -> list[dict[str, Any]]:
    return [
        cast(dict[str, Any], json.loads(p.read_text())) for p in sorted(PROBE_DIR.glob(f"{slug}_*x*.json"))
    ]


def _run_generation(recipe: Recipe, condition: str, *, raw_root: Path, steps: int) -> dict[str, Any]:
    """One generation with every guard in place; staged load; per-phase memory. Returns the result dict."""
    import mlx.core as mx
    from _comparison_memory import (
        PeakSampler,
        SamplerError,
        host_free_pct,
        mlx_resident_sampler,
        phys_footprint_bytes,
    )
    from _comparison_models import (
        DENOISER_ATTRS,
        ENCODER_ATTRS,
        assert_prompt_cache_hit,
        evaluate_modules,
        load_model,
        precompute_prompt,
        release_text_encoders,
    )
    from _comparison_recipes import SEED, prompt_for
    from _comparison_steps import decision_kinds, register_stamped_preview, split_steps
    from mlx_taef.integrations.mflux import LivePreviewCallback

    import mlx_teacache

    reset_condition_outputs(raw_root, recipe.slug, condition, trash=Path.home() / ".Trash", tag=_now_tag())
    frames_dir = frames_dir_for(raw_root, recipe.slug, condition)
    frames_dir.mkdir(parents=True, exist_ok=True)
    out_png = raw_dir_for(raw_root, recipe.slug) / f"{condition}.png"
    prompt = prompt_for(recipe)

    sampler = PeakSampler(
        sample_resident=mlx_resident_sampler(),
        sample_footprint=phys_footprint_bytes,
        sample_host_free=host_free_pct,
    )
    sampler.start()
    failed = False
    try:
        # Load: encoders first; the denoiser waits until the encoders are gone where the recipe frees them.
        t0 = time.perf_counter()
        flux = load_model(recipe)
        evaluate_modules(flux, ENCODER_ATTRS)
        preview = LivePreviewCallback(
            flux=flux,
            variant=recipe.decoder,
            every=1,
            numbered_frames=True,
            save_to=frames_dir / "step.png",
            on_error="raise",
        )
        mx.eval(preview.model.parameters())
        load_seconds = time.perf_counter() - t0
        mlx_peak_load = int(mx.get_peak_memory())
        sampler.end_phase("load")

        # Encode, then free the encoders (where set), then evaluate the transformer and VAE. The denoiser
        # eval is materialization/quantization, not prompt encoding, so its time is folded into
        # load_seconds; encode_seconds covers only precompute_prompt + release_text_encoders. The memory
        # phase boundary stays here (mlx_peak_encode_bytes is still the peak across this whole window).
        mx.reset_peak_memory()
        t1 = time.perf_counter()
        precompute_prompt(flux, recipe, prompt)
        released = release_text_encoders(flux) if recipe.free_encoders else []
        encode_seconds = time.perf_counter() - t1
        t2 = time.perf_counter()
        evaluate_modules(flux, DENOISER_ATTRS)
        load_seconds += time.perf_counter() - t2
        mlx_peak_encode = int(mx.get_peak_memory())
        sampler.end_phase("encode")

        pre, post = register_stamped_preview(flux.callbacks.register, preview)
        handle = None
        if condition == "b":
            from mlx_teacache import apply_teacache
            from mlx_teacache.errors import TeaCacheUncalibratedCheckpointWarning

            with warnings.catch_warnings():
                warnings.simplefilter("error", TeaCacheUncalibratedCheckpointWarning)
                handle = apply_teacache(flux)

        mx.clear_cache()  # so load leftovers do not sit in the generation window
        mx.reset_peak_memory()
        gen_start = time.perf_counter()
        image = flux.generate_image(
            prompt=prompt,
            seed=SEED,
            num_inference_steps=steps,
            height=recipe.height,
            width=recipe.width,
            guidance=recipe.guidance,
        )
        mx.synchronize()
        generation_seconds = time.perf_counter() - gen_start
        mlx_peak_generation = int(mx.get_peak_memory())
        sampler.end_phase("generation")
    except BaseException:
        failed = True
        raise
    finally:
        if failed:
            # A real error is already propagating; a sampler-teardown failure on top of it must not mask it.
            try:
                sampler.stop()
            except SamplerError as exc:
                print(
                    f"{recipe.slug}/{condition}: memory sampler also failed during teardown: {exc}",
                    file=sys.stderr,
                )
        else:
            memory = sampler.stop()

    assert_prompt_cache_hit(flux, recipe)
    if len(preview.saved_paths) != steps:
        raise RuntimeError(f"expected {steps} preview frames, got {len(preview.saved_paths)}")
    compute, preview_cost = split_steps(gen_start, pre.stamps, post.stamps)
    image.save(path=str(out_png), export_json_metadata=False, overwrite=True)

    result: dict[str, Any] = {
        "condition": condition,
        "width": recipe.width,
        "height": recipe.height,
        "load_seconds": load_seconds,
        "encode_seconds": encode_seconds,
        "generation_seconds": generation_seconds,
        "compute_seconds": compute,
        "preview_seconds": preview_cost,
        "mlx_peak_load_bytes": mlx_peak_load,
        "mlx_peak_encode_bytes": mlx_peak_encode,
        "mlx_peak_generation_bytes": mlx_peak_generation,
        "memory": memory,
        "frames": len(preview.saved_paths),
        "released_encoders": released,
        "mlx_teacache_version": mlx_teacache.__version__,
    }
    if handle is not None:
        from _bench_telemetry import streak_telemetry

        kinds = decision_kinds(handle.stats.last_generation.decisions, steps)
        telemetry = streak_telemetry(handle.stats)
        result.update(
            rel_l1_thresh=handle.rel_l1_thresh,
            decision_kinds=kinds,
            skipped=kinds.count("skipped"),
            computed=kinds.count("computed"),
            max_consecutive_skips=telemetry["max_consecutive_skips"],
            skip_pattern=telemetry["skip_pattern"],
        )
        handle.restore()
    return result


def _install_guards(recipe: Recipe, label: str) -> None:
    import mlx.core as mx
    from _mlx_caps import install_caps
    from _mlx_watchdog import arm_mlx_watchdog

    working_set_bytes = int(mx.device_info()["max_recommended_working_set_size"])
    soft_gb = soft_cap_gb(recipe.wired_cap_gb, recipe.cache_gb, working_set_bytes)
    wired_b, soft_b, cache_b = install_caps(
        wired_gb=recipe.wired_cap_gb, soft_gb=soft_gb, cache_gb=recipe.cache_gb
    )
    print(
        f"  [worker] {label}: caps wired={wired_b / GIB:.2f} soft={soft_b / GIB:.2f} cache={cache_b / GIB:.2f} GiB",
        flush=True,
    )

    def _on_abort(payload: dict[str, int]) -> None:
        line = json.dumps({"aborted": "active-memory watchdog", "label": label, **payload})
        print(f"{WORKER_RESULT_SENTINEL}{line}", flush=True)

    arm_mlx_watchdog(on_abort=_on_abort, headroom_gib=4.0)


def _smoke_recipe(recipe: Recipe) -> Recipe:
    from dataclasses import replace

    # 4, not 2: TeaCache always computes the first and last step (skip_first_n_steps=1 +
    # skip_last_n_steps=1), so 2 steps leave nothing outside that always-computed window and
    # condition B raises InvalidStepWindowError.
    return replace(recipe, steps=4, width=256, height=256, free_encoders=True)


def _worker_main(args: argparse.Namespace) -> None:
    from _comparison_recipes import recipe_for, recipe_stamp, with_resolution

    recipe = with_resolution(recipe_for(args.only), args.width, args.height)
    if args.smoke:
        recipe = _smoke_recipe(recipe)
    raw_root = SMOKE_ROOT if args.smoke else (PROBE_DIR / "raw" if args.probe else RAW_ROOT)
    _install_guards(recipe, f"{recipe.slug}/{args.condition}")
    steps = 3 if args.probe else recipe.steps
    result = _run_generation(recipe, args.condition, raw_root=raw_root, steps=steps)
    result["stamp"] = recipe_stamp(recipe, versions=_versions())
    result["git_sha"] = _git_sha()
    print(f"{WORKER_RESULT_SENTINEL}{json.dumps(result)}", flush=True)


def _spawn(recipe: Recipe, condition: str, *, probe: bool, smoke: bool) -> dict[str, Any]:
    """Run one worker; stream its output live (heavy-runs monitoring) while collecting it for the result line."""
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--only",
        recipe.slug,
        "--condition",
        condition,
        "--width",
        str(recipe.width),
        "--height",
        str(recipe.height),
    ]
    cmd += ["--probe"] if probe else []
    cmd += ["--smoke"] if smoke else []
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1"}
    print(
        f"\n>> worker {recipe.slug}/{condition} {recipe.width}x{recipe.height}{' probe' if probe else ''}",
        flush=True,
    )
    lines: list[str] = []
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=None, text=True, env=env) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lines.append(line)
        returncode = proc.wait()
    payload = parse_worker_line("".join(lines))
    if payload is not None and "aborted" in payload:
        return payload
    if returncode != 0 or payload is None:
        raise RuntimeError(f"worker {recipe.slug}/{condition} failed: exit {returncode}")
    return payload


def merge_probe_results(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Merge condition A and B's raw probe measurements into one worst-case reading.

    A 3-step probe of condition A alone never sees B's cached TeaCache residuals, which measurably add
    0.5-0.8 GiB on top of everything A holds. So the probe records the worse side per field -- the higher
    peak, the higher footprint, the lower host-free floor -- and an abort on either side fails the whole
    probe, the same as an A-only abort always has."""
    if "aborted" in a:
        return a
    if "aborted" in b:
        return b
    return {
        "mlx_peak_load_bytes": max(a["mlx_peak_load_bytes"], b["mlx_peak_load_bytes"]),
        "mlx_peak_encode_bytes": max(a["mlx_peak_encode_bytes"], b["mlx_peak_encode_bytes"]),
        "mlx_peak_generation_bytes": max(a["mlx_peak_generation_bytes"], b["mlx_peak_generation_bytes"]),
        "memory": {
            "peak_footprint_bytes": max(
                a["memory"]["peak_footprint_bytes"], b["memory"]["peak_footprint_bytes"]
            ),
            "min_host_free_pct": min(a["memory"]["min_host_free_pct"], b["memory"]["min_host_free_pct"]),
        },
    }


def probe_record_from_worker(
    recipe: Recipe, result: dict[str, Any], *, working_set_bytes: int, versions: dict[str, str]
) -> dict[str, Any]:
    """Map a worker's raw result (or an ``aborted`` payload) onto ``probe_record``'s inputs: the three MLX
    phase peaks, the host free-memory floor, and the OS process footprint mlx-guard actually kills on."""
    from _comparison_recipes import probe_record

    if "aborted" in result:
        return probe_record(
            recipe,
            phase_peaks={},
            min_host_free_pct=None,
            working_set_bytes=working_set_bytes,
            versions=versions,
            aborted=str(result["aborted"]),
        )
    memory = result["memory"]
    record = probe_record(
        recipe,
        phase_peaks={
            "load": result["mlx_peak_load_bytes"],
            "encode": result["mlx_peak_encode_bytes"],
            "generation": result["mlx_peak_generation_bytes"],
        },
        min_host_free_pct=memory["min_host_free_pct"],
        working_set_bytes=working_set_bytes,
        versions=versions,
        peak_footprint_bytes=memory["peak_footprint_bytes"],
    )
    record["memory"] = memory
    return record


def _probe(slug: str, *, fallback: bool) -> int:
    """3-step probe of BOTH conditions: B holds cached TeaCache residuals on top of everything A holds
    (+0.5-0.8 GiB measured), so judging the probe on A alone misses B's worse memory. The record gates on
    whichever condition peaked higher (``merge_probe_results``); an abort on either side fails it."""
    import mlx.core as mx
    from _comparison_recipes import WORKING_SET_BYTES_DEFAULT, recipe_for, with_resolution

    base = recipe_for(slug)
    if base.fallback is None:
        raise SystemExit(f"{slug} has no fallback and needs no probe")
    recipe = with_resolution(base, *base.fallback) if fallback else base
    working_set = int(mx.device_info().get("max_recommended_working_set_size", WORKING_SET_BYTES_DEFAULT))
    result_a = _spawn(recipe, "a", probe=True, smoke=False)
    result_b = _spawn(recipe, "b", probe=True, smoke=False)
    merged = merge_probe_results(result_a, result_b)
    record = probe_record_from_worker(recipe, merged, working_set_bytes=working_set, versions=_versions())
    _write_probe(record)
    print(f"probe {slug} {recipe.width}x{recipe.height}: pass={record['pass']}", flush=True)
    return 0  # a failed probe is a measurement, not a failure; resolve_resolution acts on it


def _write_probe(record: dict[str, Any]) -> None:
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    (PROBE_DIR / f"{record['slug']}_{record['width']}x{record['height']}.json").write_text(
        json.dumps(record, indent=2)
    )


def _probe_failed(slug: str, *, fallback: bool, reason: str) -> None:
    """Hand-record a probe killed from outside (mlx-guard kills the whole group, so no record was written)."""
    from _comparison_recipes import WORKING_SET_BYTES_DEFAULT, probe_record, recipe_for, with_resolution

    base = recipe_for(slug)
    recipe = with_resolution(base, *base.fallback) if fallback and base.fallback else base
    _write_probe(
        probe_record(
            recipe,
            phase_peaks={},
            min_host_free_pct=None,
            working_set_bytes=WORKING_SET_BYTES_DEFAULT,
            versions=_versions(),
            aborted=reason,
        )
    )


def _resolved(slug: str, *, smoke: bool) -> Recipe:
    from _comparison_recipes import recipe_for, resolve_resolution

    if smoke:
        return _smoke_recipe(recipe_for(slug))
    return resolve_resolution(recipe_for(slug), _load_probes(slug), versions=_versions())


def _orchestrate(
    slug: str,
    *,
    budget: int,
    smoke: bool,
    spawn: Callable[..., dict[str, Any]] | None = None,
    trash: Path | None = None,
    chunks_dir: Path | None = None,
    raw_dir: Path | None = None,
) -> int:
    from _comparison_recipes import recipe_stamp

    spawn_fn = spawn or _spawn
    trash_dir = trash if trash is not None else (Path.home() / ".Trash")
    recipe = _resolved(slug, smoke=smoke)
    chunks = chunks_dir if chunks_dir is not None else (SMOKE_ROOT / "chunks" if smoke else CHUNKS_DIR)
    raw_root = raw_dir if raw_dir is not None else (SMOKE_ROOT if smoke else RAW_ROOT)
    expected = recipe_stamp(recipe, versions=_versions())
    for condition in plan_conditions(chunks, raw_root, slug, recipe.steps, expected, budget):
        path = chunk_path(chunks, slug, condition)
        retire_chunk(path, trash=trash_dir, tag=_now_tag())
        retire_chunk(path.with_suffix(".aborted.json"), trash=trash_dir, tag=_now_tag())
        result = spawn_fn(recipe, condition, probe=False, smoke=smoke)
        if "aborted" in result:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.with_suffix(".aborted.json").write_text(json.dumps(result, indent=2))
            print(
                f"== ABORTED by the memory watchdog on {slug}/{condition}; nothing persisted ==", flush=True
            )
            return 4
        check_chunk_stamp(result, expected, path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(result, indent=2))
        tmp.replace(path)
        print(f"  chunk persisted: {path}", flush=True)
    remaining = plan_conditions(chunks, raw_root, slug, recipe.steps, expected, -1)
    return 3 if remaining else 0


def _finalize(slug: str, *, export_jpg: bool = False) -> None:
    """Both chunks done → SSIM on the raw PNGs, contact sheets, report entry, render check.

    ``export_jpg`` additionally converts the four raw PNGs to the JPGs COMPARISON.md and the per-model
    pages actually link — off by default so a routine re-finalize (step L) does not touch the committed,
    already-optimised JPGs; pass it only when regenerating the showcase images themselves."""
    import numpy as np
    from _comparison_recipes import recipe_stamp
    from _comparison_sheet import build_contact_sheet, export_jpgs
    from PIL import Image
    from skimage.metrics import structural_similarity

    recipe = _resolved(slug, smoke=False)
    a, b = load_pair(CHUNKS_DIR, RAW_ROOT, slug, recipe.steps, recipe_stamp(recipe, versions=_versions()))
    if a["git_sha"] != b["git_sha"]:
        print(f"  warning: A ran at {a['git_sha']}, B at {b['git_sha']}", flush=True)
    raw = raw_dir_for(RAW_ROOT, slug)
    with Image.open(raw / "a.png") as ia, Image.open(raw / "b.png") as ib:
        ssim = float(
            structural_similarity(
                np.asarray(ia.convert("RGB")), np.asarray(ib.convert("RGB")), channel_axis=2, data_range=255
            )
        )
    for cond, kinds in (("a", ["computed"] * recipe.steps), ("b", b["decision_kinds"])):
        build_contact_sheet(frame_paths(frames_dir_for(RAW_ROOT, slug, cond)), kinds).save(
            raw / f"steps-{cond}.png"
        )
    if export_jpg:
        export_jpgs(raw, REPO / "_artifacts" / "comparison" / slug)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    provenance = {
        "generated_at": generated_at,
        "git_sha_a": a["git_sha"],
        "git_sha_b": b["git_sha"],
        "mlx_teacache_version": a["mlx_teacache_version"],
        **{k: str(v) for k, v in a["stamp"].items() if k.startswith("version_")},
    }
    entry = assemble_entry(recipe, a, b, ssim=ssim, provenance=provenance)
    report: dict[str, Any] = (
        json.loads(REPORT_PATH.read_text()) if REPORT_PATH.exists() else {"schema_version": 2, "variants": {}}
    )
    report = {**merge_entry(report, slug, entry, generated_at=generated_at), **_report_header()}
    sys.path.insert(0, str(REPO / "docs"))
    import _generate_comparison

    _generate_comparison.render_blocks(report)  # fails loudly if the page cannot render this entry
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(
        f"{slug}: ssim={ssim:.4f} wall={entry['speedup_wall']:.2f}x steady={entry['speedup_steady']:.2f}x "
        f"preview-subtracted={entry['speedup_preview_subtracted']:.2f}x -> {REPORT_PATH}",
        flush=True,
    )


def _report_header() -> dict[str, Any]:
    import platform

    from _comparison_recipes import PROMPT, QWEN_PROMPT_SUFFIX, SEED
    from mflux.utils.apple_silicon import AppleSiliconUtil

    def sysctl(key: str) -> str:
        return subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True, check=False
        ).stdout.strip()

    return {
        "schema_version": 2,
        "prompt": PROMPT,
        "qwen_prompt_suffix": QWEN_PROMPT_SUFFIX,
        "seed": SEED,
        "protocol": "one cold generation per condition in its own process; weights and prompt embeddings evaluated "
        "before the clock; taef preview decoded every step",
        "mflux_compiles_predict": not AppleSiliconUtil.is_m1_or_m2(),
        "hardware": {
            "chip": sysctl("machdep.cpu.brand_string"),
            "ram_gb": round(int(sysctl("hw.memsize") or 0) / GIB),
            "os": f"macOS {platform.mac_ver()[0]}",
            "python": platform.python_version(),
        },
    }


def main() -> None:
    from _comparison_recipes import RECIPES

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worker", action="store_true", help="(internal) run one condition in this process")
    ap.add_argument("--only", required=True, choices=[r.slug for r in RECIPES])
    ap.add_argument("--condition", choices=CONDITIONS, default="a")
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument(
        "--probe", action="store_true", help="3-step memory probe of condition A; writes no chunk"
    )
    ap.add_argument(
        "--fallback", action="store_true", help="with --probe / --probe-failed: the fallback size"
    )
    ap.add_argument(
        "--probe-failed", action="store_true", help="record a probe killed from outside as failed"
    )
    ap.add_argument("--reason", default="killed by mlx-guard")
    ap.add_argument("--max-workers", type=positive_int, default=None)
    ap.add_argument("--finalize", action="store_true", help="SSIM + contact sheets + report entry")
    ap.add_argument(
        "--export-jpg",
        action="store_true",
        help="with --finalize: also write the showcase JPGs to _artifacts/comparison/<slug>/",
    )
    ap.add_argument(
        "--smoke", action="store_true", help="4 steps at 256x256 into tests/_artifacts/comparison_smoke"
    )
    args = ap.parse_args()
    if args.worker:
        _worker_main(args)
        return
    if args.probe_failed:
        _probe_failed(args.only, fallback=args.fallback, reason=args.reason)
        return
    if args.probe:
        raise SystemExit(_probe(args.only, fallback=args.fallback))
    if args.finalize:
        _finalize(args.only, export_jpg=args.export_jpg)
        return
    budget = args.max_workers if args.max_workers is not None else -1
    raise SystemExit(_orchestrate(args.only, budget=budget, smoke=args.smoke))


if __name__ == "__main__":
    main()
