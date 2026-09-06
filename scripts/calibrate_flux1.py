"""Calibrate, score, or sweep a FLUX.1-family model (dev, krea-dev).

The FLUX.1-dev coefficients are vendored from upstream ali-vilab TeaCache; this
script exists for the family members that share the transformer but not the
weights (Krea [dev] first). Three modes, all chunked per prompt / per threshold,
one subprocess per chunk, resumable:

  capture   uv run python scripts/calibrate_flux1.py --model krea-dev --max-prompts 1
            Records per-step (mod_in rel-L1, body_out rel-L1) pairs for each
            calibration prompt at the model's recipe, then fits a degree-4
            polynomial and, with --score-coefficients c4,c3,c2,c1,c0, reports the
            R² of a given tuple (e.g. flux1-dev's) on the same pairs.
  sweep     uv run python scripts/calibrate_flux1.py --model krea-dev --sweep 0.15,0.20,0.25,0.30
            One gated generation per threshold against a vanilla reference on the
            red-apple prompt: skips, SSIM vs vanilla, skip pattern, longest streak.

The forward is never re-walked here: the recording transformer calls the same
`_flux1_prelude` / `_flux1_extract_mod_input` / `_flux1_run_body` / `_flux1_tail`
the variant integration runs, and the pair metric is the gate's own
`mean_abs_rel_l1`. Memory: `install_caps` + the active+cache watchdog before any
model load (FLUX.1 at 512² q4 peaks ~9–12 GiB).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from calibrate_flux2 import CALIBRATION_PROMPTS, _fit_polynomial

from mlx_teacache._kernel.gate import mean_abs_rel_l1, poly_eval
from mlx_teacache.variants.flux1_dev.integration import (
    _flux1_extract_mod_input,
    _flux1_prelude,
    _flux1_run_body,
    _flux1_tail,
)

SEED = 42
HEIGHT = 512
WIDTH = 512
QUANTIZE = 4
SWEEP_PROMPT = "a red apple on a wooden table"

# Each model card's published recipe; the sweep and the bench run the same one.
RECIPES: dict[str, dict[str, Any]] = {
    "dev": {"num_inference_steps": 25, "guidance": 3.5},
    "krea-dev": {"num_inference_steps": 28, "guidance": 4.5},
}

_ARTIFACTS = Path(__file__).resolve().parent.parent / "tests" / "_artifacts"
_CANONICAL_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested without weights).
# ---------------------------------------------------------------------------


def _pairs_from_capture(mod_ins: list[mx.array], body_outs: list[mx.array]) -> list[tuple[float, float]]:
    """Consecutive-step (x, y) pairs: x = rel-L1 of the gate signal, y = rel-L1 of the body output."""
    return [
        (mean_abs_rel_l1(mod_ins[t], mod_ins[t - 1]), mean_abs_rel_l1(body_outs[t], body_outs[t - 1]))
        for t in range(1, len(mod_ins))
    ]


def _r2_score(coeffs: tuple[float, float, float, float, float], xs: list[float], ys: list[float]) -> float:
    """Coefficient of determination of a fixed polynomial on the given pairs."""
    preds = [poly_eval(coeffs, x) for x in xs]
    mean_y = sum(ys) / len(ys)
    ss_res = sum((y - p) ** 2 for y, p in zip(ys, preds, strict=True))
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _chunk_filename(model: str, idx: int) -> str:
    return f"chunk_{model}_prompt{idx:02d}.json"


def _pending_prompt_indices(chunk_dir: Path, model: str, n_prompts: int) -> list[int]:
    return [i for i in range(n_prompts) if not (chunk_dir / _chunk_filename(model, i)).exists()]


def _aggregate_pairs(chunk_dir: Path, model: str, n_prompts: int) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for i in range(n_prompts):
        chunk = json.loads((chunk_dir / _chunk_filename(model, i)).read_text())
        for x, y in chunk["pairs"]:
            xs.append(float(x))
            ys.append(float(y))
    return xs, ys


def _threshold_name(t: float) -> str:
    return f"t{t:.3f}"


def _sweep_units(thresholds: list[float]) -> list[str]:
    return ["vanilla"] + [_threshold_name(t) for t in thresholds]


def _sweep_chunk_filename(model: str, unit: str) -> str:
    return f"sweep_{model}_{unit}.json"


def _build_sweep_summary(
    chunks: list[dict[str, Any]], vanilla_seconds: float, *, model: str
) -> dict[str, Any]:
    rows = sorted(
        (
            {
                "threshold": c["threshold"],
                "wrapper_seconds": c["wrapper_seconds"],
                "speedup_vs_vanilla_single_rep": vanilla_seconds / c["wrapper_seconds"],
                "skipped": c["skipped"],
                "computed": c["computed"],
                "ssim_vs_vanilla": c["ssim_vs_vanilla"],
                "skip_pattern": str(c.get("skip_pattern", "")),
                "max_consecutive_skips": int(c.get("max_consecutive_skips", 0)),
            }
            for c in chunks
        ),
        key=lambda r: r["threshold"],
    )
    return {
        "model": model,
        "recipe": RECIPES[model],
        "quantize": QUANTIZE,
        "prompt": SWEEP_PROMPT,
        "seed": SEED,
        "height": HEIGHT,
        "width": WIDTH,
        "vanilla_seconds": vanilla_seconds,
        "thresholds": rows,
        "note": "Single-rep wall-clock (subprocess-per-threshold, cold each; thermal noise); SSIM and "
        "skip counts are deterministic per threshold. Choose DEFAULT_THRESH at the knee where SSIM "
        "holds the FLUX.1 PR-gate bar (0.90).",
    }


def _default_chunk_dir(model: str) -> Path:
    return _ARTIFACTS / "calibrate_flux1" / model


# ---------------------------------------------------------------------------
# Recording transformer (imperative shell around the integration's helpers).
# ---------------------------------------------------------------------------


class _RecordingFlux1Transformer(nn.Module):  # type: ignore[misc]
    """Runs the vanilla FLUX.1 forward through the integration's own helpers and
    records (mod_in, body_out) per step. No gating, no skipping."""

    def __init__(self, inner: Any, captures: list[tuple[mx.array, mx.array]]) -> None:
        super().__init__()
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_captures", captures)

    def __call__(
        self,
        t: int,
        config: Any,
        hidden_states: mx.array,
        prompt_embeds: mx.array,
        pooled_prompt_embeds: mx.array,
        **kwargs: Any,
    ) -> Any:
        inner = object.__getattribute__(self, "_inner")
        body_in, enc, temb, rot = _flux1_prelude(
            inner,
            t=t,
            config=config,
            hidden_states=hidden_states,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            kwargs=kwargs,
        )
        mod_in = _flux1_extract_mod_input(inner.transformer_blocks[0], body_in, temb)
        body_out = _flux1_run_body(inner, body_in, enc, temb, rot, kwargs)
        mx.eval(mod_in, body_out)
        object.__getattribute__(self, "_captures").append((mod_in, body_out))
        return _flux1_tail(inner, body_out, enc, temb)

    def freeze(self, *args: Any, **kwargs: Any) -> Any:
        return object.__getattribute__(self, "_inner").freeze(*args, **kwargs)

    def parameters(self) -> Any:
        return object.__getattribute__(self, "_inner").parameters()

    def trainable_parameters(self) -> Any:
        return object.__getattribute__(self, "_inner").trainable_parameters()

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(object.__getattribute__(self, "_inner"), name)


def _load_flux(model: str) -> Any:
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.flux.variants.txt2img.flux import Flux1

    cfg = {"dev": ModelConfig.dev, "krea-dev": ModelConfig.krea_dev}[model]()
    flux = Flux1(quantize=QUANTIZE, model_config=cfg)
    flux.freeze()
    return flux


def _guard(label: str, chunk_dir: Path) -> None:
    from _mlx_caps import install_caps
    from _mlx_watchdog import abort_handler, arm_mlx_watchdog

    install_caps(wired_gb=18, soft_gb=20)
    arm_mlx_watchdog(on_abort=abort_handler(label, chunk_dir))


def _generate(flux: Any, *, prompt: str, model: str, save_path: Path | None = None) -> float:
    recipe = RECIPES[model]
    start = time.perf_counter()
    image = flux.generate_image(
        prompt=prompt,
        seed=SEED,
        num_inference_steps=recipe["num_inference_steps"],
        height=HEIGHT,
        width=WIDTH,
        guidance=recipe["guidance"],
    )
    mx.synchronize()
    elapsed = time.perf_counter() - start
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path=str(save_path), export_json_metadata=False)
    return elapsed


# ---------------------------------------------------------------------------
# Workers.
# ---------------------------------------------------------------------------


def _capture_worker(model: str, prompt_idx: int, *, chunk_dir: Path, dry_run: bool) -> None:
    chunk_dir.mkdir(parents=True, exist_ok=True)
    out = chunk_dir / _chunk_filename(model, prompt_idx)
    prompt = CALIBRATION_PROMPTS[prompt_idx]
    steps = int(RECIPES[model]["num_inference_steps"])
    if dry_run:
        pairs = [[round(0.01 * (k + 1), 5), round(0.02 * (k + 1), 5)] for k in range(steps - 1)]
        out.write_text(
            json.dumps({"idx": prompt_idx, "prompt": prompt, "model": model, "dry_run": True, "pairs": pairs})
        )
        print(f"[worker {model}/{prompt_idx}] dry-run chunk -> {out}", flush=True)
        return
    _guard(f"calibrate_flux1-{model}-prompt{prompt_idx}", chunk_dir)
    flux = _load_flux(model)
    captures: list[tuple[mx.array, mx.array]] = []
    original = flux.transformer
    flux.transformer = _RecordingFlux1Transformer(original, captures)
    try:
        elapsed = _generate(flux, prompt=prompt, model=model)
    finally:
        flux.transformer = original
    assert len(captures) == steps, f"expected {steps} captures, got {len(captures)}"
    pairs = _pairs_from_capture([m for m, _ in captures], [b for _, b in captures])
    out.write_text(
        json.dumps(
            {
                "idx": prompt_idx,
                "prompt": prompt,
                "model": model,
                "recipe": RECIPES[model],
                "quantize": QUANTIZE,
                "height": HEIGHT,
                "width": WIDTH,
                "seed": SEED,
                "elapsed_s": elapsed,
                "peak_memory_gb": mx.get_peak_memory() / 1024**3,
                "pairs": [[x, y] for x, y in pairs],
            },
            indent=2,
        )
    )
    print(f"[worker {model}/{prompt_idx}] {len(pairs)} pairs, {elapsed:.1f}s -> {out}", flush=True)


def _sweep_worker(model: str, unit: str, *, chunk_dir: Path, dry_run: bool) -> None:
    chunk_dir.mkdir(parents=True, exist_ok=True)
    out = chunk_dir / _sweep_chunk_filename(model, unit)
    img_dir = chunk_dir / "images"
    if dry_run:
        if unit == "vanilla":
            out.write_text(json.dumps({"unit": unit, "vanilla_seconds": 10.0, "dry_run": True}))
        else:
            t = float(unit[1:])
            steps = int(RECIPES[model]["num_inference_steps"])
            out.write_text(
                json.dumps(
                    {
                        "unit": unit,
                        "threshold": t,
                        "dry_run": True,
                        "wrapper_seconds": round(10.0 - 5.0 * t, 4),
                        "skipped": int(30 * t),
                        "computed": steps - int(30 * t),
                        "ssim_vs_vanilla": round(0.999 - 0.25 * t, 4),
                        "skip_pattern": "C" * steps,
                        "max_consecutive_skips": 0,
                    }
                )
            )
        print(f"[worker {model}/{unit}] dry-run chunk -> {out}", flush=True)
        return
    import numpy as np
    from _bench_telemetry import streak_telemetry
    from PIL import Image
    from skimage.metrics import structural_similarity as ssim

    from mlx_teacache import apply_teacache

    _guard(f"calibrate_flux1-{model}-sweep-{unit}", chunk_dir)
    flux = _load_flux(model)
    if unit == "vanilla":
        seconds = _generate(flux, prompt=SWEEP_PROMPT, model=model, save_path=img_dir / "vanilla.png")
        out.write_text(json.dumps({"unit": unit, "vanilla_seconds": seconds}, indent=2))
        print(f"[worker {model}/vanilla] {seconds:.1f}s -> {out}", flush=True)
        return
    t = float(unit[1:])
    van_path = img_dir / "vanilla.png"
    if not van_path.exists():
        raise SystemExit(f"[worker {model}/{unit}] vanilla.png missing; run the vanilla unit first")
    with apply_teacache(flux, rel_l1_thresh=t) as h:
        seconds = _generate(flux, prompt=SWEEP_PROMPT, model=model, save_path=img_dir / f"{unit}.png")
        skipped, computed = h.stats.skipped_count, h.stats.computed_count
        telemetry = streak_telemetry(h.stats)

    def _load(path: Path) -> Any:
        return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)

    score = float(ssim(_load(van_path), _load(img_dir / f"{unit}.png"), channel_axis=-1, data_range=255))
    out.write_text(
        json.dumps(
            {
                "unit": unit,
                "threshold": t,
                "wrapper_seconds": seconds,
                "skipped": skipped,
                "computed": computed,
                "ssim_vs_vanilla": score,
                **telemetry,
            },
            indent=2,
        )
    )
    print(
        f"[worker {model}/{unit}] skipped={skipped} computed={computed} streak={telemetry['max_consecutive_skips']} "
        f"SSIM={score:.4f} {seconds:.1f}s -> {out}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Orchestrators.
# ---------------------------------------------------------------------------


def _spawn(args: list[str]) -> int:
    return subprocess.run([sys.executable, str(Path(__file__).resolve()), *args]).returncode


def _run_capture(
    model: str,
    *,
    chunk_dir: Path,
    n_prompts: int,
    max_prompts: int | None,
    fit_mode: str,
    score: tuple[float, float, float, float, float] | None,
    dry_run: bool,
) -> int:
    chunk_dir.mkdir(parents=True, exist_ok=True)
    pending = _pending_prompt_indices(chunk_dir, model, n_prompts)
    to_run = pending if max_prompts is None else pending[:max_prompts]
    print(
        f"[orchestrator] {model}: {n_prompts} prompts, {len(pending)} pending, running {len(to_run)}",
        flush=True,
    )
    for idx in to_run:
        extra = ["--dry-run"] if dry_run else []
        rc = _spawn(
            ["--worker", "--model", model, "--prompt-idx", str(idx), "--chunk-dir", str(chunk_dir), *extra]
        )
        if rc != 0 or not (chunk_dir / _chunk_filename(model, idx)).exists():
            raise SystemExit(
                f"[orchestrator] worker for prompt {idx} failed (rc={rc}); completed chunks are reused"
            )
    if _pending_prompt_indices(chunk_dir, model, n_prompts):
        print("[orchestrator] PARTIAL: prompts remain; re-invoke to continue", flush=True)
        return 3
    xs, ys = _aggregate_pairs(chunk_dir, model, n_prompts)
    coeffs, r2 = _fit_polynomial(xs, ys, fit_mode=fit_mode)
    result: dict[str, Any] = {
        "model": model,
        "recipe": RECIPES[model],
        "quantize": QUANTIZE,
        "height": HEIGHT,
        "width": WIDTH,
        "seed": SEED,
        "n_prompts": n_prompts,
        "n_pairs": len(xs),
        "fit_mode": fit_mode,
        "coefficients_c4_to_c0": coeffs,
        "r2": r2,
        "x_values": xs,
        "y_values": ys,
    }
    if score is not None:
        result["scored_coefficients_c4_to_c0"] = list(score)
        result["scored_r2"] = _r2_score(score, xs, ys)
    out = (
        _CANONICAL_DIR / f"_calibration_flux1_{model.replace('-', '_')}.json"
        if not dry_run and chunk_dir.resolve() == _default_chunk_dir(model).resolve()
        else chunk_dir / "calibration.json"
    )
    out.write_text(json.dumps(result, indent=2))
    print(f"[orchestrator] fit ({fit_mode}) R²={r2:.4f} coeffs={coeffs}", flush=True)
    if score is not None:
        print(f"[orchestrator] scored tuple R²={result['scored_r2']:.4f}", flush=True)
    print(f"[orchestrator] wrote {out}", flush=True)
    return 0


def _run_sweep(
    model: str, thresholds: list[float], *, chunk_dir: Path, max_units: int | None, dry_run: bool
) -> int:
    chunk_dir.mkdir(parents=True, exist_ok=True)
    units = _sweep_units(thresholds)
    pending = [u for u in units if not (chunk_dir / _sweep_chunk_filename(model, u)).exists()]
    to_run = pending if max_units is None else pending[:max_units]
    print(
        f"[orchestrator] sweep {model}: {len(units)} units, {len(pending)} pending, running {len(to_run)}",
        flush=True,
    )
    for unit in to_run:
        extra = ["--dry-run"] if dry_run else []
        rc = _spawn(
            ["--worker", "--model", model, "--sweep-unit", unit, "--chunk-dir", str(chunk_dir), *extra]
        )
        if rc != 0 or not (chunk_dir / _sweep_chunk_filename(model, unit)).exists():
            raise SystemExit(
                f"[orchestrator] sweep worker for {unit} failed (rc={rc}); completed chunks are reused"
            )
    if [u for u in units if not (chunk_dir / _sweep_chunk_filename(model, u)).exists()]:
        print("[orchestrator] PARTIAL: units remain; re-invoke to continue", flush=True)
        return 3
    vanilla = json.loads((chunk_dir / _sweep_chunk_filename(model, "vanilla")).read_text())["vanilla_seconds"]
    chunks = [json.loads((chunk_dir / _sweep_chunk_filename(model, u)).read_text()) for u in units[1:]]
    summary = _build_sweep_summary(chunks, vanilla, model=model)
    out = chunk_dir / "sweep_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    for r in summary["thresholds"]:
        print(
            f"  t={r['threshold']:.3f} skipped={r['skipped']} streak={r['max_consecutive_skips']} "
            f"SSIM={r['ssim_vs_vanilla']:.4f} speedup={r['speedup_vs_vanilla_single_rep']:.2f}x",
            flush=True,
        )
    print(f"[orchestrator] wrote {out}", flush=True)
    return 0


def _parse_floats(text: str | None) -> list[float] | None:
    if text is None:
        return None
    values = [float(x) for x in text.split(",") if x.strip()]
    if not values:
        raise SystemExit("expected at least one comma-separated number")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", choices=sorted(RECIPES), required=True)
    parser.add_argument(
        "--chunk-dir", type=Path, default=None, help="default: tests/_artifacts/calibrate_flux1/<model>"
    )
    parser.add_argument("--dry-run", action="store_true", help="synthetic chunks, no model load")
    parser.add_argument("--worker", action="store_true", help="internal: run one chunk")
    parser.add_argument("--prompt-idx", type=int, default=None, help="worker: calibration prompt index")
    parser.add_argument("--sweep-unit", default=None, help="worker: 'vanilla' or 't<thresh>'")
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="capture: run at most N pending prompts (exit 3 if more remain)",
    )
    parser.add_argument("--fit-mode", choices=["free", "origin"], default="free")
    parser.add_argument(
        "--score-coefficients", default=None, help="c4,c3,c2,c1,c0 to score on the captured pairs"
    )
    parser.add_argument(
        "--sweep", default=None, help="comma-separated thresholds; runs the sweep instead of capture"
    )
    parser.add_argument("--max-units", type=int, default=None, help="sweep: run at most N pending units")
    args = parser.parse_args()

    chunk_dir: Path = args.chunk_dir if args.chunk_dir is not None else _default_chunk_dir(args.model)
    if args.worker:
        if args.sweep_unit is not None:
            _sweep_worker(args.model, args.sweep_unit, chunk_dir=chunk_dir, dry_run=args.dry_run)
        elif args.prompt_idx is not None:
            _capture_worker(args.model, args.prompt_idx, chunk_dir=chunk_dir, dry_run=args.dry_run)
        else:
            parser.error("--worker needs --prompt-idx or --sweep-unit")
        return
    if args.sweep is not None:
        sys.exit(
            _run_sweep(
                args.model,
                _parse_floats(args.sweep) or [],
                chunk_dir=chunk_dir / "sweep",
                max_units=args.max_units,
                dry_run=args.dry_run,
            )
        )
    score = _parse_floats(args.score_coefficients)
    if score is not None and len(score) != 5:
        parser.error("--score-coefficients needs exactly five numbers")
    sys.exit(
        _run_capture(
            args.model,
            chunk_dir=chunk_dir,
            n_prompts=len(CALIBRATION_PROMPTS),
            max_prompts=args.max_prompts,
            fit_mode=args.fit_mode,
            score=tuple(score) if score is not None else None,  # type: ignore[arg-type]
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    main()
