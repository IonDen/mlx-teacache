# COMPARISON.md + global-config cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a `COMPARISON.md` visual+timing showcase at repo root with three non-distilled FLUX entries, committed webp images + a recoverable JSON of every measured number. Side: refresh `~/.claude/CLAUDE.md` and `~/.claude/skills/user-mlx-developer/` with v0.4.0 + v0.4.1 lessons.

**Architecture:** Three independent improvements bundled by theme. Item 3 (COMPARISON.md, the repo work) goes through a single branch + PR per the new release-flow rule and uses a new `scripts/bench_comparison.py` that produces both per-rep timings and webp images in one pass, persisting everything to `_artifacts/comparison_report.json`. Per plan-audit Finding 2, each (variant, condition) pair runs in a **separate subprocess** so the "cold" rep is genuinely cold — no MLX kernel state from a prior generation in the same process. The main script orchestrates 6 subprocesses (3 variants × {vanilla, wrapper}) and aggregates their stdout JSON into the final report. Items 1 + 2 are out-of-repo paragraph-pass edits to `~/.claude/` files — done in parallel, no PR.

**Tech Stack:** Python 3.11+, `mflux>=0.17,<0.18`, MLX (`mlx.core`), Pillow (already a mflux dep) for PNG→webp, `argparse`+`statistics`+`json` from stdlib. `uv run` for execution. `ruff` for lint+format.

**Spec:** [`docs/superpowers/specs/2026-05-17-comparison-doc-and-cleanup-design.md`](../specs/2026-05-17-comparison-doc-and-cleanup-design.md).

---

## File map

| File | Responsibility | Action |
|---|---|---|
| `scripts/bench_comparison.py` | Generation + timing pipeline | **Create.** Loads each of the three variants, runs vanilla×3 then wrapper×3, times each rep, captures the rep-1 PNG, converts PNG→webp, writes the JSON report. ~250 LoC. |
| `_artifacts/comparison/flux1-dev/{vanilla,wrapper}.webp` | flux1-dev showcase images | **Create.** Produced by the script. Committed. |
| `_artifacts/comparison/klein-base-4b-g1/{vanilla,wrapper}.webp` | klein-base-4b g=1.0 showcase | **Create.** Produced by the script. Committed. |
| `_artifacts/comparison/klein-base-4b-cfg/{vanilla,wrapper}.webp` | klein-base-4b CFG showcase | **Create.** Produced by the script. Committed. |
| `_artifacts/comparison_report.json` | All numbers, recoverable | **Create.** Produced by the script. Schema v1, contains every per-rep timing, skip count, hardware, recipe. Committed. |
| `COMPARISON.md` | Repo-root visual+timing doc | **Create.** Hand-edited from the script output. |
| `README.md` | Add "When to use" section + link | **Modify.** Insert section after "Supported variants" (around line 105). |
| `.gitignore` | Distinguish two _artifacts/ paths | **Modify.** Add explanatory comment above the existing `tests/_artifacts/` line. |
| `~/.claude/CLAUDE.md` | Personal global rules | **Modify** (out of repo). Paragraph-pass review; targeted edits only where stale. |
| `~/.claude/skills/user-mlx-developer/SKILL.md` and references | MLX engineering skill | **Modify** (out of repo). Fold in v0.4 lessons. |

---

## Preconditions (author machine)

1. **On main, clean tree.** v0.4.1 shipped (commit `dc93155` is on main, tag `v0.4.1` pushed).

   ```bash
   git checkout main && git pull --ff-only
   git status --short    # Expect: clean (no tracked changes)
   ```

2. **Start the feature branch.**

   ```bash
   git checkout -b feature/comparison-md-and-cleanup
   ```

3. **Local CI gate that must stay green between commits:**

   ```bash
   uv run ruff check . && uv run ruff format --check . && uv run pytest tests/ -m "not parity and not slow and not benchmark and not network"
   ```

4. **Weights available locally:** `flux1-dev` and `flux2-klein-base-4b` already downloaded (used in v0.4.x work).

   ```bash
   hf download black-forest-labs/FLUX.1-dev --dry-run | head -3
   hf download black-forest-labs/FLUX.2-klein-base-4B --dry-run | head -3
   ```

---

## Task 1: Add `_artifacts/` distinction comment to `.gitignore`

The repo currently gitignores `tests/_artifacts/` (test fixtures and bench scratch). The new `_artifacts/` at repo root is intentionally NOT ignored — it holds COMPARISON.md's committed images + the JSON report. Add a comment so future readers don't add a blanket `_artifacts/` ignore rule by mistake.

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Read the existing `.gitignore` to find the `tests/_artifacts/` line**

  ```bash
  grep -n "_artifacts" .gitignore
  ```

  Expected: one match around line 16 reading `tests/_artifacts/`.

- [ ] **Step 2: Insert the comment block above that line**

  Open `.gitignore` and edit so the relevant block reads:

  ```gitignore
  # Bench / sweep test fixtures, regenerated on every run.
  tests/_artifacts/

  # NOTE: the repo-root _artifacts/ is INTENTIONALLY NOT ignored — it holds
  # committed comparison images for COMPARISON.md and the recoverable
  # comparison_report.json. Keep them tracked.
  ```

  If the existing block has no comment, replace the bare `tests/_artifacts/` line with the version above. If there's already a comment, prepend the new note to it. The order matters: keep `tests/_artifacts/` AS the only ignore rule under the comment block.

- [ ] **Step 3: Verify the repo-root path isn't accidentally ignored**

  ```bash
  mkdir -p _artifacts/comparison && touch _artifacts/comparison/.keep
  git check-ignore _artifacts/comparison/.keep
  # Expected: empty output, exit code 1 (NOT ignored).
  rm _artifacts/comparison/.keep
  ```

  If `git check-ignore` returns a match, the comment isn't enough and we need a `!_artifacts/` rule. (Unlikely — `tests/_artifacts/` is path-specific.)

- [ ] **Step 4: Commit**

  ```bash
  git add .gitignore
  git commit -m "chore(gitignore): distinguish committed repo-root _artifacts/ from gitignored tests/_artifacts/"
  ```

---

## Task 2: Create `scripts/bench_comparison.py`

A new dedicated script that loads each of three variants, runs vanilla×3 then wrapper×3 (cold + 2 warm reps per condition), times each rep, captures the rep-1 image as webp, and writes a complete recoverable JSON report. ~250 LoC.

**Files:**
- Create: `scripts/bench_comparison.py`

This is a CONTENT-generation script (not a test target), so we don't TDD it. We write the script, smoke-test the `--help`, and the main proof is running it on the real models in Task 3.

- [ ] **Step 1: Create the file with the full implementation**

  Create `scripts/bench_comparison.py` with the following content (paste verbatim). This implementation incorporates plan-audit fixes F1 (use `.tmp.png` for the Pillow-readable intermediate path), F2 (subprocess isolation so each "cold" rep is genuinely cold), and F4 (read hardware from macOS sysctl with CLI overrides).

  ```python
  """Generate COMPARISON.md content: vanilla mflux vs mlx-teacache on three
  non-distilled FLUX variants. Produces per-variant webp images + a complete
  recoverable JSON report under _artifacts/.

  Run as:
    uv run python scripts/bench_comparison.py

  Architecture
  ------------

  Each (variant, condition) pair runs in a SEPARATE subprocess so the rep-1
  timing is genuinely "cold" — no prior mflux generation in the same Python
  process, no warm MLX kernel state. The main script orchestrates 6
  subprocesses (3 variants x {vanilla, wrapper}), reads their stdout JSON,
  and aggregates into _artifacts/comparison_report.json.

  The same script file is the orchestrator AND the per-condition worker —
  selected by --condition / --variant flags. Workers print one JSON line at
  the end of stdout (their bench result); the orchestrator parses that line
  to assemble the final report.

  Three entries (non-distilled only):
    - flux1-dev at 25 steps, guidance=3.5
    - flux2-klein-base-4b at 25 steps, guidance=1.0
    - flux2-klein-base-4b at 50 steps, guidance=4.0 (canonical upstream CFG)
  """

  from __future__ import annotations

  import argparse
  import json
  import platform
  import statistics
  import subprocess
  import sys
  import time
  from dataclasses import dataclass
  from pathlib import Path
  from typing import Any

  # Shared portrait prompt — same across every variant. Single variable = recipe.
  PROMPT = (
      "Portrait of a young woman with auburn hair and green eyes, soft "
      "golden-hour window light, photorealistic, shallow depth of field, "
      "50mm prime lens, subtle freckles, neutral background, cinematic "
      "color grading."
  )
  SEED = 42
  HEIGHT = 1024
  WIDTH = 768
  REPS = 3  # rep 1 = cold (subprocess just started); reps 2-3 = warm

  WEBP_QUALITY = 88
  WEBP_METHOD = 6  # Pillow's slowest+best encoder; ~1-2s on 768x1024.

  WORKER_RESULT_SENTINEL = "::BENCH_RESULT::"


  @dataclass(frozen=True)
  class VariantConfig:
      slug: str  # subdir name under _artifacts/comparison/
      variant_id: str  # registry id (used only for reporting clarity)
      num_inference_steps: int
      guidance: float
      loader: str  # "flux1-dev" or "klein-base-4b"


  VARIANTS: tuple[VariantConfig, ...] = (
      VariantConfig(
          slug="flux1-dev",
          variant_id="flux1-dev",
          num_inference_steps=25,
          guidance=3.5,
          loader="flux1-dev",
      ),
      VariantConfig(
          slug="klein-base-4b-g1",
          variant_id="flux2-klein-base-4b",
          num_inference_steps=25,
          guidance=1.0,
          loader="klein-base-4b",
      ),
      VariantConfig(
          slug="klein-base-4b-cfg",
          variant_id="flux2-klein-base-4b",
          num_inference_steps=50,
          guidance=4.0,
          loader="klein-base-4b",
      ),
  )


  # ---------------------------------------------------------------------------
  # WORKER side — runs in a subprocess for one (variant, condition) pair.
  # ---------------------------------------------------------------------------


  def _load_flux(loader: str) -> Any:
      from mflux.models.common.config.model_config import ModelConfig

      if loader == "flux1-dev":
          from mflux.models.flux.variants.txt2img.flux import Flux1

          flux = Flux1.from_name("dev", quantize=4)
      elif loader == "klein-base-4b":
          from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

          flux = Flux2Klein(quantize=4, model_config=ModelConfig.flux2_klein_base_4b())
      else:
          raise ValueError(f"unknown loader: {loader!r}")
      flux.freeze()
      return flux


  def _generate(flux: Any, *, num_inference_steps: int, guidance: float) -> tuple[float, Any]:
      """Time one flux.generate_image call. Flushes GPU before stopping clock."""
      import mlx.core as mx

      start = time.perf_counter()
      image = flux.generate_image(
          prompt=PROMPT,
          seed=SEED,
          num_inference_steps=num_inference_steps,
          height=HEIGHT,
          width=WIDTH,
          guidance=guidance,
      )
      mx.eval(mx.zeros(1))  # flush GPU work before stopping the clock
      elapsed = time.perf_counter() - start
      return elapsed, image


  def _save_as_webp(image: Any, dest_webp: Path) -> None:
      """Save mflux's image as webp via PNG intermediate.

      Plan-audit Finding 1 fix: the intermediate file must have a real `.png`
      suffix so Pillow can infer the format. `<stem>.tmp.png` (NOT
      `<stem>.png.tmp`) keeps `.png` as the final suffix. After Pillow writes
      the PNG we re-open it, encode as webp, and unlink the PNG so only the
      webp survives in the repo.
      """
      from PIL import Image

      dest_webp.parent.mkdir(parents=True, exist_ok=True)
      png_tmp = dest_webp.with_name(dest_webp.stem + ".tmp.png")
      image.save(path=str(png_tmp), export_json_metadata=False)
      with Image.open(png_tmp) as pil_img:
          pil_img.save(dest_webp, format="WEBP", quality=WEBP_QUALITY, method=WEBP_METHOD)
      png_tmp.unlink()


  def _run_worker_vanilla(cfg: VariantConfig, save_to: Path) -> dict[str, Any]:
      flux = _load_flux(cfg.loader)
      times: list[float] = []
      for i in range(REPS):
          elapsed, image = _generate(
              flux, num_inference_steps=cfg.num_inference_steps, guidance=cfg.guidance
          )
          times.append(elapsed)
          if i == 0:
              _save_as_webp(image, save_to)
          print(f"  vanilla rep {i + 1}: {elapsed:.2f}s", flush=True)
      return {"condition": "vanilla", "rep_seconds": times}


  def _run_worker_wrapper(cfg: VariantConfig, save_to: Path) -> dict[str, Any]:
      from mlx_teacache import apply_teacache

      flux = _load_flux(cfg.loader)
      times: list[float] = []
      skipped: list[int] = []
      computed: list[int] = []
      thresh_used: float = 0.0
      for i in range(REPS):
          with apply_teacache(flux) as handle:
              if i == 0:
                  thresh_used = handle.rel_l1_thresh
              elapsed, image = _generate(
                  flux, num_inference_steps=cfg.num_inference_steps, guidance=cfg.guidance
              )
              times.append(elapsed)
              skipped.append(handle.stats.skipped_count)
              computed.append(handle.stats.computed_count)
              if i == 0:
                  _save_as_webp(image, save_to)
          print(
              f"  wrapper rep {i + 1}: {elapsed:.2f}s "
              f"(skipped {skipped[-1]}/{cfg.num_inference_steps})",
              flush=True,
          )
      return {
          "condition": "wrapper",
          "rep_seconds": times,
          "skipped_per_rep": skipped,
          "computed_per_rep": computed,
          "rel_l1_thresh_used": thresh_used,
      }


  def _worker_main(args: argparse.Namespace) -> None:
      """Subprocess entrypoint. Runs one (variant, condition) pair and prints
      a single JSON line prefixed by WORKER_RESULT_SENTINEL on stdout."""
      cfg = next(v for v in VARIANTS if v.slug == args.variant)
      save_to: Path = Path(args.save_to)
      if args.condition == "vanilla":
          result = _run_worker_vanilla(cfg, save_to)
      elif args.condition == "wrapper":
          result = _run_worker_wrapper(cfg, save_to)
      else:
          raise ValueError(f"unknown --condition {args.condition!r}")
      print(f"{WORKER_RESULT_SENTINEL}{json.dumps(result)}", flush=True)


  # ---------------------------------------------------------------------------
  # ORCHESTRATOR side — spawns the workers and assembles the report.
  # ---------------------------------------------------------------------------


  def _mflux_version() -> str:
      try:
          from importlib.metadata import version

          return version("mflux")
      except Exception:
          return "unknown"


  def _mlx_teacache_version() -> str:
      from mlx_teacache import __version__

      return __version__


  def _macos_sysctl(key: str) -> str | None:
      """Read a macOS sysctl value as a string. Returns None on failure."""
      if sys.platform != "darwin":
          return None
      try:
          out = subprocess.run(
              ["sysctl", "-n", key], capture_output=True, text=True, check=True
          )
          return out.stdout.strip() or None
      except (FileNotFoundError, subprocess.CalledProcessError):
          return None


  def _detect_hardware(machine_label_override: str | None, ram_gb_override: int | None) -> dict[str, Any]:
      """Plan-audit Finding 4 fix: hardware provenance recorded in the JSON.

      Reads chip name + RAM via macOS sysctl. CLI flags override whatever
      sysctl reports if the marketing chip name is missing or wrong (e.g.
      a future macOS that doesn't expose `machdep.cpu.brand_string` cleanly)."""
      chip = machine_label_override or _macos_sysctl("machdep.cpu.brand_string") or platform.processor() or "Apple Silicon"
      ram_bytes_str = _macos_sysctl("hw.memsize")
      ram_gb: int | None = ram_gb_override
      if ram_gb is None and ram_bytes_str is not None:
          try:
              ram_gb = round(int(ram_bytes_str) / (1024**3))
          except ValueError:
              ram_gb = None
      return {
          "chip": chip,
          "ram_gb": ram_gb,  # may be None if neither sysctl nor override yielded a value
          "machine": platform.machine(),
          "os": f"{platform.system()} {platform.release()}",
          "mlx_teacache_version": _mlx_teacache_version(),
          "mflux_version": _mflux_version(),
          "quantize": 4,
          "dtype": "bf16",
      }


  def _run_one_worker(slug: str, condition: str, save_to: Path) -> dict[str, Any]:
      """Spawn the worker subprocess and capture its result line."""
      cmd = [
          sys.executable,
          str(Path(__file__).resolve()),
          "--worker",
          "--variant",
          slug,
          "--condition",
          condition,
          "--save-to",
          str(save_to),
      ]
      print(f"\n>> spawning worker: {slug} / {condition}", flush=True)
      proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
      # Stream child stdout/stderr to the orchestrator's stdout so progress is visible.
      if proc.stdout:
          sys.stdout.write(proc.stdout)
      if proc.stderr:
          sys.stderr.write(proc.stderr)
      if proc.returncode != 0:
          raise RuntimeError(
              f"worker failed for {slug}/{condition}: exit {proc.returncode}"
          )
      # Find the result line in stdout.
      for line in proc.stdout.splitlines():
          if line.startswith(WORKER_RESULT_SENTINEL):
              return json.loads(line[len(WORKER_RESULT_SENTINEL):])
      raise RuntimeError(
          f"worker for {slug}/{condition} did not emit a {WORKER_RESULT_SENTINEL} result line"
      )


  def _orchestrate(cfg: VariantConfig, base_dir: Path) -> dict[str, Any]:
      """Run vanilla + wrapper subprocesses for one variant; merge into a JSON entry."""
      variant_dir = base_dir / cfg.slug

      vanilla_path = variant_dir / "vanilla.webp"
      vanilla = _run_one_worker(cfg.slug, "vanilla", vanilla_path)

      wrapper_path = variant_dir / "wrapper.webp"
      wrapper = _run_one_worker(cfg.slug, "wrapper", wrapper_path)

      vanilla_times = vanilla["rep_seconds"]
      wrapper_times = wrapper["rep_seconds"]
      vanilla_cold = vanilla_times[0]
      vanilla_warm = statistics.median(vanilla_times[1:])
      wrapper_cold = wrapper_times[0]
      wrapper_warm = statistics.median(wrapper_times[1:])
      speedup_warm = vanilla_warm / wrapper_warm if wrapper_warm else 0.0
      speedup_cold = vanilla_cold / wrapper_cold if wrapper_cold else 0.0

      print(
          f"  cold: vanilla {vanilla_cold:.2f}s | wrapper {wrapper_cold:.2f}s "
          f"| speedup_cold {speedup_cold:.2f}x"
      )
      print(
          f"  warm: vanilla {vanilla_warm:.2f}s | wrapper {wrapper_warm:.2f}s "
          f"| speedup_warm {speedup_warm:.2f}x"
      )

      return {
          "variant_id": cfg.variant_id,
          "num_inference_steps": cfg.num_inference_steps,
          "guidance": cfg.guidance,
          "vanilla": {
              "rep_seconds": vanilla_times,
              "cold_seconds": vanilla_cold,
              "warm_median_seconds": vanilla_warm,
          },
          "wrapper": {
              "rep_seconds": wrapper_times,
              "cold_seconds": wrapper_cold,
              "warm_median_seconds": wrapper_warm,
              "skipped_per_rep": wrapper["skipped_per_rep"],
              "computed_per_rep": wrapper["computed_per_rep"],
              "rel_l1_thresh_used": wrapper["rel_l1_thresh_used"],
          },
          "speedup_warm": speedup_warm,
          "speedup_cold": speedup_cold,
          "image_paths": {
              "vanilla": str(vanilla_path.relative_to(base_dir.parent.parent)),
              "wrapper": str(wrapper_path.relative_to(base_dir.parent.parent)),
          },
      }


  def main() -> None:
      parser = argparse.ArgumentParser(description=__doc__)
      parser.add_argument(
          "--worker",
          action="store_true",
          help="(internal) run as a worker subprocess for one (variant, condition) pair.",
      )
      parser.add_argument("--variant", help="Variant slug (worker mode only).")
      parser.add_argument("--condition", help="vanilla or wrapper (worker mode only).")
      parser.add_argument("--save-to", help="Image destination path (worker mode only).")
      parser.add_argument(
          "--output-root",
          type=Path,
          default=Path(__file__).parent.parent / "_artifacts",
          help="Root directory for outputs. Default: <repo>/_artifacts/",
      )
      parser.add_argument(
          "--machine-label",
          default=None,
          help="Override the chip name written into comparison_report.json's hardware section "
               "(e.g. 'Apple M1 Max'). Defaults to macOS sysctl machdep.cpu.brand_string.",
      )
      parser.add_argument(
          "--ram-gb",
          type=int,
          default=None,
          help="Override the RAM-GB field. Defaults to round(hw.memsize / 1 GiB) on macOS.",
      )
      args = parser.parse_args()

      if args.worker:
          _worker_main(args)
          return

      base_dir: Path = args.output_root / "comparison"
      base_dir.mkdir(parents=True, exist_ok=True)
      report_path: Path = args.output_root / "comparison_report.json"

      from datetime import datetime, timezone

      report: dict[str, Any] = {
          "schema_version": 1,
          "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
          "hardware": _detect_hardware(args.machine_label, args.ram_gb),
          "prompt": PROMPT,
          "seed": SEED,
          "height": HEIGHT,
          "width": WIDTH,
          "reps_per_condition": REPS,
          "isolation": "subprocess-per-condition",
          "variants": {},
      }

      for cfg in VARIANTS:
          print(f"\n=== {cfg.variant_id} (slug={cfg.slug}) — "
                f"{cfg.num_inference_steps} steps, guidance={cfg.guidance} ===")
          report["variants"][cfg.slug] = _orchestrate(cfg, base_dir)

      report_path.write_text(json.dumps(report, indent=2))
      print(f"\nReport written: {report_path}")
      for slug, entry in report["variants"].items():
          print(
              f"  {slug:24s} "
              f"vanilla_warm={entry['vanilla']['warm_median_seconds']:.2f}s "
              f"wrapper_warm={entry['wrapper']['warm_median_seconds']:.2f}s "
              f"speedup_warm={entry['speedup_warm']:.2f}x "
              f"skipped[0]={entry['wrapper']['skipped_per_rep'][0]}"
          )


  if __name__ == "__main__":
      main()
  ```

- [ ] **Step 2: Argparse smoke (no model load)**

  ```bash
  uv run python scripts/bench_comparison.py --help
  ```

  Expected: prints docstring + the `--output-root` option, exits 0.

- [ ] **Step 3: Lint + format**

  ```bash
  uv run ruff check scripts/bench_comparison.py && uv run ruff format --check scripts/bench_comparison.py
  ```

  Expected: green. If ruff format wants changes, apply them with `uv run ruff format scripts/bench_comparison.py` and re-check.

- [ ] **Step 4: Type-check**

  ```bash
  uv run mypy scripts/bench_comparison.py
  ```

  Expected: success. If mypy flags the mflux import (no stubs), that's pre-existing — only flag genuinely new mypy errors introduced by this file.

- [ ] **Step 5: Pure-core sanity (the script isn't imported by tests, but make sure nothing regressed)**

  ```bash
  uv run pytest tests/ -m "not parity and not slow and not benchmark and not network"
  ```

  Expected: 123 passed.

- [ ] **Step 6: Commit**

  ```bash
  git add scripts/bench_comparison.py
  git commit -m "feat(scripts): bench_comparison.py — vanilla vs mlx-teacache on non-distilled FLUX variants"
  ```

---

## Task 3: Run the comparison bench end-to-end (MAIN THREAD)

This is the heavy ML step. Per CLAUDE.md "Heavy generations: main thread, not subagents," run from the main session with `run_in_background=true` and tee to `/tmp/`. Estimated total wall-clock: ~25-35 minutes on M1 Max (flux1-dev ~3 min × 6 reps + klein-base-4b g=1.0 ~5 min × 6 reps + klein-base-4b CFG ~15 min × 6 reps; but the script also includes per-variant warmup costs from MLX kernel JIT and the larger 1024×768 resolution).

**Files (created by the run, NOT touched by hand):**
- Create: `_artifacts/comparison/flux1-dev/{vanilla,wrapper}.webp`
- Create: `_artifacts/comparison/klein-base-4b-g1/{vanilla,wrapper}.webp`
- Create: `_artifacts/comparison/klein-base-4b-cfg/{vanilla,wrapper}.webp`
- Create: `_artifacts/comparison_report.json`

- [ ] **Step 1: Confirm weights are reachable**

  ```bash
  hf download black-forest-labs/FLUX.1-dev --dry-run | head -2
  hf download black-forest-labs/FLUX.2-klein-base-4B --dry-run | head -2
  ```

  Both should show `Will download 0 files` (already cached locally).

- [ ] **Step 2: Run the bench in the background, teed to /tmp/**

  Use the Bash tool with `run_in_background=true`:

  ```bash
  uv run python scripts/bench_comparison.py 2>&1 | tee /tmp/bench-comparison.log
  ```

  Tee to `/tmp/bench-comparison.log` so progress is observable via `tail -f` and survives the background-task watchdog. Expected: ~25-35 minutes wall-clock total. The script prints `=== <variant_id> ===` headers + per-rep timings as it goes.

- [ ] **Step 3: Verify the outputs exist and are well-formed**

  After the background task notifies completion:

  ```bash
  ls -lh _artifacts/comparison/flux1-dev/*.webp
  ls -lh _artifacts/comparison/klein-base-4b-g1/*.webp
  ls -lh _artifacts/comparison/klein-base-4b-cfg/*.webp
  ls -lh _artifacts/comparison_report.json
  ```

  Each webp should be 50-300 KB. The JSON should be a few KB.

- [ ] **Step 4: Sanity-check the JSON**

  Read the JSON file with the Read tool (don't pipe through python — we don't have it installed by default):

  ```bash
  head -40 _artifacts/comparison_report.json
  ```

  Verify:
  - `"schema_version": 1`
  - `"prompt"` field contains the auburn-hair portrait prompt
  - `"variants"` has 3 keys: `"flux1-dev"`, `"klein-base-4b-g1"`, `"klein-base-4b-cfg"`
  - Each variant has `"rep_seconds": [<a>, <b>, <c>]` for both vanilla + wrapper
  - Each variant has `"skipped_per_rep"` with non-negative ints
  - `"speedup_warm"` ≥ 1.0 for the non-CFG entries (where skips fire)

- [ ] **Step 5: Inspect the images visually**

  Open the 6 webp files in a viewer (Preview.app, browser, etc.) and check:
  - Vanilla and wrapper images for the same variant look near-identical at first glance (TeaCache preserves quality).
  - File sizes are reasonable (50-300 KB each).
  - No corrupted / black / banded outputs.

  If any image is corrupted or visually wrong, STOP and report — re-running the failing variant is cheaper than shipping bad assets.

- [ ] **Step 6: Commit the artifacts**

  ```bash
  git add _artifacts/comparison/ _artifacts/comparison_report.json
  git status --short    # Sanity: 6 webp files + 1 JSON should be staged
  git commit -m "data(comparison): generate vanilla+wrapper images and timing report for COMPARISON.md"
  ```

---

## Task 4: Write `COMPARISON.md` with real numbers from the JSON

Hand-write the doc using the numbers from `_artifacts/comparison_report.json` (Task 3 produces it). Every cited number must come from the JSON; the JSON is the recovery source.

**Files:**
- Create: `COMPARISON.md`

- [ ] **Step 1: Open `_artifacts/comparison_report.json` and extract the numbers you'll need**

  Read with the Read tool. Pull out for each of the three variants:
  - `vanilla.cold_seconds`, `vanilla.warm_median_seconds`
  - `wrapper.cold_seconds`, `wrapper.warm_median_seconds`
  - `wrapper.skipped_per_rep[0]` (the cold-rep skip count; should equal warm reps' since seed is fixed)
  - `speedup_warm`, `speedup_cold`
  - `wrapper.rel_l1_thresh_used`

- [ ] **Step 2: Create `COMPARISON.md` at repo root**

  Write `COMPARISON.md` with the template below, replacing `<X>` placeholders with the real values from the JSON. Round to 1 decimal place for seconds, 2 decimals for speedup ratios.

  ````markdown
  # Vanilla mflux vs mlx-teacache — side-by-side comparison

  Visual + timing showcase across non-distilled FLUX variants. Distilled
  schedules (FLUX.1 schnell, FLUX.2 Klein 4B and 9B at default steps) are
  excluded — they don't engage the polynomial gate. See the
  [README](README.md#when-to-use) for guidance on picking the right variant.

  ## Test machine

  - M1 Max 32GB unified memory
  - macOS 26.x
  - mlx-teacache 0.4.1, mflux 0.17.5
  - bf16 weights, `quantize=4`
  - 768 × 1024 portrait
  - Seed: 42
  - 3 reps per condition (1 cold + 2 warm), warm column is the median of reps 2 and 3

  All numbers below come from [`_artifacts/comparison_report.json`](_artifacts/comparison_report.json),
  which is regenerated by `uv run python scripts/bench_comparison.py` and committed alongside the images.

  ## Prompt

  > Portrait of a young woman with auburn hair and green eyes, soft golden-hour
  > window light, photorealistic, shallow depth of field, 50mm prime lens,
  > subtle freckles, neutral background, cinematic color grading.

  ---

  ## FLUX.1 family

  ### `flux1-dev` — 25 steps, guidance=3.5

  | | Vanilla mflux | mlx-teacache |
  |---|---|---|
  | Steps | 25 | 25 |
  | Guidance | 3.5 | 3.5 |
  | Seed | 42 | 42 |
  | **Cold (rep 1)** | <X>s | <Y>s |
  | **Warm (median reps 2-3)** | <X>s | <Y>s |
  | Skips | — | <N> / 25 |
  | Speedup (warm) | 1.00× | **<Z>×** |

  | Vanilla mflux | mlx-teacache |
  |---|---|
  | ![vanilla](_artifacts/comparison/flux1-dev/vanilla.webp) | ![wrapper](_artifacts/comparison/flux1-dev/wrapper.webp) |

  ---

  ## FLUX.2 family

  ### `flux2-klein-base-4b` — 25 steps, guidance=1.0

  | | Vanilla mflux | mlx-teacache |
  |---|---|---|
  | Steps | 25 | 25 |
  | Guidance | 1.0 | 1.0 |
  | Seed | 42 | 42 |
  | **Cold (rep 1)** | <X>s | <Y>s |
  | **Warm (median reps 2-3)** | <X>s | <Y>s |
  | Skips | — | <N> / 25 |
  | Speedup (warm) | 1.00× | **<Z>×** |

  | Vanilla mflux | mlx-teacache |
  |---|---|
  | ![vanilla](_artifacts/comparison/klein-base-4b-g1/vanilla.webp) | ![wrapper](_artifacts/comparison/klein-base-4b-g1/wrapper.webp) |

  ### `flux2-klein-base-4b` (CFG) — 50 steps, guidance=4.0

  The canonical upstream BFL recipe.

  | | Vanilla mflux | mlx-teacache |
  |---|---|---|
  | Steps | 50 | 50 |
  | Guidance | 4.0 | 4.0 |
  | Seed | 42 | 42 |
  | **Cold (rep 1)** | <X>s | <Y>s |
  | **Warm (median reps 2-3)** | <X>s | <Y>s |
  | Skips | — | <N> / 50 |
  | Speedup (warm) | 1.00× | **<Z>×** |

  | Vanilla mflux | mlx-teacache |
  |---|---|
  | ![vanilla](_artifacts/comparison/klein-base-4b-cfg/vanilla.webp) | ![wrapper](_artifacts/comparison/klein-base-4b-cfg/wrapper.webp) |

  ---

  ## Reproducing

  ```bash
  uv run python scripts/bench_comparison.py
  ```

  Regenerates all three entries end-to-end on M1 Max in about 30 minutes.
  Outputs land in `_artifacts/comparison/` (vanilla + wrapper `.webp` per variant)
  and `_artifacts/comparison_report.json` (every per-rep timing + skip count + hardware).
  ````

  Use the JSON-extracted numbers for each `<X>`/`<Y>`/`<N>`/`<Z>` slot.

- [ ] **Step 3: Run the humanizer skill over the new prose**

  Per `~/.claude/CLAUDE.md` "Public-facing docs: humanize before shipping," invoke the humanizer skill over the COMPARISON.md prose. Apply its suggestions but DO NOT change any numbers — the humanizer is for prose, not for the data tables.

  After the humanizer pass, eyeball the doc again:
  - Em-dash overuse cleaned up?
  - No "stands as / serves as" copula avoidance?
  - No rule-of-three filler?
  - Numbers still match the JSON?

- [ ] **Step 4: Pure-core sanity**

  ```bash
  uv run pytest tests/ -m "not parity and not slow and not benchmark and not network"
  ```

  Expected: 123 passed (no test should be affected by adding a markdown file).

- [ ] **Step 5: Commit**

  ```bash
  git add COMPARISON.md
  git commit -m "docs: COMPARISON.md — vanilla vs mlx-teacache side-by-side on non-distilled FLUX variants"
  ```

---

## Task 5: Add "When to use" section + COMPARISON.md link to `README.md`

The README's "Supported variants" table is the current entry point. Add a follow-on section right after it that warns away from distilled use and links to COMPARISON.md.

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Locate the existing "Supported models" section**

  Note: the README heading is `## Supported models` (per plan-audit minor correction; the spec/plan earlier called it "Supported variants"). The insertion point is between the supported-models table footnotes and the next `## ` heading.

  ```bash
  grep -n "^## Supported models\|^## Benchmarks\|^## Quick start\|^## Combining" README.md | head -5
  ```

  Identify the line numbers. The new "When to use" section goes after the supported-models footnotes, before the next `## ` heading.

- [ ] **Step 2: Insert the new section**

  Use the Edit tool. Find the last line of the supported-models section (the last footnote) and add a new section right after it.

  Plan-audit Finding 3 fix: the new section's wording must distinguish **TeaCache step-skipping** from **FLUX.2 `mx.compile`-path avoidance**. Saying "the wrapper adds about 1-2% overhead and changes nothing" is wrong on FLUX.2 distilled Klein — the wrapper still gives 1.3-1.9× wall-clock there because it sidesteps mflux's compiled `_predict`. The recommendation needs to be about the algorithmic gate, not the wrapper as a whole.

  The new section content (paste verbatim — humanizer will be run in step 3):

  ```markdown
  ## When to use

  mlx-teacache helps in two distinct ways depending on the variant: algorithmic TeaCache step-skipping (the headline feature) and FLUX.2 `mx.compile`-path avoidance (an incidental wall-clock benefit on chips where mflux compiles `_predict`).

  For TeaCache step-skipping, use non-distilled schedules. Distilled variants like FLUX.1 schnell (4 steps) or FLUX.2 Klein 4B and 9B at their distilled defaults (4-8 steps) skip zero steps at the package default threshold — adjacent transformer outputs change too much between consecutive steps for caching to fire.

  FLUX.2 distilled Klein still shows about 1.3-1.9× wall-clock improvement on Max/Ultra chips because the wrapper sidesteps mflux's compiled `_predict` (a different mechanism, documented separately). That is compile-path avoidance rather than caching. [COMPARISON.md](COMPARISON.md) focuses on the variants where the gate itself engages.

  Recommended variants for measurable step-skipping:

  - `flux1-dev` at 20-50 steps
  - `flux2-klein-base-4b` at 20-50 steps (with or without CFG)
  ```

- [ ] **Step 3: Humanizer pass on the new section**

  Invoke the humanizer skill on the new "When to use" paragraphs only. Apply suggested rewrites. Numbers (1-2%, 20-50 steps) stay as-is.

- [ ] **Step 4: Pure-core sanity**

  ```bash
  uv run pytest tests/ -m "not parity and not slow and not benchmark and not network"
  ```

  Expected: 123 passed.

- [ ] **Step 5: Commit**

  ```bash
  git add README.md
  git commit -m "docs(readme): add 'When to use' section + link to COMPARISON.md"
  ```

---

## Task 6: Open the PR + stop for review

Per the new release-flow rule in `~/.claude/CLAUDE.md`: open the PR, hand the link back, **do not** local-merge.

**Files:** none modified by this task; it pushes the branch and opens the PR.

- [ ] **Step 1: Pre-push CI sanity locally**

  ```bash
  uv run ruff check . && uv run ruff format --check . && uv run pytest tests/ -m "not parity and not slow and not benchmark and not network"
  ```

  Expected: ruff + format green, 123 tests pass.

- [ ] **Step 2: Push the branch**

  ```bash
  git push -u origin feature/comparison-md-and-cleanup
  ```

  Expected: branch created on origin, tracking set up.

- [ ] **Step 3: Open the PR**

  ```bash
  gh pr create --title "docs: COMPARISON.md side-by-side + non-distilled-only recommendation" --body "$(cat <<'EOF'
  ## Summary

  Adds a `COMPARISON.md` at repo root with three non-distilled FLUX entries (`flux1-dev` at 25 steps, `flux2-klein-base-4b` at g=1.0 / 25 steps, and `flux2-klein-base-4b` at g=4.0 / 50 steps — the canonical upstream CFG recipe). Each entry has cold (rep 1) and warm (median reps 2-3) timings, skip counts, and a side-by-side vanilla-vs-wrapper portrait image at 768×1024 webp.

  ### What's committed

  - `COMPARISON.md` at repo root, linked from the README's new "When to use" section.
  - `_artifacts/comparison/<variant>/{vanilla,wrapper}.webp` — 6 portrait webp images, ~50-300 KB each.
  - `_artifacts/comparison_report.json` — complete recoverable record: schema_version, hardware (chip, RAM, OS, library versions, dtype, quantize), prompt, seed, dimensions, and per-rep timings + skip counts for every variant. COMPARISON.md is rebuildable from this JSON if it's ever lost.
  - `scripts/bench_comparison.py` — the generator script, reproducible via `uv run python scripts/bench_comparison.py`.
  - `.gitignore` comment clarifying that the new repo-root `_artifacts/` is committed (the existing `tests/_artifacts/` stays gitignored — different purpose).
  - `README.md` "When to use" section explicitly recommends against distilled schedules (schnell, Klein 4B / 9B at defaults) and points users at `flux1-dev` or `flux2-klein-base-4b` for measurable speedups.

  ### What's measured

  All numbers in `_artifacts/comparison_report.json`. Headline results in COMPARISON.md tables. M1 Max 32GB, mflux 0.17.5, bf16, quantize=4, 768×1024 portrait, seed=42.

  ### What's not included

  Distilled variants (`flux1-schnell`, `flux2-klein-4b`, `flux2-klein-9b`) are deliberately excluded from COMPARISON.md per the new "When to use" recommendation. Their existing README "Benchmarks" rows stay as factual reference but the comparison doc focuses on where the library actually helps.

  ## Test plan

  - [x] `_artifacts/comparison/` has 6 webp images, each under 300 KB.
  - [x] `_artifacts/comparison_report.json` has all three variants with per-rep data, schema_version=1.
  - [x] `COMPARISON.md` numbers match `comparison_report.json` exactly.
  - [x] `scripts/bench_comparison.py --help` works (argparse smoke).
  - [x] Pure-core tests still pass (no code changes outside scripts/ + docs/).
  - [x] ruff + format clean.

  ## Reproducer

  ```bash
  uv run python scripts/bench_comparison.py
  ```
  EOF
  )"
  ```

- [ ] **Step 4: Watch CI**

  Use Bash with `run_in_background=true` to watch the checks:

  ```bash
  gh pr checks --watch
  ```

  Wait for all checks to land. Expected: lint, typecheck, test-pure-core, test-mflux (3.11/3.12/3.13), coverage all pass. test-parity skips (no GPU in CI).

- [ ] **Step 5: STOP — hand the PR link to the user**

  Per the release-flow rule, do NOT call `gh pr merge`. Report back with:
  - PR URL.
  - Branch name: `feature/comparison-md-and-cleanup`.
  - CI status (which checks green).
  - The human-language release summary: what the PR ships, why, the bench numbers from `comparison_report.json`, and where the new images live.

  Wait for the user to merge on GitHub.

---

## Task 7: After merge — pull main, no tag

Unlike v0.4.x releases, this is a docs-only change and does NOT cut a new tag. mlx-teacache stays at 0.4.1 on PyPI. After the human merges on GitHub:

**Files:** none.

- [ ] **Step 1: Pull merged main**

  ```bash
  git checkout main && git pull --ff-only && git log --oneline -3
  ```

  Expected: latest commit is the squash-merge of this PR.

- [ ] **Step 2: Confirm no tag is needed**

  Mentally verify: this PR adds docs + images, not user-facing code. No PyPI release. README still cites `mlx-teacache==0.4.1`.

---

## Task 8: Out-of-repo — refresh `~/.claude/CLAUDE.md`

**Files (out of repo, no PR):**
- Modify: `~/.claude/CLAUDE.md`

- [ ] **Step 1: Open `~/.claude/CLAUDE.md` and read it end-to-end**

  Use Read tool on `/Users/ionden/.claude/CLAUDE.md`. Read the full file. Note any paragraph that:
  - References specific v0.x.y behavior that's no longer current.
  - Cites a workaround for a problem that's since been fixed.
  - Repeats a rule covered better elsewhere.
  - Could be more concise.

- [ ] **Step 2: For each candidate edit, decide: edit, leave, or remove**

  Apply the rule: only edit if the current text is wrong, stale, or ambiguous. Don't restyle for its own sake.

  Specific paragraphs to evaluate from the existing CLAUDE.md (commit `742e91b`-era content):

  1. **"Git commits / no Co-Authored-By":** still load-bearing. Leave.
  2. **"HuggingFace CLI":** still load-bearing. Leave.
  3. **"MLX work / invoke user-mlx-developer skill":** mentions `mlx-taef` (a sibling worktree). If that worktree still exists, leave. If it's gone, update the list.
  4. **"Heavy generations: main thread, not subagents":** v0.4.1 confirmed this rule. Leave.
  5. **"Public-facing docs: humanize":** still load-bearing. Leave.
  6. **"Performance claims need a committed benchmark":** v0.4.1's three-way bench reinforced it. Leave.
  7. **"Release flow: branch → PR → stop":** new in this session. Leave.

  Likely net edits: zero or one. If zero, skip steps 3-4.

- [ ] **Step 3: Apply targeted edits (if any)**

  Use the Edit tool. One small edit per concern, with a clear reason in your mental model.

- [ ] **Step 4: Verify the file is still well-formed**

  ```bash
  wc -l /Users/ionden/.claude/CLAUDE.md
  head -5 /Users/ionden/.claude/CLAUDE.md
  ```

  Expected: file has a markdown header at the top, line count is in the same ballpark as before (small edit).

  Out-of-repo files don't go through git, so there's no commit step.

---

## Task 9: Out-of-repo — refresh `~/.claude/skills/user-mlx-developer/`

**Files (out of repo, no PR):**
- Modify: files under `~/.claude/skills/user-mlx-developer/`

- [ ] **Step 1: Inventory the skill files**

  ```bash
  ls -la /Users/ionden/.claude/skills/user-mlx-developer/
  ls -la /Users/ionden/.claude/skills/user-mlx-developer/references/ 2>/dev/null
  ```

  Expected: a `SKILL.md` plus a `references/` directory with topic files (e.g. `lazy-eval.md`, `compile-state-capture.md`, `mflux-callback-protocols.md`).

- [ ] **Step 2: Read `SKILL.md` end-to-end**

  Use Read tool. Note what the skill says about:
  - `mx.compile` and graph state capture
  - mflux's callback protocols
  - Polynomial coefficient calibration
  - CFG handling
  - Anything else relevant to v0.4.x work

- [ ] **Step 3: Identify additions from v0.4.0 + v0.4.1 lessons**

  Things to fold in (only if not already covered):

  1. **mx.compile graph topology fragility under eager wrapping.** v0.4.0 fixed a bug where computing `temb_mod_params_single` upfront vs inline in `_flux2_run_body` produced bit-different MLX outputs even though the math was identical. When reimplementing a function that vanilla mflux wraps in `mx.compile`, mirror the graph topology exactly — including where intermediates are computed inside the body loops, not hoisted to the prelude.

  2. **CFG per-branch caching pattern.** v0.4.1: the `mod_in` gating signal in FLUX.2 is encoder-independent at a fixed `(latents, timestep)`. That lets one shared gate decision drive two cached residuals (one per CFG branch). Worth recording as a pattern for future cache designs on diffusion transformers where the gate input is shared but the cached value isn't.

  3. **Polynomial transfer is empirical, not architectural.** v0.4.1 plan-audit Finding 3: the per-step mod_in invariant doesn't prove that g=1.0-fit coefficients work at g=4.0 — that's a trajectory-level property that has to be measured. Warn against shipping fits across recipes without an empirical validation step.

  4. **Lazy mflux imports for deferred-dep public APIs.** `apply_teacache(...)` defers all mflux imports inside the function body so `from mlx_teacache import apply_teacache` works on a machine without mflux installed. Pattern: import inside the function, not at module top, when the dep is optional.

- [ ] **Step 4: Apply targeted edits to `SKILL.md` and/or relevant `references/*.md`**

  Use Edit / Write tools. Aim for paragraph-level additions, not whole new sections unless the lesson genuinely needs one. Each addition should reference what the source incident was (e.g. "v0.4.0 fixed this in `forward.py:_flux2_run_body`") so the lesson is anchored to a verifiable artifact.

- [ ] **Step 5: If a new sub-reference file is warranted, create it**

  E.g. `references/cfg-per-branch-caching.md` if the CFG pattern is substantial enough. Otherwise tuck the note into an existing file.

- [ ] **Step 6: Skim the result**

  Re-read `SKILL.md` and any modified files. Verify nothing reads as contradicting itself after edits. Out-of-repo, no commit step.

---

## Self-review

After writing this plan, I checked it against the spec:

1. **Spec coverage:**
   - §3.1 file layout → Tasks 1 + 2 + 3 (gitignore, script, artifacts).
   - §3.2 COMPARISON.md structure → Task 4 (template embedded).
   - §3.3 generation script → Task 2 (full script body embedded).
   - §3.4 cold/warm definition → Task 2 step 1 (script implements it) + Task 4 step 2 (doc explains it).
   - §3.5 README integration → Task 5.
   - §3.6 .gitignore adjustment → Task 1.
   - JSON schema (updated post-spec) → Task 2's script and Task 3's verification.
   - Item 1 CLAUDE.md → Task 8.
   - Item 2 user-mlx-developer skill → Task 9.
   - Release flow (no local merge) → Task 6 + Task 7.

2. **Placeholder scan:** the `<X>`, `<Y>`, `<N>`, `<Z>` in Task 4's COMPARISON.md template are intentional placeholders that the engineer replaces with JSON-extracted values; the JSON is generated in Task 3 so the data is concrete by the time Task 4 runs. Not plan failures.

3. **Type consistency:** the `_run_wrapper_reps` function returns `(times, skipped, computed, thresh_used)`; Task 2's caller `_bench_one_variant` unpacks it as `wrapper_times, skipped, computed, thresh_used`. Variant `slug` field exists on `VariantConfig` (Task 2) and matches the directory name in Task 3's verification step and Task 4's image paths.

No gaps found, no fixes needed.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-17-comparison-doc-and-cleanup.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task (with the heavy-ML Task 3 still running from the main thread per CLAUDE.md), review between tasks. Fast iteration, full audit trail.
2. **Inline Execution** — execute tasks in this session, batch checkpoints. Heavier on main-thread context.

Which approach?
