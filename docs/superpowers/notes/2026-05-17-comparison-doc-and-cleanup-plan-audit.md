# COMPARISON.md + Global Config Cleanup Plan Audit

**Plan:** `docs/superpowers/plans/2026-05-17-comparison-doc-and-cleanup.md`  
**Spec:** `docs/superpowers/specs/2026-05-17-comparison-doc-and-cleanup-design.md`  
**Audit date:** 2026-05-17  
**Scope:** material plan issues only. I ignored normal execution friction and did not review this as TDD/lint/syntax work.

## Verdict

The plan's intent is good: a dedicated comparison doc, committed images, and a recoverable JSON source of truth are the right shape for a visual benchmark artifact.

I would fix three things before execution:

1. the proposed image-save path likely fails after the first expensive generation;
2. the "cold" wrapper timing is not actually cold under the scripted protocol;
3. the README "When to use" wording contradicts the current v0.4.1 docs about FLUX.2 distilled Klein speedups.

There is also one provenance fix worth making before committing benchmark artifacts: record the actual chip/RAM in JSON instead of hardcoding it in prose.

## Findings

### 1. The webp conversion path uses a temp filename that mflux/Pillow will not save as PNG

**Severity:** High  
**Files:** `docs/superpowers/plans/2026-05-17-comparison-doc-and-cleanup.md`, proposed `scripts/bench_comparison.py`

The planned `_save_as_webp()` creates:

```python
png_tmp = dest_webp.with_suffix(".png.tmp")
image.save(path=str(png_tmp), export_json_metadata=False)
with Image.open(png_tmp) as pil_img:
    pil_img.save(dest_webp, format="WEBP", quality=WEBP_QUALITY, method=WEBP_METHOD)
```

(`plan:250-261`)

In mflux 0.17.5, `GeneratedImage.save()` delegates to `ImageUtil.save_image()`, which eventually calls Pillow as `image.save(file_path)` (`generated_image.py:101-115`, `image_util.py:229-241`). Pillow infers the format from the path suffix. A file named `vanilla.png.tmp` has final suffix `.tmp`, not `.png`, so Pillow cannot infer PNG. mflux catches and logs that error (`image_util.py:256-257`) instead of raising it, which means the next `Image.open(png_tmp)` will fail because the temp file was never written.

Impact: the first variant can spend minutes generating an image, then crash at the save/convert step before any artifact is produced.

**Fix:** use a temp path with a real `.png` suffix, or bypass mflux's save wrapper:

```python
png_tmp = dest_webp.with_name(dest_webp.stem + ".tmp.png")
image.save(path=str(png_tmp), export_json_metadata=False, overwrite=True)
```

or:

```python
png_tmp = dest_webp.with_name(dest_webp.stem + ".tmp.png")
image.image.save(png_tmp, format="PNG")
```

Keep the `png_tmp.unlink()` cleanup.

### 2. The wrapper "cold" timing is warmed by the vanilla reps

**Severity:** Medium-High  
**Files:** `docs/superpowers/plans/2026-05-17-comparison-doc-and-cleanup.md`, proposed `scripts/bench_comparison.py`, `COMPARISON.md`

The spec defines cold as the first generation after `Flux1(...)` / `Flux2Klein(...).freeze()` returns, with model load excluded (`spec:232-239`). The plan's script loads one `flux` instance per variant, runs all three vanilla reps, and only then runs the wrapper reps (`plan:313-340`). It then labels `wrapper_times[0]` as `wrapper_cold` (`plan:342-348`) and publishes it as "Cold (rep 1)" in `COMPARISON.md` (`plan:595-603`, `plan:620-628`).

That wrapper rep is not cold. It follows three full vanilla generations on the same model instance, after MLX kernels and Metal dispatch paths have already been warmed. The reported `speedup_cold` will therefore compare a genuinely cold vanilla rep against a warmed wrapper rep.

Impact: the comparison doc can overstate cold-start speedup and present a number the JSON cannot honestly support.

**Fix:** choose one of these:

- Drop the cold column and report only warm medians. This is simplest and matches the existing benchmark framing.
- Keep "rep 1" but do not call it cold or compute `speedup_cold`.
- If true cold timing matters, run each condition in a separate subprocess/fresh process and record that protocol in the JSON. Re-loading a model in the same Python process is still not fully cold because MLX kernel state can be process-global.

### 3. The planned "When to use" section misstates distilled FLUX.2 behavior

**Severity:** Medium-High  
**Files:** `docs/superpowers/plans/2026-05-17-comparison-doc-and-cleanup.md`, `docs/superpowers/specs/2026-05-17-comparison-doc-and-cleanup-design.md`, `README.md`, `ROADMAP.md`

The new README section says:

> "On those models the wrapper adds about 1-2% gating overhead and skips zero steps." (`plan:735-745`)

The spec says the same more strongly:

> "the wrapper adds ~1-2% gating overhead and changes nothing for users" (`spec:23-26`)

That is true for the algorithmic TeaCache gate, but it is not true for FLUX.2 distilled Klein on the measured M1 Max path. Current v0.4.1 docs say the wrapper still runs about `1.3-1.9x` faster than vanilla mflux on FLUX.2 Klein at distilled schedules, because it sidesteps mflux's compiled `_predict` path (`README.md:10-12`, `README.md:173-181`, `README.md:216-218`; `ROADMAP.md:80-82`).

Impact: the new recommendation would contradict the current README and make supported FLUX.2 Klein behavior sound worse than it is. The real distinction is "no algorithmic step-skipping on distilled schedules," not "the wrapper is only overhead."

**Fix:** reword the section around mechanism:

> "For TeaCache step-skipping, use non-distilled schedules. Distilled schedules skip zero steps at the default threshold. FLUX.2 distilled Klein can still show wall-clock gains on chips where mflux compiles `_predict`, but that is compile-path avoidance rather than caching, so COMPARISON.md focuses on variants where the gate itself engages."

That keeps the user guidance honest without inviting users to tune distilled thresholds blindly.

### 4. Hardware provenance is not recoverable from the proposed JSON

**Severity:** Medium  
**Files:** `docs/superpowers/plans/2026-05-17-comparison-doc-and-cleanup.md`, proposed `scripts/bench_comparison.py`, `COMPARISON.md`

The plan says `_artifacts/comparison_report.json` is the recovery source of truth for every measured number (`plan:137-140`, `plan:569-606`). But the proposed report records:

```python
"chip": platform.processor() or "Apple Silicon",
"machine": platform.machine(),
"os": f"{platform.system()} {platform.release()}",
```

(`plan:410-421`)

On the current macOS host, `platform.processor()` reports `arm`, and `platform.machine()` reports `arm64`; neither records "M1 Max" or memory size. Meanwhile `COMPARISON.md` and the PR body hardcode "M1 Max 32GB" (`plan:595-603`, `plan:797-810`).

Impact: the committed JSON would not be sufficient to verify the hardware line in the doc, and future reviewers could not tell whether a comparison was regenerated on the intended machine.

**Fix:** add explicit provenance fields before running the bench:

- either CLI flags such as `--machine-label "M1 Max"` and `--ram-gb 32`;
- or macOS `sysctl` calls for `machdep.cpu.brand_string` / `hw.memsize`, with manual override if Apple does not expose the marketing chip name cleanly.

Then build the `COMPARISON.md` test-machine section from JSON rather than hardcoding it.

## Minor Correction

Task 5 says to locate `## Supported variants`, but current `README.md` uses `## Supported models` (`README.md:92`). The intended insertion point is still clear: after the supported-models table footnotes and before `## Combining with mlx-taef`.

## Confirmed Good Decisions

- A dedicated `scripts/bench_comparison.py` is better than overloading `scripts/bench_speedup.py`; release-gate benchmarking and visual showcase generation have different responsibilities.
- Excluding distilled variants from `COMPARISON.md` is reasonable as long as the README explains the difference between step-skipping and FLUX.2 compile-path avoidance.
- Committing a JSON report alongside images is the right way to keep the markdown recoverable and auditable.
- Keeping repo-root `_artifacts/` tracked while leaving `tests/_artifacts/` ignored is fine with the planned `.gitignore` comment.

## Sources Checked

- Current local repo: `README.md`, `ROADMAP.md`, `.gitignore`, `scripts/bench_speedup.py`
- Plan/spec under review: `docs/superpowers/plans/2026-05-17-comparison-doc-and-cleanup.md`, `docs/superpowers/specs/2026-05-17-comparison-doc-and-cleanup-design.md`
- Local mflux 0.17.5 source: `.venv/lib/python3.13/site-packages/mflux/utils/generated_image.py`, `.venv/lib/python3.13/site-packages/mflux/utils/image_util.py`
