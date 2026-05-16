# Roadmap and docs research audit

**Date:** 2026-05-15
**Audits:** `README.md`, `ROADMAP.md`, `docs/m3-plus-tradeoff.md`, `docs/manual-verification.md`
**Local dependency baseline:** `mflux==0.17.5`, `mlx==0.31.2` on Darwin
**Status:** Addendum. I did not edit the public docs; this file lists corrections and useful additions to fold in.

## Executive take

The updated roadmap is directionally good, but a few details should be fixed before it becomes public release guidance:

1. The M1/M2 Pro compile classification is wrong for current mflux.
2. The manual verification recipe is likely broken: mflux `GeneratedImage` has no `.latents`, and FLUX.2 threshold-zero is not bit-exact by this project's own tests.
3. README has broken internal links to `docs/calibration.md` and a missing `docs/superpowers/spikes/...` file.
4. Z-Image should move up: mflux 0.17.5 already has a Z-Image implementation, so the roadmap risk is not "needs a new mflux variant first".
5. FLUX.2 Klein 9B has license/gating and safety-filter obligations that are more important than disk size alone.

## High-impact corrections

### 1. M1/M2 Pro are eager in current mflux, not compiled

`docs/m3-plus-tradeoff.md` and the roadmap table group M1 Pro / M2 Pro with Max / Ultra as compiled. Current mflux does not do that.

Local mflux source:

- `.venv/lib/python3.13/site-packages/mflux/utils/apple_silicon.py:14-17`
- `.venv/lib/python3.13/site-packages/mflux/models/flux2/variants/txt2img/flux2_klein.py:279-281`

The predicate is:

```python
chip_name = cls._get_chip_name().lower()
if "max" in chip_name or "ultra" in chip_name:
    return False
return "apple m1" in chip_name or "apple m2" in chip_name
```

Assuming Apple reports the brand string as `Apple M1 Pro` / `Apple M2 Pro`, Pro chips return `True` and mflux uses eager `_predict`. Only Max and Ultra are explicitly excluded. M3/M4/M5 return `False` because they are not M1/M2, so they use `mx.compile`.

Suggested table:

| Chip | Vanilla mflux `_predict` in 0.17.5 | Note |
|---|---|---|
| M1 / M2 base | eager | current docs correct |
| M1 Pro / M2 Pro | likely eager | current docs likely wrong |
| M1 Max / Ultra, M2 Max / Ultra | compiled | current docs correct |
| M3 / M4 / M5 generations | compiled | current docs correct |

This matters for the benchmark expectations: Pro users should not be warned about losing mflux compile until this is rechecked on actual hardware or mflux changes the predicate.

### 2. Manual verification uses a nonexistent `.latents` attribute

`docs/manual-verification.md` currently does:

```python
r1 = flux.generate_image(...)
r2 = flux.generate_image(...)
assert mx.array_equal(r1.latents, r2.latents)
```

In mflux 0.17.5, `GeneratedImage` stores image metadata but no latent tensor. Local source: `.venv/lib/python3.13/site-packages/mflux/utils/generated_image.py:16-69`.

There is a second issue: the manual recipe uses FLUX.2 Klein 4B, but project tests now document that FLUX.2 threshold-zero wrapper parity is cosine-based, not byte-exact, because vanilla uses compiled `_predict` while the wrapper is eager. So the manual recipe is both API-invalid and semantically stricter than the test suite.

Suggested replacement:

- For a byte-exact threshold-zero smoke, use FLUX.1 and an `after_loop` callback to capture latents.
- For FLUX.2 Klein, use the same cosine gate as `tests/test_parity_flux2.py` or just smoke `skipped_count` plus finite image output.

### 3. README has broken internal references

`README.md` points to files that are not present in this repo snapshot:

- `docs/calibration.md`
- `docs/superpowers/spikes/2026-05-14-mlx-teacache-phase-0-spike.md`

The FLUX.2 calibration source that actually exists is:

- `scripts/calibrate_flux2_klein.py`
- `scripts/_calibration_flux2_klein.json`
- the provenance comment in `src/mlx_teacache/coefficients.py`

Either add `docs/calibration.md` before release or change the README link to the existing calibration script/report. The missing spike link should be removed or replaced with the shipped notes under `docs/superpowers/notes/`.

### 4. Z-Image is already in mflux 0.17.5

The roadmap says Z-Image would need a new mflux variant first. That is stale.

Local mflux has:

- `mflux.models.z_image.variants.z_image.ZImage`
- `ModelConfig.z_image()` and `ModelConfig.z_image_turbo()`
- a Z-Image `_predict` with the same eager-vs-compiled gate as FLUX.2

Relevant local sources:

- `.venv/lib/python3.13/site-packages/mflux/models/z_image/variants/z_image.py:24-213`
- `.venv/lib/python3.13/site-packages/mflux/models/common/config/model_config.py:507-528`
- `.venv/lib/python3.13/site-packages/mflux/models/z_image/README.md:1-24`

The upstream Hugging Face card for `Tongyi-MAI/Z-Image` says Z-Image is an undistilled, full-capacity foundation model with CFG, recommended 28-50 steps, Apache-2.0 license, and `ZImagePipeline`.

Roadmap implication: Z-Image base may deserve a higher rank than Chroma and maybe even than img2img if the goal is speedup-per-engineering-effort. The work is not "new mflux variant"; it is a new `mlx-teacache` integration path for a single-stream DiT plus calibration.

### 5. FLUX.2 Klein 9B risk is license/gating, not just disk

The roadmap currently calls out 52.9 GB of HF weights. Keep that, but add:

- BFL's 4B card is Apache-2.0 and says the model fits in about 13 GB VRAM.
- BFL's 9B card says non-commercial use and about 29 GB VRAM.
- Hugging Face marks the 9B repo as gated.
- BFL's model card states filters or manual review are required for 9B under the FLUX Non-Commercial License.

This affects CI, benchmark reproducibility, docs wording, and whether `flux2-klein-9b` should be presented as a normal supported variant or a gated/non-commercial variant.

### 6. Chroma is less "one day" than the roadmap suggests

`lodestones/Chroma1-HD` is Apache-2.0 and the card says it is based on FLUX.1-schnell, but it is exposed through Diffusers as `ChromaPipeline`, not as a plain FLUX.1 pipeline. A local search found no Chroma support in mflux 0.17.5.

That does not mean Chroma is impossible. It does mean "documentation + one parity test, no new code" is optimistic. First proof point should be: can mflux `Flux1` load Chroma weights without a custom mapping, and does the transformer block layout match the FLUX.1 residual boundary used by mlx-teacache?

## Benchmarking additions

### Warmup and repeat the community benchmark

The `docs/m3-plus-tradeoff.md` benchmark currently times one vanilla run and one TeaCache run. That can overstate or understate the effect because `mx.compile` has a slow first call and MLX caches compiled functions.

MLX's compile docs explicitly say the first compiled call builds the graph and code, later calls reuse the compiled function, and shape/type/input-count changes can recompile. The docs' own timing helper warms up before measuring.

Suggested benchmark protocol:

1. Print environment:
   - chip string from `sysctl -n machdep.cpu.brand_string`
   - macOS version
   - `mlx`, `mflux`, `mlx-teacache` versions
   - model, quantization, resolution, steps, guidance, threshold
2. Run one untimed warmup in each mode.
3. Run at least three timed repetitions in each mode.
4. Report median and min, not a single number.
5. Report `computed_count`, `skipped_count`, and `speedup_estimate` from the TeaCache handle.

Also use the project default threshold (`0.20`) in public docs unless the section is explicitly measuring threshold `0.25`.

### Clarify FLUX.1 vs FLUX.2 benchmark rows

The README states the flagship 1.48x number on FLUX.1-dev / 25 steps, while the m3-plus benchmark recipe uses FLUX.2 Klein 4B. That is fine, but the docs should not imply the two numbers are directly comparable. FLUX.2 at its common 4-8 step schedule has a much lower skip ceiling.

## M5 / TensorOps wording

Apple's M5 MLX article is strong evidence that:

- MLX takes advantage of M5 Neural Accelerators with the latest macOS beta / macOS 26.2 or later.
- Those accelerators provide dedicated matrix-multiplication operations.
- MLX uses Tensor Operations and Metal Performance Primitives from Metal 4.
- Apple measured more than 3.8x faster 1024x1024 FLUX-dev-4bit image generation on M5 vs M4.

The docs should be more careful with this sentence:

> Neural Accelerators are only available via the compiled path.

I would phrase it as:

> Current mflux reaches the M5 TensorOps-eligible path through `mx.compile(_predict)`. mlx-teacache's eager FLUX.2 wrapper bypasses that compiled `_predict`, so it may lose some or all of the M5 TensorOps advantage. Confirm with profiler/benchmarks before making a hard "only available" claim.

That is a better distinction between sourced Apple claims and our inference from mflux integration mechanics.

## Compile-friendly gating notes

Option C in the roadmap is the right first implementation candidate, but the constraints should be explicit:

- The eager dispatcher must remain outside the compiled callable; otherwise Python-side gate state is traced once.
- The compiled callables should be pure tensor functions with stable signatures.
- Cached residual state should live in eager Python and be passed into the skip path as an array.
- Recompile triggers are shape, dtype, and input-count changes per MLX docs. Both compiled paths need stable signatures per generation.

Option B (`mx.where(should_run, full_path, cached_path)`) should be described as a correctness experiment, not a speed path. It still builds both branch graphs and pays the full transformer cost, so it cannot preserve TeaCache speedup.

## Future model ranking adjustments

Suggested changes to `ROADMAP.md` future rows:

1. Move **Z-Image base** higher. It is Apache-2.0, already in mflux, uses 28-50 steps, and has a compile-gated `_predict` analogous to FLUX.2.
2. De-rank **Chroma** until there is a local mflux load proof. HF says `ChromaPipeline`; local mflux has no Chroma support.
3. Split **FLUX.2 base 4B** from **base 9B**. Base 4B is Apache-2.0 and already has a matching mflux config; base 9B is gated/non-commercial like the distilled 9B.
4. For video, note license/memory differences:
   - Mochi 1 preview: Apache-2.0, but official card says 42 GB VRAM for high-quality Diffusers path and about 60 GB VRAM for the reference single-GPU repo path.
   - CogVideoX 2B: Apache-2.0; CogVideoX 5B: license `other`.
   - HunyuanVideo / HunyuanVideo 1.5: license `other`.

## Small doc cleanups

- `docs/manual-verification.md` pins `mlx-teacache[mflux]==0.1.0`; changelog has `0.1.1`. Use latest patch in docs unless validating a specific old release.
- `tests/test_parity_flux2.py` has stale top-level text saying the gate is `mx.allclose(atol=0.1, rtol=0.05)`, while the actual code now uses cosine >= 0.97. Not public docs, but worth fixing before people copy the rationale into README.
- README quick start uses FLUX.2 Klein with `num_inference_steps=25`. That is fine for demonstrating caching, but a first-run quick start may be friendlier at 8 steps plus a note that longer schedules show larger gains.

## Sources checked

Primary / official:

- Apple ML Research, "Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU":
  https://machinelearning.apple.com/research/exploring-llms-mlx-m5
- MLX compilation docs:
  https://ml-explore.github.io/mlx/build/html/usage/compile.html
- Diffusers cache API:
  https://huggingface.co/docs/diffusers/api/cache
- BFL FLUX.2 Klein 4B:
  https://huggingface.co/black-forest-labs/FLUX.2-klein-4B
- BFL FLUX.2 Klein 9B:
  https://huggingface.co/black-forest-labs/FLUX.2-klein-9B
- Chroma1-HD:
  https://huggingface.co/lodestones/Chroma1-HD
- Z-Image:
  https://huggingface.co/Tongyi-MAI/Z-Image
- Mochi 1 preview:
  https://huggingface.co/genmo/mochi-1-preview

Local source:

- mflux 0.17.5 `AppleSiliconUtil`
- mflux 0.17.5 `Flux2Klein._predict`
- mflux 0.17.5 `ZImage`
- mflux 0.17.5 `ModelConfig`
- mflux 0.17.5 `GeneratedImage`
- mlx-teacache `README.md`, `ROADMAP.md`, `docs/m3-plus-tradeoff.md`, `docs/manual-verification.md`

---

## Strengthening review (claude, 2026-05-15)

I re-verified each high-impact correction against the local mflux install and the mlx-teacache repo. All five hold. This section adds (a) the verification trace so future readers can confirm, (b) findings the audit missed, and (c) a concrete prioritized action list keyed to release windows.

### Verification trace

| Audit claim | Verified how | Result |
|---|---|---|
| M1 Pro / M2 Pro use eager (not compiled) | Traced `is_m1_or_m2` against every M-series brand-string variant. Pro chips return True → eager. | ✅ Confirmed |
| `GeneratedImage` has no `.latents` | `dir(GeneratedImage)` shows save/heatmap methods only. The recipe in `docs/manual-verification.md` will fail at runtime. | ✅ Confirmed |
| `docs/calibration.md` and the spike-notes path are missing | `ls` of both paths returns ENOENT. README L73 + L111 reference them. | ✅ Confirmed |
| Z-Image is already in mflux 0.17.5 | Imported `mflux.models.z_image.variants.z_image.ZImage` cleanly; has `_predict`. `ModelConfig.z_image()` + `z_image_turbo()` exist. | ✅ Confirmed |
| Chroma is NOT in mflux 0.17.5 | `[m for m in dir(ModelConfig) if 'chroma' in m]` → empty. | ✅ Confirmed |

### Additional findings the audit didn't surface

**A1. The M1 Pro / M2 Pro miscount affects FIVE files, not one.** The audit named two; the wrong rule is also in:

- `~/.claude/skills/user-mlx-developer/references/mflux-and-local-projects.md` (the per-chip table I added in the most recent skill update)
- `mlx-teacache/CHANGELOG.md` (the 0.1.0 entry's "M3+ users lose…" line, since restored to wrong wording when historical-correction text was removed)

Both will mislead future code that consults the skill or the changelog.

**A2. The `Flux2Klein` compile gate covers FLUX.2 only.** The FLUX.1 path in mflux 0.17.5 uses a *different* `Flux1._predict` whose compile gate I haven't audited. Our docs implicitly assume the rule is identical — likely true, worth confirming. Action: read `mflux.models.flux.variants.txt2img.flux:_predict` definition and update docs if it differs.

**A3. `apply_teacache()` raises before mflux can even start a generation; the manual-verification recipe doesn't exercise the FLUX.2 cosine gate path at all.** The audit suggests an alternative recipe using `after_loop`, which is the right call. I'd add: the recipe should also cover the *distilled-step warning* once 0.2.0 lands, so the doc actually serves as a smoke test for everything user-facing.

**A4. The "FLUX.2 Klein 9B is gated on HF" claim needs re-verification.** I downloaded the full 9B repo from this machine using `hf download` without any token or terms acceptance and it succeeded. Either HF doesn't enforce gating via the CLI, the 9B is gated for *commercial use* but not for download, or the audit is reading a stale UI state. Worth a 60-second check on the model card before propagating "gated" language. The non-commercial license claim is independent of the gating claim and stands on its own.

**A5. The fast-path note in `flux2_klein.py:279` is the ONLY compile gate.** There is no second compile inside the transformer body, so Option C (pre-compile two paths) only needs to wrap the predict closure, not the entire transformer module hierarchy. This reduces the design complexity of the compile-friendly gating ROADMAP item — worth mentioning in the option C tradeoff table.

**A6. `docs/manual-verification.md` pins `0.1.0` but the example also asserts `mx.array_equal` on the FLUX.2 path.** Even if `.latents` existed, the assertion would fail because FLUX.2 wrapper parity is cosine-only by design (per `tests/test_parity_flux2.py`). Two independent bugs in the same 30-line recipe.

**A7. README quick-start uses `num_inference_steps=25` for FLUX.2 Klein.** Klein 4B is distilled — typical use is 4–8 steps. A 25-step Klein run is slow AND produces lower-quality output than 8 steps (the schedule isn't designed for it). The quick-start should use 8 steps; the 25-step value should appear only in the FLUX.1-dev benchmark section.

**A8. The audit's "Move Z-Image up" recommendation has a hidden cost.** Z-Image's `_predict` is structurally similar to FLUX.2 but the transformer block layout differs (Z-Image uses cross-attention with a different token-routing pattern). Z-Image v0.2 calibration would be a fresh polynomial fit on a different architecture. So "already in mflux, drop in our wrapper" is half-true: integration is easy, calibration is not. Estimate revised: 3–5 days, not "drop-in".

### Concrete prioritized action list

#### Critical — ship in v0.1.2 doc-fix patch release (no code change, today)

1. **Fix the M1 Pro / M2 Pro chip table** in:
   - `README.md` Limitations bullet
   - `docs/m3-plus-tradeoff.md` per-chip table
   - `ROADMAP.md` per-chip table draft
   - `~/.claude/skills/user-mlx-developer/references/mflux-and-local-projects.md`
2. **Fix README broken links** (`docs/calibration.md`, the spikes path):
   - Replace `docs/calibration.md` link with a paragraph referencing `scripts/calibrate_flux2_klein.py` + `scripts/_calibration_flux2_klein.json` + the in-source provenance comment
   - Replace the spike link with a pointer to `docs/superpowers/notes/2026-05-14-task-25-mlx-nondeterminism.md` or remove the sentence
3. **Replace `docs/manual-verification.md`** with the audit's suggested recipe (FLUX.1 `after_loop` capture for byte-exact, FLUX.2 cosine smoke). Bump pin to `mlx-teacache[mflux]==0.1.1`.
4. **Stale comment in `tests/test_parity_flux2.py`** top docstring still says `mx.allclose(atol=0.1, rtol=0.05)` was the original (failed) oracle. Fine as historical context, but the comment should be clearer that it's describing the journey, not the current gate.

These are zero-risk doc/test-comment fixes. Could ship as 0.1.2 tonight if desired.

#### Important — fold into v0.2.0 release (alongside 9B + img2img + distilled warning)

5. **Add `docs/calibration.md`** as a proper doc covering the calibration procedure, output format, and how to derive coefficients for a new variant. Material exists in the script + JSON + provenance comment; just needs assembly.
6. **Update `ROADMAP.md` future-model rankings** per audit §"Future model ranking adjustments":
   - Z-Image base → revised effort 3–5 days (not "drop-in") with calibration as the long pole
   - Chroma → de-rank to "needs upstream mflux support first; track lodestones/Chroma1-HD + a mflux integration request"
   - Split FLUX.2-klein-base-4B (Apache-2.0, comparable to Klein-4B effort) from FLUX.2-klein-base-9B (non-commercial license + larger calibration)
   - Add per-row license + memory columns to the video models row
7. **Verify the FLUX.2 Klein 9B "gated" claim** before propagating. Either confirm via HF web UI or re-mark as "non-commercial license; not gated for download in current state."
8. **Soften the M5 wording** per audit §"M5 / TensorOps wording" — "may lose some or all of the M5 TensorOps advantage" instead of "only available via the compiled path." Apply to README + skill + m3-plus-tradeoff.
9. **README quick-start** switch to 8 steps for Klein, and surface the 25-step FLUX.1-dev as the benchmark example.

#### Quality — fold into v0.2.1 or later

10. **Adopt the audit's benchmark protocol** in `docs/m3-plus-tradeoff.md`: env-print, warmup, 3 timed reps, report median + min + computed/skipped counts. This is what turns the recipe from "anecdotal one-shot" to "comparable across community submissions."
11. **Verify FLUX.1 compile gate** (audit-add A2): confirm the FLUX.1 `_predict` uses the same `is_m1_or_m2()` predicate. If yes, no doc change needed; if no, the per-chip table needs a per-variant row.
12. **Option C ROADMAP refinement** (audit-add A5): note that the compile gate is the single predict closure, not the transformer module, reducing wrap complexity.

### Pulling the audit into the v0.2.0 spec

The img2img + distilled-warning spec (`2026-05-15-img2img-and-distilled-notification-design.md`) targets v0.2.0. Audit items 5–9 above also target v0.2.0. If we want a clean v0.2.0 release, the implementation plan derived from that spec should be extended to include those doc items as additional ordered tasks. Recommendation: add a "v0.2.0 doc track" section to the plan listing items 5–9 as parallel TDD-style tasks (write the doc, verify links resolve in CI, commit). The code track (img2img + warning) and doc track can ship in the same release commit.

### Self-assessment

I am the author of several of the bugs the audit caught. Specifically:

- The M1 Pro / M2 Pro miscount was in my recent docs across five files (audit named two; the additional three are the skill, the CHANGELOG line, and the ROADMAP draft).
- The README broken-link references pre-dated my recent work (came from v0.1.0 ship) but I didn't catch them while sweeping for "M3+" cleanups.

Lesson: when correcting one technical claim across multiple docs, run `grep` for the *broader topic*, not just the specific wrong phrase. A sweep for "M[1-5]" or "compile" would have caught more of the chip-classification drift than a sweep for "M3+".
