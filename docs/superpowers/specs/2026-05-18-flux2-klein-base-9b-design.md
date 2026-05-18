# v0.5.0 — flux2-klein-base-9b support (design)

**Date:** 2026-05-18
**Status:** approved by user 2026-05-18, ready for implementation plan.
**Target release:** v0.5.0
**Predecessor:** v0.4.0 added `flux2-klein-base-4b` (non-distilled FLUX.2 Klein 4B, 25-step calibration, origin-constrained polyfit, `default_thresh=0.17`). v0.4.1 added per-branch CFG caching for FLUX.2.

## Goal

Add `flux2-klein-base-9b` (non-distilled FLUX.2 Klein 9B, FLUX Non-Commercial license) to the supported-variants list with the canonical upstream recipe (`num_inference_steps=50`, `guidance_scale=4.0`) as the validation target.

Ship by reusing base-4b's polynomial coefficients verbatim and base-4b's default threshold (0.17), then proving the reuse works with a one-time validation pass at the canonical recipe before tagging.

## Why this variant, and why now

`black-forest-labs/FLUX.2-klein-base-9B` is the non-distilled 9B sibling of `flux2-klein-base-4b`. Black Forest Labs ships it for users who want a higher-capacity image generator at the cost of slower inference. mlx-teacache currently supports the distilled `flux2-klein-9b` (v0.3.0; the gate doesn't engage; wall-clock improvement comes from `mx.compile`-path avoidance only) but not the base variant.

v0.4.1 already shipped per-branch CFG caching, which is the FLUX.2-family infrastructure base-9b needs to run gate-engaged at `guidance > 1.0`. No new mechanisms are required; we are extending coverage.

## Out of scope for this design

- Running a fresh calibration for klein-base-9b. We reuse base-4b's polynomial coefficients and validate empirically. A real calibration is deferred until validation forces it (failure mode) or until the ROADMAP "Calibration fit-quality on FLUX.2-family architectures" item is picked up.
- Threshold-sweep tooling. ROADMAP item, not v0.5.0.
- Alternative gate signals (FBCache, DiCache, TaylorSeer). ROADMAP items.
- New model architectures outside the Klein family.

## Architecture

### 1. Variant detection

**File:** `src/mlx_teacache/integrations/mflux/detect.py`

- Add `"flux2-klein-base-9b"` to the `VariantId` Literal type.
- Add `"flux2-klein-base-9b"` to the `_SUPPORTED` tuple.
- Alias handling: reuse the existing `Flux2Klein` model-class branch. The alias map already recognizes the `klein-base-9b` model-config name from mflux 0.17.5 (`ModelConfig.flux2_klein_base_9b()` returns a config whose `aliases` include `"klein-base-9b"`).

### 2. Coefficient registry

**File:** `src/mlx_teacache/coefficients.py`

- Add a `"flux2-klein-base-9b"` entry to `_REGISTRY`.
- The entry points to the **same polynomial coefficient array** as `"flux2-klein-base-4b"`. Not a copy — a literal shared reference (or an explicit equality assertion in a test) so the reuse is intentional and visible at review time.
- `Provenance.default_thresh = 0.17` (same as base-4b).
- Provenance comment cites the v0.4.0 calibration JSON the coefficients came from and links to the v0.5.0 validation artifact at `_artifacts/validation_klein_base_9b.json`.

### 3. API wiring

**File:** `src/mlx_teacache/api.py`

- Update the docstring "Supported variants" list to include `flux2-klein-base-9b` under the "non-distilled; gate engages" group.
- Update the `default_thresh=0.17` set to include klein-base-9b.

### 4. Calibration script

**File:** `scripts/calibrate_flux2.py`

- Remove the `NotImplementedError` stub for `--variant klein-base-9b`. The script is functional — anyone who wants to override the reused coefficients with a real fit can run it. v0.5.0 itself does not run it.
- Update the docstring to remove the "wired in v0.5.0" note and reflect that the variant is now runnable.

### 5. Validation script (new)

**File:** `scripts/validate_klein_base_9b.py`

A single-purpose validation harness. Not generic — the comment explicitly notes this is the one-shot script that ships with v0.5.0 to convert "coefficients transfer from base-4b" from an assumption into a measured fact.

- Single fixed prompt (reuse the COMPARISON.md portrait prompt for continuity).
- Seed 42, 1024×768, 50 steps, guidance=4.0.
- Generates vanilla and wrapper outputs sequentially in the same Python process. (Validation, not a release-gate bench. Process isolation matters less here.)
- Decodes both through the VAE.
- Computes SSIM (skimage, channel-aware).
- Writes `_artifacts/validation_klein_base_9b.json` with: prompt, seed, steps, guidance, threshold, wrapper `skipped_count` and `computed_count`, SSIM, hardware (sysctl-detected), mlx_teacache version, mflux version.
- Prints PASS / FAIL and exits non-zero if SSIM < 0.95.

The script is NOT in the test suite (it loads weights and runs heavy generation). It is a one-shot release-gate run executed manually before tagging.

### 6. Bench wiring

**File:** `scripts/bench_speedup.py`

- Add `klein-base-9b` to the `--variant` choices and the variant-config map.
- Default recipe: 50 steps, guidance=4.0 (the canonical upstream recipe; same as base-4b CFG row).
- Three-way mode default-on (vanilla / no-gate / gated), inherited from base-4b. Lets us attribute wall-clock between `mx.compile`-path avoidance (no-gate vs vanilla) and gating contribution (gated vs no-gate).

### 7. Tests

**Files:** `tests/test_detect.py`, `tests/test_coefficients.py`, `tests/test_klein.py` (or wherever the FLUX.2 parametrized test suites live)

- `test_detect.py`: replace the v0.4 rejection test (`test_flux2_klein_base_9b_rejected`) with an acceptance test (`test_flux2_klein_base_9b_recognized`).
- `test_coefficients.py`: assert that `_REGISTRY["flux2-klein-base-9b"]` resolves to the same polynomial array as `_REGISTRY["flux2-klein-base-4b"]`. Catches accidental drift from the intentional reuse.
- Real-weight test suites (gated behind `HF_TOKEN`): parametrize the existing FLUX.2 Klein test cases to also cover klein-base-9b. Reuse the same fixtures and tolerances as klein-base-4b.

### 8. Docs

**Files:** `README.md`, `CHANGELOG.md`, `docs/calibration.md`, `ROADMAP.md`

- `README.md` "Supported models": new row for `flux2-klein-base-9b` with a footnote pointing at the validation evidence and noting the FLUX NC license restriction.
- `README.md` "When to use mlx-teacache": add klein-base-9b to the list of non-distilled FLUX.2 variants the wrapper helps on.
- `CHANGELOG.md`: v0.5.0 entry. Headline: variant addition + coefficient-reuse pattern + validation SSIM. Numbers TBD until validation runs (placeholder in the plan; filled in during the validation step).
- `docs/calibration.md`: a paragraph on the coefficient-reuse approach — when it's appropriate (same architecture family + same calibration recipe + empirical validation), when it's not (cross-recipe or cross-architecture transfer).
- `ROADMAP.md`: move v0.5.0 from "Active" to "Released" with the validation SSIM, three-way bench attribution, and a one-sentence note about coefficient reuse. Add a follow-up "Active" entry for the calibration-fit-quality item if the validation passes but bench shows weak engagement.

### 9. License + safety filter

klein-base-9b is FLUX Non-Commercial license, same as the distilled klein-9b. The Hugging Face repo is gated; users must accept the license on the model page before `mflux` can download weights. mlx-teacache itself does not gate access — it relies on mflux's existing flow. The README footnote calls out the NC restriction explicitly so users see it before they pip install.

No new gating code is required.

## Data flow

For inference (unchanged from base-4b):

1. User constructs `Flux2Klein(model_config=ModelConfig.flux2_klein_base_9b())`.
2. User wraps via `apply_teacache(flux)`. `identify_variant()` returns `"flux2-klein-base-9b"`, which `coefficients.lookup()` resolves to the shared base-4b polynomial and `default_thresh=0.17`.
3. `flux.generate_image(..., num_inference_steps=50, guidance=4.0, ...)` runs.
4. v0.4.1's per-branch CFG caching engages: separate `cached_residual` (conditional) and `cached_residual_neg` (unconditional) slots. Skip decisions are made independently on each branch.
5. Stats finalize on natural completion; user reads `handle.stats.skipped_count` etc.

For validation (one-time, pre-tag):

1. Run `uv run python scripts/validate_klein_base_9b.py`.
2. Script generates vanilla and wrapped outputs at 50/g=4.0 on a fixed prompt + seed.
3. Computes SSIM, writes JSON, exits non-zero if < 0.95.
4. Engineer reviews JSON, commits it to `_artifacts/validation_klein_base_9b.json`.
5. Engineer runs `scripts/bench_speedup.py --variant klein-base-9b --three-way` for the README + CHANGELOG numbers.
6. Engineer updates docs with the measured numbers, opens PR, ships.

## Quality + skip gates

- **Validation SSIM ≥ 0.95** at 50 steps + g=4.0 between vanilla and wrapper. Hard merge gate.
- **Three-way bench attribution** must distinguish gating contribution from `mx.compile`-path avoidance. If gating contributes ≥ 1.10× (similar to base-4b's 1.16×), document as "gate engaged."
- **0-skip contingency:** if the wrapper produces 0 skips at threshold=0.17, the polynomial transfer did not engage. Ship with the honest attribution ("wall-clock improvement is from `mx.compile`-path avoidance only") rather than letting "0 skips" inherit headline credit. The v0.4.1 plan-audit lesson applies.

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Coefficients don't transfer (SSIM < 0.95 or 0 skips at 0.17) | Medium | Validation pass catches before merge. Fallback: cut a follow-up branch that runs a real calibration. v0.5.0 ships with whatever the validation shows; don't massage. |
| 9B at 50/g=4.0 is heavier than expected on M1 Max (validation overshoots 2h budget) | Low-medium | Single prompt × 2 conditions ≈ 60-90 min worst case. If it overruns, validate on a smaller resolution (768×512) for the SSIM check and document the deviation. |
| Real-weight tests fail because HF gating is unfriendly in CI | Medium | Real-weight tests already gated behind `HF_TOKEN`; CI skips when the secret is absent (same as klein-9b). Local validation is the actual gate. |
| Existing klein-base-4b tests break when parametrized for 9B (memory pressure on M1 Max 32GB) | Low | Run real-weight tests serially with `pytest -p no:xdist`. Add a `slow + memory-heavy` marker if needed. |

## Effort + timeline

| Phase | Estimate | Notes |
|---|---|---|
| Variant wiring (detect, coefficients, api) | 1-2 h | Mechanical, follows base-4b precedent line-for-line. |
| Validation script | 1 h | Single-purpose, single-prompt. |
| Bench wiring | 30 min | One variant-config entry. |
| Tests (parametrize + rejection→acceptance flip) | 1-2 h | Touches `test_detect.py`, `test_coefficients.py`, klein test suite. |
| Docs (README, CHANGELOG, calibration.md, ROADMAP) | 1 h | Placeholders for validation numbers. |
| **Engineering subtotal (no generation)** | **~5 h** | All of the above can be done today; no model load, no GPU. |
| Validation run | 1-2 h | Tomorrow. Single prompt × 2 conditions on M1 Max. |
| Three-way bench | 2-3 h | Tomorrow. Three conditions × multiple reps. |
| Doc updates with measured numbers + PR + CI + merge + tag | 1-2 h | Tomorrow. |
| **Tomorrow subtotal (generation + ship)** | **~5 h** | |

## Release packaging

Single feature branch (`feature/v0.5.0-klein-base-9b`). Single PR. Single tag (`v0.5.0`). The PR opens after both the engineering work and the validation+bench numbers are in.

Per the release-flow rule, the PR is opened and the human merges on GitHub. Tag push to publish to PyPI is a separate explicit authorization step.

## Acceptance criteria

- [ ] `identify_variant()` recognizes `flux2-klein-base-9b` from mflux's `Flux2Klein` + `ModelConfig.flux2_klein_base_9b()`.
- [ ] `coefficients.lookup("flux2-klein-base-9b")` returns the same polynomial as `coefficients.lookup("flux2-klein-base-4b")` with `default_thresh=0.17`, asserted by a unit test.
- [ ] `scripts/validate_klein_base_9b.py` exists, is runnable, and on the actual M1 Max validation run produces `_artifacts/validation_klein_base_9b.json` with SSIM ≥ 0.95.
- [ ] `scripts/bench_speedup.py --variant klein-base-9b --three-way` runs end-to-end and reports vanilla / no-gate / gated wall-clock + skip counts.
- [ ] `test_detect.py` acceptance test for klein-base-9b passes; the v0.4 rejection test is gone.
- [ ] FLUX.2 Klein parametrized test suite includes klein-base-9b, behind `HF_TOKEN`.
- [ ] README, CHANGELOG, `docs/calibration.md`, and ROADMAP are updated with measured numbers.
- [ ] PR opens with three-way bench numbers and validation SSIM in the body.
- [ ] Tag `v0.5.0` cut after human merge, with explicit user authorization for the publish.

## Audit reference

This spec is open to a plan-audit pass before implementation begins. The most likely findings:

1. Coefficient identity: is the v0.5.0 entry a literal shared reference or a copy? (The unit test in §7 makes this explicit either way.)
2. Validation prompt: one prompt is a thin gate. Consider 2-3 prompts if 9B's behavior diverges per-prompt — but a thin gate is consistent with the "reuse-and-validate" strategy chosen for v0.5.0. Real calibration is the fallback.
3. 50-step CFG run on M1 Max 32GB at 9B: memory pressure. Worth a dry-run model-load check before kickoff tomorrow.
