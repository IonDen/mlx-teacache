# v0.4.0 — `flux2-klein-base-4b` support: design

**Date:** 2026-05-17
**Status:** Approved (brainstorming complete; ready for implementation-plan handoff).
**Target release:** v0.4.0.
**Source-of-truth references:**
- v0.3 non-engagement postmortem: `docs/superpowers/notes/2026-05-16-flux2-teacache-non-engagement-postmortem.md`
- v0.3 Klein 9B integration (pattern this design mirrors): `docs/superpowers/specs/2026-05-16-klein-9b-design.md`
- ROADMAP entry: `ROADMAP.md` → "v0.4.0: `flux2-klein-base-4b`"

## Goal

Ship `flux2-klein-base-4b` (Apache-2.0, non-distilled FLUX.2 variant designed for 20-50 step generation) as the fifth supported variant in mlx-teacache. **Scope: TeaCache engagement at `guidance=1.0` only.** This is the first FLUX.2 variant where the polynomial gate is expected to engage on its own (under that constraint) — the library will deliver its headline TeaCache step-skipping feature on a FLUX.2 schedule for the first time.

**CFG / guidance scope explicitly deferred to v0.4.1.** The current FLUX.2 wrapper routes `guidance > 1.0` to `_vanilla_flux2_cfg_predict()` and records `cfg-fallback` (see `src/mlx_teacache/integrations/mflux/flux2.py:101-129`); no gate runs. Upstream BFL's base-4B model card recommends `guidance_scale=4.0, num_inference_steps=50` as the canonical recipe. v0.4.0 will explicitly document that running base-4b at upstream-recommended guidance falls back to vanilla mflux and that CFG-engaged TeaCache lands in v0.4.1 (per-branch caching). This is a deliberate scoping decision recorded during the 2026-05-17 spec audit response: ship the small, fast piece first.

Distilled FLUX.2 Klein 4B and 9B at their 4-8 step defaults remain supported but documented as "out of scope for algorithmic step-skipping by design" (wall-clock benefit comes from `mx.compile`-path avoidance, not caching). The ROADMAP "Out of scope (deliberate)" section already records this decision.

## Why this variant, and why now

1. **Architecture re-use.** mflux 0.17.5 exposes `flux2_klein_base_4b()` and the runtime `Flux2Klein` class is identical to the one used by distilled Klein 4B. mlx-teacache's `_predict` guard already matches `startswith("flux2-")`, so no integration-code surgery is needed — only data (fresh coefficients) and surface plumbing.
2. **Licensing.** base-4b keeps the Apache-2.0 commercial posture of the FLUX.2 4B family while moving from distilled 4-step inference to non-distilled 25-50 step inference. The distilled `flux2-klein-4b` we already support is also Apache-2.0; the non-commercial / BFL-safety-filter terms only attach to the 9B variants (`flux2-klein-9b` today, `flux2-klein-base-9b` in v0.5).
3. **Direct precedent.** NVIDIA's FLUX.2-dev blog reports ~32% steps skipped at `teacache_thresh=0.05` on a 50-step schedule. FLUX.2-dev and klein-base-4b share the same broader architecture family, so the polynomial gate signal is expected to be predictive on base-4b — only the calibration constants differ.
4. **Library-narrative repair.** v0.2.0 shipped FLUX.2 Klein 4B with a misleading "TeaCache step-skipping" framing. v0.3.0 corrected the framing honestly. v0.4.0 finally lets the library deliver the advertised mechanism on FLUX.2.

## Out of scope for this design

- **CFG / guidance > 1.0 caching** — deferred to v0.4.1 (per-branch caching for FLUX.2). The CFG-fallback path in `flux2.py:101-129` stays unchanged. See Goal section for why this scope split.
- `flux2-klein-base-9b`. Same approach as base-4b but with FLUX Non-Commercial license + BFL safety-filter obligations. Deferred to v0.5.0 per ROADMAP.
- New caching mechanisms (FBCache, per-step-index lookup, TaylorSeer). The 2026-05-16 postmortem proposed these as v0.4 directions; on 2026-05-17 we decided not to pursue them, since the polynomial gate is expected to work on non-distilled FLUX.2 without alternative mechanisms. The postmortem coda records this decision; the references stay for the historical record.
- Per-variant default thresholds (Approach B in brainstorming). Package default `rel_l1_thresh=0.20` stays unchanged unless the 0-skip contingency below forces it.
- SSIM-vs-threshold sweep at calibration time (Approach C). Held as a contingency option if the default-threshold gate fails to engage on base-4b at g=1.0 (see "0-skip contingency" below).

## Architecture

Five additive change sites; same shape as v0.3 Klein 9B. No structural refactoring. No new modules.

### 1. Calibration

- Remove the `NotImplementedError` guard for `klein-base-4b` in `scripts/calibrate_flux2.py` (the script's `--variant` flag and monkeypatch logic are already variant-agnostic).
- Capture 10 prompts × 25 steps × seed=42 on M1 Max, quantize=4, 512×512, guidance=1.0 (matches the v0.3 calibration recipe — CFG disabled because the wrapper falls back to vanilla at `guidance > 1.0`, so calibration data captured at g=1.0 is what the runtime gate actually sees). Origin-constrained polyfit (same constraint settled on for 9B during v0.3).
- Write `scripts/_calibration_flux2_klein_base_4b.json` with the full report (coefficients, R², empirical x/y range, fit_mode, raw `x_values` and `y_values` arrays so offline refits are possible).
- Expected bench cost on M1 Max: ~8 hours. Main-thread `run_in_background=true`; log teed to `/tmp/calibrate-klein-base-4b.log`.

### 2. Coefficient registry

- Add `_REGISTRY["flux2-klein-base-4b"]` entry in `src/mlx_teacache/coefficients.py` with the calibrated tuple `(c4, c3, c2, c1, 0.0)` and a `Provenance` dataclass entry. Use the existing schema exactly:
  - `source="builtin"` (the `Provenance.source` field is `Literal["builtin", "user"]`; in-repo built-ins use `"builtin"`).
  - `revision="in-repo-2026-05-<DD>-origin"` (date filled in at calibration time).
  - `calibration_dataset="10 prompts × 25 steps × seed=42, M1 Max 32GB, bf16, 512x512, guidance=1.0, origin-constrained polyfit"` — `guidance=1.0` because the calibration recipe captures the runtime-gate path, which the wrapper only enters when CFG is inactive (see "CFG / guidance scope" section below).
  - `fit_metric="constrained-LSQ R^2 on consecutive-step (mod_in, body_out) rel-L1 pairs (poly(0)=0)"`.
  - `fit_metric_value=<measured R²>`.
  - `reference_url="https://github.com/IonDen/mlx-teacache/blob/main/scripts/calibrate_flux2.py"`.
- Same registry shape as the existing four entries (`flux1-dev`, `flux1-schnell`, `flux2-klein-4b`, `flux2-klein-9b`). No new fields.

### 3. Detect + API wiring

The detection layer lists supported variants in three places — all three need to be updated, not just the `_SUPPORTED` tuple. Earlier draft of this spec was wrong here.

- `src/mlx_teacache/integrations/mflux/detect.py`:
  - Extend the `VariantId = Literal[...]` declaration to include `"flux2-klein-base-4b"`.
  - Extend the `_SUPPORTED` tuple with `"flux2-klein-base-4b"`.
  - In `identify_variant()` add an alias-string branch inside the `isinstance(flux, _Flux2KleinType)` block: `if "flux2-klein-base-4b" in aliases: return "flux2-klein-base-4b"`. The block is currently a string-equality cascade for `"flux2-klein-4b"` and `"flux2-klein-9b"`; without this branch, base-4b raises `IncompatibleModelError` even after `_SUPPORTED` is updated.
- `src/mlx_teacache/api.py`: the `_predict` guard is already broadened to `startswith("flux2-")`, but the docstring at `api.py:141-146` enumerates supported variants and is stale once base-4b lands — update the docstring to include the new variant. No runtime change.

### 4. Bench

- `scripts/bench_speedup.py`: add a new variant branch for `klein-base-4b`. 25-step bench by default. Same warmup + 3 reps + median pattern. Reports skip count alongside wall-clock so the row makes the cache mechanism legible.
- Saves `tests/_artifacts/bench_images/klein-base-4b/{vanilla,wrapper}.png` for visual comparison (gitignored).

### 5. Tests

- `tests/test_coefficients.py`: new assertion that `_REGISTRY["flux2-klein-base-4b"]` exists with non-empty coefficients and provenance revision matches the expected pattern.
- `tests/test_detect.py`: parametrize the existing FLUX.2 variant detection test with the new variant id.
- `tests/test_api.py`: parametrize the existing api smoke test with the new variant.
- `tests/test_image_quality_flux2.py` + `tests/test_parity_flux2.py`: parametrize **with variant-aware generation kwargs**. The existing `_gen_kwargs_klein()` helper hardcodes `num_inference_steps=8` (the distilled default); base-4b must run at the calibrated 25-step schedule, otherwise the PR gate validates a schedule the variant is never used at. Convert the helper to a dispatch keyed on variant id: distilled Klein 4B/9B keep `num_inference_steps=8`, base-4b uses `num_inference_steps=25`. PR-gate SSIM threshold ≥ 0.85 (same as existing FLUX.2 PR-gate); decision on whether to raise to 0.90 deferred to post-calibration data. Cosine ≥ 0.99 at threshold 0 for the parity oracle.

### 6. Docs

The v0.4 PR bundles four uncommitted doc-clarity edits already in the working tree:
- `README.md` — "How the speedup happens" closing paragraph; Limitations section; removed unsafe threshold-bump advice.
- `CHANGELOG.md` — v0.3.0 entry's closing paragraph updated.
- `ROADMAP.md` — "Active" section restructured; "Out of scope" entry added.
- `docs/superpowers/notes/2026-05-16-flux2-teacache-non-engagement-postmortem.md` — 2026-05-17 coda.

Plus v0.4-specific doc updates:
- `README.md` — Supported variants table: add `flux2-klein-base-4b` row with a footnote noting "TeaCache engages at `guidance=1.0`; CFG/`guidance > 1.0` falls back to vanilla mflux pending v0.4.1." Benchmarks table: add a row with measured numbers from the new bench, labeled `g=1.0`. Limitations section: explicit "CFG on base-4b lands in v0.4.1" bullet. Quick-start example for base-4b uses `guidance=1.0` and notes the limitation.
- `CHANGELOG.md` — v0.4.0 entry: "Added: `flux2-klein-base-4b` support (Apache-2.0, non-distilled, 20-50 step generation). The first FLUX.2 variant where the polynomial gate engages on its own (at `guidance=1.0`). CFG-engaged caching deferred to v0.4.1 — at `guidance > 1.0` the wrapper falls back to vanilla mflux, same behavior as Klein 4B/9B."
- `docs/calibration.md` — new row in the built-in coefficient sources table.
- `ROADMAP.md` — move v0.4.0 entry from "Active" to "Released"; promote v0.4.1 (CFG per-branch caching for FLUX.2) to "Active". v0.5.0 (`flux2-klein-base-9b`) stays as the next major variant after CFG lands.

### Forward reference: v0.4.1 — CFG per-branch caching for FLUX.2

Not part of this spec, but the decision to defer CFG support to v0.4.1 needs a placeholder in ROADMAP so users following along see when the canonical base-4b recipe (`guidance_scale=4.0, num_inference_steps=50` per upstream model card) will be accelerated. Outline:
- Replace `_vanilla_flux2_cfg_predict()` with a per-branch gated path. Each branch (positive + negative prompt) keeps its own cache state; one shared gate decision per step (FBCache-style — when "skip" fires, both branches reuse their respective cached residuals).
- Skip-window validation under CFG; stats schema for two-branch decisions.
- Validation: parity + SSIM at `guidance=4.0` on base-4b against vanilla mflux at the same config.
- Effort estimate: 4-6 days implementation + ~12 hours bench.
- Risks deferred to the v0.4.1 spec.

## Data flow

Identical to v0.3. No lifecycle change.

```
apply_teacache(Flux2Klein(model_config=ModelConfig.flux2_klein_base_4b()))
    ↓
detect.resolve_variant(flux) → "flux2-klein-base-4b"
    ↓
coefficients._REGISTRY["flux2-klein-base-4b"] → (c4, c3, c2, c1, 0.0), Provenance
    ↓
flux._predict replaced with eager closure (api._build_flux2_predict_closure)
    ↓
per-step: flux2_forward_with_gate runs polynomial gate, decides compute/skip
```

## Quality + skip gates

The release gates are:

- **PR-gate SSIM** ≥ 0.85 on the red-apple prompt at default threshold, at `num_inference_steps=25, guidance=1.0`. Same bar as the existing FLUX.2 PR-gate. Open question whether to raise to 0.90 (matching FLUX.1-dev) — decide after seeing the SSIM distribution at calibration time.
- **Skip count** ≥ 1 across 3 reps at default `rel_l1_thresh=0.20`, on the red-apple bench prompt, at 25 steps, **at g=1.0**. This is a **release blocker** — see "0-skip contingency" below. The v0.4 narrative is "first FLUX.2 variant where the polynomial gate engages on its own"; without engagement at default, that narrative fails and v0.4 must either gain a new default threshold or be reframed as a structural-only release.
- **Wall-clock speedup** ≥ 1.3× on M1 Max at 25 steps, g=1.0. FLUX.1-dev measures 1.44×; base-4b is expected in the same ballpark since the gate machinery is identical.
- **Cosine similarity** ≥ `_FLUX2_COSINE_GATE` (currently `0.97`) at threshold 0, g=1.0 (parity test). Bit-exact parity is not expected on FLUX.2 (per v0.1 design); the cosine bar is the runtime contract. Measured value on existing variants is ~0.99+; the 0.97 gate absorbs prompt-to-prompt variance.

### 0-skip contingency

If post-calibration the gate produces 0 skips at default `rel_l1_thresh=0.20`, do NOT ship as "0 skips, structural-only" — that contradicts the v0.4 goal. Pursue one of these resolution paths before tagging:

1. **Recalibrate.** Inspect the empirical y-range from the JSON report. If `y_min` is just slightly above 0.20, an alternative origin-constrained fit (e.g. wider prompt set, different polynomial degree) may produce coefficients whose minimum is below 0.20. Re-run calibration with the adjusted recipe; ship at default if it engages.
2. **Per-variant default threshold (API change, otherwise out of scope).** If `y_min ≥ 0.25` consistently (similar to Klein 9B's situation but on a non-distilled schedule, which would be surprising), introduce a `default_thresh` field in the `Provenance` dataclass and look it up in `apply_teacache`. Set base-4b's default to a value that produces ≥ 1 skip at SSIM ≥ 0.85. This was Approach B in the brainstorming and was deferred; making it a contingency keeps it permitted.
3. **Reframe v0.4.0 as structural-only release.** Drop the "first FLUX.2 variant where the polynomial gate engages" claim from README + CHANGELOG. Ship base-4b as supported but with the same "0 skips at default; bump threshold at your own risk" framing as distilled Klein. This is the explicit fallback if (1) and (2) both fail or are rejected.

Decision is made post-calibration based on the actual measurements. The implementation plan should encode this as a branch point with a measurement gate.

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| 0 skips at default 0.20 at g=1.0 (no engagement) | Low | Non-distilled schedule is exactly the regime TeaCache was designed for; NVIDIA shows engagement on FLUX.2-dev at 0.05. Pursue 0-skip contingency paths (1), (2), or (3) above — do NOT ship as "0 skips, structural-only" by default. |
| Polynomial fit converges with low R² (similar to Klein 9B's 0.47) | Medium | Investigate root cause; could be a property of FLUX.2-family coefficients vs FLUX.1's well-fit polynomial. Document and ship — the polynomial doesn't have to be highly predictive to be useful, only to gate correctly on average. |
| Users hit `cfg-fallback` and conclude v0.4 doesn't work | Medium-high | This is the F1 audit finding made into a v0.4.0 product reality. README + CHANGELOG must lead with "v0.4.0 engages on base-4b at `guidance=1.0`; CFG (guidance > 1.0) falls back to vanilla, full CFG-engaged support lands in v0.4.1." `bench_speedup.py --variant klein-base-4b` runs at g=1.0 and labels its row "g=1.0 only". Doc-clarity diff in the v0.4 PR includes this messaging. |
| Calibration bench takes > 8 hours (thermal throttling) | Medium | Tee log to `/tmp/`; resume if interrupted. The captured `(x, y)` arrays in the JSON output let us refit offline if the bench crashes mid-run. |
| Real-weight tests need ~15 GB HuggingFace download | Low | Pre-download via `hf download black-forest-labs/FLUX.2-klein-base-4B` overnight before the implementation starts. |
| The new bench row reveals a wall-clock speedup that is mostly compile-avoidance (not step-skipping) | Low | If wall-clock improvement is real but skip count is low at g=1.0, fall into the 0-skip contingency path above. |

## Effort + timeline

- Weight download: ~30 min for 15 GB at typical home bandwidth.
- Calibration bench: ~8 hours overnight on M1 Max.
- Code changes (detect, registry, bench, tests): ~3-4 hours.
- Real-weight testing (parity + SSIM + bench rerun): ~2 hours.
- Doc updates + PR + review: ~1-2 hours.
- **Total: ~1-2 working days plus one overnight calibration run.**

## Release packaging

Single PR for v0.4.0:
1. The four doc-clarity edits already in the working tree (committed in the same PR for cleanliness)
2. The base-4b implementation (calibration JSON, coefficients entry, detect/api/bench/test updates)
3. CHANGELOG v0.4.0 entry
4. README benchmarks table updated with the new row from `bench_speedup.py`
5. Tag v0.4.0 from the merge commit; release.yml triggers PyPI publish, same flow as v0.3.0

## Acceptance criteria

v0.4.0 is ready to tag when:
- [ ] `scripts/calibrate_flux2.py --variant klein-base-4b --fit-mode origin` produces a valid JSON report (coefficients, R², empirical x/y range, raw `x_values` / `y_values`).
- [ ] `_REGISTRY["flux2-klein-base-4b"]` returns the new coefficients with `Provenance(source="builtin", ...)` matching the schema described in §2.
- [ ] `apply_teacache(Flux2Klein(model_config=ModelConfig.flux2_klein_base_4b()))` returns a working handle (no `IncompatibleModelError`). Requires the three detect-layer updates in §3 (Literal type, `_SUPPORTED`, alias branch).
- [ ] `tests/test_image_quality_flux2.py` SSIM PR-gate passes at default threshold, at `num_inference_steps=25, guidance=1.0`, via the variant-aware `_gen_kwargs_klein()` dispatch in §5.
- [ ] `tests/test_parity_flux2.py` cosine ≥ `_FLUX2_COSINE_GATE` (0.97) at threshold 0, g=1.0, on the PR-gate parity tests (`test_paired_parity_klein_pr_gate`, `test_paired_parity_at_threshold_zero_klein_pr_gate`).
- [ ] `scripts/bench_speedup.py --variant klein-base-4b` reports skip count ≥ 1 across 3 reps at default threshold AT g=1.0 — OR — the 0-skip contingency path is fully resolved per §"0-skip contingency" before tagging.
- [ ] CI green on the PR (lint + typecheck + pure-core + mflux × 3 Python versions + coverage).
- [ ] README, CHANGELOG, `docs/calibration.md`, and ROADMAP reflect the new variant AND state the CFG-fallback scope.

## Audit reference

This spec was reviewed by `docs/superpowers/notes/2026-05-17-flux2-klein-base-4b-spec-audit.md` (six findings). Resolution summary:

| Finding | Resolution |
|---|---|
| 1. CFG/guidance scope | Scoped: v0.4.0 is `guidance=1.0`-only; CFG caching deferred to v0.4.1 (Path C in scope decision). |
| 2. Detect/API wiring | Fixed in §3 — explicit Literal/_SUPPORTED/alias-branch edits enumerated. |
| 3. Provenance fields | Fixed in §2 — `source="builtin"`, `calibration_dataset` says `guidance=1.0`. |
| 4. 0-skip framing | Fixed in §"Quality + skip gates" — skip ≥ 1 at default is now a release blocker with three explicit contingency paths. |
| 5. Test plan hardcodes 8-step | Fixed in §5 — variant-aware `_gen_kwargs_klein()` dispatch (distilled at 8, base-4b at 25). |
| 6. Licensing narrative | Fixed in §"Why this variant" — claim narrowed to "Apache-2.0 commercial posture of the 4B family, moving from distilled to non-distilled inference." |
