# v0.4.1 — CFG per-branch caching for FLUX.2: design

**Date:** 2026-05-17
**Status:** Draft (brainstorming complete; awaiting user review before implementation-plan handoff).
**Target release:** v0.4.1.
**Source-of-truth references:**
- v0.4.0 design (parent variant integration): `docs/superpowers/specs/2026-05-17-flux2-klein-base-4b-design.md`
- v0.3 non-engagement postmortem (CFG-fallback origin): `docs/superpowers/notes/2026-05-16-flux2-teacache-non-engagement-postmortem.md`
- ROADMAP entry: `ROADMAP.md` → "v0.4.1: CFG per-branch caching for FLUX.2"
- Current CFG-fallback code: `src/mlx_teacache/integrations/mflux/flux2.py:32-129`

## Goal

Replace the current `_vanilla_flux2_cfg_predict()` path with a per-branch gated forward so the canonical upstream recipe (`flux2-klein-base-4b` at `guidance_scale=4.0, num_inference_steps=50`) can actually skip steps. After v0.4.1, CFG steps are gate-active and can be cached — that is the feature.

**Framing correction (per audit Finding 1).** Today's `_vanilla_flux2_cfg_predict()` is *not* a fallback to mflux's vanilla `_predict`. It is vanilla CFG math executed inside our eager `_predict` replacement. mflux 0.17.5's original `Flux2Klein._predict` wraps its closure in `mx.compile` on M3+/Max/Ultra chips (`.venv/.../flux2_klein.py:279-281`); ours never does. So a v0.4 base-4b user at `guidance=4.0` already gets the "eager wrapper / compile-path-avoided" wall-clock effect, just without any step-skipping. **v0.4.1's headline win is therefore "CFG steps become gateable," not "compile avoidance becomes available under CFG"** — the latter is already true in v0.4.

Distilled Klein 4B / 9B users running with `guidance > 1.0` get the unified machinery; they still produce ~0 skips on their 4-8 step distilled schedules (per the v0.3 postmortem), so the wall-clock benefit they already had (compile-path avoidance) is unchanged. The code path is unified, but their user-facing behavior does not change.

## Why this is tractable (and where the argument stops)

The mod_in gating signal does **not** depend on `encoder_hidden_states`. From `src/mlx_teacache/integrations/mflux/forward.py:258-304`:

```
body_in = inner.x_embedder(latents)                  # shared across branches
temb_mod_params_img = inner.double_stream_modulation_img(temb)   # shared
norm_in = inner.transformer_blocks[0].norm1(body_in) # shared
return (1.0 + scale_msa) * norm_in + shift_msa       # shared
```

Latents are the same at step `t` regardless of branch; `temb` is the same (timestep is shared, no guidance arg). So at a fixed `(latents_t, timestep_t)`, `mod_in` is **byte-identical** for the positive and negative branches.

**What this proves:** at a single step, we only need one gate input signal, so we only need one gate decision. The architecture choice "one shared decision, two cached residuals" rests on this.

**What this does NOT prove (per audit Finding 3):** that the g=1.0 polynomial coefficients transfer to a g=4.0 trajectory. The invariant is local. Under CFG the latent sequence diverges from g=1.0 from step 1 onward (because guidance reshapes the noise update), so the *distribution* of mod_in rel-L1 deltas and the per-branch body_out rel-L1 deltas can differ from what the g=1.0 polynomial was fit on. The two branches also have potentially different temporal smoothness even though they share a gate input, and CFG amplifies branch-level error through `negative + g * (positive - negative)`.

**Consequence:** we treat g=1.0 → g=4.0 transfer as an *empirical hypothesis* (validated by the bench gates in §"Quality + skip gates"), not as a mathematical consequence. The architecture is sound; whether the existing polynomial fires the gate at the same rate is something to measure.

## Out of scope for this design

- **Default-path recalibration at `guidance=4.0`.** v0.4.1 ships the existing g=1.0 polynomial. If the empirical bench at g=4.0 (§"Quality + skip gates") shows the gate engages with acceptable quality, we ship. If it doesn't, the CFG-aware recalibration path (§"0-skip / quality contingency") fires as a release-blocking task. Treating recalibration as a contingency is a scope choice, not a claim about transfer.
- **Per-branch separate polynomial gate decisions.** Considered (Approach B below) and rejected — see "Approaches" §B.
- **FBCache / per-block caching.** Different mechanism; v0.4 postmortem coda already ruled this out as not the chosen direction.
- **Per-variant default threshold tuning under CFG.** v0.4.0 ships `flux2-klein-base-4b` with `default_thresh=0.17` tuned for g=1.0. v0.4.1 keeps that default. If the empirical sweep at g=4.0 shows the default is wrong under CFG, file a follow-up; do not block v0.4.1 on retuning.
- **Asymmetric CFG cache (skipping only the unconditional branch).** FasterCache does this with frequency-domain reconstruction (`alpha_low_frequency`, `alpha_high_frequency`); TeaCache's premise is residual caching, not branch approximation. Out of scope.
- **Stats schema change beyond what's strictly needed.** No new `StepDecision.kind`, no new public counters. Only the internal `cfg_was_active` derivation changes (§3 below).

## Approaches considered

### A — Shared gate decision, per-branch cached residual (Recommended)

One polynomial gate evaluation per step on the shared `mod_in`. If the gate says "compute," run both transformer calls and update both cached residuals. If "skip," reuse both cached residuals. Combine via the usual `negative + guidance * (positive - negative)` math.

**Pros:** Minimal state addition (one extra `mx.array` on `TeaCacheState`). Architecture grounded by the per-step mod_in invariant above. One `StepDecision` per step — stats schema unchanged. Whether the polynomial transfers from g=1.0 calibration to g=4.0 trajectories is an empirical question answered by §"Quality + skip gates" — not assumed here.

**Cons:** When the gate says "skip" we save 100% of one full transformer call (per branch). Combined with the two branches, a skip step saves ~2× the per-call cost of a non-CFG run, but we only get one decision per step — we can't separately skip more of one branch than the other.

### B — Per-branch independent gate

Run the polynomial gate twice per step, once per branch, with separate cached residuals and separate `accumulated_distance` accumulators. Branches can drift to different skip schedules.

**Pros:** Theoretically can extract more skips on one branch than the other if the trajectories diverge.

**Cons:** Loses the mod_in invariant simplification — would need to recompute mod_in for the negative branch using the negative encoder embeddings... except mod_in doesn't depend on those, so the input signal is *identical* and the gate would always produce the same decision anyway, just with two `accumulated_distance` accumulators that drift relative to each other only through floating-point noise. Net: more state, no real benefit. **Rejected.**

### C — Approximate the negative branch from the positive branch (FasterCache-style)

Compute the positive branch's body output every step, approximate the negative branch by low/high-frequency reconstruction.

**Pros:** Halves per-step transformer cost on every step the approximation fires (not just skipped steps).

**Cons:** Requires alpha tuning per model, frequency-domain reconstruction code we don't have, fundamentally different mechanism from TeaCache. Larger code surface, larger quality risk. **Out of scope.**

**Recommendation: Approach A.** Reasoning matches the architectural mod_in invariant; the implementation is bounded; the calibration carries over; the stats schema is unchanged.

## Architecture (Approach A)

Five change sites. No new modules.

### 1. Cache state (`src/mlx_teacache/cache.py`)

Add one field to `TeaCacheState`:

```python
@dataclass
class TeaCacheState:
    # ... existing fields ...
    cached_residual: mx.array | None = None       # positive branch (unchanged role)
    cached_residual_neg: mx.array | None = None   # NEW — negative branch
    # ... existing fields ...
```

`previous_mod_input`, `step_counter`, `accumulated_distance`, `last_timestep`, `skip_window_validated`, `num_steps` stay as-is (shared across branches — see "Why this is tractable" above).

`reset_for_new_generation` clears `cached_residual_neg` alongside `cached_residual`.

### 2. New gated CFG forward (`src/mlx_teacache/integrations/mflux/forward.py`)

Add `flux2_cfg_forward_with_gate(...)` next to `flux2_forward_with_gate`. Signature mirrors the CFG-fallback function so the predict-closure swap is clean:

```python
def flux2_cfg_forward_with_gate(
    inner: Any,
    handle: Any,
    *,
    hidden_states: mx.array,
    prompt_embeds: mx.array,
    text_ids: mx.array,
    negative_prompt_embeds: mx.array,
    negative_text_ids: mx.array,
    guidance: float,
    timestep: mx.array,
    img_ids: mx.array,
) -> mx.array: ...
```

Internal flow per step:

1. **Shared prelude.** Compute `body_in`, `temb`, `temb_mod_params_img`, `temb_mod_params_txt`, `concat_rotary_emb` *once* (these are branch-independent). Compute `mod_in` once.
2. **Fast path** (`rel_l1_thresh <= 0`): run both transformer bodies with no caching, combine via CFG math, record one `"computed"` `StepDecision`, return.
3. **Gate.** Single `gate_step` call on the shared `mod_in`. Record one `StepDecision`.
4. **Compute path.** If `decision.should_compute`: run both transformer bodies, optionally update both `cached_residual` / `cached_residual_neg` per `decision.should_update_cache`.
5. **Skip path.** If not `should_compute`: reuse both `cached_residual_{pos,neg}`. Raise `InternalStateError` if either is `None` (gate logic guarantees this can't happen after seed step).
6. **Tail.** Apply `norm_out` + `proj_out` to both body outputs, then combine via `negative_noise + guidance * (noise - negative_noise)`. The tail must run on both branches because `norm_out` depends on `temb` (shared) but the body output differs per branch.
7. **State updates.** `state.step_counter += 1`, `state.last_timestep = float(timestep.flatten()[0])`.

The existing `_flux2_run_body` is reused — it's already branch-agnostic (takes `encoder_hidden_states` per call).

### 3. Predict closure rewrite (`src/mlx_teacache/integrations/mflux/flux2.py`)

The current CFG branch (lines 101-129) calls `_vanilla_flux2_cfg_predict` and records `"cfg-fallback"`. Replace with a call to `flux2_cfg_forward_with_gate`. The non-CFG branch (lines 130-150) is unchanged.

**Behavior change — skip-window validation under CFG.** Today, all-CFG generations bypass `InvalidStepWindowError` because validation is gated behind the first non-CFG call. In v0.4.1, CFG steps are gated steps, so validation must fire on the first gated step regardless of CFG. The pragmatic fix: move the lazy validation block (currently in `flux2.py:131-140`) one level up so it runs in the predict closure before dispatching to either `flux2_forward_with_gate` or `flux2_cfg_forward_with_gate`. New user-visible effect: a v0.4.0 user who set `skip_first + skip_last >= num_inference_steps` and ran all-CFG previously got silent vanilla behavior; in v0.4.1 they now get `InvalidStepWindowError`. CHANGELOG calls this out under "Behavior changes."

`_vanilla_flux2_cfg_predict` stays in the file as a **test-only parity reference** — used by the new threshold-zero parity test in §6 to assert that `flux2_cfg_forward_with_gate` at `rel_l1_thresh<=0` produces the same output (within Metal noise) as the vanilla CFG path. It's no longer called from production code.

### 4. Stats: `cfg_was_active` derivation (`src/mlx_teacache/stats.py` + `src/mlx_teacache/integrations/mflux/lifecycle.py`)

Today, `cfg_was_active` at `finalize_last_generation` time is derived from `_staging.cfg_fallback > 0`. With v0.4.1 there are no `cfg-fallback` records anymore (the path is gone). Two options:

**Option 4a (recommended): explicit CFG flag on the staging buffer.** Add `cfg_was_active: bool = False` to `_Staging`. The predict closure sets it to `True` the first time it enters the CFG branch (or `False` if no CFG call occurs). `lifecycle.call_after_loop` reads `_staging.cfg_was_active` directly when building `PendingFinalize`. `cfg_fallback_steps` stays as a public field but is **always 0 in v0.4.1+** (kept for backward compatibility with v0.4.0 consumers; documented in CHANGELOG as deprecated and slated for removal in v1.0).

**Option 4b (rejected): repurpose `cfg_fallback_steps` to mean "CFG steps."** Backwards-incompatible semantic change — existing consumers that read `cfg_fallback_steps` to detect fallback would silently break.

We go with 4a. `_staging.clear()` resets `cfg_was_active` to `False`. `discard_current_generation` does the same via the existing `_staging.clear()` path.

Lifecycle's `call_before_loop` distilled-step warning currently suppresses on `flux2_cfg_fallback = (variant.startswith("flux2-") and guidance > 1.0)`. Now that CFG isn't a fallback, this suppression should be **removed** — the regular `possible_skips == 0` check is the source of truth. (See §6 "Lifecycle warning" for the test that locks this in.)

### 5. Bench (`scripts/bench_speedup.py`) — three-way protocol

Per audit Finding 1, the v0.4.1 win has to be separable from the v0.4 "eager wrapper / compile-path avoidance" win that's already present at `guidance > 1.0`. The bench therefore runs **three conditions** on `klein-base-4b` at `guidance=4.0, num_inference_steps=50` and reports all three:

| Condition | What runs | Attribution |
|---|---|---|
| **A. Vanilla mflux** | Original `flux.generate_image` with no `apply_teacache` wrapper. mflux's `_predict` is whatever mflux chose (compiled on M3+/Max/Ultra; eager on M1/M2). | Baseline. |
| **B. Wrapped, no gate** | `apply_teacache(flux, rel_l1_thresh=0)`. The eager `_predict` replacement runs; the gate is hard-short-circuited to "always compute, never cache" via the existing threshold-zero fast path in `flux2_cfg_forward_with_gate`. | Isolates the eager-wrapper / compile-path-avoidance effect — what v0.4 already gives a CFG user. |
| **C. Wrapped, gated** | `apply_teacache(flux)` at default threshold `0.17`. Full v0.4.1 gated path. | Adds CFG step-skipping on top of (B). |

Report wall-clock medians, skip counts, and the two ratios: `A / B` (compile-avoidance win, attributable to v0.4) and `B / C` (gating win, attributable to v0.4.1). README's v0.4.1 row carries the `B / C` number explicitly — that is the headline. `A / C` is also reported as "combined speedup over upstream vanilla" for completeness.

Implementation note: the three-condition harness lives behind a `--three-way` flag (default `True` on `klein-base-4b`). Other variants stay on the existing two-condition (vanilla vs wrapper) bench.

Saves `tests/_artifacts/bench_images/klein-base-4b/{vanilla,wrapper_nogate,wrapper_gated}.png`.

The g=1.0 / 25-step numbers from v0.4.0 stay reproducible via `--guidance 1.0 --num-inference-steps 25` overrides (the existing v0.4 bench row stays valid).

### 6. Tests

- **`tests/test_parity_flux2.py`:** Two CFG parity tests on base-4b at `guidance=4.0`, 50 steps, `rel_l1_thresh=0`:
  1. **Release-blocker — paired same-process vanilla-vs-wrapper.** Generate one image with no wrapper (real mflux `_predict`, compiled or eager per chip), generate another in the same process with `apply_teacache(flux, rel_l1_thresh=0)`, compare. Cosine ≥ `_FLUX2_COSINE_GATE` (0.97); pixel mismatch ratio ≤ 0.15. This matches the existing FLUX.2 paired-parity pattern (`tests/test_parity_flux2.py:46-63`) and is what user-facing parity actually means — per audit Finding 4, the local helper test alone is too weak because the helper shares assumptions with the new gated function.
  2. **Diagnostic — gated path vs `_vanilla_flux2_cfg_predict`.** Same threshold-zero generation but the reference is the in-repo helper. Looser bound acceptable (cosine ≥ 0.99 since both run inside our eager `_predict` replacement and share more graph topology). This test isolates "the new gated function matches our helper" from "our helper matches mflux." Useful for debugging when the paired parity test fails.
- **`tests/test_image_quality_flux2.py`:** Add CFG SSIM test on base-4b at default threshold (`0.17`), `guidance=4.0`, 50 steps. SSIM ≥ 0.85. Skip count ≥ 1 (re-asserts the v0.4.0 engagement claim now also holds under CFG).
- **`tests/test_api.py`:** Add `test_apply_teacache_cfg_smoke_klein_base_4b` — apply, generate at g=4.0, assert `handle.stats.cfg_was_active` would be True after generation (use the actual `GenerationStats.cfg_was_active` field). Verify no `cfg-fallback` decisions in the recorded `decisions` tuple.
- **`tests/test_stats.py` (new or extension of existing):** Verify `_staging.cfg_was_active` is set correctly through the predict-closure → lifecycle → finalize path. Use the existing fake-flux test scaffolding from `test_api.py`.
- **Lifecycle warning test:** Add `test_no_benefit_warning_does_not_suppress_under_cfg` — base-4b at g=4.0 with `skip_first=skip_last=0` and a 1-step generation should warn `TeaCacheNoBenefitWarning` (the `flux2_cfg_fallback` suppression is gone, so the normal `possible_skips == 0` path fires).
- **Update `_gen_kwargs_klein()`** in test files: base-4b kwargs now have a `cfg` variant alongside the existing `g=1.0` variant. Existing tests stay green at g=1.0; new tests exercise g=4.0, 50 steps.

### 7. Docs

- **`README.md`**: Update the base-4b row in the supported-variants table to drop the "CFG falls back to vanilla pending v0.4.1" footnote. Update the Benchmarks table with the new g=4.0, 50-step row. Replace the v0.4.0 Limitations bullet ("CFG on base-4b lands in v0.4.1") with the v0.4.1 status. Update Quick-start example for base-4b to use `guidance=4.0, num_inference_steps=50` (the canonical upstream recipe).
- **`CHANGELOG.md`**: v0.4.1 entry — "CFG-engaged TeaCache for FLUX.2. The canonical upstream recipe (`guidance_scale=4.0, num_inference_steps=50`) on `flux2-klein-base-4b` is now accelerated end-to-end. `_vanilla_flux2_cfg_predict()` no longer runs in production paths. `cfg_fallback_steps` deprecated (always 0; slated for removal in v1.0). Per-branch cached residual added to `TeaCacheState` (`cached_residual_neg`)."
- **`docs/calibration.md`**: Add a note that the base-4b polynomial calibrated at g=1.0 transfers to g=4.0 because mod_in is encoder-independent. If empirical drift is observed (bench numbers below), document the threshold or recalibration in a follow-up section.
- **`ROADMAP.md`**: Move v0.4.1 from Active to Released; promote v0.5.0 (`flux2-klein-base-9b`) to top of Active.
- **`docs/superpowers/notes/2026-05-17-v0.4.0-branch-audit.md`**: Add a 2026-MM-DD post-v0.4.1 coda noting the audit's Finding 1 (mechanism over-attribution) is no longer in tension with CFG performance claims because the same machinery now drives CFG.

### Data flow

```
apply_teacache(Flux2Klein(model_config=ModelConfig.flux2_klein_base_4b()))
    ↓
flux._predict = make_teacache_predict_factory(handle)
    ↓
flux.generate_image(guidance=4.0, num_inference_steps=50, ...)
    ↓
per-step (predict closure):
  if cfg_active: flux2_cfg_forward_with_gate(...)    ← NEW gated CFG path
  else:          flux2_forward_with_gate(...)        ← unchanged
    ↓
flux2_cfg_forward_with_gate:
  1. Shared prelude: mod_in, body_in, temb, mods
  2. gate_step(mod_in) → one decision
  3a. compute → both transformer calls + (optional) update both cached_residual_{pos,neg}
  3b. skip    → reuse both cached_residual_{pos,neg}
  4. tail (norm_out + proj_out) on both branches
  5. combined = negative + guidance * (positive - negative)
```

## Quality + skip gates (release blockers)

- **PR-gate SSIM** ≥ 0.85 on the red-apple prompt at default threshold `0.17`, `num_inference_steps=50, guidance=4.0`, on `flux2-klein-base-4b`.
- **Skip count** ≥ 1 across 3 reps at default threshold, on the red-apple bench prompt, at 50 steps + g=4.0.
- **Wall-clock speedup** ≥ 1.2× on M1 Max at 50 steps + g=4.0. Lower bar than v0.4.0's 1.3× because CFG doubles the per-step cost (two transformer calls per gated step) — the same skip count produces less wall-clock improvement.
- **CFG parity cosine** ≥ `_FLUX2_COSINE_GATE` (0.97) at threshold=0 on the **paired same-process vanilla-vs-wrapper** test (real mflux generation, not the in-repo helper). This is the user-facing "we didn't break the math" gate. The diagnostic helper-vs-gated test (§6) carries a tighter cosine ≥ 0.99 bound but is not the release blocker.
- **`cfg_was_active` correctness**: at `guidance > 1.0` the GenerationStats records `cfg_was_active=True`; at `guidance == 1.0` it records `False`.

## 0-skip / quality contingency

If post-bench:

1. **Skip count is 0 at default threshold under CFG, OR SSIM < 0.85.** Run a CFG-aware calibration. Per audit Finding 2, this is **not** a flag-on-the-existing-script change — the current capturing closure (`scripts/calibrate_flux2.py:139-192`) discards `negative_prompt_embeds` and returns positive-only noise to the scheduler, which at `guidance=4.0` would generate a non-CFG trajectory and produce misleading data. The CFG-aware calibration must:
   - **Compute both branches.** The closure runs the transformer for both positive and negative encoder embeddings each step, like `_vanilla_flux2_cfg_predict` does.
   - **Return CFG-combined noise to the scheduler.** `negative_noise + guidance * (positive_noise - negative_noise)` — the scheduler's next-latent depends on the real CFG output; without this, every subsequent step's mod_in / body_out captures are off-trajectory.
   - **Capture per-branch body_out_concat alongside the shared mod_in.** Save `body_out_pos` and `body_out_neg` per step. The shared mod_in is captured once.
   - **Fit policy:** the polynomial maps mod_in_rel_l1 → body_out_rel_l1. Under CFG we have two body_out trajectories. The per-step y target is `max(rel_l1(body_out_pos), rel_l1(body_out_neg))` — the worst-branch delta. Fitting on the worst-branch ensures the gate decision is safe for whichever branch drives quality. Alternative fit modes (average, positive-only, negative-only) are recorded in the JSON for offline analysis but the default fit is worst-branch.
   - **JSON output schema extension:** add `fit_branch_policy` field ("worst" / "average" / "positive" / "negative"), `x_values_shared`, `y_values_pos`, `y_values_neg` arrays so refits are possible offline.
   - **CLI:** `uv run python scripts/calibrate_flux2.py --variant klein-base-4b --fit-mode origin --guidance 4.0 --fit-branch-policy worst`.

   Implementation note: the CFG-aware closure is ~30 LoC, structurally similar to `_vanilla_flux2_cfg_predict` plus the existing capture logic. Reuses `_flux2_extract_mod_input` and `_flux2_run_body` directly.

2. **Wall-clock speedup `B / C` < 1.2× (gating contribution).** Acceptable to ship as long as skip count ≥ 1 and SSIM passes — the feature is real, the win is bounded by CFG's 2× per-step cost. Document the realistic speedup in the bench row using the three-way numbers.

3. **`B / C` ratio is ~1.0× (gate doesn't engage at all under CFG).** Same as (1): run the CFG-aware calibration before tagging. Do not ship v0.4.1 with the headline `B / C` win equal to 1.0× — that would mean the feature does nothing.

If recalibration is performed, the new coefficients land in `_REGISTRY["flux2-klein-base-4b"]` with an updated `revision` field and a re-run of `sweep_threshold_klein_base_4b.py` (or its CFG counterpart) to confirm the per-variant default still produces the best skip/SSIM tradeoff under CFG.

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Polynomial calibrated at g=1.0 doesn't transfer to g=4.0 (drift in skip count or SSIM) | Medium | Empirical bench + SSIM gate above. Contingency: recalibrate at g=4.0 via the new `--guidance` flag on the calibration script. |
| `cached_residual_neg` doubles cache memory footprint per generation | Low | Each cached residual is one `body_out_concat` tensor (~50 MB at 512×512 base-4b q4). Doubling to ~100 MB is fine on 16 GB+ unified memory. Documented in CHANGELOG. |
| `_vanilla_flux2_cfg_predict` removal regresses some unknown test | Low | Keep the function in the file as a test-only parity reference. Only the production call site changes. Existing tests that import it stay green. |
| CFG parity test fails at threshold=0 due to MLX graph topology differences (same class of bug as v0.4.0's `_flux2_run_body` upfront-vs-inline modulation) | Medium | The new `flux2_cfg_forward_with_gate` reuses `_flux2_run_body` unchanged, so the topology that produced byte-exact parity in v0.4.0 carries over per branch. The CFG combination math is a single `mx.add` + `mx.subtract` + `mx.multiply` triplet — also no graph-topology surprises. Parity test catches any regression. |
| Stats schema confusion: `cfg_fallback_steps` always 0 in v0.4.1+ but kept for backward compat | Low | CHANGELOG explicitly calls this out with the deprecation. Field gets a docstring note: "Deprecated since v0.4.1; always 0. Use `GenerationStats.cfg_was_active` instead. Slated for removal in v1.0." |
| Distilled Klein 4B/9B at g > 1.0 regresses because they now hit the gated CFG path instead of vanilla | Low | The polynomial gate at default threshold produces 0 skips on distilled schedules (per the v0.3 postmortem), so the new path runs both branches at every step — identical to the old `_vanilla_flux2_cfg_predict` work, with one extra polynomial-gate evaluation per step (~µs). No wall-clock regression. Add a smoke test at distilled g=4.0 (Klein 4B). |
| `mx.compile` interaction (we're still eager, but the new function adds more eager ops) | Low | The full `_predict` was already eager-replaced in v0.1; v0.4.1 doesn't change that. Adding more eager work doesn't reintroduce a compile path. |

## Effort + timeline

- Code (cache field, new forward function, predict closure swap, stats flag): ~1 day.
- Lifecycle warning suppression removal: ~30 min.
- Calibration-script `--guidance` flag (contingency-ready, not necessarily used): ~30 min.
- Bench update (50-step + g=4.0 default for base-4b): ~30 min.
- Tests (parity, SSIM, smoke, stats invariant, lifecycle warning): ~1 day.
- Bench run at g=4.0, 50 steps (cold cache + 3 reps): ~1.5 hours.
- Contingency: if recalibration is needed, +6-8 hours of calibration time.
- Docs (README, CHANGELOG, calibration.md, ROADMAP): ~2-3 hours.
- PR + review + CI: ~2-3 hours.
- **Total: 3-4 working days if calibration transfers; 4-5 days if recalibration is needed.**

## Release packaging

Single PR for v0.4.1:
1. Cache + forward + predict-closure changes.
2. Stats `cfg_was_active` flag wiring.
3. Lifecycle warning suppression removal.
4. Calibration script `--guidance` flag (precautionary, even if g=1.0 polynomial transfers).
5. Bench update for base-4b → 50 steps + g=4.0.
6. Test additions (parity, SSIM, smoke, stats, lifecycle warning).
7. README + CHANGELOG + calibration.md + ROADMAP updates.
8. Tag v0.4.1 from the merge commit; release.yml triggers PyPI publish, same flow as v0.3.0 / v0.4.0.

## Acceptance criteria

v0.4.1 is ready to tag when:
- [ ] `flux2_cfg_forward_with_gate` exists in `forward.py` and the predict closure routes CFG steps through it (no production call to `_vanilla_flux2_cfg_predict`).
- [ ] `TeaCacheState.cached_residual_neg` added and cleared in `reset_for_new_generation`.
- [ ] `_Staging.cfg_was_active` added; lifecycle reads it; `GenerationStats.cfg_was_active` is correct for both g=1.0 and g=4.0 generations.
- [ ] Lifecycle's `flux2_cfg_fallback` warning suppression is gone.
- [ ] `tests/test_parity_flux2.py` CFG parity test passes (cosine ≥ 0.97 at threshold=0, g=4.0).
- [ ] `tests/test_image_quality_flux2.py` CFG SSIM test passes (≥ 0.85 at default threshold, g=4.0, 50 steps, skip count ≥ 1).
- [ ] `tests/test_api.py` CFG smoke test passes.
- [ ] `tests/test_stats.py` (or existing test extension) confirms `cfg_was_active` invariant.
- [ ] `scripts/bench_speedup.py --variant klein-base-4b` reports wall-clock speedup ≥ 1.2× and skip count ≥ 1, OR the 0-skip / quality contingency is fully resolved.
- [ ] CI green on the PR.
- [ ] README, CHANGELOG, `docs/calibration.md`, and ROADMAP reflect the new behavior.

## Open questions for the user (non-blocking, can be answered during plan execution)

- **Default bench recipe.** v0.4.0 bench is `g=1.0, 25 steps`. v0.4.1 default should be `g=4.0, 50 steps` (canonical upstream recipe) — but the README v0.4.0 row stays valid. Confirm we publish both rows side-by-side or replace the v0.4.0 row.
- **Polynomial recalibration as v0.4.1 task vs. follow-up.** If the bench at g=4.0 shows acceptable skip count + SSIM with the existing polynomial, we ship. If it doesn't, we recalibrate as part of v0.4.1 (adds ~8h calibration). Default plan assumes the polynomial transfers; the script's `--guidance` flag is added precautionarily.
