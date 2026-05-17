# v0.4.1 CFG Per-Branch Caching Spec Audit

**Spec:** `docs/superpowers/specs/2026-05-17-flux2-cfg-per-branch-caching-design.md`  
**Audit date:** 2026-05-17  
**Scope:** design review for v0.4.1 FLUX.2 CFG caching; material current-release issues only.

## Verdict

The shared-gate / two-residual architecture is directionally sound. The local `mod_in` invariant checks out against the current FLUX.2 forward path: at a fixed latent and timestep, the image-stream gate signal is independent of prompt encoder embeddings.

I would fix three spec issues before handing this to a plan:

1. the spec misstates the current CFG fallback baseline and therefore risks over-attributing v0.4.1 speedups;
2. the proposed `--guidance 4.0` calibration contingency cannot be implemented as a small flag on the current calibration script;
3. the transfer argument from g=1.0 to g=4.0 is too strong: the invariant is local per step, not a guarantee over the CFG denoising trajectory.

## Findings

### 1. The spec treats v0.4 CFG fallback as vanilla mflux, but the wrapper already replaces `_predict`

**Severity:** High  
**Files:** `docs/superpowers/specs/2026-05-17-flux2-cfg-per-branch-caching-design.md`, `ROADMAP.md`, `src/mlx_teacache/integrations/mflux/flux2.py`, `src/mlx_teacache/api.py`, `scripts/bench_speedup.py`

The spec goal says v0.4.1 means "no FLUX.2 generation falls back to vanilla mflux" (`spec:14`), and the roadmap says CFG users of distilled Klein will newly get the compile-avoidance benefit (`ROADMAP.md:18`).

That is not quite the current code reality. `apply_teacache()` replaces `flux._predict` with `make_teacache_predict_factory(handle)` for every FLUX.2 variant (`api.py:287-288`). Inside that replacement, the CFG branch calls `_vanilla_flux2_cfg_predict()` directly (`flux2.py:101-129`). That function performs vanilla CFG math, but it does not call mflux's original `_predict` closure. In mflux 0.17.5, the original `Flux2Klein._predict()` returns `mx.compile(predict)` on non-M1/M2 chips (`.venv/.../flux2_klein.py:279-281`).

So the v0.4 path is better described as:

- **vanilla CFG math**;
- **no TeaCache gate / no skip decisions**;
- still inside the eager TeaCache `_predict` replacement, not necessarily vanilla mflux's compiled `_predict`.

Impact: a v0.4.1 benchmark of vanilla mflux vs wrapper at `guidance=4.0` will mix at least two effects: existing eager-wrapper / compile-path change plus new CFG step-skipping. That repeats the v0.4.0 attribution problem unless the spec requires a three-way comparison:

1. vanilla mflux;
2. v0.4-style CFG fallback / eager no-gate path;
3. v0.4.1 gated CFG path.

**Fix:** revise the goal and bench sections. The v0.4.1 feature is "CFG steps become gate-active and can skip," not "compile avoidance becomes available under CFG." Add a benchmark control that disables skips under the v0.4.1 path (`rel_l1_thresh=0`) or preserves the v0.4 fallback path behind a local bench flag, so the docs can separate compile/eager effects from actual CFG caching.

### 2. The g=4.0 recalibration contingency is under-specified and would produce invalid data if implemented as written

**Severity:** High  
**Files:** `docs/superpowers/specs/2026-05-17-flux2-cfg-per-branch-caching-design.md`, `scripts/calibrate_flux2.py`

The spec says that if CFG skip/quality fails, recalibration can be done with:

```bash
uv run python scripts/calibrate_flux2.py --variant klein-base-4b --fit-mode origin --guidance 4.0
```

and calls this a small `--guidance` flag addition (`spec:213-218`, `spec:239`, `spec:254`).

The current calibration capture cannot support that by just adding a CLI flag. `_make_capturing_closure()` explicitly discards `guidance`, `negative_prompt_embeds`, and `negative_text_ids` (`calibrate_flux2.py:139-150`). It records only the positive branch body output (`calibrate_flux2.py:176-188`) and returns only the positive branch noise to the scheduler (`calibrate_flux2.py:189-192`). At `guidance=4.0`, mflux's real path computes both positive and negative branches and combines them (`.venv/.../flux2_klein.py:267-276`).

If the current capture is run with `--guidance 4.0`, the denoising trajectory is not a CFG trajectory at all; it is positive-only noise returned into a config whose guidance says 4.0. Any coefficients or y-ranges from that run would be misleading.

**Fix:** make the contingency explicitly CFG-aware:

- encode/capture both positive and negative branch body outputs;
- return the CFG-combined noise to the scheduler so the next latent follows the real g=4.0 trajectory;
- record branch-specific `body_out_concat` deltas, or at least the max/worst of positive and negative branch deltas, because v0.4.1 will cache both residuals from one shared decision;
- write the calibration JSON so it says whether the fit was positive-only, negative-only, combined, or worst-branch.

This is not necessarily a huge implementation, but it is more than a small flag and should be reflected in the plan.

### 3. The `mod_in` invariant is local; it does not prove g=1.0 coefficients transfer to the CFG trajectory

**Severity:** Medium-High  
**Files:** `docs/superpowers/specs/2026-05-17-flux2-cfg-per-branch-caching-design.md`, `docs/calibration.md`

The spec correctly observes that at a fixed step, `mod_in` is independent of positive vs negative encoder states (`spec:20-33`). The current implementation supports that: `mod_in` uses `x_embedder(hidden_states)`, timestep embeddings, image-stream modulation, and block-0 image-stream norm; encoder states enter later through attention (`forward.py:258-304`, `.venv/.../transformer_block.py:35-43`).

But the spec then leans too hard on that invariant:

- "Skip-rate matches non-CFG runs at the same threshold" (`spec:50`);
- g=1.0 polynomial "transfers naturally" / recalibration is out of scope unless gates fail (`spec:37-40`, `spec:175`);
- the benchmark expects the canonical `guidance=4.0, num_inference_steps=50` path to validate the same default threshold (`spec:201-207`).

The invariant is only per-step and conditional on the current latent. Under CFG, the noise update after the first step is different, so the subsequent latent sequence and `mod_in_t -> mod_in_{t-1}` distribution can differ from the g=1.0 calibration trajectory. The negative branch body-output trajectory can also have a different temporal smoothness than the positive branch even though the input gate signal is shared. CFG then amplifies branch errors through `negative + guidance * (positive - negative)`.

Impact: the shared-gate design can still be valid, but the spec should present transfer as a hypothesis to be measured, not as a mathematical consequence of encoder-independent `mod_in`. Given the v0.4.0 base-4B fit is already low-R2 and threshold-sensitive, over-trusting transfer is a real release risk.

**Fix:** update the design language:

- keep "one shared gate decision" as the implementation choice;
- remove "skip-rate matches non-CFG runs";
- require the g=4.0 bench report to include actual skip counts and quality at the canonical 50-step trajectory;
- if quality or skip count is marginal, run the CFG-aware calibration/sweep before release instead of filing it as a follow-up.

### 4. The parity oracle should still include actual vanilla mflux, not only `_vanilla_flux2_cfg_predict`

**Severity:** Medium  
**Files:** `docs/superpowers/specs/2026-05-17-flux2-cfg-per-branch-caching-design.md`, `tests/test_parity_flux2.py`

The spec says `_vanilla_flux2_cfg_predict` remains as the test-only parity reference (`spec:131`) and the release blocker is "CFG parity cosine >= 0.97 at threshold=0 against `_vanilla_flux2_cfg_predict`" (`spec:206`).

That reference is useful, but by itself it is weaker than the existing paired-parity pattern. A local reference can share the same implementation assumptions as the new gated function. The existing FLUX.2 parity tests deliberately compare wrapper output against same-process vanilla generation because graph topology and eager/compiled MLX dispatch have caused real differences before (`tests/test_parity_flux2.py:46-63`, `forward.py:316-324`).

Impact: `flux2_cfg_forward_with_gate(rel_l1_thresh=0)` could match the local reference and still drift materially from actual mflux generation. The user-facing contract is "wrapper matches mflux vanilla math closely enough," not merely "new function matches our helper."

**Fix:** keep the helper-level test, but make the release blocker a paired same-process vanilla-vs-wrapper CFG test at `guidance=4.0`, `num_inference_steps=50`, `rel_l1_thresh=0`. Use the same cosine/image-quality style already accepted for FLUX.2 numerical parity. The helper test can remain a narrower diagnostic.

## Confirmed Assumptions

- The upstream base-4B model-card framing is correct: Hugging Face lists Apache-2.0, describes base-4B as full-capacity / undistilled, and shows a Diffusers example with `guidance_scale=4.0` and `num_inference_steps=50`.
- BFL docs distinguish distilled API Klein models from undistilled Base variants; Base 4B is Apache-2.0 and not step-distilled.
- mflux 0.17.5 creates negative prompt embeddings when `guidance > 1.0` and combines the two transformer calls with CFG math in `_predict()`.
- The shared per-step `mod_in` signal is independent of positive vs negative prompt embeddings in current mflux 0.17.5.

Sources:

- https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B
- https://docs.bfl.ml/flux_2/flux2_overview
- local mflux 0.17.5 source under `.venv/lib/python3.13/site-packages/mflux`

## Suggested Spec Edits Before Planning

1. Reframe v0.4.1 as "CFG gate engagement and skip support" rather than "no vanilla mflux fallback / compile avoidance now available."
2. Add a three-way benchmark protocol: vanilla mflux, no-skip eager wrapper, gated wrapper.
3. Replace the calibration contingency with a CFG-aware capture design that records both branches and returns CFG-combined noise.
4. Treat g=1.0 -> g=4.0 coefficient transfer as empirical; remove language implying the skip-rate should match.
5. Make paired same-process vanilla-vs-wrapper CFG parity the release blocker; keep `_vanilla_flux2_cfg_predict` as a diagnostic reference.
