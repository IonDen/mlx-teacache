# v0.4.1 CFG Per-Branch Caching Plan Audit

**Plan:** `docs/superpowers/plans/2026-05-17-flux2-cfg-per-branch-caching.md`  
**Audit date:** 2026-05-17  
**Scope:** implementation-plan review for material current-release issues only. I ignored normal development friction: stale tests, helper names, lint/typing details, and placeholder scaffolding unless it changed the release claim.

## Verdict

The plan is largely aligned with the audited spec. The core architecture remains sound: one shared `mod_in` gate decision, separate positive/negative cached residuals, and a three-way benchmark that separates v0.4 compile/eager effects from v0.4.1 gate engagement.

I would fix two things before handing this to implementation:

1. the CFG recalibration contingency currently calibrates the wrong step schedule;
2. the fallback release branch still allows a "structural-only" v0.4.1 even if the feature does not engage.

There is also one public-doc wording correction to avoid reintroducing the earlier `mod_in` transfer overclaim.

## Findings

### 1. CFG recalibration would still run at 25 steps, not the failing 50-step release target

**Severity:** High  
**Files:** `docs/superpowers/plans/2026-05-17-flux2-cfg-per-branch-caching.md`, `scripts/calibrate_flux2.py`, `scripts/sweep_threshold_klein_base_4b.py`

Task 11 makes the v0.4.1 release gate the canonical base-4B CFG recipe: `guidance=4.0`, `num_inference_steps=50` (`plan:1276-1306`). That matches the upstream FLUX.2 Klein Base 4B model card, which shows `guidance_scale=4.0` and `num_inference_steps=50`.

But Task 8 only adds `--guidance` and `--fit-branch-policy` to `scripts/calibrate_flux2.py` (`plan:969-1018`). It does not add a `--num-inference-steps` override. The current calibration script's base-4B config is still hard-coded to 25 steps:

- `scripts/calibrate_flux2.py:99-103`: `klein-base-4b` uses `"num_inference_steps": 25`;
- `scripts/calibrate_flux2.py:197-210`: `_capture_one_prompt()` passes that value into `generate_image()`.

So Task 12's contingency command:

```bash
uv run python scripts/calibrate_flux2.py --variant klein-base-4b --fit-mode origin --guidance 4.0 --fit-branch-policy worst
```

would capture a **25-step CFG trajectory**, while the release failure it is supposed to fix happened on a **50-step CFG trajectory**. The plan even estimates Task 12 as `50 steps x 10 prompts x 2 transformer calls` (`plan:1321-1325`), but the script path as specified would not do that.

Impact: if the 50-step CFG bench has 0 skips or poor SSIM, a 25-step CFG refit can produce coefficients that look valid in the calibration report but still fail the actual release gate. Since TeaCache's gate depends on adjacent-step `mod_in` and body-output deltas, changing the denoising schedule changes the data distribution the polynomial sees.

**Fix:** add a calibration `--num-inference-steps` override in Task 8 and make Task 12 run:

```bash
uv run python scripts/calibrate_flux2.py \
  --variant klein-base-4b \
  --fit-mode origin \
  --guidance 4.0 \
  --num-inference-steps 50 \
  --fit-branch-policy worst
```

Also update the JSON/provenance to record both guidance and step count clearly. If threshold retuning is needed, `scripts/sweep_threshold_klein_base_4b.py` should be made CFG/50-step aware before selecting a new `default_thresh`; its current constants are `STEPS = 25` and `guidance=1.0` (`sweep_threshold_klein_base_4b.py:41-61`).

### 2. The plan still permits a release that fails the feature's own engagement gate

**Severity:** Medium-High  
**Files:** `docs/superpowers/plans/2026-05-17-flux2-cfg-per-branch-caching.md`, `README.md`, `CHANGELOG.md`, `ROADMAP.md`

The plan's goal is explicit: v0.4.1 should make the canonical base-4B CFG path gate-active so it "can skip steps" (`plan:5`). Task 10 and Task 11 correctly encode that as release evidence: SSIM must pass and skip count must be at least 1 at `guidance=4.0`, 50 steps (`plan:1192-1255`, `plan:1296-1306`).

Task 12 weakens that contract at the end:

> "If still failing, lower `default_thresh` further or document v0.4.1 as a structural-only release" (`plan:1340-1347`).

That path would let the project tag v0.4.1 after the contingency still fails the core release claim. A structural-only implementation may be a useful internal PR, but it is not the release described by this plan, README row, changelog entry, or roadmap move-to-released text.

Impact: the release could repeat the earlier non-engagement problem under a new name: production CFG no longer records `cfg-fallback`, but the canonical user recipe still gets no real skip benefit. That would make the v0.4.1 docs misleading even if the code path is cleaner.

**Fix:** remove the structural-only release branch. If Task 12 still fails skip/SSIM after CFG/50 calibration and sweep, either hold v0.4.1, retarget the PR as an internal architecture change without user-facing benchmark claims, or explicitly rename/rescope the release. The `skip count >= 1` gate should remain a release blocker for this version.

### 3. The calibration-doc wording reintroduces the transfer overclaim

**Severity:** Medium  
**Files:** `docs/superpowers/plans/2026-05-17-flux2-cfg-per-branch-caching.md`, `docs/calibration.md`

The plan's main body correctly treats g=1.0 to g=4.0 transfer as empirical. However, the docs task says to append:

> "Polynomial calibrated at `guidance=1.0`; the per-step mod_in gating signal is encoder-independent so the same coefficients ship under CFG." (`plan:1400-1404`)

That wording makes encoder-independence sound like the reason coefficient transfer is valid. It is not. The invariant only proves that positive and negative branches can share one gate input at a fixed latent/timestep. It does not prove that a g=1.0 polynomial fits the g=4.0 denoising trajectory.

Impact: this would put the same overclaim corrected in the spec audit into public calibration docs. Future threshold or calibration decisions could lean on the wrong premise.

**Fix:** change the docs wording to:

> "Polynomial calibrated at `guidance=1.0`; v0.4.1 reuses it under CFG only because the g=4.0 / 50-step release bench passed the skip and SSIM gates. The encoder-independent `mod_in` invariant justifies one shared branch decision per step; coefficient transfer remains empirical."

If Task 12 runs, include `--num-inference-steps 50` in the calibration example as part of Finding 1's fix.

## Confirmed Good Decisions

- The plan's three-way benchmark is the right attribution model: vanilla mflux, wrapped no-gate (`rel_l1_thresh=0`), and wrapped gated (`plan:746-845`).
- Keeping `_vanilla_flux2_cfg_predict()` as a diagnostic reference while making same-process vanilla-vs-wrapper parity the release blocker is correct (`plan:1090-1148`).
- The shared decision plus two residuals approach matches the local FLUX.2 transformer structure: `mod_in` is computed from latents/timestep/image-stream modulation before prompt encoder states enter the branch-specific attention body (`forward.py:258-304`, mflux transformer lines 67-133).

## Sources

- Upstream model card: https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B
- BFL FLUX.2 overview: https://docs.bfl.ml/flux_2/flux2_overview
- Local mflux 0.17.5 source: `.venv/lib/python3.13/site-packages/mflux`
- Current local code under `src/mlx_teacache/`, `scripts/`, and `docs/superpowers/`
