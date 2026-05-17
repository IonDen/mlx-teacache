# v0.4.1 — CFG Per-Branch Caching for FLUX.2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current `_vanilla_flux2_cfg_predict()` non-gated CFG path with a per-branch gated forward, so the canonical upstream recipe (`flux2-klein-base-4b` at `guidance_scale=4.0, num_inference_steps=50`) can skip steps. After v0.4.1, CFG steps are gate-active. `cfg_fallback_steps` is deprecated (always 0); `GenerationStats.cfg_was_active` is the new signal.

**Architecture:** Approach A from the spec — one shared gate decision per step, two cached residuals (positive + negative), one CFG combination step. The gate input (`mod_in`) is byte-identical across branches because it depends only on shared latents + timestep, not on encoder embeddings. Five change sites: `cache.py` (one new field), `forward.py` (new gated CFG function), `flux2.py` (predict-closure rewrite + skip-window validation lift), `lifecycle.py` (remove CFG-fallback warning suppression), `stats.py` (`cfg_was_active` staging field). Plus bench three-way protocol, optional CFG-aware calibration extension (release-blocker contingency only), tests, and docs.

**Tech Stack:** Python 3.11+, `mflux>=0.17,<0.18`, MLX, pytest, ruff. Existing TDD discipline: tests before implementation; ruff + mypy clean before each commit.

**Spec:** [`docs/superpowers/specs/2026-05-17-flux2-cfg-per-branch-caching-design.md`](../specs/2026-05-17-flux2-cfg-per-branch-caching-design.md). Spec audit: [`docs/superpowers/notes/2026-05-17-flux2-cfg-per-branch-caching-spec-audit.md`](../notes/2026-05-17-flux2-cfg-per-branch-caching-spec-audit.md).

---

## File map

| File | Responsibility | Action |
|---|---|---|
| `src/mlx_teacache/cache.py` | Per-handle cache state | Add `cached_residual_neg: mx.array \| None = None`; clear it in `reset_for_new_generation`. |
| `src/mlx_teacache/stats.py` | Stats schema | Add `cfg_was_active: bool = False` to `_Staging`; clear in `_Staging.clear()`; deprecate `cfg_fallback_steps` (always 0 from v0.4.1+). |
| `src/mlx_teacache/integrations/mflux/forward.py` | Gated transformer forward | Add `flux2_cfg_forward_with_gate(...)` next to `flux2_forward_with_gate`. Reuses `_flux2_run_body` per branch. |
| `src/mlx_teacache/integrations/mflux/flux2.py` | `_predict` replacement closure | Rewrite CFG branch to call `flux2_cfg_forward_with_gate`. Move skip-window validation up so it runs on first gated call regardless of CFG. Set `_staging.cfg_was_active = True` on first CFG branch entry. Keep `_vanilla_flux2_cfg_predict` as a test-only diagnostic reference. |
| `src/mlx_teacache/integrations/mflux/lifecycle.py` | Lifecycle warning | Remove `flux2_cfg_fallback` suppression; read `_staging.cfg_was_active` for `PendingFinalize.cfg_was_active`. |
| `scripts/bench_speedup.py` | Reproducible bench | Add three-way protocol (`--three-way`, default `True` on `klein-base-4b`): vanilla mflux, wrapped-no-gate (`rel_l1_thresh=0`), wrapped-gated (default threshold). Default `klein-base-4b` config moves to `g=4.0, 50 steps`; preserve g=1.0 / 25-step via `--guidance` and `--num-inference-steps` overrides. |
| `scripts/calibrate_flux2.py` | Calibration driver | Add `--guidance` and `--fit-branch-policy` flags AND a CFG-aware capture closure. Used only if the v0.4.1 release-gate bench fails to engage. The CFG-aware closure computes both branches, returns CFG-combined noise to the scheduler, captures per-branch `body_out` plus the shared `mod_in`. JSON output gains `fit_branch_policy`, `y_values_pos`, `y_values_neg`. |
| `tests/test_api.py` | Public API smoke (parity-marked) | Add `test_apply_teacache_cfg_smoke_klein_base_4b` (g=4.0 generation produces `GenerationStats.cfg_was_active=True`; no `cfg-fallback` decisions). |
| `tests/test_stats.py` (NEW or extension) | Stats unit | Verify `_staging.cfg_was_active` toggles correctly; verify `cfg_fallback_steps` stays 0 in v0.4.1+. |
| `tests/test_parity_flux2.py` | Parity oracle (parity-marked) | Two new CFG parity tests: (1) **release-blocker** paired same-process vanilla-vs-wrapper at `rel_l1_thresh=0, g=4.0, 50 steps` (cosine ≥ `_FLUX2_COSINE_GATE`, pixel mismatch ≤ 0.15); (2) **diagnostic** gated path vs `_vanilla_flux2_cfg_predict` (cosine ≥ 0.99). |
| `tests/test_image_quality_flux2.py` | SSIM PR-gate (parity-marked) | Add CFG SSIM test for base-4b at default threshold, g=4.0, 50 steps. SSIM ≥ 0.85, skip count ≥ 1. |
| `tests/test_lifecycle.py` or `tests/test_api.py` | Lifecycle warning | Add `test_no_benefit_warning_fires_under_cfg_when_window_invalid` — base-4b at g=4.0 with a misconfigured skip window now raises `InvalidStepWindowError` (was silent in v0.4.0). |
| `README.md` | User-facing docs | Drop the CFG-fallback footnote on base-4b row. Update Benchmarks table with new g=4.0/50-step row (B/C ratio as headline). Update Limitations / Quick-start to use the canonical upstream recipe. Bump install pin to 0.4.1. |
| `CHANGELOG.md` | Release notes | Add `## [0.4.1]` entry — CFG gate engagement, three-way bench protocol, `cached_residual_neg`, `cfg_was_active` field, `cfg_fallback_steps` deprecation, skip-window behavior change. |
| `docs/calibration.md` | Calibration procedure | Update base-4b row: note polynomial calibrated at g=1.0 is shipped under CFG by default; CFG-aware calibration available via `--guidance 4.0 --fit-branch-policy worst` if needed. |
| `ROADMAP.md` | Roadmap | Move v0.4.1 from Active to Released; promote v0.5.0 (`flux2-klein-base-9b`) to top of Active. |

---

## Preconditions (author machine)

1. **Clean main branch.** v0.4.0 is shipped (tag `v0.4.0` from merge commit `a58859a`). Spec + audit notes are untracked at `docs/superpowers/specs/2026-05-17-flux2-cfg-per-branch-caching-design.md` and `docs/superpowers/notes/2026-05-17-flux2-cfg-per-branch-caching-spec-audit.md`. These will be committed as Task 1.

   ```bash
   git status --short
   # Expect: only the spec + audit notes as ?? entries (plus older untracked review notes from prior PRs).
   git checkout -b feature/v0.4.1-cfg-per-branch
   ```

2. **base-4b weights already downloaded** (v0.4.0 release-gate work). Sanity check:

   ```bash
   hf download black-forest-labs/FLUX.2-klein-base-4B --dry-run | head -3
   ```

3. **Local CI gate that must stay green between commits:**

   ```bash
   uv run ruff check . && uv run ruff format --check . && uv run pytest tests/ -m "not parity and not slow and not benchmark and not network"
   ```

---

## Task 1: Commit spec + audit notes

Land the design + audit response in git history before any code changes, so reviewers see the rationale upfront.

**Files:**
- Create (already on disk, untracked): `docs/superpowers/specs/2026-05-17-flux2-cfg-per-branch-caching-design.md`
- Create (already on disk, untracked): `docs/superpowers/notes/2026-05-17-flux2-cfg-per-branch-caching-spec-audit.md`

- [ ] **Step 1: Verify working-tree state**

  ```bash
  ls docs/superpowers/specs/2026-05-17-flux2-cfg-per-branch-caching-design.md
  ls docs/superpowers/notes/2026-05-17-flux2-cfg-per-branch-caching-spec-audit.md
  ```

- [ ] **Step 2: Stage + commit the two doc files only**

  ```bash
  git add docs/superpowers/specs/2026-05-17-flux2-cfg-per-branch-caching-design.md \
          docs/superpowers/notes/2026-05-17-flux2-cfg-per-branch-caching-spec-audit.md
  git commit -m "$(cat <<'EOF'
  docs(superpowers): v0.4.1 CFG per-branch caching spec + audit

  Spec: shared gate decision + per-branch cached residual under CFG for
  FLUX.2. Lights up the canonical base-4b recipe (guidance=4.0,
  num_inference_steps=50). Three-way bench protocol separates the v0.4
  compile-avoidance win from the new gating win. Polynomial transfer
  g=1.0 → g=4.0 treated as empirical hypothesis; CFG-aware recalibration
  available as a release-blocking contingency.

  Audit findings F1 (framing/three-way bench), F2 (CFG-aware capture),
  F3 (transfer-as-hypothesis), F4 (paired vanilla-vs-wrapper parity) all
  resolved in the spec before plan handoff.
  EOF
  )"
  ```

---

## Task 2: Add `cached_residual_neg` to `TeaCacheState` (test-first)

**Files:**
- Modify: `src/mlx_teacache/cache.py`
- Test: `tests/test_cache.py` (add to existing file)

- [ ] **Step 1: Locate / inspect existing cache tests**

  ```bash
  ls tests/test_cache.py 2>/dev/null && head -30 tests/test_cache.py || echo "no test_cache.py — will create"
  ```

  If `tests/test_cache.py` does not exist, create it (a minimal file with one fixture and the new test). If it exists, append.

- [ ] **Step 2: Write the failing test**

  Add to `tests/test_cache.py`:

  ```python
  def test_reset_for_new_generation_clears_cached_residual_neg():
      """cached_residual_neg must be cleared alongside cached_residual when a
      generation starts. Prevents cross-generation pollution under CFG."""
      import mlx.core as mx

      from mlx_teacache.cache import TeaCacheState

      state = TeaCacheState()
      state.cached_residual = mx.zeros((1, 4))
      state.cached_residual_neg = mx.zeros((1, 4))
      state.reset_for_new_generation(num_steps=10)
      assert state.cached_residual is None
      assert state.cached_residual_neg is None
  ```

- [ ] **Step 3: Run the test (expect AttributeError on `cached_residual_neg`)**

  ```bash
  uv run pytest tests/test_cache.py::test_reset_for_new_generation_clears_cached_residual_neg -v
  ```

  Expected: FAIL with `AttributeError: 'TeaCacheState' object has no attribute 'cached_residual_neg'`.

- [ ] **Step 4: Implement the field**

  Edit `src/mlx_teacache/cache.py`. After the existing `cached_residual: mx.array | None = None` line, add:

  ```python
      cached_residual_neg: mx.array | None = None
  ```

  In `reset_for_new_generation`, after `self.cached_residual = None`, add:

  ```python
          self.cached_residual_neg = None
  ```

  Update the docstring of `reset_for_new_generation` to mention both fields. Update the file-top docstring if it enumerates fields.

- [ ] **Step 5: Run the test (expect PASS)**

  ```bash
  uv run pytest tests/test_cache.py -v
  ```

- [ ] **Step 6: Commit**

  ```bash
  uv run ruff check . && uv run ruff format --check . && uv run pytest tests/ -m "not parity and not slow and not benchmark and not network"
  git add src/mlx_teacache/cache.py tests/test_cache.py
  git commit -m "feat(cache): add cached_residual_neg for v0.4.1 CFG per-branch caching"
  ```

---

## Task 3: Add `cfg_was_active` flag to `_Staging` (test-first)

`_staging.cfg_was_active` is the new internal signal that drives `GenerationStats.cfg_was_active` at finalize time. `cfg_fallback_steps` stays as a public field but is always 0 in v0.4.1+ (the predict closure no longer records `cfg-fallback`).

**Files:**
- Modify: `src/mlx_teacache/stats.py`
- Test: `tests/test_stats.py` (existing or NEW)

- [ ] **Step 1: Locate existing stats tests**

  ```bash
  ls tests/test_stats.py 2>/dev/null && wc -l tests/test_stats.py || echo "no test_stats.py — will create"
  ```

- [ ] **Step 2: Write the failing tests**

  Append to `tests/test_stats.py` (create the file if it doesn't exist; minimal imports):

  ```python
  from mlx_teacache.stats import StepDecision, TeaCacheStats, _Staging


  def test_staging_cfg_was_active_defaults_false():
      st = _Staging()
      assert st.cfg_was_active is False


  def test_staging_cfg_was_active_clears_on_clear():
      st = _Staging()
      st.cfg_was_active = True
      st.clear()
      assert st.cfg_was_active is False


  def test_finalize_records_cfg_was_active_from_staging():
      """finalize_last_generation must propagate _staging.cfg_was_active to
      GenerationStats.cfg_was_active. Replaces the v0.4.0 derivation from
      cfg_fallback_steps > 0 which is no longer correct in v0.4.1+."""
      stats = TeaCacheStats()
      stats._staging.cfg_was_active = True
      # Record one synthetic computed step so the length-invariant passes.
      stats.record(
          StepDecision(
              step_idx=0, timestep=1.0, rel_l1=None, accumulated_distance=0.0, decision="computed"
          )
      )
      stats.finalize_last_generation(num_inference_steps=1, cfg_was_active=True)
      assert stats.last_generation is not None
      assert stats.last_generation.cfg_was_active is True
      # cfg_fallback_steps stays at 0 in v0.4.1+ — feature is gone.
      assert stats.cfg_fallback_steps == 0
  ```

  Note: the existing `finalize_last_generation` signature takes `cfg_was_active` as an explicit kwarg — this is fine; lifecycle will read `_staging.cfg_was_active` and pass it through. The test above verifies the kwarg propagation.

- [ ] **Step 3: Run the tests (expect first two to FAIL with AttributeError)**

  ```bash
  uv run pytest tests/test_stats.py -v
  ```

- [ ] **Step 4: Implement the field**

  Edit `src/mlx_teacache/stats.py`. In the `_Staging` dataclass, after `cfg_fallback: int = 0`, add:

  ```python
      cfg_was_active: bool = False
  ```

  In `_Staging.clear()`, after `self.cfg_fallback = 0`, add:

  ```python
          self.cfg_was_active = False
  ```

  Update the module docstring to mention the new field and to note that `cfg_fallback_steps` is deprecated (always 0 in v0.4.1+).

  Update the `TeaCacheStats.cfg_fallback_steps` public-counter docstring with: `"Deprecated since v0.4.1; always 0. Use GenerationStats.cfg_was_active. Slated for removal in v1.0."`

- [ ] **Step 5: Run the tests (expect PASS)**

  ```bash
  uv run pytest tests/test_stats.py -v
  ```

- [ ] **Step 6: Commit**

  ```bash
  uv run ruff check . && uv run ruff format --check . && uv run pytest tests/ -m "not parity and not slow and not benchmark and not network"
  git add src/mlx_teacache/stats.py tests/test_stats.py
  git commit -m "feat(stats): add _Staging.cfg_was_active; deprecate cfg_fallback_steps"
  ```

---

## Task 4: Implement `flux2_cfg_forward_with_gate` (test-first)

Add the gated CFG forward function in `forward.py`. This is the core change.

**Files:**
- Modify: `src/mlx_teacache/integrations/mflux/forward.py`
- Test: `tests/test_forward_unit.py` (new or extension of an existing unit-test file)

The function mirrors `flux2_forward_with_gate` structurally but runs **both** transformer bodies and combines via CFG math. Reuses `_flux2_run_body`, `_flux2_extract_mod_input`, and `gate_step` unchanged.

- [ ] **Step 1: Write the failing test (shape + invariant only — no real model)**

  Append to `tests/test_forward_unit.py` (or create it):

  ```python
  """Pure-shape unit tests for the new CFG forward. Real-model parity is in
  tests/test_parity_flux2.py."""

  import pytest


  def test_flux2_cfg_forward_with_gate_is_importable():
      """v0.4.1 contract: forward.py exposes flux2_cfg_forward_with_gate."""
      from mlx_teacache.integrations.mflux.forward import flux2_cfg_forward_with_gate

      assert callable(flux2_cfg_forward_with_gate)
  ```

  Keep this test light — the function takes a real mflux transformer; full behavior is exercised by the parity test in Task 10. This unit test just locks in the public symbol.

- [ ] **Step 2: Run the test (expect ImportError)**

  ```bash
  uv run pytest tests/test_forward_unit.py::test_flux2_cfg_forward_with_gate_is_importable -v
  ```

- [ ] **Step 3: Implement `flux2_cfg_forward_with_gate`**

  Edit `src/mlx_teacache/integrations/mflux/forward.py`. After the existing `flux2_forward_with_gate` function, add:

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
  ) -> Any:
      """v0.4.1: gated CFG forward for FLUX.2.

      One shared polynomial-gate decision per step (mod_in is encoder-
      independent — see forward.py:258-304). Two cached residuals
      (positive + negative). CFG combination math runs after the tail on
      both branches.

      Replaces _vanilla_flux2_cfg_predict in production. The vanilla helper
      stays in flux2.py as a test-only diagnostic reference."""
      from mflux.models.common.config.model_config import ModelConfig

      state = handle._state.cache
      stats = handle._state.stats

      # 1. Shared prelude (mirrors flux2_forward_with_gate lines 367-394
      #    minus encoder_hidden_states handling, which is per-branch).
      if not isinstance(timestep, mx.array):
          timestep = mx.array(timestep, dtype=hidden_states.dtype)
      if timestep.ndim == 0:
          timestep = mx.full((hidden_states.shape[0],), timestep, dtype=hidden_states.dtype)
      timestep = timestep.astype(hidden_states.dtype)
      timestep_scale = mx.where(mx.max(timestep) <= 1.0, 1000.0, 1.0).astype(hidden_states.dtype)
      timestep = timestep * timestep_scale
      temb = inner.time_guidance_embed(timestep, None)
      temb = temb.astype(ModelConfig.precision)

      body_in = inner.x_embedder(hidden_states)
      if img_ids.ndim == 3:
          img_ids = img_ids[0]
      image_rotary_emb = inner.pos_embed(img_ids)
      temb_mod_params_img = inner.double_stream_modulation_img(temb)
      temb_mod_params_txt = inner.double_stream_modulation_txt(temb)

      # 2. Per-branch encoder + text-rotary prep. These differ per branch.
      enc_pos = inner.context_embedder(prompt_embeds)
      txt_ids_pos = text_ids[0] if text_ids.ndim == 3 else text_ids
      txt_rot_pos = inner.pos_embed(txt_ids_pos)
      concat_rot_pos = (
          mx.concatenate([txt_rot_pos[0], image_rotary_emb[0]], axis=0),
          mx.concatenate([txt_rot_pos[1], image_rotary_emb[1]], axis=0),
      )

      enc_neg = inner.context_embedder(negative_prompt_embeds)
      txt_ids_neg = negative_text_ids[0] if negative_text_ids.ndim == 3 else negative_text_ids
      txt_rot_neg = inner.pos_embed(txt_ids_neg)
      concat_rot_neg = (
          mx.concatenate([txt_rot_neg[0], image_rotary_emb[0]], axis=0),
          mx.concatenate([txt_rot_neg[1], image_rotary_emb[1]], axis=0),
      )

      timestep_val = float(timestep.flatten()[0])

      # 3. Fast path (threshold <= 0): run both bodies, no caching.
      if handle.rel_l1_thresh <= 0.0:
          body_out_pos = _flux2_run_body(
              inner, body_in, enc_pos, temb, temb_mod_params_img, temb_mod_params_txt, concat_rot_pos
          )
          body_out_neg = _flux2_run_body(
              inner, body_in, enc_neg, temb, temb_mod_params_img, temb_mod_params_txt, concat_rot_neg
          )
          stats.record(
              StepDecision(
                  step_idx=state.step_counter,
                  timestep=timestep_val,
                  rel_l1=None,
                  accumulated_distance=state.accumulated_distance,
                  decision="computed",
              )
          )
          state.last_timestep = timestep_val
          state.step_counter += 1
          return _flux2_apply_tail_and_combine(
              inner, body_out_pos, body_out_neg, enc_pos, enc_neg, temb, guidance
          )

      # 4. Slow path: build mod_in, run gate ONCE on shared signal.
      mod_in = _flux2_extract_mod_input(inner, body_in, temb_mod_params_img)
      if state.previous_mod_input is not None and mod_in.shape != state.previous_mod_input.shape:
          from mlx_teacache.errors import TransformerShapeError

          raise TransformerShapeError(
              step_idx=state.step_counter,
              expected=state.previous_mod_input.shape,
              actual=mod_in.shape,
          )

      from mlx_teacache.gate import gate_step

      decision = gate_step(
          state,
          rel_l1_thresh=handle.rel_l1_thresh,
          coefficients=handle.coefficients,
          skip_first=handle.skip_first_n_steps,
          skip_last=handle.skip_last_n_steps,
          num_steps=handle._gen_ctx.active_num_steps,
          step_idx=state.step_counter,
          mod_in=mod_in,
      )
      stats.record(
          _step_decision_from_gate(decision, step_idx=state.step_counter, timestep=timestep_val)
      )
      state.last_timestep = timestep_val

      # 5. Compute / skip — applied uniformly across both branches.
      body_in_concat_pos = mx.concatenate([enc_pos, body_in], axis=1)
      body_in_concat_neg = mx.concatenate([enc_neg, body_in], axis=1)
      if decision.should_compute:
          body_out_pos = _flux2_run_body(
              inner, body_in, enc_pos, temb, temb_mod_params_img, temb_mod_params_txt, concat_rot_pos
          )
          body_out_neg = _flux2_run_body(
              inner, body_in, enc_neg, temb, temb_mod_params_img, temb_mod_params_txt, concat_rot_neg
          )
          if decision.should_update_cache:
              state.cached_residual = body_out_pos - body_in_concat_pos
              state.cached_residual_neg = body_out_neg - body_in_concat_neg
              state.previous_mod_input = mod_in
      else:
          from mlx_teacache.errors import InternalStateError

          if state.cached_residual is None or state.cached_residual_neg is None:
              raise InternalStateError(
                  "cached_residual or cached_residual_neg is None on a skipped CFG step; "
                  "gate logic should guarantee seed-step caching before any skip."
              )
          body_out_pos = body_in_concat_pos + state.cached_residual
          body_out_neg = body_in_concat_neg + state.cached_residual_neg

      state.step_counter += 1
      return _flux2_apply_tail_and_combine(
          inner, body_out_pos, body_out_neg, enc_pos, enc_neg, temb, guidance
      )


  def _flux2_apply_tail_and_combine(
      inner: Any,
      body_out_pos: mx.array,
      body_out_neg: mx.array,
      enc_pos: mx.array,
      enc_neg: mx.array,
      temb: mx.array,
      guidance: float,
  ) -> mx.array:
      """Apply Flux2 norm_out + proj_out tail to each branch independently,
      then combine via CFG math: negative + guidance * (positive - negative).

      norm_out + proj_out are branch-independent ops parameterized by temb;
      we apply them once per branch because the body_out per branch differs.
      The CFG math is the same triplet mflux uses (flux2_klein.py:267-276)."""
      noise_pos = body_out_pos[:, enc_pos.shape[1] :, ...]
      noise_pos = inner.norm_out(noise_pos, temb)
      noise_pos = inner.proj_out(noise_pos)

      noise_neg = body_out_neg[:, enc_neg.shape[1] :, ...]
      noise_neg = inner.norm_out(noise_neg, temb)
      noise_neg = inner.proj_out(noise_neg)

      return noise_neg + guidance * (noise_pos - noise_neg)
  ```

  Note on graph topology: per the existing `_flux2_run_body` docstring, computing the single-stream modulation inline (inside `_flux2_run_body`) is required to keep MLX graph topology byte-identical with vanilla. We reuse `_flux2_run_body` exactly, so each branch keeps the correct topology.

- [ ] **Step 4: Run the import test (expect PASS)**

  ```bash
  uv run pytest tests/test_forward_unit.py -v
  ```

- [ ] **Step 5: Run lint + typecheck**

  ```bash
  uv run ruff check src/mlx_teacache/integrations/mflux/forward.py
  uv run ruff format --check src/mlx_teacache/integrations/mflux/forward.py
  ```

- [ ] **Step 6: Commit**

  ```bash
  uv run ruff check . && uv run ruff format --check . && uv run pytest tests/ -m "not parity and not slow and not benchmark and not network"
  git add src/mlx_teacache/integrations/mflux/forward.py tests/test_forward_unit.py
  git commit -m "feat(forward): add flux2_cfg_forward_with_gate (shared decision, per-branch residual)"
  ```

---

## Task 5: Rewrite predict closure CFG branch + lift skip-window validation

Switch the production CFG path from `_vanilla_flux2_cfg_predict` to the new gated function. Move the lazy skip-window validation up so it runs on the **first gated step regardless of CFG**, and set `_staging.cfg_was_active = True` on first CFG branch entry. Keep `_vanilla_flux2_cfg_predict` in the file as a test-only diagnostic.

**Files:**
- Modify: `src/mlx_teacache/integrations/mflux/flux2.py`
- Test: `tests/test_api.py` — add a CFG smoke test

- [ ] **Step 1: Write the failing test**

  Append to `tests/test_api.py`:

  ```python
  @pytest.mark.parity
  def test_apply_teacache_cfg_records_cfg_was_active_klein_base_4b():
      """v0.4.1: at guidance > 1.0, the gated CFG forward fires; the staging
      buffer's cfg_was_active flag flips True. GenerationStats.cfg_was_active
      then propagates from staging at finalize time."""
      from mflux.models.common.config.model_config import ModelConfig
      from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

      from mlx_teacache import apply_teacache

      flux = Flux2Klein(quantize=4, model_config=ModelConfig.flux2_klein_base_4b())
      flux.freeze()
      handle = apply_teacache(flux)
      try:
          flux.generate_image(
              prompt="a red apple",
              seed=42,
              num_inference_steps=4,  # small for unit speed; full bench is 50
              height=512,
              width=512,
              guidance=4.0,
          )
          assert handle.stats.last_generation is not None
          assert handle.stats.last_generation.cfg_was_active is True
          # No cfg-fallback decisions — production path no longer records them.
          kinds = {d.decision for d in handle.stats.last_generation.decisions}
          assert "cfg-fallback" not in kinds
      finally:
          handle.restore()
  ```

- [ ] **Step 2: Run the test (expect FAIL — current closure still records cfg-fallback)**

  ```bash
  uv run pytest tests/test_api.py::test_apply_teacache_cfg_records_cfg_was_active_klein_base_4b -v
  ```

  Expected: FAIL because v0.4.0 closure routes `guidance > 1.0` to `_vanilla_flux2_cfg_predict` and records `"cfg-fallback"`. (This test requires a real model; mark with `@pytest.mark.parity` so it stays in the parity-only test bucket.)

- [ ] **Step 3: Rewrite the closure**

  Edit `src/mlx_teacache/integrations/mflux/flux2.py`. Update the imports and rewrite the predict closure. The new structure:

  ```python
  from mlx_teacache.errors import InvalidStepWindowError, MissingGenerationContextError
  from mlx_teacache.integrations.mflux.forward import (
      flux2_cfg_forward_with_gate,
      flux2_forward_with_gate,
  )
  from mlx_teacache.stats import StepDecision  # still used by the fast-path stat record
  ```

  Then the predict closure body becomes (replacing lines 75-150):

  ```python
          def predict(
              latents: mx.array,
              latent_ids: mx.array,
              prompt_embeds: mx.array,
              text_ids: mx.array,
              negative_prompt_embeds: mx.array | None,
              negative_text_ids: mx.array | None,
              guidance: float,
              timestep: mx.array,
          ) -> Any:
              nonlocal context_consumed
              ctx = handle._gen_ctx

              if not context_consumed:
                  if ctx.active_num_steps is None or ctx.consumed_at_token == ctx.token:
                      raise MissingGenerationContextError()
                  ctx.consumed_at_token = ctx.token
                  context_consumed = True

              # Lazy skip-window validation: now runs on the first gated call
              # regardless of CFG. In v0.4.0 this was gated behind the non-CFG
              # branch only; in v0.4.1 CFG is also gated, so all-CFG runs that
              # had a misconfigured skip window now raise rather than silently
              # running vanilla. Documented in CHANGELOG as a behavior change.
              if not handle._state.cache.skip_window_validated:
                  if handle.skip_first_n_steps + handle.skip_last_n_steps >= ctx.active_num_steps:
                      raise InvalidStepWindowError(
                          skip_first=handle.skip_first_n_steps,
                          skip_last=handle.skip_last_n_steps,
                          num_steps=ctx.active_num_steps,
                      )
                  handle._state.cache.skip_window_validated = True

              cfg_active = negative_prompt_embeds is not None and negative_text_ids is not None
              if cfg_active:
                  assert negative_prompt_embeds is not None
                  assert negative_text_ids is not None
                  handle._state.stats._staging.cfg_was_active = True
                  return flux2_cfg_forward_with_gate(
                      transformer,
                      handle,
                      hidden_states=latents,
                      prompt_embeds=prompt_embeds,
                      text_ids=text_ids,
                      negative_prompt_embeds=negative_prompt_embeds,
                      negative_text_ids=negative_text_ids,
                      guidance=guidance,
                      timestep=timestep,
                      img_ids=latent_ids,
                  )

              return flux2_forward_with_gate(
                  transformer,
                  handle,
                  hidden_states=latents,
                  encoder_hidden_states=prompt_embeds,
                  timestep=timestep,
                  img_ids=latent_ids,
                  txt_ids=text_ids,
              )
  ```

  Keep `_vanilla_flux2_cfg_predict` as-is in the file with an updated docstring stating it's a test-only diagnostic reference for parity tests (no longer called from production).

- [ ] **Step 4: Run the CFG smoke test (expect PASS — parity-marked)**

  ```bash
  uv run pytest tests/test_api.py::test_apply_teacache_cfg_records_cfg_was_active_klein_base_4b -v -m parity
  ```

- [ ] **Step 5: Run the broader non-parity tests to confirm no regression**

  ```bash
  uv run pytest tests/ -m "not parity and not slow and not benchmark and not network" -v
  ```

  Expected: all green. The non-CFG path is untouched; pure-core tests should not have been affected by closure rewrites.

- [ ] **Step 6: Commit**

  ```bash
  uv run ruff check . && uv run ruff format --check .
  git add src/mlx_teacache/integrations/mflux/flux2.py tests/test_api.py
  git commit -m "feat(flux2): route CFG through gated path; lift skip-window validation"
  ```

---

## Task 6: Read `cfg_was_active` from staging in lifecycle; remove suppression

Lifecycle reads `_staging.cfg_was_active` (set by the predict closure in Task 5) when building `PendingFinalize`. The old `flux2_cfg_fallback` warning suppression goes away.

**Files:**
- Modify: `src/mlx_teacache/integrations/mflux/lifecycle.py`
- Test: `tests/test_api.py` or `tests/test_lifecycle.py`

- [ ] **Step 1: Write the failing test (warning fires under bad skip window at CFG)**

  Append to `tests/test_api.py`:

  ```python
  @pytest.mark.parity
  def test_invalid_skip_window_raises_under_cfg_klein_base_4b():
      """v0.4.1 behavior change: an all-CFG generation with skip_first + skip_last
      >= num_inference_steps used to silently run vanilla in v0.4.0. In v0.4.1 the
      CFG path is gated, so the lazy skip-window validation fires and raises."""
      from mflux.models.common.config.model_config import ModelConfig
      from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

      from mlx_teacache import InvalidStepWindowError, apply_teacache

      flux = Flux2Klein(quantize=4, model_config=ModelConfig.flux2_klein_base_4b())
      flux.freeze()
      # 4 steps with skip_first=2 + skip_last=2 means 4 >= 4 — invalid window.
      handle = apply_teacache(flux, skip_first_n_steps=2, skip_last_n_steps=2)
      try:
          with pytest.raises(InvalidStepWindowError):
              flux.generate_image(
                  prompt="a red apple",
                  seed=42,
                  num_inference_steps=4,
                  height=512,
                  width=512,
                  guidance=4.0,
              )
      finally:
          handle.restore()
  ```

- [ ] **Step 2: Run the test (expect FAIL — currently does NOT raise on all-CFG)**

  ```bash
  uv run pytest tests/test_api.py::test_invalid_skip_window_raises_under_cfg_klein_base_4b -v -m parity
  ```

  Note: this test will only pass after Task 5's predict-closure rewrite. If Task 5 already shipped, this should now PASS without changing lifecycle. But the lifecycle change in this task (Step 4) is independent — `cfg_was_active` derivation moves from `cfg_fallback > 0` to `_staging.cfg_was_active`.

- [ ] **Step 3: Update lifecycle to use the new derivation**

  Edit `src/mlx_teacache/integrations/mflux/lifecycle.py`:

  - In `call_before_loop`, **remove** the `flux2_cfg_fallback` suppression block (the `(self._handle.variant_id.startswith("flux2-") and float(getattr(config, "guidance", 1.0) or 1.0) > 1.0)` block and the corresponding early `return` in the `if flux2_cfg_fallback or window_invalid: return` line — keep only `window_invalid` suppression).
  - In `call_after_loop`, replace `cfg_was_active=self._handle._state.stats._staging.cfg_fallback > 0` with `cfg_was_active=self._handle._state.stats._staging.cfg_was_active`.
  - Add a docstring note: "cfg_was_active comes from staging (set by the predict closure on first CFG branch entry). The old `cfg_fallback > 0` derivation is obsolete in v0.4.1+ (no production code records `cfg-fallback` decisions)."

- [ ] **Step 4: Run all non-parity tests**

  ```bash
  uv run pytest tests/ -m "not parity and not slow and not benchmark and not network" -v
  ```

- [ ] **Step 5: Run the parity tests added in Tasks 5 + 6**

  ```bash
  uv run pytest tests/test_api.py -v -m parity -k "cfg or skip_window"
  ```

- [ ] **Step 6: Commit**

  ```bash
  uv run ruff check . && uv run ruff format --check .
  git add src/mlx_teacache/integrations/mflux/lifecycle.py tests/test_api.py
  git commit -m "refactor(lifecycle): drop CFG-fallback warning suppression; read cfg_was_active from staging"
  ```

---

## Task 7: Bench three-way protocol

Extend `scripts/bench_speedup.py` so the `klein-base-4b` row separates compile-avoidance (v0.4) from gating (v0.4.1).

**Files:**
- Modify: `scripts/bench_speedup.py`

- [ ] **Step 1: Update the `klein-base-4b` `_variant_config`**

  Change the existing `klein-base-4b` branch to default to upstream-recommended recipe:

  ```python
      if variant == "klein-base-4b":
          # Canonical upstream recipe (v0.4.1+). Override with
          # --guidance 1.0 --num-inference-steps 25 to reproduce the v0.4.0 row.
          return {
              "loader": _load_flux2_klein,
              "num_inference_steps": 50,
              "guidance": 4.0,
          }
  ```

- [ ] **Step 2: Add `--guidance` and `--num-inference-steps` CLI overrides**

  In `main()`'s argparse setup (just after `--reps`):

  ```python
      parser.add_argument(
          "--guidance",
          type=float,
          default=None,
          help="Override the variant's default guidance value.",
      )
      parser.add_argument(
          "--num-inference-steps",
          type=int,
          default=None,
          help="Override the variant's default step count.",
      )
      parser.add_argument(
          "--three-way",
          action="store_true",
          default=None,
          help="Run vanilla + wrapped-no-gate + wrapped-gated conditions. Default True on klein-base-4b.",
      )
  ```

  Resolve the overrides after `cfg = _variant_config(args.variant)`:

  ```python
      if args.guidance is not None:
          guidance = args.guidance
      else:
          guidance = cfg["guidance"]
      if args.num_inference_steps is not None:
          num_inference_steps = args.num_inference_steps
      else:
          num_inference_steps = cfg["num_inference_steps"]
      three_way = args.three_way if args.three_way is not None else (args.variant == "klein-base-4b")
  ```

- [ ] **Step 3: Add the wrapped-no-gate condition**

  After the existing "Vanilla x{args.reps}" loop and before the existing "TeaCache wrapper x{args.reps}" loop, add (only when `three_way` is True):

  ```python
      nogate_times: list[float] = []
      if three_way:
          print(f"\n== Wrapped (no gate, rel_l1_thresh=0) x{args.reps} ==")
          for i in range(args.reps):
              save = bench_dir / "wrapper_nogate.png" if i == 0 else None
              with apply_teacache(flux, rel_l1_thresh=0.0) as h:
                  t, _ = _generate(
                      flux,
                      num_inference_steps=num_inference_steps,
                      guidance=guidance,
                      save_path=save,
                  )
                  nogate_times.append(t)
              suffix = f"  (saved {save.name})" if save else ""
              print(f"  rep {i + 1}: {t:.2f}s  (rel_l1_thresh=0, no skipping){suffix}")
  ```

  Also rename `wrapper.png` → `wrapper_gated.png` (only the saved-image filename, only when `three_way` is True; preserve `wrapper.png` for the non-three-way path).

- [ ] **Step 4: Extend the Summary to report all three medians and both ratios**

  After computing `vanilla_med` and `wrapper_med`, when `three_way`:

  ```python
      if three_way:
          nogate_med = statistics.median(nogate_times)
          compile_avoidance_ratio = vanilla_med / nogate_med   # v0.4 effect
          gating_ratio = nogate_med / wrapper_med              # v0.4.1 effect
          combined_ratio = vanilla_med / wrapper_med
          print(f"  three-way medians: vanilla {vanilla_med:.2f}s | no-gate {nogate_med:.2f}s | gated {wrapper_med:.2f}s")
          print(f"  compile-avoidance (vanilla / no-gate): {compile_avoidance_ratio:.2f}x  [v0.4 effect]")
          print(f"  gating          (no-gate / gated):    {gating_ratio:.2f}x  [v0.4.1 effect]")
          print(f"  combined        (vanilla / gated):     {combined_ratio:.2f}x")
  ```

  Add the corresponding fields to the optional `--report` JSON when `three_way`.

- [ ] **Step 5: Lint + smoke-test the script (no model load — just argparse)**

  ```bash
  uv run python scripts/bench_speedup.py --help | head -30
  ```

  Confirm `--guidance`, `--num-inference-steps`, and `--three-way` are documented.

- [ ] **Step 6: Commit**

  ```bash
  uv run ruff check scripts/bench_speedup.py && uv run ruff format --check scripts/bench_speedup.py
  git add scripts/bench_speedup.py
  git commit -m "feat(bench): three-way protocol on klein-base-4b (vanilla / no-gate / gated)"
  ```

---

## Task 8: Add CFG-aware capture to `scripts/calibrate_flux2.py` (precautionary)

Ship the CFG-aware capture and `--guidance` / `--fit-branch-policy` flags now even if the bench shows the existing g=1.0 polynomial transfers, so the recalibration contingency is one command away. Per audit Finding 2, this is **not** a flag-only addition — it's a new capturing closure that runs both branches and returns CFG-combined noise.

**Files:**
- Modify: `scripts/calibrate_flux2.py`

- [ ] **Step 1: Add the CFG-aware capturing closure**

  Add (do NOT delete the existing non-CFG closure):

  ```python
  def _make_cfg_capturing_closure(
      inner: Any, captures: list[dict[str, Any]], ModelConfig: Any
  ) -> Any:
      """CFG-aware capture (v0.4.1). Runs BOTH branches per step, returns
      CFG-combined noise to the scheduler so the next latent follows the
      real g>1 trajectory, captures the shared mod_in plus per-branch
      body_out_concat."""

      def predict(
          latents: mx.array,
          latent_ids: mx.array,
          prompt_embeds: mx.array,
          text_ids: mx.array,
          negative_prompt_embeds: mx.array | None,
          negative_text_ids: mx.array | None,
          guidance: float,
          timestep: mx.array,
      ) -> mx.array:
          assert negative_prompt_embeds is not None, "CFG capture requires negative embeds"
          assert negative_text_ids is not None

          ts = timestep
          if not isinstance(ts, mx.array):
              ts = mx.array(ts, dtype=latents.dtype)
          if ts.ndim == 0:
              ts = mx.full((latents.shape[0],), ts, dtype=latents.dtype)
          ts = ts.astype(latents.dtype)
          ts_scale = mx.where(mx.max(ts) <= 1.0, 1000.0, 1.0).astype(latents.dtype)
          ts = ts * ts_scale
          temb = inner.time_guidance_embed(ts, None)
          temb = temb.astype(ModelConfig.precision)

          body_in = inner.x_embedder(latents)
          img_ids = latent_ids[0] if latent_ids.ndim == 3 else latent_ids
          image_rotary_emb = inner.pos_embed(img_ids)
          temb_mod_params_img = inner.double_stream_modulation_img(temb)
          temb_mod_params_txt = inner.double_stream_modulation_txt(temb)

          # Shared gate signal.
          mod_in = _flux2_extract_mod_input(inner, body_in, temb_mod_params_img)

          # Positive branch.
          enc_pos = inner.context_embedder(prompt_embeds)
          txt_ids_pos = text_ids[0] if text_ids.ndim == 3 else text_ids
          txt_rot_pos = inner.pos_embed(txt_ids_pos)
          concat_rot_pos = (
              mx.concatenate([txt_rot_pos[0], image_rotary_emb[0]], axis=0),
              mx.concatenate([txt_rot_pos[1], image_rotary_emb[1]], axis=0),
          )
          body_out_pos = _flux2_run_body(
              inner, body_in, enc_pos, temb, temb_mod_params_img, temb_mod_params_txt, concat_rot_pos
          )

          # Negative branch.
          enc_neg = inner.context_embedder(negative_prompt_embeds)
          txt_ids_neg = negative_text_ids[0] if negative_text_ids.ndim == 3 else negative_text_ids
          txt_rot_neg = inner.pos_embed(txt_ids_neg)
          concat_rot_neg = (
              mx.concatenate([txt_rot_neg[0], image_rotary_emb[0]], axis=0),
              mx.concatenate([txt_rot_neg[1], image_rotary_emb[1]], axis=0),
          )
          body_out_neg = _flux2_run_body(
              inner, body_in, enc_neg, temb, temb_mod_params_img, temb_mod_params_txt, concat_rot_neg
          )

          mx.eval(mod_in, body_out_pos, body_out_neg)
          captures.append({"mod_in": mod_in, "body_out_pos": body_out_pos, "body_out_neg": body_out_neg})

          # Tail + CFG combine for the scheduler.
          noise_pos = body_out_pos[:, enc_pos.shape[1] :, ...]
          noise_pos = inner.norm_out(noise_pos, temb)
          noise_pos = inner.proj_out(noise_pos)
          noise_neg = body_out_neg[:, enc_neg.shape[1] :, ...]
          noise_neg = inner.norm_out(noise_neg, temb)
          noise_neg = inner.proj_out(noise_neg)
          return noise_neg + guidance * (noise_pos - noise_neg)

      return predict
  ```

  Add a wrapper factory:

  ```python
  def _build_cfg_capturing_predict_factory(captures: list[dict[str, Any]]) -> Any:
      from mflux.models.common.config.model_config import ModelConfig

      def factory(transformer: Any) -> Any:
          return _make_cfg_capturing_closure(transformer, captures, ModelConfig)

      return factory
  ```

- [ ] **Step 2: Add `--guidance`, `--num-inference-steps`, and `--fit-branch-policy` flags + branching**

  Per plan-audit Finding 1: the recalibration must run at the same step schedule as the failing release bench. Adding only `--guidance` would silently calibrate at the variant's hardcoded default (25 steps) while the release gate runs at 50 — the resulting coefficients would not match the failing trajectory.

  In `main()`, after the existing `--fit-mode` arg:

  ```python
      parser.add_argument(
          "--guidance",
          type=float,
          default=1.0,
          help="Guidance value for calibration (1.0 = no CFG / positive only; >1 enables CFG capture path).",
      )
      parser.add_argument(
          "--num-inference-steps",
          type=int,
          default=None,
          help="Override the variant's hardcoded step count. Required when calibrating for a recipe that differs from the variant's default (e.g. base-4b CFG @ 50 steps, not the default 25).",
      )
      parser.add_argument(
          "--fit-branch-policy",
          default="worst",
          choices=["worst", "average", "positive", "negative"],
          help="Under CFG calibration, which per-step y target to fit: worst-branch (default), average, positive only, or negative only.",
      )
  ```

  Resolve the step-count override after looking up the variant config:

  ```python
      cfg = _VARIANTS[args.variant]
      variant_id: str = cfg["variant_id"]
      num_inference_steps: int = args.num_inference_steps if args.num_inference_steps is not None else cfg["num_inference_steps"]
      output_json: str = cfg["output_json"]
      fit_mode: str = args.fit_mode
  ```

  Branch on `args.guidance > 1.0` to swap in the CFG-aware factory:

  ```python
      def _capture_one_prompt(
          flux: Any, prompt: str, *, num_inference_steps: int, guidance: float
      ) -> list[dict[str, Any]]:
          captures: list[dict[str, Any]] = []
          had_instance_attr = "_predict" in vars(flux)
          original = flux._predict if had_instance_attr else None
          if guidance > 1.0:
              flux._predict = _build_cfg_capturing_predict_factory(captures)
          else:
              flux._predict = _build_capturing_predict_factory(captures)
          try:
              flux.generate_image(
                  prompt=prompt,
                  seed=SEED,
                  num_inference_steps=num_inference_steps,
                  height=HEIGHT,
                  width=WIDTH,
                  guidance=guidance,
              )
          finally:
              if had_instance_attr:
                  flux._predict = original
              else:
                  del flux._predict
          return captures
  ```

  Update the main loop's `_capture_one_prompt` call to pass `guidance=args.guidance`.

- [ ] **Step 3: Compute branch-aware y targets**

  In the pair-computation loop, when `args.guidance > 1.0`, use:

  ```python
      for t in range(1, len(capture)):
          x = _rel_l1(capture[t]["mod_in"], capture[t - 1]["mod_in"])
          if args.guidance > 1.0:
              y_pos = _rel_l1(capture[t]["body_out_pos"], capture[t - 1]["body_out_pos"])
              y_neg = _rel_l1(capture[t]["body_out_neg"], capture[t - 1]["body_out_neg"])
              if args.fit_branch_policy == "worst":
                  y = max(y_pos, y_neg)
              elif args.fit_branch_policy == "average":
                  y = 0.5 * (y_pos + y_neg)
              elif args.fit_branch_policy == "positive":
                  y = y_pos
              else:  # negative
                  y = y_neg
              ys_pos.append(y_pos)
              ys_neg.append(y_neg)
          else:
              y = _rel_l1(capture[t]["body_out"], capture[t - 1]["body_out"])
          xs.append(x)
          ys.append(y)
          prompt_pairs.append((x, y))
  ```

  Initialize `ys_pos: list[float] = []` and `ys_neg: list[float] = []` at the top of the main loop. Include them in the JSON report when `args.guidance > 1.0`.

- [ ] **Step 4: Extend the JSON report**

  Add to the `report` dict when `args.guidance > 1.0`:

  ```python
      report["fit_branch_policy"] = args.fit_branch_policy
      report["y_values_pos"] = [float(y) for y in ys_pos]
      report["y_values_neg"] = [float(y) for y in ys_neg]
  ```

  Always include the (existing) `guidance` field (matches `args.guidance` rather than the module-level `GUIDANCE` constant). Also ensure the resolved `num_inference_steps` (post-override) is what's serialized into the report's `num_inference_steps` field — calibration provenance must be honest about the exact schedule the polynomial was fit on.

- [ ] **Step 5: Lint + smoke-test the script's argparse**

  ```bash
  uv run python scripts/calibrate_flux2.py --variant klein-base-4b --help
  ```

  Confirm `--guidance` and `--fit-branch-policy` appear.

- [ ] **Step 6: Commit**

  ```bash
  uv run ruff check scripts/calibrate_flux2.py && uv run ruff format --check scripts/calibrate_flux2.py
  git add scripts/calibrate_flux2.py
  git commit -m "feat(calibrate): CFG-aware capture + per-branch y targets (release-blocker contingency)"
  ```

---

## Task 9: CFG parity tests (release-blocker + diagnostic)

Per audit Finding 4: the release blocker is paired same-process vanilla-vs-wrapper CFG parity. Add both that test and the lighter diagnostic against `_vanilla_flux2_cfg_predict`.

**Files:**
- Modify: `tests/test_parity_flux2.py`

- [ ] **Step 1: Read the existing paired-parity scaffolding**

  ```bash
  head -120 tests/test_parity_flux2.py
  ```

  Identify the existing `test_paired_parity_klein_pr_gate` / `test_paired_parity_at_threshold_zero_klein_pr_gate` patterns. The new CFG tests follow the same structure but with `guidance=4.0, num_inference_steps=50`.

- [ ] **Step 2: Add the release-blocker CFG paired parity test**

  Append:

  ```python
  @pytest.mark.parity
  def test_paired_cfg_parity_at_threshold_zero_klein_base_4b_pr_gate():
      """v0.4.1 release blocker: at rel_l1_thresh=0, the gated CFG path must
      produce the same image (within Metal noise) as real mflux generation
      at guidance=4.0, num_inference_steps=50 on flux2-klein-base-4b. Per
      audit Finding 4, the in-repo _vanilla_flux2_cfg_predict helper is too
      weak as the release oracle because it shares assumptions with the
      gated function; this test uses real mflux."""
      from mflux.models.common.config.model_config import ModelConfig
      from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

      from mlx_teacache import apply_teacache

      flux = Flux2Klein(quantize=4, model_config=ModelConfig.flux2_klein_base_4b())
      flux.freeze()

      # 1. Vanilla mflux (no wrapper).
      vanilla_img = flux.generate_image(
          prompt="a red apple on a wooden table",
          seed=42,
          num_inference_steps=50,
          height=512,
          width=512,
          guidance=4.0,
      )

      # 2. Wrapped at threshold=0 (no skips). Same process, same flux instance.
      handle = apply_teacache(flux, rel_l1_thresh=0.0)
      try:
          wrapped_img = flux.generate_image(
              prompt="a red apple on a wooden table",
              seed=42,
              num_inference_steps=50,
              height=512,
              width=512,
              guidance=4.0,
          )
      finally:
          handle.restore()

      # Compare via the existing helper. _FLUX2_COSINE_GATE = 0.97 currently.
      _assert_paired_cosine_and_mismatch(
          vanilla_img,
          wrapped_img,
          cosine_gate=_FLUX2_COSINE_GATE,
          mismatch_threshold=0.15,
      )
  ```

  If `_assert_paired_cosine_and_mismatch` doesn't exist with that name, mirror whatever helper the existing FLUX.2 paired-parity tests use (read the file before writing; do not invent helper names).

- [ ] **Step 3: Add the diagnostic test against `_vanilla_flux2_cfg_predict`**

  Append:

  ```python
  @pytest.mark.parity
  def test_diagnostic_cfg_gated_matches_in_repo_vanilla_helper():
      """v0.4.1 diagnostic: gated CFG forward at rel_l1_thresh=0 must match
      the in-repo _vanilla_flux2_cfg_predict reference (cosine ≥ 0.99). This
      isolates 'gated function vs our helper' from 'helper vs real mflux'.
      Useful when the release-blocker test fails — tells us whether the bug
      is in the new function or in the helper that approximates mflux."""
      # Tight comparison: both code paths run inside the same eager _predict
      # replacement, share _flux2_run_body, share _flux2_extract_mod_input.
      # Any drift here points at a topology bug in flux2_cfg_forward_with_gate
      # or _flux2_apply_tail_and_combine.
      # Implementation: dispatch one synthetic generation through each path
      # and compare pixel-by-pixel.
      # NOTE: full implementation deferred to the parity-test author who
      # owns the existing scaffolding pattern — the test_id is locked in here
      # so the release gate enumerates it.
      pytest.skip("Diagnostic test scaffolding lands with Task 9 if the release-blocker test fails. Implementation follows the existing patterns in this file.")
  ```

  Rationale: ship the diagnostic test as a placeholder so the release checklist has a row for it; the full implementation is only needed if the release-blocker test fails. If the release-blocker test passes on the first run, this `pytest.skip` stays.

- [ ] **Step 4: Run pure-core tests to confirm no collateral damage**

  ```bash
  uv run pytest tests/ -m "not parity and not slow and not benchmark and not network"
  ```

- [ ] **Step 5: Commit**

  ```bash
  uv run ruff check tests/test_parity_flux2.py && uv run ruff format --check tests/test_parity_flux2.py
  git add tests/test_parity_flux2.py
  git commit -m "test(parity): paired vanilla-vs-wrapper CFG release blocker + diagnostic"
  ```

---

## Task 10: CFG SSIM PR-gate

Per spec §"Quality + skip gates": SSIM ≥ 0.85 at default threshold (0.17), g=4.0, 50 steps; skip count ≥ 1.

**Files:**
- Modify: `tests/test_image_quality_flux2.py`

- [ ] **Step 1: Read the existing `_gen_kwargs_klein` dispatch**

  ```bash
  grep -n "_gen_kwargs_klein\|flux2_klein fixture\|num_inference_steps" tests/test_image_quality_flux2.py | head -30
  ```

  v0.4.0 made this variant-aware (distilled at 8, base-4b at 25). v0.4.1 adds a `cfg` keyword:

- [ ] **Step 2: Extend `_gen_kwargs_klein` to accept a `cfg` kwarg**

  Update the helper signature:

  ```python
  def _gen_kwargs_klein(variant_id: str, *, cfg: bool = False) -> dict[str, Any]:
      """Variant-aware generation kwargs. cfg=True uses the upstream CFG
      recipe (guidance=4.0) on base-4b at the calibrated 50-step schedule.
      Distilled variants and g=1.0 base-4b are unchanged."""
      if variant_id == "flux2-klein-base-4b" and cfg:
          return {"num_inference_steps": 50, "guidance": 4.0}
      if variant_id == "flux2-klein-base-4b":
          return {"num_inference_steps": 25, "guidance": 1.0}
      # distilled Klein 4B / 9B
      return {"num_inference_steps": 8, "guidance": 1.0}
  ```

  All existing callers pass `cfg=False` (the default), so behavior is unchanged.

- [ ] **Step 3: Add the new CFG SSIM PR-gate test**

  ```python
  @pytest.mark.parity
  def test_ssim_pr_gate_cfg_klein_base_4b():
      """v0.4.1 release blocker: at default rel_l1_thresh (0.17), CFG-engaged
      generation on flux2-klein-base-4b at g=4.0/50 steps must produce SSIM
      >= 0.85 vs vanilla AND fire >= 1 skip. Skip-count assertion locks in
      the v0.4.1 engagement claim — without it the test would pass with 0
      skips and the feature would be dormant (v0.3 postmortem lesson)."""
      from mflux.models.common.config.model_config import ModelConfig
      from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

      from mlx_teacache import apply_teacache

      flux = Flux2Klein(quantize=4, model_config=ModelConfig.flux2_klein_base_4b())
      flux.freeze()
      kwargs = _gen_kwargs_klein("flux2-klein-base-4b", cfg=True)
      vanilla_img = flux.generate_image(prompt=_PR_GATE_PROMPT, seed=42, height=512, width=512, **kwargs)
      handle = apply_teacache(flux)  # uses per-variant default rel_l1_thresh=0.17
      try:
          wrapped_img = flux.generate_image(prompt=_PR_GATE_PROMPT, seed=42, height=512, width=512, **kwargs)
          assert handle.stats.skipped_count >= 1, (
              f"Expected ≥1 skip under CFG; got {handle.stats.skipped_count}. "
              f"If this fires reliably, fall into the 0-skip contingency: "
              f"run CFG-aware calibration via scripts/calibrate_flux2.py --guidance 4.0."
          )
          _assert_ssim_at_least(vanilla_img, wrapped_img, threshold=0.85)
      finally:
          handle.restore()
  ```

  (`_PR_GATE_PROMPT` and `_assert_ssim_at_least` are the existing helpers in this file — if the names differ, mirror what's there.)

- [ ] **Step 4: Pure-core sanity**

  ```bash
  uv run pytest tests/ -m "not parity and not slow and not benchmark and not network"
  ```

- [ ] **Step 5: Commit**

  ```bash
  uv run ruff check tests/test_image_quality_flux2.py && uv run ruff format --check tests/test_image_quality_flux2.py
  git add tests/test_image_quality_flux2.py
  git commit -m "test(image-quality): CFG PR-gate at default threshold on klein-base-4b"
  ```

---

## Task 11: Release-gate bench (real weights)

Run the three-way bench on the real model. This is the v0.4.1 headline measurement.

**Files:** none modified by this task; produces `scripts/_bench_report_klein_base_4b_v0_4_1.json` (untracked) plus images under `tests/_artifacts/bench_images/klein-base-4b/`.

- [ ] **Step 1: Run the bench (main thread, run_in_background)**

  ```bash
  uv run python scripts/bench_speedup.py --variant klein-base-4b --report scripts/_bench_report_klein_base_4b_v0_4_1.json 2>&1 | tee /tmp/bench-klein-base-4b-v0.4.1.log
  ```

  Use Bash `run_in_background=true` from the main session, NOT a subagent (per CLAUDE.md "heavy generations" rule). Approximate runtime: 50 steps × 9 generations (1 warmup + 3 vanilla + 3 no-gate + 3 gated) ≈ 30-45 minutes on M1 Max at q4. Monitor via `tail -f /tmp/bench-klein-base-4b-v0.4.1.log`.

- [ ] **Step 2: Inspect the report**

  ```bash
  cat scripts/_bench_report_klein_base_4b_v0_4_1.json | python -m json.tool
  ```

  Required for the v0.4.1 release narrative:
  - `skipped_counts` median ≥ 1 (engagement evidence)
  - `gating` ratio (`nogate_med / wrapper_med`) ≥ 1.2× (v0.4.1 effect)
  - `vanilla_median / wrapper_median` ≥ 1.4× (combined, comparable to v0.4.0's 1.41× at g=1.0)
  - Visual: `tests/_artifacts/bench_images/klein-base-4b/{vanilla,wrapper_nogate,wrapper_gated}.png` show no visible degradation between vanilla and wrapper_gated.

- [ ] **Step 3: Branch point — choose path based on bench**

  - **If skip ≥ 1, SSIM passes (Task 10), gating ratio ≥ 1.2×:** proceed to Task 13 (docs).
  - **If skip is 0 OR SSIM < 0.85:** proceed to Task 12 (contingency calibration).
  - **If skip ≥ 1 but gating ratio < 1.2×:** document the realistic speedup in CHANGELOG/README (skip the 1.2× claim; cite the measured number).

---

## Task 12 (CONDITIONAL): CFG-aware recalibration

Only run if Task 11 step 3 sends us here. Skip if the existing g=1.0 polynomial engages acceptably under CFG.

**Files:**
- Modify: `src/mlx_teacache/coefficients.py` (if the new fit ships)
- Modify: `scripts/_calibration_flux2_klein_base_4b.json` (overwritten with CFG fit)
- Modify: `scripts/sweep_threshold_klein_base_4b.py` (if per-variant default needs retuning)

- [ ] **Step 1: Run the CFG-aware calibration at the failing release-bench schedule**

  Per plan-audit Finding 1, the recalibration MUST match the schedule of the failing release bench (50 steps, g=4.0), not the v0.4.0 calibration default (25 steps). The `--num-inference-steps 50` override added in Task 8 is mandatory here.

  ```bash
  uv run python scripts/calibrate_flux2.py \
    --variant klein-base-4b \
    --fit-mode origin \
    --guidance 4.0 \
    --num-inference-steps 50 \
    --fit-branch-policy worst \
    2>&1 | tee /tmp/calibrate-klein-base-4b-cfg.log
  ```

  Estimated: ~12-16 hours on M1 Max (50 steps × 10 prompts × 2 transformer calls per step). Background; monitor via `tail -f`.

- [ ] **Step 2: Inspect the fit metrics in the JSON**

  Check `fit_r_squared`, `y_min`, `y_max`, `coefficients_c4_to_c0`. If `y_min ≥ 0.20` (similar to the Klein 9B situation), threshold tuning won't help and we drop to per-variant-default re-tuning.

- [ ] **Step 3: Bake new coefficients into `_REGISTRY`**

  Update `_REGISTRY["flux2-klein-base-4b"]` in `src/mlx_teacache/coefficients.py` with the new tuple. Update the `Provenance` entry:
  - `revision="in-repo-2026-MM-DD-cfg-origin"` (today's date)
  - `calibration_dataset` — replace `guidance=1.0` with `guidance=4.0, num_inference_steps=50, fit_branch_policy=worst` (the recipe + branch policy must both be recorded for honest provenance)
  - `fit_metric_value` — the new R²

  The `default_thresh` field may need re-tuning if the new polynomial's y-range differs significantly. The existing `scripts/sweep_threshold_klein_base_4b.py` is hardcoded to `STEPS = 25` and `guidance=1.0`; per plan-audit Finding 1 it must be made CFG/50-aware before selecting a new threshold:

  ```bash
  # Add (mirror Task 7's pattern) --guidance and --num-inference-steps CLI flags to
  # scripts/sweep_threshold_klein_base_4b.py. Resolve them at the top of the script
  # the same way bench_speedup.py does. The threshold list itself stays unchanged.
  uv run python scripts/sweep_threshold_klein_base_4b.py --guidance 4.0 --num-inference-steps 50
  ```

  Pick the new `default_thresh` from the CFG sweep table (SSIM ≥ 0.85, skip count ≥ 1, smallest threshold above the cliff). Update the inline `default_thresh=...` literal and its explanatory comment in `_REGISTRY["flux2-klein-base-4b"]` to cite the CFG sweep evidence + measured numbers.

- [ ] **Step 4: Re-run Task 10 + Task 11 with the new coefficients**

  ```bash
  uv run pytest tests/test_image_quality_flux2.py::test_ssim_pr_gate_cfg_klein_base_4b -v -m parity
  uv run python scripts/bench_speedup.py --variant klein-base-4b --report scripts/_bench_report_klein_base_4b_v0_4_1.json
  ```

  Now they must pass. Per plan-audit Finding 2 there is **no structural-only escape hatch** for v0.4.1: the release narrative is "CFG steps become gate-active and can skip." If skip count is still 0 or SSIM still fails at default threshold after CFG/50-step recalibration and threshold sweep:

  1. Try lowering `default_thresh` further to a value the CFG sweep evidence supports (down to NVIDIA's published `teacache_thresh=0.05` ballpark for non-distilled FLUX.2-dev) so long as SSIM stays ≥ 0.85.
  2. If no threshold in the sweep produces both skip ≥ 1 AND SSIM ≥ 0.85, **hold the release**. Do not retag v0.4.1 as structural-only — that would repeat the v0.2.0 / v0.3.0 misframing the postmortem corrected.

  Allowed exits if step 4 still fails:
  - **Hold and investigate.** File a follow-up task that explores alternative gate signals (FBCache-style first-block residual, per-step-index lookup table from the postmortem references) before retagging anything.
  - **Rescope.** Reopen the spec, scope v0.4.1 down to an architecture-only PR (predict-closure refactor + `cached_residual_neg` + tests), publish it as `v0.5.0-architecture` or similar, and explicitly do not claim CFG step-skipping in the docs. Requires user authorization and a fresh README/CHANGELOG/ROADMAP wording pass — not a silent fallback from this plan.

  Do not tag `v0.4.1` until skip ≥ 1 and SSIM ≥ 0.85 are both green at the canonical recipe.

- [ ] **Step 5: Commit**

  ```bash
  git add src/mlx_teacache/coefficients.py scripts/_calibration_flux2_klein_base_4b.json scripts/sweep_threshold_klein_base_4b.py
  git commit -m "calib: CFG-aware recalibration for klein-base-4b at guidance=4.0 (contingency)"
  ```

---

## Task 13: Docs — README, CHANGELOG, calibration.md, ROADMAP

**Files:**
- Modify: `README.md`
- Modify: `CHANGELOG.md`
- Modify: `docs/calibration.md`
- Modify: `ROADMAP.md`

- [ ] **Step 1: README — Benchmarks table + Limitations + install pin**

  - Add a v0.4.1 row to the Benchmarks table: `| flux2-klein-base-4b (CFG)⁴ | 50 | <vanilla>s | <gated>s | **<combined>×** | **<skips> / 50** | step-skipping + mx.compile avoidance |`. Add a footnote `⁴` explaining the bench is at `guidance=4.0` (the canonical upstream recipe) and that the `<combined>` ratio is over vanilla mflux; the "v0.4.1 gating contribution" (nogate→gated) is `<gating_ratio>×`.
  - Remove the v0.4.0 footnote on the base-4b row that says "CFG falls back to vanilla pending v0.4.1."
  - Update Limitations: remove the "CFG on base-4b lands in v0.4.1" bullet.
  - Quick-start example for base-4b: switch to `guidance=4.0, num_inference_steps=50` (the canonical recipe).
  - Bump install pin from 0.4.0 to 0.4.1 (`pip install mlx-teacache==0.4.1`).
  - Run `/humanizer` over the new/edited prose (per CLAUDE.md "public-facing docs" rule).

- [ ] **Step 2: CHANGELOG — v0.4.1 entry**

  Append:

  ```markdown
  ## [0.4.1] — 2026-MM-DD

  ### Added
  - CFG-engaged TeaCache for FLUX.2. The canonical upstream recipe (`guidance_scale=4.0, num_inference_steps=50`) on `flux2-klein-base-4b` is now accelerated end-to-end via a shared-decision / per-branch-residual gated forward.
  - `TeaCacheState.cached_residual_neg` for the negative branch under CFG.
  - `GenerationStats.cfg_was_active` derives from a new `_Staging.cfg_was_active` flag set by the predict closure on first CFG branch entry.
  - Three-way bench protocol on `scripts/bench_speedup.py --variant klein-base-4b`: vanilla mflux / wrapped-no-gate (compile-avoidance only) / wrapped-gated (full v0.4.1). Separates the v0.4 effect from the v0.4.1 effect.
  - `scripts/calibrate_flux2.py --guidance` and `--fit-branch-policy` flags + CFG-aware capturing closure (computes both branches, returns CFG-combined noise, captures per-branch `body_out`).

  ### Changed
  - `_vanilla_flux2_cfg_predict()` no longer runs in production paths. It remains in `src/mlx_teacache/integrations/mflux/flux2.py` as a test-only diagnostic reference.
  - Lifecycle's distilled-step no-benefit warning no longer suppresses on `guidance > 1.0` for FLUX.2 — the regular `possible_skips == 0` check is the source of truth.
  - **Behavior change (skip-window validation under CFG):** an all-CFG generation with `skip_first_n_steps + skip_last_n_steps >= num_inference_steps` previously silently ran vanilla math; v0.4.1 raises `InvalidStepWindowError`. Same validation path as non-CFG v0.4.0.

  ### Deprecated
  - `TeaCacheStats.cfg_fallback_steps`. Always 0 from v0.4.1+. Use `GenerationStats.cfg_was_active` instead. Slated for removal in v1.0.
  ```

  Fill in the date and benchmark numbers from Task 11 / 12.

- [ ] **Step 3: docs/calibration.md — base-4b row update**

  Update the `flux2-klein-base-4b` row of the "Built-in coefficient sources" table:
  - If we kept the g=1.0 polynomial (Task 11 passed without contingency): append the note (verbatim from plan-audit Finding 3 — do not paraphrase, because the earlier wording reintroduced the transfer overclaim):

    > "Polynomial calibrated at `guidance=1.0`; v0.4.1 reuses it under CFG only because the g=4.0 / 50-step release bench passed the skip and SSIM gates. The encoder-independent `mod_in` invariant justifies one shared branch decision per step; coefficient transfer remains empirical."

  - If we recalibrated under CFG (Task 12 ran): replace the row's `calibration_dataset` description with the CFG-aware run (`10 prompts × 50 steps × seed=42, ..., guidance=4.0, fit_branch_policy=worst`). Cite the new R², `y_min`, `y_max` from the JSON report.

  In the "Producing new coefficients" section, add a CFG example with the step-count override (per plan-audit Finding 1, the step count must match the failing release schedule):

  ```bash
  # CFG-aware calibration (v0.4.1+). Use when the g=1.0 polynomial does not
  # engage acceptably at the canonical upstream recipe. Step count MUST match
  # the release recipe (50 for base-4b under CFG) — calibrating at 25 steps
  # and shipping at 50 would put off-trajectory coefficients in production.
  uv run python scripts/calibrate_flux2.py \
    --variant klein-base-4b \
    --fit-mode origin \
    --guidance 4.0 \
    --num-inference-steps 50 \
    --fit-branch-policy worst
  ```

- [ ] **Step 4: ROADMAP — move v0.4.1 to Released, promote v0.5.0**

  - Move the v0.4.1 entry from "Active" to "Released" (above v0.4.0 in the Released list). One-line summary: "CFG per-branch caching for FLUX.2. Canonical upstream recipe (g=4.0, 50 steps) on `flux2-klein-base-4b` now gate-engaged. `_vanilla_flux2_cfg_predict()` retired from production paths. Three-way bench separates v0.4 compile-avoidance from v0.4.1 gating contribution."
  - Promote v0.5.0 (`flux2-klein-base-9b`) to top of "Active" section.

- [ ] **Step 5: Run /humanizer over new prose chunks**

  Per CLAUDE.md: README, CHANGELOG highlights, calibration.md changes are public-facing. Invoke `/humanizer` over the new sections, apply rewrites, then re-verify the technical numbers were not changed by the rewrite.

- [ ] **Step 6: Commit**

  ```bash
  uv run ruff check . && uv run ruff format --check . && uv run pytest tests/ -m "not parity and not slow and not benchmark and not network"
  git add README.md CHANGELOG.md docs/calibration.md ROADMAP.md
  git commit -m "docs: v0.4.1 release entries (CFG per-branch caching)"
  ```

---

## Task 14: Open PR + CI + merge + tag v0.4.1

- [ ] **Step 1: Push branch + open PR**

  ```bash
  git push -u origin feature/v0.4.1-cfg-per-branch
  gh pr create --title "v0.4.1: CFG per-branch caching for FLUX.2" --body "$(cat <<'EOF'
  ## Summary
  - Replaces the `_vanilla_flux2_cfg_predict()` non-gated CFG path with a shared-decision / per-branch-residual gated forward (`flux2_cfg_forward_with_gate`).
  - Lights up TeaCache step-skipping on the canonical upstream recipe (`flux2-klein-base-4b` at `guidance_scale=4.0, num_inference_steps=50`).
  - Three-way bench protocol (`vanilla / no-gate / gated`) separates the v0.4 compile-avoidance effect from the new v0.4.1 gating effect — addresses spec-audit Finding 1.
  - CFG-aware calibration ships precautionarily (`scripts/calibrate_flux2.py --guidance 4.0 --fit-branch-policy worst`) — addresses spec-audit Finding 2.
  - Paired same-process vanilla-vs-wrapper CFG parity is the release blocker — addresses spec-audit Finding 4.
  - `cfg_fallback_steps` deprecated (always 0 from v0.4.1+). Use `GenerationStats.cfg_was_active` instead.
  - **Behavior change:** all-CFG generation with a misconfigured skip window now raises `InvalidStepWindowError` instead of running vanilla silently.

  ## Test plan
  - [ ] Pure-core tests green (ruff + format + pytest non-parity/slow/benchmark/network).
  - [ ] CFG parity release-blocker test green (`test_paired_cfg_parity_at_threshold_zero_klein_base_4b_pr_gate`, cosine ≥ 0.97).
  - [ ] CFG SSIM PR-gate test green (`test_ssim_pr_gate_cfg_klein_base_4b`, SSIM ≥ 0.85, skip count ≥ 1).
  - [ ] `scripts/bench_speedup.py --variant klein-base-4b` three-way numbers landed in README.
  - [ ] Distilled Klein 4B/9B smoke under CFG still works (no regression).
  EOF
  )"
  ```

- [ ] **Step 2: Wait for CI**

  ```bash
  gh pr checks --watch
  ```

- [ ] **Step 3: Merge + tag (requires user authorization for the tag-push since release.yml triggers PyPI publish)**

  ```bash
  gh pr merge --squash --delete-branch
  git checkout main && git pull --ff-only
  git tag v0.4.1 -m "v0.4.1: CFG per-branch caching for FLUX.2"
  # Wait for explicit user OK on the tag push (PyPI publish) before:
  # git push origin v0.4.1
  ```

  Authorization gate per CLAUDE.md: do not push the tag without explicit user confirmation. Once authorized:

  ```bash
  git push origin v0.4.1
  gh workflow view release.yml --web   # confirm the release workflow fired
  ```

- [ ] **Step 4: Verify PyPI publish**

  ```bash
  curl -s https://pypi.org/pypi/mlx-teacache/json | python -m json.tool | grep -A2 '"version"'
  ```

  Expect `0.4.1` as latest.

---

## Self-review checklist (run after completing the plan write-out)

1. **Spec coverage:** Each spec section maps to a task — cache (Task 2), stats (Task 3), forward (Task 4), predict-closure + skip-window lift (Task 5), lifecycle (Task 6), bench three-way (Task 7), calibration CFG-aware (Task 8), parity tests (Task 9), SSIM (Task 10), bench gate (Task 11), contingency (Task 12), docs (Task 13), release (Task 14). ✓
2. **Placeholders:** Task 9 step 3 contains a `pytest.skip` placeholder — this is intentional (the diagnostic test is only fleshed out if the release-blocker fails). Task 12 step 3 has `revision="in-repo-2026-MM-DD-cfg-origin"` — date filled in at execution time. No other placeholders. ✓
3. **Type consistency:** `flux2_cfg_forward_with_gate` signature in Task 4 matches what Task 5's predict-closure calls. `_Staging.cfg_was_active` is read in Task 6's lifecycle update exactly as written in Task 3. `--guidance` flag in Tasks 7 + 8 is the same name. ✓
4. **Order dependencies:** Task 2 (cache) before Task 4 (forward uses cached_residual_neg). Task 3 (staging field) before Task 5 (closure sets it) before Task 6 (lifecycle reads it). Task 4 (function) before Task 5 (closure calls it). Tasks 7-10 land before Task 11 (real bench). Task 12 conditional on Task 11 outcome. Tasks 13-14 last. ✓
