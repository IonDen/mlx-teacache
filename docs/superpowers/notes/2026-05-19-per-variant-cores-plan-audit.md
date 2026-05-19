# Audit: per-variant cores implementation plan

Source: `docs/superpowers/plans/2026-05-19-per-variant-cores.md`
Date: 2026-05-19
Scope: Material execution risks in the v0.6.0 per-variant-core implementation plan, checked against the local v0.5.x codebase and the prior spec audit.

## Findings

### 1. The kernel gate/state tasks rewrite the TeaCache algorithm instead of extracting it

Severity: High
Refs: `docs/superpowers/plans/2026-05-19-per-variant-cores.md:75`, `docs/superpowers/plans/2026-05-19-per-variant-cores.md:196`, `docs/superpowers/plans/2026-05-19-per-variant-cores.md:367`, `docs/superpowers/plans/2026-05-19-per-variant-cores.md:451`, `src/mlx_teacache/gate.py:45`, `src/mlx_teacache/gate.py:113`, `src/mlx_teacache/gate.py:114`, `src/mlx_teacache/gate.py:115`, `src/mlx_teacache/gate.py:117`, `src/mlx_teacache/cache.py:23`

Evidence: The plan's `_kernel.gate.polynomial_gate()` evaluates the polynomial against `accumulated_rel_l1`; current `gate_step()` evaluates the polynomial against the current step's `rel_l1`, clamps the predicted output distance, adds that prediction to `state.accumulated_distance`, and resets the accumulator after a compute. The plan's `_kernel.state.TeaCacheState` also drops current runtime fields: `step_counter`, `previous_mod_input`, `skip_window_validated`, `last_timestep`, and `num_steps`.

Impact: This is not a refactor-preserving extraction. It changes skip decisions, loses forced-window behavior, loses threshold-zero fast-path behavior, and breaks img2img/active-step indexing. Bench and SSIM parity can drift even if the new code is internally consistent.

Fix: Start by moving the current `gate_step()` and current `cache.TeaCacheState` semantics verbatim into `_kernel`, then split smaller helpers only after equivalence tests compare old-vs-new decisions for threshold-zero, first-step seeding, forced windows, numerical misses, accumulated-distance reset, and CFG state reset.

### 2. The stats/lifecycle rewrite breaks the public stats contract and finalizes at the wrong time

Severity: High
Refs: `docs/superpowers/plans/2026-05-19-per-variant-cores.md:506`, `docs/superpowers/plans/2026-05-19-per-variant-cores.md:537`, `docs/superpowers/plans/2026-05-19-per-variant-cores.md:553`, `docs/superpowers/plans/2026-05-19-per-variant-cores.md:561`, `docs/superpowers/plans/2026-05-19-per-variant-cores.md:645`, `docs/superpowers/plans/2026-05-19-per-variant-cores.md:705`, `docs/superpowers/plans/2026-05-19-per-variant-cores.md:1611`, `docs/superpowers/plans/2026-05-19-per-variant-cores.md:1978`, `src/mlx_teacache/stats.py:34`, `src/mlx_teacache/stats.py:43`, `src/mlx_teacache/stats.py:70`, `src/mlx_teacache/stats.py:114`, `src/mlx_teacache/stats.py:131`, `src/mlx_teacache/integrations/mflux/lifecycle.py:183`

Evidence: The plan replaces current mutable/staged `TeaCacheStats` with an immutable tuple-of-generations model, changes `StepDecision` fields from `step_idx/timestep/rel_l1/accumulated_distance/decision` to `step/skipped/rel_l1/predicted_rel_l1/threshold`, and expects `skipped_count` to raise with no generations. Current `TeaCacheStats` returns `speedup_estimate == 1.0` before any generation, exposes aggregate counters (`computed_count`, `forced_count`, `skipped_count`, `numerical_miss_count`, `cfg_fallback_steps`), and commits stats only after `generate_image` returns naturally via the wrapped generation lifecycle. The plan's variant finalizers call `fsm.finalize_generation()` from `handle.restore()`, even if no generation ran or a generation failed.

Impact: This would break existing stats consumers, count empty generations on context exit, and lose the current guarantee that failed/interrupted generations leave no public stats trace. The design also removes `cfg_was_active` and `num_steps` from `GenerationStats`, both part of the v0.4.1 stats behavior.

Fix: Move current `stats.py` behavior into `_kernel.stats` without changing field names or commit/discard semantics. Keep generation finalization attached to the `generate_image` wrapper and mflux callbacks; `VariantPatch` should own teardown only, not decide that a generation completed.

### 3. The new dispatcher drops explicit public API parameters and weakens the compatibility gate

Severity: High
Refs: `docs/superpowers/plans/2026-05-19-per-variant-cores.md:1590`, `docs/superpowers/plans/2026-05-19-per-variant-cores.md:1950`, `docs/superpowers/plans/2026-05-19-per-variant-cores.md:2663`, `docs/superpowers/plans/2026-05-19-per-variant-cores.md:2885`, `src/mlx_teacache/api.py:137`, `src/mlx_teacache/api.py:142`, `src/mlx_teacache/api.py:190`

Evidence: Current `apply_teacache()` has explicit public keyword parameters: `rel_l1_thresh`, `coefficients`, `skip_first_n_steps`, and `skip_last_n_steps`. The plan rewrites the facade as `apply_teacache(flux, **kwargs)` and the variant `apply()` examples only handle `rel_l1_thresh`. The proposed signature test allows `rel_l1_thresh` to be absent because it treats `**kwargs` as acceptable.

Impact: This hides a public signature change and risks silently losing custom coefficients and skip-window controls. Those controls are not internal: README documents custom coefficients, and skip windows drive correctness for short schedules/img2img.

Fix: Keep the explicit `apply_teacache(flux, *, rel_l1_thresh=..., coefficients=None, skip_first_n_steps=1, skip_last_n_steps=1)` facade signature. Variant `apply()` must receive the resolved or raw public options, including custom coefficients and skip windows. Update the public API test to require exact parameter names and keyword behavior, not just `**kwargs`.

### 4. The plan hard-codes wrong coefficient tuples

Severity: High
Refs: `docs/superpowers/plans/2026-05-19-per-variant-cores.md:1296`, `docs/superpowers/plans/2026-05-19-per-variant-cores.md:2273`, `docs/superpowers/plans/2026-05-19-per-variant-cores.md:2347`, `src/mlx_teacache/coefficients.py:35`, `src/mlx_teacache/coefficients.py:50`, `src/mlx_teacache/coefficients.py:76`

Evidence: The plan's `flux1_dev/config.py` tuple is not the current shipped `_UPSTREAM_FLUX_COEFFS`; it uses values from the previously incorrect transcription path. The plan also includes placeholder/incorrect FLUX.2 Klein 4B/9B tuples while saying to verify them. Current 4B is `(236.9190176, -201.4740136, 66.9135424, -11.1479674, 1.2674506)` and current 9B is `(-523.8412981, 530.2492513, -177.6438573, 20.8932650, 0.0)`.

Impact: If implemented literally, v0.6.0 changes calibration data while claiming architectural no-op refactor. That invalidates all parity, benchmark, and documentation claims before the integration code is even considered.

Fix: Remove explicit coefficient literals from the plan except where they are copied from current source in the same task. Add a guard that compares every new `variants/*/config.py::COEFFICIENTS` to the v0.5.x registry before deleting the old registry.

### 5. Several tasks point at non-existent legacy modules and miss the real lifecycle owner

Severity: Medium-High
Refs: `docs/superpowers/plans/2026-05-19-per-variant-cores.md:367`, `docs/superpowers/plans/2026-05-19-per-variant-cores.md:711`, `docs/superpowers/plans/2026-05-19-per-variant-cores.md:2978`, `src/mlx_teacache/cache.py:1`, `src/mlx_teacache/integrations/mflux/lifecycle.py:1`

Evidence: The plan tells executors to read and later delete `src/mlx_teacache/state.py` and `src/mlx_teacache/lifecycle.py`, but those files do not exist in the current tree. The current cache state is `src/mlx_teacache/cache.py`, and the generation lifecycle owner is `src/mlx_teacache/integrations/mflux/lifecycle.py`.

Impact: Executors following the plan will extract the wrong shape of state/lifecycle, skip the current active-step/img2img reset behavior, and later run deletion steps that do not remove the real legacy lifecycle code.

Fix: Replace every `state.py` reference with `cache.py`, and every top-level `lifecycle.py` reference with `integrations/mflux/lifecycle.py` unless the plan intentionally creates a new top-level shim first. The cleanup tasks should delete or shim files that actually exist.

### 6. The variant integration skeletons invent helper boundaries that do not exist in the current forward code

Severity: Medium-High
Refs: `docs/superpowers/plans/2026-05-19-per-variant-cores.md:1518`, `docs/superpowers/plans/2026-05-19-per-variant-cores.md:1561`, `docs/superpowers/plans/2026-05-19-per-variant-cores.md:1564`, `docs/superpowers/plans/2026-05-19-per-variant-cores.md:1569`, `docs/superpowers/plans/2026-05-19-per-variant-cores.md:1628`, `src/mlx_teacache/integrations/mflux/forward.py:90`, `src/mlx_teacache/integrations/mflux/forward.py:142`, `src/mlx_teacache/integrations/mflux/forward.py:186`, `src/mlx_teacache/integrations/mflux/forward.py:222`, `src/mlx_teacache/integrations/mflux/forward.py:245`

Evidence: The FLUX.1 integration task asks workers to copy `_extract_mod_in`, `_compute_residual`, and `_apply_cached_residual` from the legacy forward module, but those helper functions do not exist. The current code is a monolithic `flux1_forward_with_gate()` that mirrors mflux prelude/body/tail ordering and caches `body_out_concat - body_in_concat` before always running the tail. The plan's skeleton calls `_inner(*args, **kwargs)` and then tries to compute/apply residuals after the full transformer call, which is not where TeaCache's residual lives.

Impact: The worker can either fail to find the referenced helpers or implement a residual at the wrong boundary. In MLX diffusion wrappers this is not a harmless refactor: graph topology, buffer lifetime, and body/tail placement are part of the parity surface.

Fix: Make the first integration port copy the current `flux1_forward_with_gate()` and `ProxyFlux1Transformer` behavior directly, then extract reusable pure helpers only after parity. Do not ask workers to invent new helper seams during the same port.
