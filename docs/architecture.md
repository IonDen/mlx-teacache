# Architecture

How `mlx-teacache` is put together, for anyone reading the source or adding a variant. The user-facing "How it works" summary lives in the [README](../README.md#how-it-works); this page is the contributor's map.

The whole library is one idea applied per model: run the expensive transformer body only on the steps that matter, reuse the previous step's body output on the rest, and decide which is which with a small polynomial gate. Everything below is the machinery that makes that safe to drop into mflux and easy to extend.

## The public entry: `apply_teacache`

`api.py::apply_teacache(flux, **kwargs)` is the only public function, and it always runs the same sequence:

1. Validate the keyword arguments statically (threshold range, coefficient shape, skip-window ints).
2. Check the already-patched sentinel (`flux._teacache_handle`). A second apply on a patched model raises rather than nesting.
3. Walk the variant registry and take the first entry whose `detect.matches(flux)` returns True. No match raises `IncompatibleModelError`.
4. Warn at apply time if the matched variant has no built-in default threshold (the distilled Klein variants) and the caller passed no coefficients: the dispatcher raises `TeaCacheNoBenefitWarning`. Under `filterwarnings = error` it raises as an error, so parity-lane apply sites wrap it.
5. Lazily import the winning variant's `integration.py` and call its `apply()`. A variant can emit its own warning here — `qwen-image` raises `TeaCacheUncalibratedCheckpointWarning` once when the model was loaded from a checkpoint its coefficients were not calibrated on (Qwen-Image-2512 on mflux 0.19).
6. Attach `variant_id` and a rollback that clears the sentinel, and hand back a `TeaCacheHandle`.

The 4-keyword signature (`rel_l1_thresh`, `coefficients`, `skip_first_n_steps`, `skip_last_n_steps`) is snapshot-tested, so it does not drift between releases.

## Variant registry and the mflux-free import contract

Every model is a three-file subpackage under `src/mlx_teacache/variants/<id>/`:

- `config.py` — a `META` dict (`variant_id`, display name, license, recipes), the degree-4 `COEFFICIENTS` tuple (high-to-low), and `DEFAULT_THRESH`. **Imports no mflux.**
- `detect.py` — `matches(flux) -> bool`, duck-typing `flux.model_config`. **Imports no mflux.**
- `integration.py` — `apply(flux, **kwargs) -> TeaCacheHandle`. **Imports mflux**, and is loaded lazily.

`variants/__init__.py::_build_registry()` walks every subpackage at import time but eagerly imports only `config` and `detect`, registering a lazy `load_integration` thunk for the rest. A variant's `integration.py` is imported only after its `detect.matches()` wins. That is the contract that keeps `import mlx_teacache` working without the `[mflux]` extra installed: nothing on the import path from the package root touches mflux until a real model asks for it. The rule for anyone adding code: do not import mflux from `config`, `detect`, the package root, or `_kernel/`.

## `_kernel/` is canonical; the top-level modules are shims

The pure-math primitives — `gate_step`, `TeaCacheState`, `Provenance`, `TeaCacheStats` — live in `src/mlx_teacache/_kernel/` and import only `mlx.core`, never mflux. The top-level `gate.py`, `cache.py`, `coefficients.py`, and `stats.py` are thin re-export shims kept for API stability (they were extracted into `_kernel/` in v0.6.0). Edit the `_kernel/` copy; the shims just forward.

## The gate

`_kernel/gate.py::gate_step()` is the single source of skip logic. It returns a frozen `GateDecision(kind, should_compute, should_update_cache, rel_l1, predicted_distance, accumulated_distance)`, and the forwards act on that decision rather than deciding for themselves.

The core behaviour:

- **Signal.** Each step measures the relative-L1 distance between its gate signal (for the FLUX variants, the modulated block-0 input) and the previous gated step's — computed or skipped. That consecutive delta is exactly the quantity the calibration scripts fit their polynomials on.
- **Accumulate and threshold.** The polynomial maps that delta to a predicted body-output change; the gate accumulates predictions across consecutive skips and recomputes once the running total would cross `rel_l1_thresh`, then resets it to zero. The accumulator is monotonic within a generation.
- **Anchoring.** The gate owns `state.previous_mod_input` and advances it on every signal-bearing step, computed or skipped, so the delta stays consecutive. No variant forward writes the anchor — a change that keeps all the anchoring logic in one place.
- **Runaway guard.** `MAX_CONSECUTIVE_SKIPS = 8` forces a recompute after eight skips in a row regardless of the accumulator. The origin-constrained fits cross zero at large deltas, past the range they were fit on, where the clamp would otherwise read a large real change as no change and stall the accumulator. This is a deliberate departure from upstream, which has no such cap. It sits above every shipped operating point (the longest observed streak at a default is 4, on Qwen).
- **Fast paths.** A threshold at or below zero always computes and never caches; a forced skip-window step or a non-finite signal computes but does not update the cache; a numerical miss (a non-finite prediction from finite inputs) drops the cached residual and zeroes both the accumulator and the streak, so the next finite step re-seeds instead of reusing a stale residual.

Residuals are written only through `TeaCacheState.store_residuals(pos=, neg=)`, which evaluates them at the one place they are created (a lazy `body_out - body_in` would pin both operands for the life of the handle), and `release_arrays()` drops them after the denoising loop and on restore.

## Patch strategies, by model family

The three families differ in where mflux exposes a seam, so the wrapper hooks each differently.

**FLUX.1** (`variants/flux1_*`, including Krea). Swap `flux.transformer` for a `ProxyFlux1Transformer` (an `nn.Module`) whose `__call__` runs `flux1_forward_with_gate`, inserting the gate between the transformer body and the norm/projection tail. The real module is held as `_inner` through `object.__setattr__`, so MLX's `parameters()` filter does not recurse into it — which is why `flux.parameters()` at the parent level can miss transformer parameters while the wrapper is active (use `flux.transformer.parameters()` or `restore()` first).

**FLUX.2 Klein and Z-Image** (`variants/flux2_*`, `variants/z_image_base`). Replace `flux._predict` with an instance-level eager closure. This deliberately bypasses mflux's `mx.compile` of `_predict`, which is what keeps per-step gating live (a compiled `_predict` would trace the Python gate once and never run it again). It is also the source of the separate "compile-avoidance" wall-clock effect, which the docs keep attributed apart from step-skipping. Under CFG the closure keeps two cached residuals (positive and negative branch) and shares one gate decision per step.

**Qwen-Image** (`variants/qwen_image`). Like FLUX.1 it proxies `flux.transformer` (`ProxyQwenTransformer` running `qwen_forward_with_gate`), because Qwen has no `_predict` factory and no `mx.compile`. But mflux's Qwen `generate_image` calls the transformer twice per step — positive then negative caption, combined outside the transformer — so the forward threads a `CfgBranchPairer` (`pairing.py`): the gate decision is computed once on the positive branch and reused on the negative, with a cached residual per branch. Sharing one decision is exact rather than approximate, because the gate signal depends on the latents and timestep, not the caption.

## Lifecycle: who owns the stats

`integrations/mflux/lifecycle.py` keeps per-generation bookkeeping out of the handle:

- `GenerationContextCallback` (registered on `flux.callbacks`) owns the per-generation cache reset and stats staging. It computes the **active** step count as `num_inference_steps - init_time_step`, so img2img windows are handled correctly, and emits `TeaCacheNoBenefitWarning` when no step in the window can be skipped.
- `wrap_generate_image` wraps `flux.generate_image` in a `try`/`finally`. Stats are committed only when a generation finishes naturally and discarded on any exception or `KeyboardInterrupt`, which preserves the invariant that a committed generation has exactly one decision per step. It also checks the lifecycle callback is still registered, raising `MissingGenerationContextError` if a caller removed it.

## Teardown

Each `apply()` returns a `VariantPatch(rollbacks, finalizers, on_restored)` and a variant-agnostic `TeaCacheHandle` (`handle.py`). `restore()` (and `__exit__`) run every rollback in reverse install order, then every finalizer. If any teardown action fails, the first error is re-raised, the handle stays retryable, the success-only `on_restored` hook does not run, and the double-apply sentinel stays set on the half-restored model. After a clean teardown, `on_restored` clears the sentinel, the stats object freezes, and the handle is marked torn down. Installs are transactional too: if a mutation inside `apply()` raises, the rollbacks gathered so far run before the error propagates, so a failed apply leaves the model as it found it.

## Coefficients, calibration, provenance

FLUX.1 coefficients are vendored from upstream ali-vilab TeaCache; the FLUX.2 and Z-Image tuples are calibrated in-repo by `scripts/calibrate_*.py`, which captures per-step `(gate_signal, body_output)` relative-L1 pairs from a no-skip run and fits a degree-4 polynomial. `flux2-klein-base-9b` reuses `base-4b`'s tuple by object identity, justified by the shared architecture and calibration recipe and validated per release. Every built-in tuple carries a `Provenance` record, and the committed calibration JSONs are pinned by tests. Procedure and per-variant fit quality live in [`docs/calibration.md`](calibration.md) and the per-variant pages under [`docs/variants/`](variants).

## Quality gates: SSIM, not bit-exactness

Replacing an `mx.compile`d function with an eager one costs about one ULP per element of Metal-dispatch divergence, which compounds across steps, so FLUX.2 parity is numerical rather than bit-exact (cosine stays at or above 0.97 at threshold 0). The guarantee the library actually makes is end-to-end image quality, measured by SSIM: the red-apple PR-gate holds ≥ 0.90 on FLUX.1 and ≥ 0.85 on FLUX.2, and the wider prompt suite holds ≥ 0.80 to absorb high-frequency-detail variance. Those gates run against real weights in `tests/test_image_quality_*.py` and `tests/test_parity_*.py`.

## Adding a variant

The moving parts, in order: create `variants/<id>/{config,detect,integration}.py` (config and detect mflux-free); calibrate coefficients or vendor them; add a `docs/variants/<id>.md` page; regenerate the README "Supported models" table with `docs/_generate_supported_models.py`; and expect the coverage floor to need a small downward nudge, since a new `integration.py` forward is only reachable with real weights.
