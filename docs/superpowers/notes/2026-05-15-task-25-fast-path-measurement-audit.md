# Task 25 - fast-path measurement audit and research

**Date:** 2026-05-15
**Audits:** `2026-05-14-task-25-fast-path-measurement.md`
**Local code reviewed:** `cb71978 Add threshold-zero fast path in flux1_forward_with_gate`
**Status:** Addendum. Keep the 2026-05-14 note, but revise the claims below before turning it into spec or test work.

## Executive take

I agree with the main decision in the measurement note: the committed Task 24 FLUX.1 latents should not be the correctness oracle for threshold-zero TeaCache on MLX. The measured cosine around `0.18` and max_abs around `5.8` is not "slightly too strict tolerance"; it is a different denoising trajectory. A gate that passes by raising `atol` high enough would be meaningless.

The important adjustment is causal humility. The fast path not shrinking divergence falsifies the `cached_residual` ref-liveness theory as the dominant cause, but it does not prove that the proxy, callback, or `generate_image` wrapper is the cause. The current evidence proves a narrower and stronger point: same-process vanilla and threshold-zero wrapper can be equivalent, while saved cross-process fixtures are not stable enough to act as a math oracle.

My recommendation: keep the fast path, stop testing threshold-zero against committed latents, and replace it with same-process paired parity plus restore controls. Keep committed fixtures only as drift fingerprints.

## Audit corrections

### 1. Diffusers FirstBlockCache is tested, but not with bit-exact fixtures

The measurement note says Diffusers FirstBlockCache has no tests. That is too broad.

Current Diffusers main at `2375f70f67bb49cd82ac9d04983650f8266fcea8` has `FirstBlockCacheTesterMixin` in `tests/pipelines/test_pipelines_common.py`. The test builds CPU dummy pipelines, runs a 4-step generation without cache, then with cache enabled, then after disabling cache. It asserts:

- enabled cache output slice is `np.allclose(..., atol=0.1)`;
- disabled cache output slice is `np.allclose(..., atol=1e-4)`.

So the correct statement is:

> Diffusers has a tolerant, in-process, dummy-pipeline regression test for FirstBlockCache. It does not have a bit-exact full latent/image fixture parity test for real models.

That correction matters because it supports the redesign: use in-process controls and calibrated tolerances, not long-lived exact artifacts.

### 2. MLX still has no PyTorch-style forward hooks

MLX `nn.Module` at `046217bcae7347aa814665f39a8f0e404029ddb0` is dict-like and declares `__call__: Callable`; it does not provide a central `forward` method or `register_forward_hook` equivalent. A local search of `python/mlx/nn` found no hook API.

This means a Diffusers-style hook registry cannot be ported directly. Diffusers can rewrite `module.forward` because PyTorch has `nn.Module.__call__ -> forward`. In mflux/MLX, the transformer is called through `transformer(...)`, so interception is about `__call__`, not `forward`.

### 3. Instance-level `__call__` replacement is not a real option

The measurement note lists "method swap on instance (`flux.transformer.__call__ = bound_wrapper`)" as possible. For `obj(...)`, Python special method lookup is on the type, not the instance dictionary. I verified this locally:

```text
a.__call__() -> instance
a()          -> class
```

This follows the Python data model's special-method lookup rule. So instance-level `__call__` replacement would not reliably intercept `flux.transformer(...)`.

Real FLUX.1 patch options are:

1. Current per-instance proxy: replace `flux.transformer` with an object whose class defines `__call__`.
2. Class-level monkey patch: assign `Transformer.__call__ = patched_call`, global within the process.
3. Dynamic one-instance subclass: mutate `flux.transformer.__class__` to a generated subclass with patched `__call__`, risky and likely not worth it.
4. Change mflux call sites, which is outside this package's integration boundary.

ComfyUI-TeaCache's CogVideoX path assigns `transformer.forward = ...`, but that is a PyTorch `forward` patch. It is not analogous to MLX `__call__`.

### 4. Upstream TeaCache libraries optimize for quality, not exactness

`ali-vilab/TeaCache` at `7c10efc4702c6b619f47805f7abe4a7a08085aa0` patches Diffusers FLUX with a class-level `FluxTransformer2DModel.forward = teacache_forward` style and stores several TeaCache fields on the transformer class. I did not find a FLUX parity test suite there; the repo presents visual examples and metric/eval scripts.

`welltop-cn/ComfyUI-TeaCache` at `91dff8e31684ca70a5fda309611484402d8fa192` ships ComfyUI node patching and example workflows. For the main Comfy FLUX node, `rel_l1_thresh == 0` returns the original model unchanged. For other paths it patches `forward`/`forward_orig` inside Comfy's wrapper system. I did not find pytest-style correctness tests in that repo.

Diffusers is the most disciplined of the three, and even there the cache correctness test is tolerant and in-process.

## Local fast-path assessment

The current fast path in `src/mlx_teacache/integrations/mflux/forward.py` is directionally right:

- threshold `<= 0` never builds `mod_in`;
- it never builds `body_in_concat`;
- it never writes `cached_residual`;
- it records every step as `computed`;
- `gate.py` also returns `should_update_cache=False` for non-positive threshold.

Even if this did not fix cross-fixture divergence, it removes unnecessary work and avoids preserving body/tail intermediates when the cache can never be consumed. I would keep it.

The doc comment in `forward.py` should eventually be softened. It currently says the removed tensors "can perturb downstream kernel dispatch" and points to the nondeterminism notes. The measurement now shows this was not the dominant perturbation. Better wording: this branch avoids unnecessary cache bookkeeping and removes one previously suspected source of MLX allocation/refcount perturbation.

## Test-design recommendation

### Tier 1 - synthetic unit tests

Keep the existing pure/synthetic tests for `gate_step`, lifecycle, restore, proxy delegation, and stats. These catch logic regressions cheaply and should stay exact.

### Tier 2 - threshold-zero same-process parity

Replace saved-latent parity with paired same-process parity:

```python
vanilla_before = capture(flux, prompt=prompt, ...)
with apply_teacache(flux, rel_l1_thresh=0.0) as handle:
    wrapper = capture(flux, prompt=prompt, ...)
vanilla_after = capture(flux, prompt=prompt, ...)

assert mx.array_equal(vanilla_before, wrapper)
assert mx.array_equal(vanilla_before, vanilla_after)
assert handle.stats.skipped_count == 0
```

The `vanilla_after` control is important. It proves `restore()` returned the shared module fixture to the same observable state and catches hidden callback/proxy/sentinel leakage.

For one prompt, also consider the reverse order with a fresh model when practical:

```text
wrapper -> restore -> vanilla
```

This guards against a test that only passes because vanilla ran first and warmed MLX state in a favorable order.

### Tier 3 - default-threshold quality against a same-process baseline

Default threshold is allowed to skip, so exact parity is the wrong target. Compare default-threshold TeaCache against a same-process vanilla baseline, not against committed fixtures.

Minimum gate:

- generation completes;
- all latents are finite;
- `handle.stats.skipped_count >= 1` for calibrated prompts;
- latent similarity/image metric is above a measured threshold.

Do not hard-code `cosine >= 0.985` until it is calibrated against same-process vanilla for the current implementation. If image-level metrics are added, prefer SSIM/LPIPS on decoded images as the slower nightly gate.

### Tier 4 - fixture drift, not correctness

Keep committed `.safetensors` only for artifact integrity or drift diagnostics:

- "Did this environment reproduce the old fingerprint?"
- "Did a dependency or MLX version change output bytes?"

Do not make those fingerprints fail `apply_teacache(rel_l1_thresh=0)` correctness.

## Specific test-file edits implied

`tests/test_parity_flux1.py` is currently untracked and still fixture-based. I would change it before landing:

- `test_threshold_zero_bit_exact_dev`: compare same-process vanilla vs threshold-zero wrapper, then vanilla-after-restore.
- `test_threshold_zero_bit_exact_schnell`: same redesign.
- `test_default_threshold_cosine_similarity_dev`: compare default threshold to same-process vanilla, not `_load_reference`.
- `test_threshold_zero_with_negative_coefficients_no_skip`: same-process parity plus `skipped_count == 0`; optionally assert the live cache never stores `cached_residual`.
- `test_failed_generation_retry_no_stale_cache`: after the simulated crash, compare the retry against a same-process vanilla baseline instead of a committed fixture.
- Leave restore/idempotency/callback/lifecycle tests exact; they are not affected by MLX numeric drift.

If the parity suite is expensive, use one prompt for PR-level parity and all five prompts for a local/nightly parity mark.

Cost note: if one 25-step generation is about 2.5 minutes, a paired vanilla-plus-wrapper test is about 5 minutes per prompt. Five prompts are about 25 minutes total, not 12.5 minutes total, unless the note is only counting incremental cost over an already-running wrapper generation.

## Diagnostic matrix, if we still want causality

I would run this as a separate localizer script, one prompt first:

| Case | What changes from vanilla | Purpose |
|---|---|---|
| A | vanilla, no `mlx_teacache` import | current fixture baseline |
| B | import `mlx_teacache`, no apply | catches import-time/environment effects |
| C | register no-op before/after callback only | isolates callback list and pre-loop work |
| D | wrap `generate_image` only | isolates outer Python frame and finally block |
| E | delegating proxy only | `flux.transformer = Proxy(inner)` where `__call__` just calls `inner(...)` |
| F | proxy plus threshold-zero reimplemented forward | isolates our forward topology |
| G | full `apply_teacache(rel_l1_thresh=0)` | current behavior |

For every case, collect:

- same-process `vanilla_before` vs case output;
- same-process `vanilla_after` vs `vanilla_before`;
- old committed fixture vs case output;
- first divergent step if a per-step callback is cheap enough.

If same-process controls pass while old fixture comparison fails, stop trying to recover cross-process bit-exactness. The fixture is measuring process/setup drift, not TeaCache math.

## References checked

- Local Diffusers clone: `2375f70f67bb49cd82ac9d04983650f8266fcea8`
  - `src/diffusers/hooks/first_block_cache.py`
  - `src/diffusers/hooks/hooks.py`
  - `src/diffusers/models/cache_utils.py`
  - `tests/pipelines/test_pipelines_common.py`
- Local MLX clone: `046217bcae7347aa814665f39a8f0e404029ddb0`
  - `python/mlx/nn/layers/base.py`
  - `python/mlx/nn/utils.py`
- Local TeaCache clone: `7c10efc4702c6b619f47805f7abe4a7a08085aa0`
  - `TeaCache4FLUX/teacache_flux.py`
- Local ComfyUI-TeaCache clone: `91dff8e31684ca70a5fda309611484402d8fa192`
  - `nodes.py`
  - `nodes_diffusers.py`
- Python data model, special method lookup:
  - https://docs.python.org/3/reference/datamodel.html#special-method-lookup
- Diffusers caching docs:
  - https://huggingface.co/docs/diffusers/api/cache
