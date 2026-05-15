# Task 25 — fast-path measurement + upstream survey

**Date:** 2026-05-14
**Status:** Investigation continuing. Decision pending.
**Continues:** `2026-05-14-task-25-mlx-nondeterminism.md` and `…-addendum.md`.

## TL;DR

The threshold-zero fast path (commit `cb71978`) landed and is unit-tested. But measuring the wrapper's output against the committed Task 24 vanilla fixtures for all 5 dev prompts shows **the divergence did NOT shrink** — cosine ~0.18 / max_abs ~5.8, identical to before the fast path. The addendum's `cached_residual` ref-keeping was not the dominant cause. Something else in `apply_teacache`'s setup (proxy / callback registration / generate_image wrap) shifts MLX dispatch. Upstream caching libraries (ali-vilab TeaCache, ComfyUI-TeaCache, diffusers FirstBlockCache) **don't have bit-exact parity tests at all** — they rely on quality metrics (SSIM, LPIPS, visual diff). Our spec's bit-exact gate may have been the wrong oracle from the start.

## Diff measurement against committed fixtures

Script: `/tmp/measure_dev_tolerance.py`. Log: `/tmp/measure_dev_tolerance.log`. Wrapper at `rel_l1_thresh=0.0, skip_first=0, skip_last=0` (post-fast-path), `_LatentCapture` registers only `call_after_loop`.

| Prompt | sha_match | max_abs | mean_abs | max_rel | cosine |
|---|---|---|---|---|---|
| a red apple on a wooden table | ❌ | 5.68 | 1.13 | 2.62e+04 | 0.181 |
| mountain landscape at sunset | ❌ | 5.73 | 1.23 | 7.91e+04 | 0.164 |
| portrait of a woman | ❌ | 5.82 | 1.13 | 4.02e+04 | 0.191 |
| abstract pattern with circles | ❌ | 5.31 | 0.89 | 1.08e+05 | 0.105 |
| text saying HELLO | ❌ | 5.52 | 0.93 | 6.46e+04 | 0.256 |
| **Worst across 5** | — | **5.82** | — | **1.08e+05** | **0.105** |

The pre-fast-path diagnostics (`dev_diagnose.py`, `diverge_25step.py`) produced the same SHA (`0db096b2…`) as today's measurement. So the fast path commit changed the math identically for both — i.e., it didn't change observable output bytes at all. The ref-count donation theory predicted (some) shift; we observed none.

cosine ≤ 0.26 means: as latents, today's wrapper output and the committed Task 24 vanilla fixture are essentially uncorrelated. A `np.testing.assert_allclose(atol=1e-5, rtol=1e-5)` test would fail catastrophically, and any tolerance loose enough to pass (`atol > 5.8`) would also pass a completely broken implementation. **`assert_allclose` against the committed fixture is not a viable correctness gate.**

## Within-process equivalence holds

Per the earlier per-step diagnostic (`/tmp/per_step_diff.log`), within a single Python process vanilla and wrapper produce identical in-loop latents at all 25 steps. And `in_vs_after_loop.py` showed both vanilla and wrapper produce the same SHA when measured side-by-side in one process — that SHA was `0db096b2…`, not the committed `45471c34…`.

This is the key: today's wrapper IS observationally equivalent to today's vanilla *within a script*. The committed fixture is reproducible **only** by a script that doesn't touch `apply_teacache` (or anything else that perturbs MLX dispatch). The script-level setup itself shifts MLX kernel selection enough to change the output.

## How upstream caching layers solve this

We surveyed three reference implementations to see how they handle parity.

### ali-vilab/TeaCache (the paper's reference)

[`TeaCache4FLUX/teacache_flux.py`](https://github.com/ali-vilab/TeaCache/blob/main/TeaCache4FLUX/teacache_flux.py)

- **Patch mechanism:** class-level monkey patch — `FluxTransformer2DModel.forward = teacache_forward`. Replaces the forward method on the class itself, affecting all instances globally.
- **State:** stored as **class attributes** (`pipeline.transformer.__class__.cnt`, `…rel_l1_thresh`, etc.), not instance attributes.
- **No proxy/wrapper module.** They edit the original module's forward in place.
- **Threshold-zero behavior:** no explicit short-circuit; the `should_calc` check would always pass at threshold=0, so every step recomputes. They still build `previous_residual` but rely on the natural code path.
- **Tests:** none. No parity assertions. They show visual side-by-side and quote 2× speedup; quality validation is by eye.

### ComfyUI-TeaCache (welltop-cn)

[`ComfyUI-TeaCache/nodes.py`](https://github.com/welltop-cn/ComfyUI-TeaCache/blob/main/nodes.py)

- Uses ComfyUI's patch-node system (`ApplyTeaCachePatch` with `ForwardOverrider` nodes). Architecturally similar to ali-vilab's class patch but routed through ComfyUI's node graph.
- No bit-exact parity tests. Quality validation via visual output in workflows.

### HuggingFace Diffusers FirstBlockCache (`a-r-r-o-w`, PR #11180)

[`src/diffusers/hooks/first_block_cache.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/hooks/first_block_cache.py)

- **Patch mechanism:** custom `HookRegistry` system — `registry.register_hook(hook, _FBC_LEADER_BLOCK_HOOK)`. Registers a `ModelHook` base-class object that intercepts the module's forward via `ModelHook.new_forward`. Conceptually similar to PyTorch's `register_forward_hook` but implemented in-library since they need more than vanilla PyTorch hooks expose.
- **Threshold behavior:** the leader-block hook computes diff against the previous step; if `diff > threshold` it skips remaining blocks (the inverse direction of ours). At very low thresholds (`threshold → 0`) the cache effectively never engages and computation always runs.
- **Tests:** **`FirstBlockCacheTesterMixin` in `tests/pipelines/test_pipelines_common.py`** (per 2026-05-15 audit). Builds CPU dummy pipelines, runs 4-step generation without cache, then with cache enabled, then after disabling. Asserts `np.allclose(..., atol=0.1)` for cache-enabled and `np.allclose(..., atol=1e-4)` for cache-disabled. **Tolerant, in-process, dummy-pipeline regression test.** No bit-exact full-latent fixture parity for real models.

### Pattern across all three

**No upstream diffusion-caching project tests bit-exact equivalence against a real-model committed fixture.** What they do:
- ali-vilab / ComfyUI-TeaCache: visual side-by-side, no pytest-style correctness tests.
- Diffusers: tolerant `np.allclose` in-process tests on CPU dummy pipelines; quality comparison in the PR for real models.

Our spec v2.5 specified bit-exact parity at threshold=0 against a real-model fixture as the canonical correctness gate. **None of the upstream implementations have an equivalent claim**, and on MLX/Metal we now have direct evidence it isn't achievable. The python-ml-testing skill's tolerance table assumes PyTorch CUDA where `atol=1e-5/rtol=1e-5` *is* achievable; that assumption doesn't carry over.

## MLX has no register_forward_hook

PyTorch's `register_forward_hook` / `register_forward_pre_hook` are the natural mechanism diffusers uses (with extensions). **MLX has no equivalent in `nn.Module`.** The MLX docs page on `Module` lists `parameters`, `update`, `freeze`, `apply`, etc., but no hook registration. Our `ProxyFlux1Transformer` is the natural MLX-equivalent — it's the same idea (intercept forward, delegate everything else) without language-level support.

That means diffusers' "lighter touch" approach isn't available to us. The choices that exist (audit-corrected list):

1. **Per-instance proxy** (our current approach — replace `flux.transformer` with `Proxy(_inner=original)` where `Proxy.__class__` defines `__call__`).
2. **Class-level monkey patch** (ali-vilab's approach — `Transformer.__call__ = teacache_forward`). Pollutes the class globally within the process; multiple Flux1 instances share state.
3. **Dynamic per-instance subclass** — mutate `flux.transformer.__class__` to a generated subclass whose `__call__` is patched. Risky and likely not worth it.
4. **Change mflux call sites.** Outside this package's integration boundary.

(The previous draft listed "method swap on instance — `flux.transformer.__call__ = bound_wrapper`" as an option. That doesn't work: Python looks up dunder methods on the *type*, not the instance, so `flux.transformer(...)` would never see the patched `__call__`. Dropped per audit.)

## Hypotheses for what's perturbing MLX dispatch

We have *not* yet pinpointed the cause of the cosine-0.18 shift. Candidates ranked by suspicion:

1. **`ProxyFlux1Transformer` replacing `flux.transformer`** — even though our proxy delegates 1:1 at threshold-zero, it's a *different Python object*. mflux's `self.transformer(...)` call lands on `Proxy.__call__` → `flux1_forward_with_gate` (a regular function), rather than `Transformer.__call__` (an `nn.Module.__call__` dispatch on the original module). Different Python-frame topology around the per-step compute. Even though nn.Module.__call__ is itself just method dispatch, the proxy is a *new* nn.Module instance — its dict is empty, parameters delegation passes through — and Python's identity / reference graph differs.
2. **`apply_teacache` adds extra global state**: handle objects, ref-cycles between handle and flux, `_teacache_handle` sentinel attribute, etc. Some of this state lives across the generation loop and may affect MLX's tracking of array refcounts (per the addendum's mechanism).
3. **`_GenerationContextCallback` registered on flux.callbacks**: even though it has no `call_in_loop`, it does run `call_before_loop` once before the diffusion loop starts. That extra Python work + reference holding might shift MLX's first-allocation state for the run.
4. **`wrap_generate_image` wrapping flux.generate_image**: adds a Python frame around the entire generation. Mostly affects pre/post bookkeeping, less likely to perturb per-step dispatch.

The minimal experiment to localize this: in one Python script, apply only ONE of {proxy alone, callback alone, generate_image-wrap alone} and measure the diff against the committed fixture. Each isolation needs a single 25-step generation (~2.5 min). Total ~10 min walltime + script writing.

## Implications for test design

Given:
- Bit-exact across scripts is not achievable.
- `assert_allclose(atol=1e-5)` is far too tight (cosine 0.18 / max_abs 5.8 is real and unavoidable).
- Loosening `atol` to ~6.0 would accept arbitrary garbage; useless as a correctness gate.
- Within-process wrapper-vs-vanilla IS bit-exact.
- Upstream projects don't use bit-exact gates; they use quality metrics.

The viable options:

### Option A — intra-process parity (recommended)
The parity test loads `flux1_dev` once per session. For each prompt: capture vanilla output (call `flux1_dev.transformer` directly without applying TeaCache) and wrapper output (with `apply_teacache(rel_l1_thresh=0.0)`) in the SAME process. Assert `mx.array_equal(wrapper_out, vanilla_out)`. This works because intra-process MLX dispatch is consistent.

The committed `.safetensors` fixtures stay for fixture-integrity drift detection (already in `test_fixtures_integrity.py`), but are NOT the parity oracle.

**Pros:** strict bit-exact gate that catches real math regressions; no false positives from cross-script MLX drift.
**Cons:** can't pre-bake fixtures and ship them; each parity test must re-run vanilla, ~2.5× generation cost per prompt.

### Option B — image-level metric oracle (upstream-style)
Decode the final latent through the VAE on both vanilla and wrapper paths, compute SSIM or LPIPS, assert the metric is below a threshold (e.g., `LPIPS < 0.01`). Matches how ali-vilab and diffusers validate.

**Pros:** robust against fp ordering; matches industry practice; tests what users actually care about (image quality).
**Cons:** adds VAE decode (~2 GB more memory + a few seconds per prompt); LPIPS dependency; thresholds need calibration.

### Option C — class-level monkey patch (ali-vilab style)
Refactor to match ali-vilab's approach: monkey-patch `Transformer.__call__` directly on the mflux class, store TeaCache state as class attributes, restore by reassignment. Eliminates the proxy and most of `apply_teacache`'s state. **May or may not restore bit-exact parity** (we'd have to measure). Pollutes mflux globally inside the Python process; multiple Flux1 instances share state.

**Pros:** removes proxy as a perturbation source.
**Cons:** global mutation is risky; mflux is `nn.Module`-based and class-level patches need careful state management.

### Option D — accept "verified by inspection" and skip parity tests
Document the bit-exact-is-unsafe conclusion in the spec, replace `test_parity_flux1.py` with a `test_smoke_flux1.py` that just checks "generation runs to completion without errors and produces a finite latent", and rely on `test_fixtures_integrity.py` for fixture-pinning. Adds zero correctness gate beyond unit tests.

**Pros:** simplest.
**Cons:** no parity gate at all; relies entirely on unit-test coverage.

## Recommendation

**Decision (2026-05-15): use both A and B.** A is the fast strict gate; B is the slower visual-quality gate that matches industry practice. Together they cover "wrapper math is identical to upstream when configured to be" (A) and "wrapper output is perceptually equivalent to upstream end-to-end, including the VAE decode" (B).

**Audit-corrected design** (per `2026-05-15-task-25-fast-path-measurement-audit.md`):

- **A** = paired same-process parity:
  ```
  vanilla_before = capture(flux, prompt=...)
  with apply_teacache(flux, rel_l1_thresh=0.0) as h:
      wrapper = capture(flux, prompt=...)
  vanilla_after = capture(flux, prompt=...)

  assert mx.array_equal(vanilla_before, wrapper)
  assert mx.array_equal(vanilla_before, vanilla_after)   # restore control
  assert h.stats.skipped_count == 0
  ```
  The `vanilla_after` control is critical — it proves `handle.restore()` returned the shared model fixture to the same observable state and catches hidden callback/proxy/sentinel leakage. For at least one prompt, also test the reverse order (`wrapper → restore → vanilla`) to guard against warm-state ordering bias.
- **B** = image-level metric on a same-process baseline (NOT against committed fixtures): decode both vanilla and wrapper latents through the VAE, assert SSIM / LPIPS within calibrated thresholds. Per python-ml-testing skill: full VAE SSIM ≥ 0.90, LPIPS ≤ 0.10. Calibrate against measured same-process diffs first; don't hard-code without measurement.

**Cost (audit-corrected)**: a paired vanilla + wrapper test is ~2× generation time per prompt, i.e. ~5 min/prompt at 25 steps. Five prompts = ~25 min walltime, not ~12 min. Use one prompt for PR-level parity and all five (gated by `slow` or a dedicated mark) for the nightly / release gate.

We should still investigate which `apply_teacache` step causes the cross-process shift (the localization experiment below) — but as a **diagnostic curiosity**, not as a blocker for landing A+B. The audit's diagnostic matrix (cases A-G) is the cleanest plan if we want causality.

Committed `.safetensors` fixtures remain — but only as **drift fingerprints**, not as a correctness gate. `tests/test_fixtures_integrity.py` exact-SHA-checks them on every run (catches accidental fixture drift on disk). `tests/test_hf_revisions.py` checks the HF cache hasn't drifted out from under us. Neither is a correctness oracle for the wrapper.

## Open files (uncommitted)

- `tests/test_parity_flux1.py` (untracked) — still uses `mx.array_equal` against the committed fixture. Needs the redesign.
- `docs/superpowers/notes/2026-05-14-task-25-fast-path-measurement.md` (this file).

## Audit corrections applied (2026-05-15)

Per [`2026-05-15-task-25-fast-path-measurement-audit.md`](2026-05-15-task-25-fast-path-measurement-audit.md):

- **Diffusers DOES test FirstBlockCache** — `FirstBlockCacheTesterMixin` in `tests/pipelines/test_pipelines_common.py`. Tolerant `np.allclose` (atol=0.1 enabled, atol=1e-4 disabled) on CPU dummy pipelines. Updated in § "Pattern across all three" above.
- **Instance-level `__call__` swap is not viable**: Python looks up dunder methods on the type, not the instance. Removed from the options list.
- **Forward.py fast-path docstring softened** (commit pending) — measurement showed the cached_residual ref-keeping was NOT the dominant perturbation; rewording avoids overclaim.
- **Cost estimate corrected**: paired same-process parity is ~25 min for 5 prompts, not ~12 min.
- **Recommendation tightened**: A + B both, with the `vanilla_after_restore` control and reverse-order check per audit.

## References

- [ali-vilab/TeaCache/TeaCache4FLUX/teacache_flux.py](https://github.com/ali-vilab/TeaCache/blob/main/TeaCache4FLUX/teacache_flux.py) — class-level monkey patch, class-attribute state, no parity tests.
- [welltop-cn/ComfyUI-TeaCache nodes.py](https://github.com/welltop-cn/ComfyUI-TeaCache/blob/main/nodes.py) — ComfyUI patch-node integration.
- [HuggingFace Diffusers FirstBlockCache (`src/diffusers/hooks/first_block_cache.py`)](https://github.com/huggingface/diffusers/blob/main/src/diffusers/hooks/first_block_cache.py) — custom HookRegistry; no parity tests in the file.
- [Diffusers caching API docs](https://huggingface.co/docs/diffusers/api/cache) — PyramidAttentionBroadcast, FasterCache, FirstBlockCache, TaylorSeer all hook-based.
- [Diffusers PR #11180 (FirstBlockCache integration)](https://github.com/huggingface/diffusers/pull/11180) — validation is via visual comparison.
- [MLX `nn.Module` docs](https://ml-explore.github.io/mlx/build/html/python/nn/module.html) — no `register_forward_hook` / `register_forward_pre_hook`.
- python-ml-testing skill, § 7 — image-gen metric thresholds (SSIM ≥ 0.90 for full VAE; LPIPS ≤ 0.10) — option B's calibration baseline.
- Earlier notes: `2026-05-14-task-25-mlx-nondeterminism.md`, `…-addendum.md`.
