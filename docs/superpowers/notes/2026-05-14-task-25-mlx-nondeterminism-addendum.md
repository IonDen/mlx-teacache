# Task 25 addendum - MLX nondeterminism and parity-test design

**Date:** 2026-05-14
**Author:** Codex research addendum
**Scope:** Follow-up on `2026-05-14-task-25-mlx-nondeterminism.md`.

## Executive conclusion

The original note's practical conclusion is right: `threshold=0` should not be
specified or tested as cross-process byte identity on MLX/Metal. The test suite
should move from `mx.array_equal` to a measured numerical tolerance gate.

The most valuable correction is the causal explanation. MLX evaluation walks
backward from requested outputs, so an unreturned TeaCache residual is not
literally a downstream graph branch that the DFS traverses. The better
mechanism is:

1. MLX constructs dynamic lazy graphs and evaluates only when an output is
   requested.
2. `mlx-teacache` threshold-zero currently still builds extra MLX arrays:
   `body_in_concat`, `mod_in`, and `state.cached_residual =
   body_out_concat - body_in_concat`.
3. The cached residual array holds references to `body_out_concat` and
   `body_in_concat` through its `inputs`.
4. MLX uses reference-count based buffer donation. Extra live references can
   change whether inputs are donated/reused or freshly allocated.
5. That can alter memory aliases, temporary lifetimes, command-buffer contents,
   and sometimes kernel dispatch conditions without changing the mathematical
   forward pass.

This is a more concrete, testable explanation for why wrapper and vanilla can
be numerically equivalent but not byte-identical. I did not prove this donation
path with a Metal capture in this sandbox, so treat it as the leading mechanism
to verify rather than a completed root-cause proof.

## What I verified

### MLX does not promise byte reproducibility

The official README says MLX computations are lazy and graphs are constructed
dynamically. The lazy-evaluation docs state that operations record a compute
graph and computation happens only on `eval()`. They also warn that unused
outputs are not computed, but the graph for unused outputs is still built.

Official docs also document deterministic PRNG keys, but that only controls
random number generation. It does not guarantee deterministic floating-point
kernel reductions, dispatch ordering, memory reuse, or byte identity of an
entire diffusion inference graph.

I did not find an official MLX deterministic-mode API analogous to PyTorch's
`use_deterministic_algorithms`. MLX does expose environment knobs such as
`MLX_BFS_MAX_WIDTH`, `MLX_ENABLE_TF32`, and `MLX_SDPA_BLOCKS`, but these are
scheduler/kernel-selection controls, not a determinism contract.

### Evaluation is output-rooted

In current MLX source, `eval_impl` creates a synchronizer whose inputs are the
requested outputs, then walks inputs by DFS to collect dependencies and later
builds a width-limited BFS execution tape. It does not walk "consumer" edges
from an intermediate to arrays that happen to use the intermediate elsewhere.

Implication: the sentence "MLX's discovery walk sees the cached residual branch"
is too strong. If the final latent does not depend on `state.cached_residual`,
then `state.cached_residual` should not be scheduled just because it exists.

The residual still matters because it keeps its input arrays alive and therefore
changes reference counts and donation eligibility.

### Reference-count donation is a credible mechanism

MLX `array` is a shared graph node. `array::is_donatable()` returns true only
when the array descriptor and data buffer are uniquely owned. Metal backend
helpers use that to decide whether an output can reuse an input buffer.

This matters for TeaCache:

```python
state.cached_residual = body_out_concat - body_in_concat
```

That residual array stores `body_out_concat` and `body_in_concat` in its inputs.
Even if the residual is never evaluated at threshold zero, it increases the
reference counts of exactly the arrays around the body/tail boundary. The tail
then runs with a different donation/memory-reuse situation than vanilla mflux.

This is not a semantic bug. It is a reason bit-exact parity is the wrong oracle.

### FLUX.1 mflux already evaluates once per denoising step

mflux 0.17.5 calls `mx.eval(latents)` inside the denoising loop after callbacks.
That means each denoising step is an evaluation boundary. `mlx-teacache` changes
the per-step graph and object-lifetime environment before that `mx.eval`.

This supports the observed finding that per-step in-process comparison can pass
while cross-process fixture byte identity fails.

## Recommended implementation change

Add a true threshold-zero fast path. This is worth doing even if the tests move
to tolerance, because it removes unnecessary work and shrinks the numerical
surface area.

Current threshold-zero behavior:

- computes `mod_in`;
- builds `body_in_concat`;
- computes and stores `cached_residual`;
- updates `previous_mod_input`;
- can never skip.

Proposed behavior:

- if `rel_l1_thresh <= 0.0`, bypass TeaCache gating entirely;
- do not compute `mod_in`;
- do not build `body_in_concat`;
- do not compute or store `cached_residual`;
- do not update `previous_mod_input`;
- record a "computed" decision for stats, with no rel-L1 distance;
- run the vanilla-equivalent prelude/body/tail.

Concretely:

1. In `gate_step`, consider changing the threshold-zero decision to
   `should_update_cache=False`, because the cache can never be consumed while
   threshold is non-positive.
2. In `flux1_forward_with_gate`, branch before `_flux1_extract_mod_input()` and
   before `body_in_concat = mx.concatenate(...)`.
3. Keep the existing reset and invalid skip-window behavior if stats/lifecycle
   require it, but avoid all cache tensors.
4. Add a unit test asserting threshold zero leaves `state.cached_residual is
   None` after a generation or after a small fake forward.

This may or may not restore byte identity for current fixtures. It should not be
sold as a bit-exact guarantee. It is still the cleaner implementation.

## Recommended test policy

### 1. Replace byte identity with calibrated tolerance

Use an assertion helper that reports:

- shape and dtype;
- `max_abs`;
- `mean_abs`;
- `max_rel` with an epsilon denominator;
- cosine similarity;
- reference SHA and output SHA if useful.

Start from `np.testing.assert_allclose(..., rtol=1e-5, atol=1e-5)` or
`mx.allclose(..., rtol=1e-5, atol=1e-5)`, but treat this as a provisional
threshold until measured against the five committed prompts on the target
machines. MLX's documented default for `mx.allclose` is `rtol=1e-5`,
`atol=1e-8`; the larger `atol=1e-5` is useful for values near zero.

If the measured threshold-zero diff exceeds `1e-5`, do not keep relaxing
blindly. First try the threshold-zero fast path above and log the diff
distribution.

### 2. Split fixture integrity from parity correctness

The committed `.safetensors` SHAs are good for fixture drift detection. They are
not a good byte-level oracle for wrapper correctness on MLX/Metal.

Suggested split:

- `test_fixtures_integrity.py`: exact SHA checks only for committed files.
- `test_threshold_zero_numerical_parity_*`: allclose gate against fixtures.
- optional diagnostic test/script: run vanilla and wrapper in the same process
  with identical callback shape and print diff metrics.

### 3. Keep a same-process diagnostic

A useful debug script should compare vanilla and wrapper latents in the same
Python process, with the same mflux callback shape, and report per-step metrics.
This catches real wrapper math mistakes while minimizing process-level MLX
scheduler variance.

It should be marked diagnostic or slow, not the main CI oracle.

### 4. Keep threshold-greater-than-zero tests metric-based

For the default TeaCache threshold, output is intentionally approximate. Use
cosine similarity and skip-count/stat invariants. Do not require allclose for
`rel_l1_thresh > 0`.

## What not to overclaim

Do not say official MLX documentation states "MLX is nondeterministic." The
official material I found establishes the necessary pieces:

- lazy dynamic graphs;
- output-rooted evaluation;
- width-limited scheduling;
- shape/device/env-dependent kernel dispatch;
- reference-count based donation;
- random keys only control PRNG state.

The stronger nondeterminism statement is an inference from those pieces plus
the observed FLUX.1 experiment. The third-party MLX nondeterminism article is
useful supporting evidence, but it should not be the only basis for project
policy.

## Source anchors

- MLX README, lazy computation and dynamic graph construction:
  https://github.com/ml-explore/mlx#mlx
- MLX lazy evaluation docs:
  https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html
- MLX random docs, explicit PRNG keys:
  https://ml-explore.github.io/mlx/build/html/python/random.html
- MLX `allclose` docs:
  https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.allclose.html
- MLX streams docs:
  https://ml-explore.github.io/mlx/build/html/usage/using_streams.html
- MLX source inspected at `046217bcae7347aa814665f39a8f0e404029ddb0`
  on 2026-05-14:
  - `mlx/transforms.cpp`: output-rooted DFS, BFS tape, dispatch, detach:
    https://github.com/ml-explore/mlx/blob/046217bcae7347aa814665f39a8f0e404029ddb0/mlx/transforms.cpp#L74-L344
  - `mlx/array.h`: graph-node comment and `is_donatable()`:
    https://github.com/ml-explore/mlx/blob/046217bcae7347aa814665f39a8f0e404029ddb0/mlx/array.h#L26-L28
    https://github.com/ml-explore/mlx/blob/046217bcae7347aa814665f39a8f0e404029ddb0/mlx/array.h#L293-L296
  - `mlx/backend/common/binary.h`: donation-sensitive binary output storage:
    https://github.com/ml-explore/mlx/blob/046217bcae7347aa814665f39a8f0e404029ddb0/mlx/backend/common/binary.h#L37-L95
  - `mlx/backend/metal/normalization.cpp`: donation-sensitive norm output
    storage:
    https://github.com/ml-explore/mlx/blob/046217bcae7347aa814665f39a8f0e404029ddb0/mlx/backend/metal/normalization.cpp#L220-L244
  - `mlx/backend/metal/matmul.cpp`: shape/device/env-dependent GEMM dispatch:
    https://github.com/ml-explore/mlx/blob/046217bcae7347aa814665f39a8f0e404029ddb0/mlx/backend/metal/matmul.cpp#L841-L1036
  - `mlx/backend/metal/scaled_dot_product_attention.cpp`: shape/device/env-
    dependent attention dispatch:
    https://github.com/ml-explore/mlx/blob/046217bcae7347aa814665f39a8f0e404029ddb0/mlx/backend/metal/scaled_dot_product_attention.cpp#L646-L788
- mflux 0.17.5 local source:
  - `mflux/models/flux/variants/txt2img/flux.py`: in-loop `mx.eval(latents)`.
  - `mflux/callbacks/generation_context.py`: callback timing.
