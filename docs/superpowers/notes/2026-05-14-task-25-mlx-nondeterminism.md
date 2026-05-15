# Task 25 investigation — MLX non-determinism kills bit-exact parity

**Date:** 2026-05-14
**Status:** Investigation complete; fix not yet implemented.
**Trigger:** Task 25 FLUX.1 parity tests — `test_threshold_zero_bit_exact_dev[*]` failed 5/5; schnell passed 5/5.

> See also: `2026-05-14-task-25-mlx-nondeterminism-addendum.md`, which corrects the causal mechanism and refines the test-policy recommendations in this note. Read both together.

## TL;DR

`forward.py` is mathematically correct. The bit-exact gate in spec v2.5 is **not achievable** on Apple Silicon — MLX/Metal kernel dispatch is sensitive to reference counts, buffer donation eligibility, and shape/env-dependent kernel selection, and our wrapper's extra arrays (`body_in_concat`, `mod_in`, `cached_residual`) perturb those even at threshold=0 where the extra work is dead weight. The test should switch to a measured numerical-tolerance gate, and `forward.py` should grow a true threshold-zero fast path that builds none of the cache tensors.

## What happened, in order

1. Parity test reported 5/5 dev bit-exact failures, 5/5 schnell bit-exact passes, 5/5 dev cosine-similarity failures.
2. Three real pre-existing bugs were uncovered and fixed along the way (commits `8648686`, `38ca4ca`, `3b93d34`):
   - `detect.identify_variant` used `model_name` instead of `aliases` to identify variants — wrong for mflux 0.17.
   - `_callback_present_by_identity` / `_remove_callback_by_identity` checked the wrong attribute names on mflux's CallbackRegistry (`*_callbacks` are methods, not lists).
   - `_flux1_extract_mod_input` called `AdaLayerNormZero` with an `emb=` kwarg that doesn't exist.
3. A rogue subagent overwrote the 5 committed flux1-dev reference fixtures with its TeaCache-wrapped outputs while trying to "fix stale fixtures". We restored from `git checkout`. A new rule landed in CLAUDE.md (heavy generations run in main thread, not subagents).
4. HF cache for FLUX.1-dev was missing snapshot symlinks for shards 1+2; user repaired via `hf download`.
5. We added `tests/hf_revisions.toml` + `tests/test_hf_revisions.py` to detect upstream HF model drift (commit `595fb8c`).
6. Then we hit the real, irreducible issue: bytes differ across processes even when the math is equivalent.

## Investigation chain

| Setup | Final latent SHA |
|---|---|
| Task 24 commit (vanilla, only `call_after_loop` callback) | `45471c34...` |
| Today: `generate_references.py --variant flux1-dev` (vanilla, only `call_after_loop`) | `45471c34...` |
| Today: dev with `apply_teacache(threshold=0)` + only `call_after_loop` | `0db096b2...` |
| Today: dev with `apply_teacache(threshold=0)` skip=0,0 vs skip=1,1 | both `0db096b2...` |
| Today: vanilla (no TeaCache) with a callback having both `call_in_loop` + `call_after_loop` | `0db096b2...` |
| Today: wrapper with the same dual-callback shape | `0db096b2...` |
| Today: per-step in-loop diff (vanilla vs wrapper, single process) all 25 steps | bit-exact ✅ |

Within a single Python process, vanilla and wrapper agree per-step. Across two different processes whose only material difference is the *shape of the registered callback class* (whether it defines `call_in_loop`), the final bytes diverge by cosine 0.18.

## Why this happens — the corrected mechanism

The original framing of this note said "MLX's discovery walk sees the cached residual branch." That's wrong. Per inspection of `mlx/transforms.cpp` (`eval_impl` builds a synchronizer rooted on the requested outputs, then DFS-walks *inputs* of those outputs), MLX evaluation does not traverse from an intermediate to arrays that happen to consume the intermediate elsewhere. If `state.cached_residual` isn't downstream of `latents`, it isn't scheduled.

The actual mechanism is **reference-count-driven buffer donation**:

1. MLX's `array` is a shared graph node. `array::is_donatable()` returns true only when the array descriptor and data buffer are uniquely owned. Metal backend helpers (binary ops, normalization, etc.) use this to decide whether an output can reuse an input buffer in-place instead of allocating fresh.
2. Threshold-zero TeaCache currently still builds:
   - `body_in_concat = mx.concatenate([encoder_hidden_states, body_in], ...)`
   - `mod_in = _flux1_extract_mod_input(block_0, body_in, text_embeddings)` (runs `block_0.norm1`)
   - `state.cached_residual = body_out_concat - body_in_concat`
   - assigns `state.previous_mod_input = mod_in`
3. The `cached_residual` array stores `body_out_concat` and `body_in_concat` in its `inputs`. Even when nothing ever consumes `cached_residual`, the residual's existence increases the ref count of `body_out_concat` and `body_in_concat`, which lives until garbage collection or `state` is rewritten on the next step.
4. The tail (`norm_out`, `proj_out`) then runs with a different donation eligibility for the arrays at the body/tail boundary. Buffer aliases shift, temporary lifetimes shift, Metal command-buffer contents shift.
5. Combined with shape/device/env-dependent kernel dispatch (matmul, SDPA, normalization all branch on shape and Metal heuristics), the resulting fp summation order changes. Non-associativity over 25 steps amplifies tiny per-step deltas into cosine 0.18.

This is not a wrapper math bug; it's a memory-and-dispatch interaction. The fix is to stop creating the extra arrays when they're not needed.

## What MLX docs actually say

Important to be precise here: the official MLX material does not contain a sentence "MLX is non-deterministic." What it does document, and what supports the inference above:

- Lazy dynamic graphs, built on the fly, evaluated only when an output is requested.
- Output-rooted evaluation (DFS from requested outputs through `inputs`, then BFS with a width limit for the execution tape).
- Random keys control PRNG state; they do not promise deterministic kernel reductions.
- Shape/device/env-dependent kernel dispatch (matmul, SDPA, norm — confirmed in MLX source at `046217bcae7347aa814665f39a8f0e404029ddb0`).
- Reference-count-based donation (`array::is_donatable`).

There is no public MLX deterministic-mode API analogous to PyTorch's `use_deterministic_algorithms`. Environment knobs like `MLX_BFS_MAX_WIDTH`, `MLX_ENABLE_TF32`, `MLX_SDPA_BLOCKS` are scheduler/kernel controls, not a determinism contract.

The third-party article ([adityakarnam.com/mlx-non-determinism-apple-silicon](https://adityakarnam.com/mlx-non-determinism-apple-silicon/)) and the [`mlx-deterministic`](https://github.com/ProbioticFarmer/mlx-deterministic) project are useful supporting evidence that the bit-exactness gap is real and acknowledged in the community, but they should not be cited as the sole basis for project policy.

## Recommended fix to `forward.py`: true threshold-zero fast path

At `rel_l1_thresh <= 0.0`, the gate already short-circuits to "computed, update cache." But the current wrapper still computes `mod_in`, `body_in_concat`, and `cached_residual` that the gate's decision says are unused. Skip them.

Proposed behavior (FLUX.1 path; FLUX.2 mirrors):

- If `handle.rel_l1_thresh <= 0.0`, bypass TeaCache gating entirely:
  - Do not compute `mod_in`.
  - Do not build `body_in_concat`.
  - Do not compute or store `cached_residual`.
  - Do not update `previous_mod_input`.
  - Record a `StepDecision(kind="computed", rel_l1=None, accumulated_distance=0.0)` for stats.
  - Run the vanilla-equivalent prelude/body/tail.
- Keep `state.reset_for_new_generation(...)` and the skip-window validation at `t == 0`.
- Optionally tighten `gate_step` so that the threshold-zero return also sets `should_update_cache=False` (the cache can never be consumed at non-positive threshold).

Add a unit test that asserts: after a threshold-zero forward, `state.cached_residual is None` and `state.previous_mod_input is None`.

This may or may not restore byte identity for the current committed fixtures. The shape/dispatch ecosystem still depends on the parent flux module's behavior, and Metal dispatch can still vary. Don't sell the fast path as bit-exact. It is, however, a real performance win at threshold=0 (no useless extra arrays, no useless subtractions) and it shrinks the numerical surface area we're testing.

## Recommended test policy

### 1. Replace byte identity with calibrated tolerance — but measure first

Move from `mx.array_equal` to `np.testing.assert_allclose(np.asarray(out), reference, ...)` or `mx.allclose(out, reference, ...)`. Provisional starting tolerances, treating fp32 latents in a 25-step diffusion: `rtol=1e-5, atol=1e-5`. (`mx.allclose`'s default is `rtol=1e-5, atol=1e-8`; bump `atol` for stable behavior near zero.)

Crucially: **measure the actual threshold-zero diff against the 5 committed prompts on the target hardware before locking the threshold in**. If `assert_allclose` fails at `1e-5/1e-5` for one prompt, do not blindly relax — first implement the threshold-zero fast path, regenerate the diff distribution, then choose. The tolerance should be the smallest value that absorbs Metal-level fp noise without absorbing a real bug.

Write a helper that reports shape, dtype, max-abs, mean-abs, max-rel (with epsilon denominator), cosine similarity, and reference/output SHAs. Use it both as the assertion engine and as the diagnostic when a test fails.

### 2. Split fixture integrity from parity correctness

- `tests/test_fixtures_integrity.py` (already exists): exact SHA-256 of committed `.safetensors`. This is drift detection for the file on disk, nothing more.
- `tests/test_parity_flux1.py` (untracked): tolerance-based comparison against the loaded reference. The fixture isn't a byte-level oracle; it's a numerical-truth oracle within a measured tolerance.

The two tests answer different questions: "did the file change?" vs. "is the wrapper numerically equivalent?".

### 3. Keep a same-process diagnostic for real math bugs

Within one Python process, run vanilla and wrapper with the same callback shape and compare per-step. This is bit-exact (Metal makes the same kernel choices within a process), so it's a fast tripwire for actual forward.py math errors. Mark it as a diagnostic or `slow` test, not the main CI oracle.

### 4. Default-threshold tests stay metric-based

For `rel_l1_thresh > 0` the wrapper is intentionally approximate. Cosine similarity ≥ 0.985 and skip-count/stats invariants stay the right gate. Don't require `assert_allclose` for non-zero thresholds.

## What NOT to do

- Don't `mx.eval(state.cached_residual)` to "force the graph to settle". The ref-count donation issue remains because the array still holds references to its inputs after eval.
- Don't introduce `mlx-deterministic` for v0.1 — 7–31% overhead defeats the speedup goal.
- Don't claim "MLX docs say it's non-deterministic". The docs document the building blocks; the non-determinism is an inference from them plus this experiment. Cite the building blocks.
- Don't keep the bit-exact gate hoping fp ordering will be stable. Even on the same machine, the same operation can dispatch differently depending on graph shape.

## Open files (uncommitted)

- `tests/test_parity_flux1.py` — 12 tests, ruff-clean, currently uses `mx.array_equal`. Needs the tolerance update (and ideally per-prompt diff measurement) before commit.

## References

- Addendum: `docs/superpowers/notes/2026-05-14-task-25-mlx-nondeterminism-addendum.md` — corrects the mechanism and supplies MLX source anchors.
- [MLX README — lazy computation and dynamic graphs](https://github.com/ml-explore/mlx#mlx)
- [MLX Lazy Evaluation docs](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html)
- [MLX random docs (explicit PRNG keys)](https://ml-explore.github.io/mlx/build/html/python/random.html)
- [MLX `allclose` docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.allclose.html)
- MLX source (@`046217bcae7347aa814665f39a8f0e404029ddb0`):
  [`transforms.cpp` — output-rooted DFS + BFS tape + dispatch + detach](https://github.com/ml-explore/mlx/blob/046217bcae7347aa814665f39a8f0e404029ddb0/mlx/transforms.cpp#L74-L344),
  [`array.h` — graph-node + `is_donatable()`](https://github.com/ml-explore/mlx/blob/046217bcae7347aa814665f39a8f0e404029ddb0/mlx/array.h#L293-L296),
  [`backend/common/binary.h` — donation-sensitive output storage](https://github.com/ml-explore/mlx/blob/046217bcae7347aa814665f39a8f0e404029ddb0/mlx/backend/common/binary.h#L37-L95),
  [`backend/metal/matmul.cpp` — shape/env-dependent GEMM dispatch](https://github.com/ml-explore/mlx/blob/046217bcae7347aa814665f39a8f0e404029ddb0/mlx/backend/metal/matmul.cpp#L841-L1036),
  [`backend/metal/scaled_dot_product_attention.cpp` — shape/env-dependent SDPA dispatch](https://github.com/ml-explore/mlx/blob/046217bcae7347aa814665f39a8f0e404029ddb0/mlx/backend/metal/scaled_dot_product_attention.cpp#L646-L788).
- [The Hidden Problem With MLX (Aditya Karnam)](https://adityakarnam.com/mlx-non-determinism-apple-silicon/) — supporting evidence, not a primary source.
- [`mlx-deterministic` (batch-invariant Metal kernels)](https://github.com/ProbioticFarmer/mlx-deterministic) — 7-31% overhead; not appropriate for v0.1.
- [TeaCache upstream (ali-vilab)](https://github.com/ali-vilab/TeaCache) — confirms our caching approach matches the paper.
- python-ml-testing skill, § 3 (Numerical tolerances).
- user-mlx-developer skill, `references/gotchas.md` § 24 and `references/mflux-and-local-projects.md` (updated with this incident).
