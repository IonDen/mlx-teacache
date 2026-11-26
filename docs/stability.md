# Stability

This page describes the policy planned for 1.0. It is not in force yet. Until 1.0, a 0.x minor release may still change things, and it warns first when it does.

## The promise

Your code keeps working across 1.x. How many steps TeaCache skips, and the image you get, can change when mflux or
MLX changes the model's numbers, or when a minor release of mlx-teacache changes its decision rule (how the gate
decides, how the default threshold is picked, the cap on skips in a row, or which steps are never skipped). A patch
release never changes them on the same mflux and MLX. Each handle records the decision-rule revision and the mflux
version (`handle.decision_rule_revision`, `handle.mflux_version`), so you can tell which one changed. A later
mflux-teacache release will also write both into the metadata of the images its commands save. mflux-teacache, the
command-line plugin, is still 0.x and makes its own promise.

## What is covered

Everything in `mlx_teacache.__all__` and in `mlx_teacache.testing`: the names, their signatures, the stats types, the values of the `decision` field of `StepDecision`, and the exception and warning classes. Code that imports these and uses them as documented keeps working through 1.x.

`restore()` can refuse, and it raises `RestoreRefusedError` when it does. It refuses in two cases:

- A generation still has steps to run. Call `restore()` after `generate_image` returns, or from a callback that runs after the loop.
- `flux.generate_image` or `flux._predict` was replaced after `apply_teacache`. Restoring would remove that replacement. Undo your replacement first, then call `restore()` again.

The `with apply_teacache(...) as handle:` form calls `restore()` when the block exits. If a later wrapper is still installed at that point, the exit raises `RestoreRefusedError`, and that error replaces any exception raised inside the block.

A refusal undoes nothing, so the handle stays as it was and you can call `restore()` again later. The error has a `reason`, either `"generation-in-flight"` or `"patched-over"`. For `"patched-over"`, `attribute` names the replaced method. For `"generation-in-flight"`, `remaining_steps` and `active_num_steps` give the steps left and the steps in the generation. The attributes that do not apply to the reason are `None`.

One case is not caught. Say a callback was registered before `apply_teacache` and calls `restore()` just before the denoising loop starts. `restore()` is not refused then, no error is raised, and that generation runs without TeaCache. Register such a callback after `apply_teacache` and `restore()` refuses it, or call `restore()` outside a generation.

## What is not promised

- Skip counts and images across minor releases. See the promise above for when they can change.
- The text of warning messages, and which events produce a warning.
- `mlx_teacache._kernel` and `mlx_teacache.variants`.
- The root modules `mlx_teacache.gate`, `mlx_teacache.cache`, `mlx_teacache.stats` and `mlx_teacache.coefficients`. Import the public names from `mlx_teacache` instead.
- The `TeaCacheHandle` constructor and `VariantPatch`. Get a handle from `apply_teacache`.
- The formats of script and calibration files.
- mflux's own types.

## Deprecations

A removal always warns for at least one minor release first, and it happens only at a major release. A value that no release produces gets a CHANGELOG notice instead of a run-time warning. The removals marked 1.0 below land with the first 1.0 release candidate. The 0.16 deprecations:

| Deprecated | Use instead | Timing |
|---|---|---|
| `TeaCacheStats(cfg_fallback_steps=...)` and `TeaCacheStats.cfg_fallback_steps` (still accepted, with a warning) | `GenerationStats.cfg_was_active` | removed in 1.0 |
| `TeaCacheStats.total_steps_seen` | `total_active_steps` (same number, since CFG fallback ended in 0.4.1) | removed in 1.0 |
| the `"cfg-fallback"` decision string (not produced since 0.4.1; no runtime warning, this table and the CHANGELOG are the notice) | none | dropped from the decision vocabulary in 1.0 |
| `mlx_teacache.gate`, `.cache`, `.stats`, `.coefficients` | the public names in `mlx_teacache`; the rest have no public replacement | outside the 1.0 promise, removed in 2.0 |
| setting `handle.rel_l1_thresh`, `handle.stats`, `handle.provenance` or `handle.coefficients` (the gate never read the written values; `coefficients` was a writable attribute in 0.15) | pass the values to `apply_teacache` | an error from 1.0, when the handle becomes read-only |
| `InvalidStepWindowError(num_steps=...)` | `active_num_steps=` | removed in 2.0 |

Writing `handle.variant_id` warns from 0.18 (see the table below). The CHANGELOG lists each deprecation in the release that introduced it.

Added in 0.18:

| Deprecated | Use instead | Timing |
|---|---|---|
| `apply_teacache(..., coefficients=...)` without `rel_l1_thresh` (warns with `FutureWarning`) | pass `rel_l1_thresh` with the coefficients | an error from 1.0 |
| setting `handle.variant_id` (it never changed which variant was patched; it warns with `DeprecationWarning`, and the value is still stored) | read `handle.variant_id`; it is set when `apply_teacache` returns | an error from 1.0, when the handle becomes read-only |

The `coefficients`-without-threshold warning is a `FutureWarning`, which Python shows by default, so you will see it without any flag. It names the threshold `apply_teacache` picked: 0.30 on FLUX.1 Krea [dev] and 0.20 on every other model. Under `-W error` or pytest's `filterwarnings = error` the call raises before the model is patched, so you can retry on the same model.

Python hides a `DeprecationWarning` unless it is raised from `__main__`, so a library user may not see these. To see them, run Python with `-W default::DeprecationWarning` or set `PYTHONWARNINGS=default::DeprecationWarning`. pytest shows them without any flag.

## Reading the decision-rule revision

Each handle carries the revision of the decision rule and the mflux version it was built on:

```python
handle = apply_teacache(flux)
print(handle.decision_rule_revision, handle.mflux_version)
```

Four things bump the revision: a change to how the gate decides, to how the default threshold is picked, to the cap on skips in a row, or to which steps are never skipped. That only happens in a minor release. Its CHANGELOG entry names the variant and gives the old and new skip count from a committed benchmark.

## Which variant is chosen

mlx-teacache keeps an ordered registry of variants and uses the first one that matches the model. Every real model configuration matches at most one variant.

## mflux and MLX

Dependencies have a floor, never a ceiling. A newer mflux installs, and `apply_teacache` emits `TeaCacheUntestedMfluxWarning` once per process until that release has been checked. TeaCache still applies.

`apply_teacache` and `match_variant` take mflux's objects. On an mflux that has not been checked, a variant may stop matching, or `apply_teacache` may raise `IncompatibleModelError`. From 0.18.1, a generation on Z-Image or FLUX.2 Klein can also stop with it. That is the documented outcome for a model mlx-teacache does not support, and it does not break this policy.

Raising the mflux, MLX or Python floor is a minor release, and its CHANGELOG entry names the last mlx-teacache version that supports the old floor.

The base install asks for MLX 0.31 or newer with no upper bound. On an `[mflux]` install, mflux decides the MLX version (for mflux 0.24 that is 0.32, below 0.33). The test suite runs on the MLX that mflux 0.24 installs.

## What a parity pass means

A parity pass is a result from one machine and one mflux version. It shows the images matched on that machine and says nothing about others.
