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

Writing `handle.variant_id` does not warn in 0.16; a warning for it comes in a later minor release. The CHANGELOG lists each deprecation in the release that introduced it.

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

`apply_teacache` and `match_variant` take mflux's objects. On an mflux that has not been checked, a variant may stop matching, or `apply_teacache` may raise `IncompatibleModelError`. That is the documented outcome for a model mlx-teacache does not support, and it does not break this policy.

Raising the mflux, MLX or Python floor is a minor release, and its CHANGELOG entry names the last mlx-teacache version that supports the old floor.

The base install asks for MLX 0.31 or newer with no upper bound. On an `[mflux]` install, mflux decides the MLX version (for mflux 0.24 that is 0.32, below 0.33). The test suite runs on the MLX that mflux 0.24 installs.

## What a parity pass means

A parity pass is a result from one machine and one mflux version. It shows the images matched on that machine and says nothing about others.
