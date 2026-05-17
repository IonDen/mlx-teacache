# FLUX.2 Klein base-4B spec audit

**Date:** 2026-05-17  
**Spec:** `docs/superpowers/specs/2026-05-17-flux2-klein-base-4b-design.md`  
**Scope:** Review for real implementation and release-contract issues only.

## Findings

### 1. The central "FLUX.2 step-skipping" claim is only true for `guidance=1.0`, while the base-4B model card shows CFG-style usage

The spec frames `flux2-klein-base-4b` as the first FLUX.2 variant where mlx-teacache delivers algorithmic step-skipping (`spec:13`) and calibrates at `guidance=1.0` because the current wrapper falls back to vanilla at `guidance > 1.0` (`spec:38`).

That fallback is real: `src/mlx_teacache/integrations/mflux/flux2.py:101-129` routes CFG generations through `_vanilla_flux2_cfg_predict()` and records `cfg-fallback`, so no TeaCache gate runs when negative prompt embeddings are present. mflux creates those negative embeddings when `guidance > 1.0`.

The problem is that the upstream `black-forest-labs/FLUX.2-klein-base-4B` model card's Diffusers example uses `guidance_scale=4.0` and `num_inference_steps=50`. The card also describes base-4B as trained without step or guidance distillation, which makes CFG usage part of the expected base-model path, not an edge case.

Impact: a user following the official base-4B example gets no step-skipping with the current mlx-teacache FLUX.2 path. v0.4 can still ship no-CFG base-4B support, but the README/CHANGELOG claim must say that explicitly, or the design needs to add real FLUX.2 CFG caching before claiming normal base-4B acceleration.

Recommended spec change: make one of these explicit:

- v0.4 supports `flux2-klein-base-4b` TeaCache only for `guidance=1.0`; CFG/base-card usage remains fallback/no-acceleration.
- Or v0.4 includes per-branch CFG TeaCache for FLUX.2 and calibrates/tests the published `guidance_scale=4.0`, 50-step path.

### 2. The detect/API wiring section is wrong: adding `_SUPPORTED` is not enough

The spec says detection only needs `"flux2-klein-base-4b"` added to `_SUPPORTED` because the predicate is already `startswith("flux2-")` friendly (`spec:49-50`). Current code does not work that way.

Current detector contract:

- `src/mlx_teacache/integrations/mflux/detect.py:15` has `VariantId = Literal["flux1-dev", "flux1-schnell", "flux2-klein-4b", "flux2-klein-9b"]`.
- `src/mlx_teacache/integrations/mflux/detect.py:75-84` returns only when aliases contain `"flux2-klein-4b"` or `"flux2-klein-9b"`; every other `Flux2Klein` alias raises `IncompatibleModelError`.
- mflux does expose the base-4B config and aliases at `.venv/lib/python3.13/site-packages/mflux/models/common/config/model_config.py:371-399`.

Impact: implementing the spec literally will still reject `Flux2Klein(model_config=ModelConfig.flux2_klein_base_4b())`.

Recommended spec change: add explicit tasks to update `VariantId`, `_SUPPORTED`, and the `Flux2Klein` alias branch in `identify_variant()`; then update the stale API docstring at `src/mlx_teacache/api.py:141-146`.

### 3. The proposed Provenance entry would fail the existing type contract

The spec proposes:

```text
Provenance(source="in-repo", ...)
```

at `spec:44`. Current `Provenance.source` is `Literal["builtin", "user"]` (`src/mlx_teacache/coefficients.py:85-92`), and every in-repo built-in coefficient entry uses `source="builtin"` (`src/mlx_teacache/coefficients.py:122-142`).

The same line also sets `calibration_dataset="10x25xseed42_M1Max_bf16_q4_512_g4.0"`, but the calibration section says `guidance=1.0` (`spec:38`) and the script constant is `GUIDANCE = 1.0` (`scripts/calibrate_flux2.py:52-55`).

Impact: `source="in-repo"` is a mypy failure or a registry-shape drift, and the `g4.0` provenance string would make the calibration metadata lie about the data actually captured.

Recommended spec change: use `source="builtin"` unless the dataclass is intentionally extended, and set the dataset string to `guidance=1.0` / `g1.0`.

### 4. The release gates contradict the stated goal when `skipped_count == 0`

The spec sets **Skip count >= 1** as a release gate (`spec:99`) but immediately says v0.4 is not blocked if the default threshold produces 0 skips, and that an override can be documented. The risk table repeats this: 0 skips is "Low" and "do not block release" (`spec:108`).

That conflicts with the goal at `spec:13`: "first FLUX.2 variant where the polynomial gate is expected to engage on its own." If default-threshold base-4B has 0 skips, then v0.4 is not that release. It may still be structural support for a new variant, but it cannot be marketed as the FLUX.2 step-skipping repair.

This also conflicts with scope: per-variant defaults and SSIM-vs-threshold sweeps are explicitly out of scope (`spec:28-29`), so documenting an override after 0 skips would lack the quality evidence needed to recommend it.

Recommended spec change: make 0 skips at default threshold block the algorithmic-speedup claim. Either block the release until default-threshold engagement is proven, or permit the release only after rewording it as "base-4B structural support; no default TeaCache engagement yet."

### 5. The test plan will accidentally validate the old 8-step distilled schedule unless it adds variant-specific generation kwargs

The spec says to parametrize existing FLUX.2 parity and SSIM tests with `klein-base-4b` (`spec:62-63`). Today those tests hard-code the distilled Klein schedule:

- `tests/test_image_quality_flux2.py:80-88`: `_gen_kwargs_klein()` uses `num_inference_steps=8`, `guidance=1.0`.
- `tests/test_parity_flux2.py:112-122`: same 8-step helper.

But base-4B is being designed and calibrated around 25-step local runs, while the official model card example uses 50 steps. If the plan only adds base-4B to the existing fixture params, PR gates can pass on an 8-step path that is not the release target and may not exercise meaningful step-skipping.

Recommended spec change: add variant-aware generation kwargs. Distilled Klein can stay at 8 steps; base-4B should use the chosen release target, probably 25 for local PR gates and a documented 50-step optional/nightly/manual check if that remains the upstream reference.

### 6. The licensing narrative is inaccurate for the already-supported distilled 4B model

The spec says base-4B "unlocks commercial use of mlx-teacache on FLUX.2 for the first time" and that distilled Klein 4B and 9B "both ship under FLUX-family non-commercial terms" (`spec:20`).

External source check:

- `black-forest-labs/FLUX.2-klein-4B` Hugging Face card lists `License: apache-2.0` and says open weights are available for commercial use.
- `black-forest-labs/FLUX.2-klein-base-4B` also lists `License: apache-2.0`.
- BFL's official `flux2` repo model table says both 4B models are Apache-2.0 and both 9B models are FLUX Non-Commercial.

Impact: v0.4 does not unlock first commercial FLUX.2 use in mlx-teacache; the already-supported `flux2-klein-4b` did. The real v0.4 licensing story is narrower: it adds a non-distilled Apache-2.0 FLUX.2 variant.

Recommended spec change: replace the commercial-use claim with: "base-4B keeps the Apache-2.0 commercial posture of the 4B family while moving from distilled 4-step inference to non-distilled 25-50 step inference."

## Validated Direction

- mflux 0.17.5 exposes `ModelConfig.flux2_klein_base_4b()` and uses the same `Flux2Klein` runtime class/transformer shape family as distilled 4B.
- Base-4B is undistilled, Apache-2.0, and the official card shows a 50-step Diffusers example.
- NVIDIA's FLUX.2-dev TeaCache result is valid precedent for long-schedule FLUX.2-family caching, but it is not direct evidence that mlx-teacache's default `rel_l1_thresh=0.20` will engage on base-4B.

## Sources Checked

- Local spec: `docs/superpowers/specs/2026-05-17-flux2-klein-base-4b-design.md`
- Local detector/API/registry/test files cited above.
- Hugging Face base-4B model card: https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B
- Hugging Face distilled 4B model card: https://huggingface.co/black-forest-labs/FLUX.2-klein-4B
- BFL official FLUX.2 repo: https://github.com/black-forest-labs/flux2
- NVIDIA FLUX.2-dev TeaCache blog: https://developer.nvidia.com/blog/scaling-nvfp4-inference-for-flux-2-on-nvidia-blackwell-data-center-gpus/
