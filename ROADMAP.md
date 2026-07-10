# mlx-teacache Roadmap

A non-binding sketch of where the library is headed beyond the shipped v0.1.x line. Each item lists effort, value, and key risks so we can re-rank when priorities change.

## Released

- **v0.9.2** — Output-neutral hardening release. Failed FLUX.1/Qwen callback registration now restores the original transformer, and `restore()` completes every teardown action while keeping double-apply blocked until restoration succeeds. The test memory guard scales below each machine's MLX working-set ceiling; the sdist no longer ships tests without their reference fixtures; CI verifies the sdist and runs the fast lane against the newest supported mflux. Public install, artifact, threshold-default, and example documentation is corrected.
- **v0.9.1** — Maintenance release: supports mflux 0.18.x (pin widened to `>=0.17.5,<0.19`). No library code or generated-image change — the integration ports the same mflux internals, which 0.18.0 leaves intact; compatibility was checked per variant against the installed 0.18.0 source and confirmed by generating across FLUX.1, FLUX.2 Klein, Z-Image, and Qwen-Image. Adds a weight-free mflux-contract test (callback-registry layout, per-variant model-config aliases, precision default) and a CI job that runs the fast lane against the lowest supported mflux (0.17.5); PyPI keywords and description now name Qwen-Image and Z-Image.
- **v0.9.0** — `qwen-image` support: Alibaba's ~20B dual-stream MMDiT (Apache-2.0). The first variant that proxies `flux.transformer` (the FLUX.1 pattern) and also runs true two-pass CFG — `generate_image` calls the transformer twice per step and combines outside it, so a branch-pairing state machine shares one gate decision across the pair and keeps two cached residuals. Qwen is FLUX-shaped, so the FLUX-canonical modulated-input gate calibrates at R² 0.849 (vs Z-Image's 0.40). Per-variant default `rel_l1_thresh=0.30`; **1.74× warm-median** at the shared 768² portrait recipe (25 of 48 active steps skipped, SSIM 0.987) — entirely step-skipping (Qwen has no `mx.compile` path to bypass). On a 32 GB Mac, stock 4-bit Qwen is grainy, so the showcase portraits use a mixed-precision build (8-bit edge blocks + bf16 embeddings); mlx-teacache stays quantization-agnostic.
- **v0.8.1** — Maintenance release; no output or benchmark-number change (gate math and coefficients untouched). Defensive teardown + registry hardening: teardown restores a pre-existing instance `_predict` instead of deleting it (FLUX.2 + Z-Image); a malformed variant now fails `import mlx_teacache` with a `CalibrationError` naming the subpackage rather than an opaque error; the FLUX.1 img2img defensive window-check fallback uses the active denoising count, not the nominal schedule. Internal hygiene: the package root imports its public types from the canonical kernel modules (not the deprecated shims), and `from __future__ import annotations` is removed package-wide with a guard test.
- **v0.8.0** — Correctness release: transactional `apply()` on every variant (a mid-apply failure now leaves the model pristine); the whole exception surface rooted under `TeaCacheError` (`TeaCacheValueError` for invalid arguments, still a `ValueError`; `CalibrationError` backing an import-time builtin-coefficient self-check); custom coefficients validated at the call site (nan/inf raise immediately, any 5-element sequence coerced to tuple) with `source="user"` provenance on all seven variants; a one-time `TeaCacheDisabledWarning` when `rel_l1_thresh=0.0` disables caching. Test suite rebuilt to catch a dormant cache: per-recipe measured SSIM floors with skip-engagement asserts on the base variants (base-4b 3/25 skips SSIM 0.9927, base-9b 7/25 skips SSIM 0.9920), the flux1-dev 5–7 skip-count band, CFG per-branch cache-independence and skip-reconstruction pins, and the fast fake-based tests moved into per-PR CI. No output or benchmark-number change — gate math and coefficients untouched.
- **v0.7.0** — `z-image-base` support: the first non-FLUX variant and the first calibrated on a latent-dependent internal signal rather than a modulation input. Z-Image's adaLN is timestep-only, so the gate taps the first-main-layer residual (Signal B, in-repo origin fit R² 0.400; the caption-independent noise-refiner tap was rejected at R² 0.069). Self-contained per-variant mini-kernel with no sibling imports. Per-variant default `rel_l1_thresh=0.12`, set at the SSIM knee. **1.17× combined** at the 512² red-apple recipe (q8, 15/48 skips, SSIM 0.991, peak 17.2 → 11.9 GB) and **1.33× warm** at the 640×896 COMPARISON portrait recipe; the win is entirely gating (`mx.compile`-path avoidance is not a tailwind on Z-Image). Calibration, sweep, three-way bench, parity gate, and COMPARISON row all committed.
- **v0.6.0** — Per-variant cores + shared algorithmic kernel. Adding a new FLUX variant now lives in `src/mlx_teacache/variants/<id>/{config,detect,integration}.py` plus matching tests — no more cross-cutting edits to `api.py`, `forward.py`, `coefficients.py::_REGISTRY`, and `detect.py`. Pure-algorithm primitives moved to `src/mlx_teacache/_kernel/` and import-clean (gated by `tests/_kernel/test_kernel_no_mflux_import.py`); `TeaCacheHandle` collapsed to a variant-agnostic context manager driven by per-variant `VariantPatch` rollback/finalizer callbacks. The 4-kwarg `apply_teacache` public signature is preserved byte-equivalent (snapshot-tested). `scripts/bench_speedup.py` refactored to subprocess-per-rep, which the 9B + CFG memory-safety story depends on. The subprocess-isolated three-way bench on `flux2-klein-base-9b` (50 steps + g=4.0, M1 Max 32 GB) measures **1.36× combined** (1.34× gating + 1.02× compile-avoidance), correcting v0.5.0's inflated 2.68× headline (caused by same-process MLX state leakage in the v0.5.x harness). Sanity check on `flux2-klein-base-4b` against the v0.4.1 baseline pending.
- **v0.5.0** — `flux2-klein-base-9b` (non-distilled FLUX.2 Klein 9B, FLUX NC license). Ships by reusing `flux2-klein-base-4b`'s polynomial coefficients verbatim and the same `rel_l1_thresh=0.17` default — justified by the shared architecture family and identical non-distilled 25-step / g=1.0 calibration recipe. Validated empirically at the canonical 50-step CFG recipe on M1 Max 32GB: SSIM 0.986 vs vanilla, **2.68× combined wall-clock** (vanilla 2744s, wrapper 1025s; 12/48 active steps skipped). Clean attribution between gating contribution and `mx.compile`-path avoidance is deferred to v0.5.1 (the existing `bench_speedup.py` runs 9 same-process generations and isn't memory-safe at 9B on 32GB; needs a subprocess-per-rep refactor first). New `scripts/validate_klein_base_9b.py` with subprocess-per-condition isolation + explicit MLX memory cap. New "Memory guardrails for heavy generations on 32 GB" rule documented after a same-process OOM crashed the machine during pre-release work.
- **v0.4.1** — CFG per-branch caching for FLUX.2. Canonical upstream recipe (`guidance_scale=4.0, num_inference_steps=50`) on `flux2-klein-base-4b` is now gate-engaged: 1.26× combined wall-clock on M1 Max (1.16× gating contribution, 1.09× `mx.compile`-path avoidance), 9/50 skips, SSIM ≥ 0.99 vs vanilla. `_vanilla_flux2_cfg_predict()` retired from production paths. Three-way bench protocol (`vanilla` / `no-gate` / `gated`) separates the v0.4 compile-avoidance effect from the v0.4.1 gating contribution. `cfg_fallback_steps` deprecated.
- **v0.4.0** — `flux2-klein-base-4b` (Apache-2.0, non-distilled, 25-step calibration). First FLUX.2 variant where the polynomial gate engages at the package default. Per-variant `default_thresh=0.17` ships via `Provenance.default_thresh` (3/25 skips on M1 Max; wrapper measures 1.41× wall-clock — both FLUX.2 mechanisms contribute: ~12% from step-skipping plus `mx.compile`-path avoidance; SSIM > 0.99). CFG-engaged caching deferred to v0.4.1. Per-variant default-threshold mechanism added (now a permanent API).
- **v0.3.0** — `flux2-klein-9b` support (in-repo calibration, origin-constrained polyfit). Calibration script parameterized via `--variant` so v0.4 / v0.5 are additive. `Img2ImgNotSupportedError` (deprecated in v0.2.0) removed. Honest performance framing for FLUX.2 Klein: distilled schedules don't algorithmically step-skip; wall-clock improvement comes from `mx.compile`-path avoidance. `scripts/bench_speedup.py` committed as reproducible source of truth for all README benchmark numbers.
- **v0.2.0** — img2img support for FLUX.1 dev/schnell + FLUX.2 Klein 4B (single bundled PR). `TeaCacheNoBenefitWarning` for distilled-schedule + skip-window misconfigurations. Per-chip `Performance by chip` section in README with M1 Pro / M2 Pro classification corrected (they're eager, not compiled). `docs/calibration.md` written. `docs/manual-verification.md` rewritten with a working recipe.
- **v0.1.0 / v0.1.1** — Initial public release. FLUX.1 dev/schnell, FLUX.2 Klein 4B. Calibrated coefficients. Five-tier test pyramid. Trusted-Publishing pipeline.

## Active

Nothing in flight.

---

## Future improvements (no fixed release target)

Concrete improvement ideas with a clear failure mode they address. Each is a candidate for a future minor release; none are committed.

### Calibration fit-quality on FLUX.2-family architectures

FLUX.2 calibration produces consistently lower R² than FLUX.1 (klein-9b origin fit: R² = 0.471; klein-4b free fit: R² = 0.653). The polynomial form may be a bad fit for the FLUX.2 mod_in → body_out mapping. Worth investigating: (a) higher polynomial degree, (b) piecewise fit by step-index range, (c) different signal entirely (first-block residual delta — see "Alternative gate signals" below). If the fit improves, default-threshold engagement on FLUX.2 may become more reliable.

### SSIM-vs-threshold sweep tooling

A `scripts/sweep_threshold.py` that captures (threshold, skip_count, SSIM) triplets at calibration time for a given variant. Produces an evidence-backed threshold recommendation per variant. ~3 hours of additional bench cost on top of calibration. Held for a future release where threshold characterization matters more than time-to-ship.

### Alternative gate signals (non-distilled only)

FBCache (first-block residual delta), DiCache (shallow-layer probe), TaylorSeer (fixed-interval extrapolation). These were considered as v0.4 directions for distilled Klein and dropped (the distilled regime is fundamentally hostile to any caching mechanism — see "Out of scope" below). On *non-distilled* schedules where the polynomial gate works but has low R² (i.e. FLUX.2 family), they may give better engagement / quality than the polynomial. Worth a research spike if v0.4.0's polynomial-on-base-4b engagement is unsatisfying.

### img2img calibration

v0.2.0 reuses txt2img coefficients for img2img generation. The polynomial captures an architectural property so this should generalize, but a dedicated img2img calibration may follow if SSIM gates on real img2img workloads show drift. Not a frequent user request yet.

### Compile-friendly gating

Keep mflux's `mx.compile` on `_predict` while running TeaCache gating in eager Python. Three sketches:

- **A — Compile body, eager wrapper:** isolate the pure-tensor computation (`transformer(...)`) into a separately compiled function. The wrapper calls the compiled body or returns the cached residual. Skip decisions stay in Python. Most invasive — requires extracting an mflux internal as a standalone compileable callable.
- **B — Branch inside the graph:** use `mx.where(should_run, full_path, cached_path)` so both paths compute, but only one is selected. Trades all skip savings for keeping the compile fast path. Net-negative whenever caching actually engages (we paid for both paths). Useful only as a correctness experiment, not a speed path.
- **C — Pre-compile two paths, eager dispatcher (likely first attempt):** compile a full-forward function and a skip function separately at integration time. The wrapper is a tiny eager Python function that does the rel_l1 / threshold check and dispatches to whichever pre-compiled callable matches the step's decision. Pattern that production cache layers (PyTorch FirstBlockCache, ComfyUI-TeaCache) converge on. The eager dispatcher must remain outside the compiled callable; cached residual lives in eager scope and is passed into the skip path as an array. The compile gate in mflux is one predict closure, not the transformer module, so wrapping is bounded.

**Tradeoffs across A and C (the viable options):**

| Aspect | A — split body | C — two pre-compiled paths |
|---|---|---|
| Code surgery | High — must surface mflux internals as a standalone function | Lower — wrap the *whole* `_predict` in compile for each path |
| Shape stability | Inherits mflux's shape contract | Same as A; both pre-compiled paths see the same input shapes per generation |
| Cache-tensor lifecycle | Must cross compile boundary cleanly | Cleaner — cached residual lives in eager scope, fed to whichever compiled path runs |
| Recompile risk on shape change | Lower — body sees fixed shapes per generation | Same — both paths recompile if seq-len/batch changes mid-generation |
| Likely net speedup vs vanilla on M5 | Best | Near-best (small dispatcher overhead) |

Pick A or C after measuring vanilla compile-loss on representative M3/M4/M5 hardware via community benchmarks.

## Future model coverage (no fixed release)

Other Apple-Silicon-friendly models worth covering after the current FLUX.2 pipeline lands. Ranked by value-per-effort. `FLUX.2-klein-base-4B` shipped in v0.4.0, `FLUX.2-klein-base-9B` in v0.5.0, and `Z-Image base` in v0.7.0 (all under "Released"). Don't pick from this table while Active items are in flight.

| # | Model | Effort | License | Why it matters | Risks |
|---|---|---|---|---|---|
| 1 | **Chroma1-HD** (lodestones community FLUX-schnell fine-tune) | Week | Apache-2.0 | Popular community model; FLUX-architecture-compatible in spirit. | **Not in mflux 0.17.5** — exposed upstream as `ChromaPipeline` in Diffusers. Requires either a mflux integration request or a custom weight-loading path that maps Chroma's safetensors onto mflux's `Flux1` mapping. Prove loadability first; if mflux's `Flux1` accepts Chroma weights with no custom mapping, drops to 1–2 days. |
| 2 | **SD3.5 medium** (Stability AI, 2.5B DiT) | Week+ | Stability AI Community License | DiT-family, runs on `mlx-stable-diffusion` rather than mflux. Different upstream. | Less actively maintained upstream; may require upstream patches. |
| 3 | **SDXL** | Week+ | OpenRAIL-M | Largest installed base on Apple Silicon, but standard step counts (30–40) are lower than FLUX — speedup ceiling is lower. | Same upstream issue as SD3.5. |
| 4 | **AuraFlow** (~6B DiT) | 1–2 weeks | Apache-2.0 | Open license, FLUX-like, room for community uptake. | Not in mflux — needs a new mflux variant first. |
| 5 | **HunyuanVideo / Mochi / CogVideoX** | — | varies (see below) | Video diffusion: TeaCache concept is even more valuable (30+ steps, very expensive per step). | Doesn't fit 32 GB unified memory on this machine — needs a 64 GB+ M-series chip to develop and test. License notes: Mochi 1 preview Apache-2.0 (42 GB VRAM Diffusers / 60 GB reference); CogVideoX-2B Apache-2.0; CogVideoX-5B "other"; HunyuanVideo / HunyuanVideo 1.5 "other". |

### Process / infra items

- **`workflow_dispatch` nightly slow suite** — one-click GH Actions run of `pytest -m "slow and parity"` with HF_TOKEN
- **Community benchmark table** — accept PR-submitted measurements and embed in README, with the protocol from `docs/m3-plus-tradeoff.md` (env-print, warmup, ≥3 timed reps, report median + min + computed/skipped)
- **Benchmark protocol formalization** — bake warmup + repeat-run + env-print conventions into `docs/m3-plus-tradeoff.md` so community submissions are comparable

## Out of scope (deliberate)

- **Algorithmic step-skipping on distilled schedules.** This includes FLUX.2 Klein 4B + 9B at their 4-8 step defaults, FLUX.1 schnell at 4 steps, and Z-Image-Turbo. The polynomial gate's premise (adjacent steps are similar enough that the residual can be reused) does not hold on distilled trajectories where each step does a much larger share of the work. v0.3.0 documented this for Klein 4B + 9B explicitly; the v0.2.0 `TeaCacheNoBenefitWarning` is the runtime signal for users hitting this regime. Note: research efforts proposed in the 2026-05-16 postmortem (FirstBlockCache port, per-step-index lookup, fixed-interval caching as alternatives for distilled Klein) were considered and dropped in favor of shipping `flux2-klein-base-4b` (non-distilled) in v0.4.0. The structural wall-clock benefit on distilled Klein from sidestepping mflux's `mx.compile`-wrapped `_predict` is preserved (~1.2-1.9× measured on M1 Max), so the variants remain supported — they just don't get algorithmic step-skipping by design.
- **Server / API layer** — mlx-teacache is a library, not an inference service.
- **PyTorch backend** — TeaCache for PyTorch already exists upstream (ali-vilab); mlx-teacache stays MLX-native.

## How to use this doc

When picking up new work:
1. **Active items first.** Items under `## Active` are committed. Finish the current Active item before pulling the next one in.
2. **Future improvements next.** Items under `## Future improvements` are pre-vetted improvement ideas with documented failure modes they address. Each can be lifted into an Active release when its time comes.
3. **Future model coverage after that.** The model-coverage table is a menu, not a queue; pick from it based on community demand + license posture + bench cost.
4. **Out of scope is durable.** Items under `## Out of scope (deliberate)` represent intentional non-goals, not deferred work. Re-opening one requires evidence that the original reasoning no longer holds.

When closing a release, move its Active entry to Released with a one-line summary and pull the next Active item into the top slot.

---

By Denis Ineshin · [ineshin.space](https://ineshin.space)
