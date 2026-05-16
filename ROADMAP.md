# mlx-teacache Roadmap

A non-binding sketch of where the library is headed beyond the shipped v0.1.x line. Each item lists effort, value, and key risks so we can re-rank when priorities change.

## Released

- **v0.3.0** — `flux2-klein-9b` support (in-repo calibration). Calibration
  script parameterized via `--variant` so v0.4 / v0.5 are additive.
  `Img2ImgNotSupportedError` (deprecated in v0.2.0) removed.
- **v0.1.0 / v0.1.1** — Initial public release. FLUX.1 dev/schnell, FLUX.2 Klein 4B. Calibrated coefficients. Five-tier test pyramid. Trusted-Publishing pipeline.

## Active

### v0.4.0: FLUX.2 caching research (postmortem-driven)

Discovered during v0.3.0 release that on FLUX.2 Klein 4B and Klein 9B at their distilled 8-step schedules the polynomial gate produces zero step skips at the package default `rel_l1_thresh=0.20`. The wrapper's wall-clock improvement comes from `mx.compile` avoidance, not from caching. Full writeup at `docs/superpowers/notes/2026-05-16-flux2-teacache-non-engagement-postmortem.md`.

v0.4.0 investigates whether a different caching approach can engage on Klein:

- **FirstBlockCache (FBCache)** port to mflux on Apple Silicon. Already in diffusers mainline; no polynomial calibration, gates on first-transformer-block residual absmean. Promising because it sidesteps the broken continuous gate.
- **Per-step-index lookup table.** Offline profile of Klein 8-step trajectories to identify reliably skippable step indices; bake those into a hardcoded per-step decision.
- **TaylorSeer / DiCache adaptation.** Fixed-interval and online-probe alternatives to polynomial-fit gating.

The research postmortem includes 10 specific references (ali-vilab/TeaCache, NVIDIA's FLUX.2-dev blog with `thresh=0.05`, diffusers CacheMixin, SeaCache + DiCache papers, FBCache implementation). v0.4 plan starts from those.

- **Effort:** large — at least one new caching strategy ported + benchmarked + calibrated; bench numbers measured by `scripts/bench_speedup.py`.
- **Value:** unblocks real step-skipping benefit on FLUX.2 Klein variants. Compile-avoidance wall-clock is a real win but should not be the only mechanism the library offers on FLUX.2.
- **Risks:** the consensus across published work is that no caching technique demonstrably engages on 4-8 step distilled FLUX-class models. Research may conclude TeaCache-class caching is fundamentally incompatible with distilled schedules and the library should declare "no step-skipping on distilled FLUX.2" and ship a different mechanism (e.g. a clean `mx.compile`-avoidance wrapper without a polynomial gate at all).

### v0.4.0 also: `flux2-klein-base-4b`

Apache-2.0 commercial-friendly variant. Same Flux2Klein class; non-distilled schedule (25-50 steps). Fresh calibration via `scripts/calibrate_flux2.py --variant klein-base-4b`. With longer schedules, the polynomial gate is much more likely to engage, so base-4B may be the first FLUX.2 variant where TeaCache step-skipping actually works.

### v0.5.0: `flux2-klein-base-9b`

Non-distilled 9B. FLUX Non-Commercial license + BFL safety filter. Fresh calibration. Same approach as base-4B.

---

Below this line: historical v0.2.0 / v0.3.0 plan content kept for reference.

Two feature tracks plus a doc track ship together as v0.2.0.

### 1. img2img support (spec: 2026-05-15-img2img-and-distilled-notification-design.md)

Lift v0.1's blanket img2img rejection. The three currently-supported variants (FLUX.1 dev/schnell, FLUX.2 Klein 4B) accept `image_path` + `image_strength > 0` with TeaCache engaged on the active denoising window (`config.num_inference_steps - config.init_time_step`, per mflux's `Config` semantics).

- **Effort:** medium — delete one rejection block, wire active-step-count through lifecycle + stats finalization, fix FLUX.1 forward's absolute-timestep step indexing, add SSIM gates for img2img
- **Value:** the single biggest user-visible gap closed
- **Risks:** existing coefficients are reused (calibrated on txt2img); v0.2.0 ships with this as a documented approximation, recalibration possible in v0.2.1 if SSIM gates show drift
- **Compatibility note:** `Img2ImgNotSupportedError` stays exported (deprecated) for one release; removal in v0.3.0

### 2. Distilled-step notification (spec: 2026-05-15-img2img-and-distilled-notification-design.md)

`TeaCacheNoBenefitWarning` (subclass of `UserWarning`) emitted once per handle when the current configuration cannot produce any possible skip — i.e., when `eligible - 1 <= 0` (need ≥1 seed step + ≥1 skip candidate). Catches schnell-at-4-steps and aggressive skip-window configs. Scoped to schedule-shape issues; CFG-fallback no-benefit is a separate concern tracked elsewhere.

- **Effort:** small — one warning class, one branch in `GenerationContextCallback.call_before_loop`
- **Value:** removes a foot-gun for first-time users at distilled defaults
- **Risks:** none material

### Doc track (folds in alongside the code tracks, Task #64)

Audit findings from `docs/superpowers/notes/2026-05-15-roadmap-docs-research-audit.md`:

- Fix M1 Pro / M2 Pro classification in the chip table (they're eager, not compiled — `is_m1_or_m2()` returns True for Pro variants because the predicate only excludes Max + Ultra)
- Write `docs/calibration.md` (the README link points to it but it doesn't exist)
- Remove or repoint the broken spike-notes link in the README
- Rewrite `docs/manual-verification.md` — current recipe uses a non-existent `.latents` attribute and asserts byte-exact parity on FLUX.2 (which is cosine-only by design)
- Soften M5 wording from "only available via the compiled path" to "may lose some or all of the M5 TensorOps advantage"
- README quick-start: switch Klein example from 25 steps to 8 steps (Klein is distilled; 25 steps degrades quality and slows down generation)

### Performance-by-chip section (Task #54, lands in v0.2.0 README)

Add a `Performance by chip` section to the README showing per-generation speedup expectations and inviting community measurements for chips we don't have access to.

**Draft language to include in 0.2.0:**

> ### Performance by chip
>
> mlx-teacache replaces mflux's compiled `_predict` with an eager Python wrapper so step-skipping decisions can run per step. The wrapper is correct on every Apple Silicon chip, but the speedup magnitude depends on how much mflux's compile pass would have helped you. Mflux's compile gate (in 0.17.5) uses `is_m1_or_m2()` which excludes only Max and Ultra variants — so M1/M2 base **and Pro** all run eager, while Max/Ultra and every M3/M4/M5 chip get compiled `_predict`.
>
> | Chip | Vanilla `_predict` in mflux 0.17.5 | Expected speedup |
> |---|---|---|
> | Apple M1 / M2 (base) | eager | ≈ pure skip fraction (~1.5–1.6×) |
> | M1 Pro / M2 Pro | eager | ≈ pure skip fraction — same as base |
> | M1 Max / Ultra, M2 Max / Ultra | compiled | **1.48× measured** on M1 Max FLUX.1-dev / 25 steps |
> | M3 / M3 Pro / M3 Max / Ultra | compiled | Likely 1.1–1.3× — untested |
> | M4 / M4 Pro / M4 Max | compiled | Likely 1.1–1.3× — untested |
> | M5+ (Neural Accelerators / TensorOps) | compiled + accelerator | May approach 1.0× — eager wrapper may lose some or all of the M5 TensorOps advantage. Confirm with profiler before treating as fact. |
>
> Output correctness is preserved on every chip. If you can measure speedup on M2+, please open an issue with your chip + numbers — we'll fold them into this table.

### Compile-friendly gating (deferred from v0.2.0)

Investigate keeping `_predict` compiled while gating runs in eager Python. Three sketches:

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

## Future (no fixed release)

### Other Apple-Silicon models worth covering

Ranked by value-per-effort. None of these are committed; they're a menu for picking up after v0.2.0.

| # | Model | Effort | License | Why it matters | Risks |
|---|---|---|---|---|---|
| 1 | **Z-Image base** (Tongyi-MAI, full step schedule) | 3–5 days | Apache-2.0 | Already in mflux 0.17.5 as `ZImage` with a compile-gated `_predict`. Apache-2.0 license. Recommended 28–50 steps → high skip ceiling. Popular on Apple Silicon. | Different architecture from FLUX (single-stream DiT with cross-attention) — needs a fresh polynomial fit, not coefficient reuse. Integration is short; calibration is the long pole. |
| 2 | **FLUX.2-klein-base-4B** (non-distilled) | 1–2 days | Apache-2.0 | Runs 25–50 steps vs Klein-4B's 8 → larger absolute skip savings. Same `Flux2Klein` class. Apache-2.0 unlocks commercial use. | Fresh calibration; ~15 GB disk. |
| 3 | **FLUX.2-klein-base-9B** (non-distilled) | 1–2 days | FLUX Non-Commercial | Same as base-4B with more capacity. | Same license obligations as Klein 9B (safety filter, non-commercial). ~30 GB disk. |
| 4 | **Chroma1-HD** (lodestones community FLUX-schnell fine-tune) | Week | Apache-2.0 | Popular community model; FLUX-architecture-compatible in spirit. | **Not in mflux 0.17.5** — exposed upstream as `ChromaPipeline` in Diffusers. Requires either a mflux integration request or a custom weight-loading path that maps Chroma's safetensors onto mflux's `Flux1` mapping. Prove loadability first; if mflux's `Flux1` accepts Chroma weights with no custom mapping, drops to 1–2 days. |
| 5 | **SD3.5 medium** (Stability AI, 2.5B DiT) | Week+ | Stability AI Community License | DiT-family, runs on `mlx-stable-diffusion` rather than mflux. Different upstream. | Less actively maintained upstream; may require upstream patches. |
| 6 | **SDXL** | Week+ | OpenRAIL-M | Largest installed base on Apple Silicon, but standard step counts (30–40) are lower than FLUX — speedup ceiling is lower. | Same upstream issue as SD3.5. |
| 7 | **AuraFlow** (~6B DiT) | 1–2 weeks | Apache-2.0 | Open license, FLUX-like, room for community uptake. | Not in mflux — needs a new mflux variant first. |
| 8 | **HunyuanVideo / Mochi / CogVideoX** | — | varies (see below) | Video diffusion: TeaCache concept is even more valuable (30+ steps, very expensive per step). | Doesn't fit 32 GB unified memory on this machine — needs a 64 GB+ M-series chip to develop and test. License notes: Mochi 1 preview Apache-2.0 (42 GB VRAM Diffusers / 60 GB reference); CogVideoX-2B Apache-2.0; CogVideoX-5B "other"; HunyuanVideo / HunyuanVideo 1.5 "other". |

### Process / infra items

- **`workflow_dispatch` nightly slow suite** — one-click GH Actions run of `pytest -m "slow and parity"` with HF_TOKEN
- **Community benchmark table** — accept PR-submitted measurements and embed in README, with the protocol from `docs/m3-plus-tradeoff.md` (env-print, warmup, ≥3 timed reps, report median + min + computed/skipped)
- **Benchmark protocol formalization** — bake warmup + repeat-run + env-print conventions into `docs/m3-plus-tradeoff.md` so community submissions are comparable

## Out of scope (deliberate)

- **Z-Image-Turbo** and **FLUX.1-schnell at 4 steps** — distilled schedules have nothing to skip; the new `TeaCacheNoBenefitWarning` (v0.2.0) is the signal we give users for these
- **Server / API layer** — mlx-teacache is a library, not an inference service
- **PyTorch backend** — TeaCache for PyTorch already exists upstream (ali-vilab); mlx-teacache stays MLX-native

## Deferred work (no current release target)

These items are real and tracked, but not in the current release window. No 0.3.0 plan is committed yet.

## How to use this doc

When picking up new work, find the matching row above, copy the effort/value/risk line into the implementation plan header, and link back here. When closing v0.2.0, move the two Active tracks into "Released" and pull whichever "Future" row is next.
