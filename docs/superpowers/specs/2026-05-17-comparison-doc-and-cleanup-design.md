# COMPARISON.md + global-config cleanup: design

**Date:** 2026-05-17
**Status:** Draft (brainstorming complete; awaiting user review).
**Target:** post-v0.4.1 side improvement, single feature branch + PR for the repo piece; out-of-repo `~/.claude/` updates done in parallel.
**Source-of-truth references:**
- v0.4.1 release: `docs/superpowers/specs/2026-05-17-flux2-cfg-per-branch-caching-design.md`
- v0.3 non-engagement postmortem: `docs/superpowers/notes/2026-05-16-flux2-teacache-non-engagement-postmortem.md`
- Existing README "Supported variants" + "Benchmarks" tables
- Global `~/.claude/CLAUDE.md`
- `~/.claude/skills/user-mlx-developer/` skill content

## Goal

Three independent improvements bundled by theme ("post-v0.4.1 cleanup + showcase"):

1. **Refresh `~/.claude/CLAUDE.md`** — remove stale guidance, fix ambiguities, fold in the v0.4.0 + v0.4.1 lessons (paragraph-by-paragraph pass).
2. **Refresh `~/.claude/skills/user-mlx-developer/`** — same pass on the MLX engineering skill, integrating the lessons from the v0.4.0 calibration work, v0.4.1 CFG branch architecture, the `mx.compile`-avoidance discovery, and the polynomial-transfer-as-hypothesis stance.
3. **Create `COMPARISON.md` at repo root** — visual side-by-side of vanilla mflux vs mlx-teacache for every **non-distilled** supported recipe, with timing + image columns. Linked from `README.md`. Image artifacts live in `_artifacts/comparison/` (non-gitignored), `.webp` format.

Items 1 and 2 are out-of-repo (personal-config); items 3 + the README link go through a single branch + PR per the new release-flow rule (`~/.claude/CLAUDE.md` "Release flow" section added 2026-05-17).

## Out of scope

- **Distilled variants in COMPARISON.md.** `flux1-schnell` (4 steps), `flux2-klein-4b` (8 steps), `flux2-klein-9b` (8 steps) are excluded. They get no algorithmic skipping (postmortem evidence); the wrapper adds ~1-2% gating overhead and changes nothing for users. README gains a one-line recommendation: "mlx-teacache is for non-distilled schedules — `flux1-dev` and `flux2-klein-base-4b`. Don't reach for it on schnell or distilled Klein."
- **Changing the existing README "Benchmarks" table.** Those numbers stay locked as the release-gate evidence (red-apple prompt, 512×512, the bench script's protocol). COMPARISON.md runs the portrait prompt at a different resolution and is presented as a visual showcase, not the release-gate.
- **PyPI page redesign.** README is the PyPI landing; COMPARISON.md gets a link, that's enough.
- **Generating images for other future variants.** When v0.5.0 ships `flux2-klein-base-9b`, we'll add its row to COMPARISON.md as part of that release's plan, not now.

## Item 3: COMPARISON.md design

### 3.1 File layout

```
mlx-teacache/
├── COMPARISON.md                 # NEW — repo root, linked from README
├── _artifacts/                   # NEW — committed (NOT gitignored)
│   └── comparison/
│       ├── flux1-dev/
│       │   ├── vanilla.webp
│       │   └── wrapper.webp
│       ├── klein-base-4b-g1/
│       │   ├── vanilla.webp
│       │   └── wrapper.webp
│       └── klein-base-4b-cfg/
│           ├── vanilla.webp
│           └── wrapper.webp
└── scripts/
    └── bench_comparison.py       # NEW — generates timings + images for COMPARISON.md
```

`.gitignore` keeps `tests/_artifacts/` ignored (that's where bench / sweep test fixtures live, generated on every CI / bench run). The new `_artifacts/` at repo root is NOT ignored — those files are part of the repo content. Add an explicit comment to `.gitignore` to make the distinction obvious to future readers.

### 3.2 COMPARISON.md structure

```markdown
# Vanilla mflux vs mlx-teacache — side-by-side comparison

> Visual + timing showcase across non-distilled FLUX variants. Distilled
> schedules (FLUX.1 schnell, FLUX.2 Klein 4B/9B at default steps) are
> excluded — they don't engage the polynomial gate; see [the README](README.md#when-to-use)
> for guidance on when mlx-teacache is the right tool.

## Test machine

- M1 Max 32GB unified memory
- macOS 26.x
- mlx-teacache 0.4.1, mflux 0.17.5
- bf16 weights, `quantize=4`
- 768 × 1024 portrait
- Seed: 42
- Three reps per condition (1 cold + 2 warm)
- Shared prompt across all variants — see below

## Prompt

> Portrait of a young woman with auburn hair and green eyes, soft golden-hour
> window light, photorealistic, shallow depth of field, 50mm prime lens,
> subtle freckles, neutral background, cinematic color grading.

---

## FLUX.1 family

### `flux1-dev` — 25 steps, guidance=3.5

| | Vanilla mflux | mlx-teacache |
|---|---|---|
| Steps | 25 | 25 |
| Guidance | 3.5 | 3.5 |
| Seed | 42 | 42 |
| **Cold (rep 1)** | <Xs> | <Ys> |
| **Warm (median reps 2-3)** | <Xs> | <Ys> |
| Skips | — | <N> / 25 |
| Speedup (warm) | 1.00× | **<Z>×** |
| Image | ![vanilla](_artifacts/comparison/flux1-dev/vanilla.webp) | ![wrapper](_artifacts/comparison/flux1-dev/wrapper.webp) |

Brief observations: <one sentence on visual diff or quality preservation>.

---

## FLUX.2 family

### `flux2-klein-base-4b` — 25 steps, guidance=1.0

| | Vanilla mflux | mlx-teacache |
|---|---|---|
| Steps | 25 | 25 |
| Guidance | 1.0 | 1.0 |
| Seed | 42 | 42 |
| **Cold (rep 1)** | <Xs> | <Ys> |
| **Warm (median reps 2-3)** | <Xs> | <Ys> |
| Skips | — | <N> / 25 |
| Speedup (warm) | 1.00× | **<Z>×** |
| Image | ![vanilla](_artifacts/comparison/klein-base-4b-g1/vanilla.webp) | ![wrapper](_artifacts/comparison/klein-base-4b-g1/wrapper.webp) |

### `flux2-klein-base-4b` (CFG) — 50 steps, guidance=4.0

The canonical upstream BFL recipe.

| | Vanilla mflux | mlx-teacache |
|---|---|---|
| Steps | 50 | 50 |
| Guidance | 4.0 | 4.0 |
| Seed | 42 | 42 |
| **Cold (rep 1)** | <Xs> | <Ys> |
| **Warm (median reps 2-3)** | <Xs> | <Ys> |
| Skips | — | <N> / 50 |
| Speedup (warm) | 1.00× | **<Z>×** |
| Image | ![vanilla](_artifacts/comparison/klein-base-4b-cfg/vanilla.webp) | ![wrapper](_artifacts/comparison/klein-base-4b-cfg/wrapper.webp) |

---

## Reproducing

```bash
uv run python scripts/bench_comparison.py
```

Generates the three entries above end-to-end on M1 Max in ~30 minutes.
Outputs go to `_artifacts/comparison/` (vanilla + wrapper `.webp` per variant)
and `_artifacts/comparison_report.json` (full timing record).
```

The placeholder `<Xs>`, `<Ys>`, `<N>`, `<Z>` are filled in by the script's output. The doc is committed with real numbers once the bench runs.

### 3.3 Generation script — `scripts/bench_comparison.py`

A new dedicated script — NOT an extension of `bench_speedup.py`. Reasons:
- `bench_speedup.py` is the release-gate bench (3-rep median, red-apple, 512×512, JSON output to `scripts/_bench_report_*.json`). Adding a "cold rep separately" mode there mixes the showcase purpose into a release-gate tool.
- The comparison bench is a content-generation tool, not a release artifact. It runs end-to-end across three variants, saves webp images, and emits a fragment of the COMPARISON.md table.

Script behavior per variant (`flux1-dev`, `klein-base-4b-g1`, `klein-base-4b-cfg`):

1. Load the model (one-time cost; not timed).
2. **Vanilla condition:**
   - Rep 1: timed `flux.generate_image(...)` → cold time + save PNG.
   - Reps 2-3: timed `flux.generate_image(...)` → warm times.
3. **Wrapped condition:**
   - `apply_teacache(flux)` once per the 3 reps (context-manager-restored between reps so cold/warm semantics are honest).
   - Rep 1: cold time + save PNG + record `handle.stats.skipped_count`.
   - Reps 2-3: warm times.
4. Convert both saved PNGs to webp via Pillow `Image.save(path, format="WEBP", quality=88, method=6)`.
5. Compute warm median + speedup ratio.

Output:
- `_artifacts/comparison/<variant_slug>/vanilla.webp`
- `_artifacts/comparison/<variant_slug>/wrapper.webp`
- `_artifacts/comparison_report.json` — **persistent, committed, complete** record. Designed for recovery: if `COMPARISON.md` is ever lost or someone edits it by hand and we need to verify, the JSON alone is enough to rebuild the document.

**JSON schema** (one top-level object, three entries under `"variants"`):

```jsonc
{
  "schema_version": 1,
  "generated_at": "2026-05-17T<HH:MM>Z",
  "hardware": {
    "chip": "M1 Max",
    "ram_gb": 32,
    "os": "macOS 26.x",
    "mlx_teacache_version": "0.4.1",
    "mflux_version": "0.17.5",
    "quantize": 4,
    "dtype": "bf16"
  },
  "prompt": "Portrait of a young woman with auburn hair...",
  "seed": 42,
  "height": 1024,
  "width": 768,
  "variants": {
    "flux1-dev": {
      "variant_id": "flux1-dev",
      "num_inference_steps": 25,
      "guidance": 3.5,
      "vanilla": {
        "rep_seconds": [<r1>, <r2>, <r3>],
        "cold_seconds": <r1>,
        "warm_median_seconds": <median(r2, r3)>
      },
      "wrapper": {
        "rep_seconds": [<r1>, <r2>, <r3>],
        "cold_seconds": <r1>,
        "warm_median_seconds": <median(r2, r3)>,
        "skipped_per_rep": [<n1>, <n2>, <n3>],
        "computed_per_rep": [<c1>, <c2>, <c3>],
        "rel_l1_thresh_used": 0.20
      },
      "speedup_warm": <vanilla_warm / wrapper_warm>,
      "speedup_cold": <vanilla_cold / wrapper_cold>,
      "image_paths": {
        "vanilla": "_artifacts/comparison/flux1-dev/vanilla.webp",
        "wrapper": "_artifacts/comparison/flux1-dev/wrapper.webp"
      }
    },
    "klein-base-4b-g1": { ... },
    "klein-base-4b-cfg": { ... }
  }
}
```

Every individual rep is preserved, not just the medians — if anyone re-derives the medians later or wants to plot per-rep variance, the data is in the JSON. `schema_version=1` lets future bumps (when v0.5.0 adds a new entry, schema changes for whatever reason) declare a clean break.

The JSON is committed to the repo at `_artifacts/comparison_report.json` (NOT gitignored) so it survives across sessions and is part of the v0.4.x release evidence trail.

The PNG step is intermediate (mflux saves PNG natively). We convert PNG → webp in-process and delete the PNG so only webp survives.

**Webp quality:** `quality=88, method=6`. Method=6 is Pillow's slowest+best encoder; on a 768×1024 image this is ~1-2 seconds, well under noise. Quality 88 keeps subtle gradients (skin tones) without obvious banding while compressing to ~150-250 KB per image — sub-MB total for the 6 images we ship.

**Variant slugs in the dir tree:**
- `flux1-dev/` for the flux1-dev row
- `klein-base-4b-g1/` for the FLUX.2 g=1.0 / 25-step row
- `klein-base-4b-cfg/` for the FLUX.2 g=4.0 / 50-step row

Slug is what goes in the dir name only; the COMPARISON.md table uses the actual variant id + recipe descriptors.

### 3.4 Cold vs warm definition

"Cold" = first generation after `Flux1(...)` or `Flux2Klein(...).freeze()` returns (no prior generation in the process). Model load is NOT included in the cold time — we measure from `flux.generate_image(...)` entry to return. This isolates MLX kernel JIT + first-call overhead from the model-load disk I/O.

"Warm" = median of reps 2 and 3. After cold rep finishes, the MLX kernel cache is hot and the Metal command queue is primed; reps 2-3 reflect steady-state behavior.

**For the wrapper condition specifically:** each rep wraps `apply_teacache(flux)` in a `with` block, so `restore()` runs between reps. The CACHED RESIDUAL is also cleared between reps (it's per-generation, owned by the lifecycle callback). Cold-vs-warm for the wrapper measures MLX kernel state, not TeaCache cache state. This is the right comparison — TeaCache's per-generation cache resets in real-world usage too.

### 3.5 README integration

Add to `README.md` after the "Supported variants" section (around line 105):

```markdown
## When to use

mlx-teacache is for **non-distilled diffusion schedules**. The polynomial
gate doesn't engage on distilled variants like FLUX.1 schnell (4 steps) or
FLUX.2 Klein 4B/9B at their distilled defaults (4-8 steps) — adjacent
transformer outputs change too much between consecutive steps for caching
to help. On those models the wrapper adds ~1-2% gating overhead and skips
zero steps.

Recommended variants for measurable speedup:
- `flux1-dev` at 20-50 steps
- `flux2-klein-base-4b` at 20-50 steps (with or without CFG)

See [COMPARISON.md](COMPARISON.md) for vanilla-vs-wrapper side-by-side
images and timings on the recommended variants.
```

And add a single line to the "Supported variants" footnotes / Limitations section that points readers to the "When to use" section.

### 3.6 .gitignore adjustment

Add a comment block to clarify the two `_artifacts` paths:

```gitignore
# Bench / sweep test fixtures, regenerated on every run.
tests/_artifacts/

# NOTE: the repo-root _artifacts/ is INTENTIONALLY NOT ignored — it holds
# committed comparison images for COMPARISON.md. Keep them tracked.
```

## Item 1: CLAUDE.md cleanup

Scope is a paragraph-by-paragraph editorial pass on `~/.claude/CLAUDE.md`:

- **Git commits / no Co-Authored-By:** stays as-is; rule still relevant.
- **HuggingFace CLI:** stays as-is.
- **MLX work / invoke user-mlx-developer skill:** rephrase to also call out `mlx-teacache` repo work explicitly (today says "any file in an mlx-taef / mlx-teacache / mflux working tree" — fine).
- **Heavy generations: main thread, not subagents:** stays; v0.4.1 reaffirmed this is the right rule (heavy bench + parity runs on main thread worked perfectly).
- **Public-facing docs: humanize before shipping:** stays.
- **Performance claims need a committed benchmark:** stays; v0.4.1 used `bench_speedup.py` exactly as the rule prescribed.
- **Release flow: branch → PR → stop for review:** added in 2026-05-17 session; new rule, no change.

Net change to CLAUDE.md is small — possibly nothing. Open a small edit only if a paragraph reads as stale on review (e.g. references to behavior fixed in v0.4.x or workarounds that no longer apply). Document the read-through in this spec's "implementation" section but don't pre-commit to specific edits.

## Item 2: user-mlx-developer skill refresh

Scope is the same kind of paragraph-by-paragraph review on `~/.claude/skills/user-mlx-developer/`. Specific additions to fold in from v0.4.0 + v0.4.1 work:

- **mx.compile graph topology is fragile under eager wrapping.** v0.4.0 fixed a bug where computing `temb_mod_params_single` upfront vs inline produced bit-different outputs even though the math was identical (see `forward.py:_flux2_run_body` docstring). The skill should mention this as an MLX-specific reflex: when reimplementing a function that vanilla wraps in `mx.compile`, mirror the graph topology exactly, including where intermediate values are computed inside loops.
- **CFG per-branch caching pattern** (new in v0.4.1): the encoder-independent `mod_in` invariant lets one gate decision drive two cached residuals. Worth recording as a pattern for future cache designs on diffusion transformers.
- **Polynomial transfer is empirical, not architectural.** The skill should warn against assuming offline-fit coefficients transfer between generation recipes (g=1.0 → g=4.0, different step counts) without empirical validation. v0.4.1 plan-audit Finding 3 caught this.
- **Lazy mflux imports for deferred-dep public APIs.** The `apply_teacache(...)` function defers all mflux imports so `from mlx_teacache import apply_teacache` works on a machine without mflux installed. Worth noting as a pattern.

These are notes-to-self fed back into the skill so the next MLX-touching task starts smarter.

## Implementation outline

Single PR for item 3 (the repo piece):
1. `scripts/bench_comparison.py` — new script (~150-200 LoC).
2. Run the script to generate the 6 webp images + report JSON. **Main-thread `run_in_background=true`, teed to `/tmp/bench-comparison.log`.** Estimated ~25-30 min wall-clock total.
3. `COMPARISON.md` — write the doc with real numbers + image refs. Run `/humanizer` on the new prose.
4. `README.md` — add "When to use" section + COMPARISON.md link. Re-humanize the new paragraph.
5. `.gitignore` — add the comment block explaining the two `_artifacts/` paths.
6. Commit, push branch, open PR, **stop** for human review + merge per new release-flow rule.

Items 1 and 2 (CLAUDE.md + skill) are done in parallel — out of the repo, no PR. They land as direct edits to `~/.claude/`.

## Acceptance criteria

COMPARISON.md side:
- [ ] `_artifacts/comparison/{flux1-dev,klein-base-4b-g1,klein-base-4b-cfg}/{vanilla,wrapper}.webp` exist, all 6 files committed.
- [ ] Each webp is < 300 KB.
- [ ] `_artifacts/comparison_report.json` exists with all per-rep timings, skip counts, hardware metadata, and recipe details for every variant — committed alongside the images. Schema version locked.
- [ ] `COMPARISON.md` has 3 entries with real timing + skip + speedup numbers (no placeholder `<X>` strings); every number in the doc is also present in `comparison_report.json` (the doc is rebuildable from the JSON if lost).
- [ ] `scripts/bench_comparison.py` exists and is reproducible (`uv run python scripts/bench_comparison.py` regenerates all artifacts).
- [ ] `README.md` "When to use" section exists with a link to COMPARISON.md.
- [ ] `.gitignore` has the explanatory comment about the two `_artifacts/` paths.
- [ ] CI green on the PR (ruff + typecheck + tests + coverage).

CLAUDE.md + skill side:
- [ ] CLAUDE.md read-through done. If edits made, they're applied directly to `~/.claude/CLAUDE.md` and the memory index is updated if a new rule was added.
- [ ] user-mlx-developer skill read-through done. New v0.4.0 + v0.4.1 lessons folded in.

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Generation times blow out (klein-base-4b CFG @ 768×1024 takes longer than budgeted) | Medium | Background run with progress monitor; ~25-30 min is the estimate but could be 40+. Tee log so the bench is interruptible / resumable. |
| Webp encoding loses subtle detail noticed in reviews | Low | `quality=88, method=6` is a known-good preset; can re-encode at higher quality if visible artifacts. PNG → webp is one Pillow call. |
| `_artifacts/` at repo root collides with some other convention | Low | New directory; nothing in the repo uses it today. Comment in `.gitignore` makes the convention explicit. |
| The cold/warm split shows surprisingly small cold-warm difference (= no useful information) | Low | If cold ≈ warm on all variants, drop the split and ship single-rep median per cell. Decision made post-run. |
| FLUX.1-dev at 768×1024 produces a noticeably worse portrait than at 512×512 | Low | dev is high-res-capable. If quality is off, increase steps from 25 to 28-30 (a v0.4.0-style adjustment). |
