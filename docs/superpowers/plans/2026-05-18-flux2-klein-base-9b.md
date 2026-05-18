# v0.5.0 — flux2-klein-base-9b implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** `docs/superpowers/specs/2026-05-18-flux2-klein-base-9b-design.md`

**Goal:** Add `flux2-klein-base-9b` support to mlx-teacache by reusing base-4b's polynomial coefficients verbatim and validating empirically before shipping v0.5.0.

**Architecture:** Variant addition with coefficient reuse. No new mechanisms (v0.4.1's per-branch CFG caching already covers FLUX.2 generically). Engineering today, validation + bench + ship tomorrow.

**Tech Stack:** Python 3.11+, MLX, mflux 0.17.x, pytest 8, ruff, mypy strict.

---

## Phase 1 — Engineering (today, no generation)

### Task 1: Add variant to detect.py

**Files:**
- Modify: `src/mlx_teacache/integrations/mflux/detect.py`
- Test: `tests/test_detect.py`

- [ ] **Step 1: Replace rejection test with acceptance test**

In `tests/test_detect.py`, find `test_flux2_klein_base_9b_rejected` (around line 73) and replace it:

```python
def test_flux2_klein_base_9b_recognized():
    """v0.5.0 added klein-base-9b as a supported variant."""
    assert identify_variant(_FakeFlux2Klein("flux2-klein-base-9b")) == "flux2-klein-base-9b"
```

- [ ] **Step 2: Run test, verify it fails**

```bash
uv run pytest tests/test_detect.py::test_flux2_klein_base_9b_recognized -v
```

Expected: FAIL with `UnsupportedVariantError` because the variant isn't in `_SUPPORTED` yet.

- [ ] **Step 3: Update detect.py**

In `src/mlx_teacache/integrations/mflux/detect.py`:

```python
VariantId = Literal[
    "flux1-dev",
    "flux1-schnell",
    "flux2-klein-4b",
    "flux2-klein-9b",
    "flux2-klein-base-4b",
    "flux2-klein-base-9b",
]

_SUPPORTED: tuple[str, ...] = (
    "flux1-dev",
    "flux1-schnell",
    "flux2-klein-4b",
    "flux2-klein-9b",
    "flux2-klein-base-4b",
    "flux2-klein-base-9b",
)
```

And in the alias-handling block (where `"flux2-klein-base-4b" in aliases:` is checked), add a parallel branch for klein-base-9b:

```python
if "flux2-klein-base-9b" in aliases or "klein-base-9b" in aliases:
    return "flux2-klein-base-9b"
```

(Place this before the `klein-base-4b` check or after — order matters only if alias sets overlap, which they shouldn't here.)

- [ ] **Step 4: Run test, verify it passes**

```bash
uv run pytest tests/test_detect.py::test_flux2_klein_base_9b_recognized -v
```

Expected: PASS.

- [ ] **Step 5: Run full detect test file**

```bash
uv run pytest tests/test_detect.py -v
```

Expected: all tests pass, no regressions.

- [ ] **Step 6: Lint + typecheck**

```bash
uv run ruff check src/mlx_teacache/integrations/mflux/detect.py tests/test_detect.py
uv run ruff format --check src/mlx_teacache/integrations/mflux/detect.py tests/test_detect.py
uv run mypy src/mlx_teacache/integrations/mflux/detect.py
```

Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add src/mlx_teacache/integrations/mflux/detect.py tests/test_detect.py
git commit -m "feat(detect): recognize flux2-klein-base-9b (v0.5.0 prep)"
```

---

### Task 2: Register coefficients for klein-base-9b

**Files:**
- Modify: `src/mlx_teacache/coefficients.py`
- Test: `tests/test_coefficients.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_coefficients.py`:

```python
def test_klein_base_9b_reuses_base_4b_polynomial():
    """v0.5.0 ships klein-base-9b with base-4b's coefficients verbatim.

    The reuse is intentional. If this test fails, either the 9B entry was
    edited (and the change should be deliberate, not silent), or the 4B
    entry drifted.
    """
    from mlx_teacache.coefficients import _REGISTRY

    coeffs_4b = _REGISTRY["flux2-klein-base-4b"].coefficients
    coeffs_9b = _REGISTRY["flux2-klein-base-9b"].coefficients
    import numpy as np
    np.testing.assert_array_equal(np.asarray(coeffs_4b), np.asarray(coeffs_9b))


def test_klein_base_9b_default_thresh_017():
    """klein-base-9b ships with the same default threshold as base-4b."""
    from mlx_teacache.coefficients import _REGISTRY

    entry = _REGISTRY["flux2-klein-base-9b"]
    assert entry.default_thresh == 0.17
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
uv run pytest tests/test_coefficients.py::test_klein_base_9b_reuses_base_4b_polynomial tests/test_coefficients.py::test_klein_base_9b_default_thresh_017 -v
```

Expected: FAIL with `KeyError: 'flux2-klein-base-9b'`.

- [ ] **Step 3: Add the registry entry**

In `src/mlx_teacache/coefficients.py`, after the `"flux2-klein-base-4b"` entry, add a new entry. The exact shape depends on the existing dataclass — read the file first to confirm. Typical shape:

```python
"flux2-klein-base-9b": Provenance(
    coefficients=_KLEIN_BASE_4B_COEFS,   # SHARED REFERENCE — intentional reuse
    default_thresh=0.17,
    # Comment block — see below
),
```

Add a doc-block comment above the entry explaining the reuse:

```python
# flux2-klein-base-9b at the 50-step CFG canonical recipe (non-distilled).
# Coefficients REUSED VERBATIM from flux2-klein-base-4b. Justification:
# same architecture family (FLUX.2 Klein), same calibration recipe
# (25 steps, guidance=1.0, origin-constrained polyfit). The reuse was
# validated empirically before v0.5.0 shipped — see
# _artifacts/validation_klein_base_9b.json for the SSIM ≥ 0.95 evidence
# at 50 steps + guidance=4.0.
# default_thresh=0.17 likewise reused from base-4b's v0.4.0 SSIM sweep.
```

Make sure the name `_KLEIN_BASE_4B_COEFS` matches what the file actually uses (rename the existing constant if needed).

- [ ] **Step 4: Run tests, verify they pass**

```bash
uv run pytest tests/test_coefficients.py -v
```

Expected: all coefficient tests pass.

- [ ] **Step 5: Lint + typecheck**

```bash
uv run ruff check src/mlx_teacache/coefficients.py tests/test_coefficients.py
uv run ruff format --check src/mlx_teacache/coefficients.py tests/test_coefficients.py
uv run mypy src/mlx_teacache/coefficients.py
```

Expected: green.

- [ ] **Step 6: Commit**

```bash
git add src/mlx_teacache/coefficients.py tests/test_coefficients.py
git commit -m "feat(coefficients): register flux2-klein-base-9b reusing base-4b polynomial"
```

---

### Task 3: API surface — docstrings + default-thresh set

**Files:**
- Modify: `src/mlx_teacache/api.py`, `src/mlx_teacache/__init__.py`

- [ ] **Step 1: Read api.py to find the variant lists**

```bash
grep -n "flux2-klein-base-4b\|klein-base\|default_thresh\|0.17" src/mlx_teacache/api.py src/mlx_teacache/__init__.py
```

Identify (a) the docstring "Supported variants" list and (b) any conditional that gates `default_thresh=0.17`.

- [ ] **Step 2: Update api.py docstring**

In the docstring section listing non-distilled variants (look for the existing `flux2-klein-base-4b` mention), add `flux2-klein-base-9b` alongside it. Match the surrounding indentation and prose style.

Also update the comment "Currently set for flux2-klein-base-4b (0.17)" to read "Currently set for flux2-klein-base-4b and flux2-klein-base-9b (0.17)".

- [ ] **Step 3: Update __init__.py docstring if it lists variants too**

Same edit pattern.

- [ ] **Step 4: Verify the threshold-defaulting code path covers the new variant**

If api.py has explicit `if variant == "flux2-klein-base-4b": default = 0.17` style code, add klein-base-9b to the condition (or refactor to read from `_REGISTRY[variant].default_thresh`, which is the cleaner path).

- [ ] **Step 5: Run the API + integration tests**

```bash
uv run pytest tests/ -v -k "api or integration" -m "not slow"
```

Expected: all pass.

- [ ] **Step 6: Lint + typecheck the changed files**

```bash
uv run ruff check src/mlx_teacache/api.py src/mlx_teacache/__init__.py
uv run mypy src/mlx_teacache/api.py src/mlx_teacache/__init__.py
```

Expected: green.

- [ ] **Step 7: Commit**

```bash
git add src/mlx_teacache/api.py src/mlx_teacache/__init__.py
git commit -m "feat(api): expose flux2-klein-base-9b in docstrings + default-thresh map"
```

---

### Task 4: Remove the NotImplementedError stub in calibrate_flux2.py

**File:** `scripts/calibrate_flux2.py`

- [ ] **Step 1: Inspect the stub**

```bash
grep -n "NotImplementedError\|klein-base-9b\|wired in v0.5" scripts/calibrate_flux2.py
```

Find the `klein-base-9b` block (around line 105 per `coefficients.py:154` precedent) and the `NotImplementedError` raise (around line 78).

- [ ] **Step 2: Remove the NotImplementedError**

The klein-base-9b config block in `_VARIANTS` already exists. Find the conditional that raises NotImplementedError when `--variant klein-base-9b` is selected and remove it (or replace with the normal model-load path, mirroring klein-base-4b).

- [ ] **Step 3: Update the script docstring**

Remove the line "klein-base-9b is declared but raises NotImplementedError (wired in v0.5.0)" — the variant is now runnable. Replace with a note about its current status:

```python
# klein-base-9b shares coefficients with klein-base-4b (v0.5.0 ship decision).
# Run this script only if you want to override the reused coefficients with a
# fresh fit. See docs/calibration.md for when that's warranted.
```

- [ ] **Step 4: Dry-run the help output**

```bash
uv run python scripts/calibrate_flux2.py --help
```

Expected: help text lists `klein-base-9b` as a valid `--variant`. **Do not actually run the calibration.**

- [ ] **Step 5: Lint**

```bash
uv run ruff check scripts/calibrate_flux2.py
uv run ruff format --check scripts/calibrate_flux2.py
```

Expected: green.

- [ ] **Step 6: Commit**

```bash
git add scripts/calibrate_flux2.py
git commit -m "feat(scripts): unblock klein-base-9b in calibrate_flux2.py"
```

---

### Task 5: Write the validation script

**File (new):** `scripts/validate_klein_base_9b.py`

- [ ] **Step 1: Write the script**

Single file. Reuses the `Flux2Klein` loader pattern from `scripts/bench_speedup.py`. Does NOT use subprocess isolation (this is a validation, not a release-gate bench).

Skeleton:

```python
"""One-shot validation that flux2-klein-base-9b's reused base-4b coefficients
work at the canonical 50-step CFG recipe.

Generates one fixed prompt at seed 42, 1024x768, num_inference_steps=50,
guidance=4.0, both vanilla and wrapped via apply_teacache. Decodes through
the VAE, computes SSIM, writes _artifacts/validation_klein_base_9b.json.
Exits non-zero if SSIM < 0.95.

Usage:
  uv run python scripts/validate_klein_base_9b.py
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Reuse the comparison prompt for continuity with COMPARISON.md
PROMPT = (
    "Portrait of a young woman with auburn hair and green eyes, soft "
    "golden-hour window light, photorealistic, shallow depth of field, "
    "50mm prime lens, subtle freckles, neutral background, cinematic "
    "color grading."
)
SEED = 42
HEIGHT = 1024
WIDTH = 768
STEPS = 50
GUIDANCE = 4.0
SSIM_THRESHOLD = 0.95


def _detect_hardware() -> dict[str, Any]:
    chip = subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
    ).stdout.strip() or "Apple Silicon"
    ram_bytes = int(
        subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout.strip() or "0"
    )
    return {
        "chip": chip,
        "ram_gb": round(ram_bytes / (1024**3)),
        "machine": platform.machine(),
        "os": f"{platform.system()} {platform.release()}",
    }


def _load_flux() -> Any:
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

    flux = Flux2Klein(quantize=4, model_config=ModelConfig.flux2_klein_base_9b())
    flux.freeze()
    return flux


def _generate(flux: Any) -> Any:
    import mlx.core as mx
    image = flux.generate_image(
        prompt=PROMPT,
        seed=SEED,
        num_inference_steps=STEPS,
        height=HEIGHT,
        width=WIDTH,
        guidance=GUIDANCE,
    )
    mx.eval(mx.zeros(1))
    return image


def _to_numpy(image: Any) -> Any:
    import numpy as np
    pil = image.image  # mflux GeneratedImage.image is a PIL.Image
    return np.array(pil).astype(np.float32) / 255.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "_artifacts" / "validation_klein_base_9b.json",
    )
    args = parser.parse_args()

    from mlx_teacache import apply_teacache
    from skimage.metrics import structural_similarity as ssim

    print(f"=== klein-base-9b validation: {STEPS} steps, guidance={GUIDANCE} ===")
    flux = _load_flux()

    print(">> vanilla generation")
    t0 = time.perf_counter()
    vanilla_image = _generate(flux)
    vanilla_seconds = time.perf_counter() - t0
    print(f"   {vanilla_seconds:.1f}s")
    vanilla_np = _to_numpy(vanilla_image)

    print(">> wrapper generation")
    t0 = time.perf_counter()
    with apply_teacache(flux) as handle:
        wrapper_image = _generate(flux)
        wrapper_seconds = time.perf_counter() - t0
        skipped = handle.stats.skipped_count
        computed = handle.stats.computed_count
        thresh = handle.rel_l1_thresh
    print(f"   {wrapper_seconds:.1f}s, skipped={skipped}/{computed + skipped}, thresh={thresh}")
    wrapper_np = _to_numpy(wrapper_image)

    score = ssim(vanilla_np, wrapper_np, data_range=1.0, channel_axis=-1)
    passed = score >= SSIM_THRESHOLD

    report = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%MZ"),
        "hardware": _detect_hardware(),
        "prompt": PROMPT,
        "seed": SEED,
        "height": HEIGHT,
        "width": WIDTH,
        "num_inference_steps": STEPS,
        "guidance": GUIDANCE,
        "rel_l1_thresh_used": thresh,
        "vanilla_seconds": vanilla_seconds,
        "wrapper_seconds": wrapper_seconds,
        "wrapper_skipped": skipped,
        "wrapper_computed": computed,
        "ssim": float(score),
        "ssim_threshold": SSIM_THRESHOLD,
        "passed": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"\nReport written: {args.output}")
    print(f"SSIM: {score:.4f} (threshold {SSIM_THRESHOLD})")
    print("RESULT: PASS" if passed else "RESULT: FAIL")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Verify the script imports and parses without running**

```bash
uv run python scripts/validate_klein_base_9b.py --help
```

Expected: argparse help text, no model load.

- [ ] **Step 3: Lint + typecheck**

```bash
uv run ruff check scripts/validate_klein_base_9b.py
uv run ruff format scripts/validate_klein_base_9b.py
uv run mypy scripts/validate_klein_base_9b.py
```

Apply formatter changes if any. Expected: green.

- [ ] **Step 4: Commit**

```bash
git add scripts/validate_klein_base_9b.py
git commit -m "feat(scripts): one-shot validation harness for klein-base-9b coefficient reuse"
```

---

### Task 6: Wire klein-base-9b into bench_speedup.py

**File:** `scripts/bench_speedup.py`

- [ ] **Step 1: Find the variant table**

```bash
grep -n "klein-base-4b\|klein-9b\|--variant\|VARIANT_CONFIGS\|variant ==" scripts/bench_speedup.py | head -40
```

Identify how variants are registered (choices list + variant-config dict / if-elif tree).

- [ ] **Step 2: Add klein-base-9b config entry**

Mirror the klein-base-4b CFG entry. Default recipe: 50 steps, guidance=4.0, three-way mode default-on.

Concrete edit (adapt to actual file shape):

- Add `"klein-base-9b"` to the `--variant` argparse choices list.
- Add a model-loader branch:
  ```python
  elif variant == "klein-base-9b":
      cfg = ModelConfig.flux2_klein_base_9b()
      return Flux2Klein(quantize=4, model_config=cfg)
  ```
- Add a default-recipe block keyed off `klein-base-9b` returning `(num_inference_steps=50, guidance=4.0, three_way=True)`.

- [ ] **Step 3: Dry-run --help**

```bash
uv run python scripts/bench_speedup.py --help
```

Expected: `klein-base-9b` listed in `--variant` choices. **Do not run the bench.**

- [ ] **Step 4: Lint**

```bash
uv run ruff check scripts/bench_speedup.py
uv run ruff format --check scripts/bench_speedup.py
```

Expected: green.

- [ ] **Step 5: Commit**

```bash
git add scripts/bench_speedup.py
git commit -m "feat(scripts): wire klein-base-9b into bench_speedup.py (50 steps, g=4.0, three-way)"
```

---

### Task 7: Parametrize FLUX.2 Klein real-weight tests

**Files:**
- Modify: existing parametrized tests under `tests/` that hit real Klein weights (typically named `test_*_klein*.py` or similar; gated by `@pytest.mark.slow` and HF_TOKEN check)

- [ ] **Step 1: Find the parametrized test fixtures**

```bash
grep -rn "klein-base-4b\|flux2_klein_base_4b\|HF_TOKEN" tests/ | head -30
```

Identify the parametrize markers / pytest_generate_tests hooks that drive variant coverage on real weights.

- [ ] **Step 2: Add klein-base-9b to the parametrize lists**

Mirror what was done for klein-base-4b in v0.4.0. Typically a `@pytest.mark.parametrize("variant", [..., "flux2-klein-base-9b"])` or a `_VARIANTS` constant.

If the test uses model-config loaders, add a klein-base-9b branch:
```python
elif variant == "flux2-klein-base-9b":
    return Flux2Klein(quantize=4, model_config=ModelConfig.flux2_klein_base_9b())
```

- [ ] **Step 3: Confirm test discovery picks up the new parametrization**

```bash
uv run pytest tests/ --collect-only -q -k "klein_base_9b or klein-base-9b" 2>&1 | head -30
```

Expected: at least one collected test ID mentions klein-base-9b. **Do not run; these are heavy weight-loading tests.**

- [ ] **Step 4: Lint**

```bash
uv run ruff check tests/
uv run ruff format --check tests/
```

Expected: green.

- [ ] **Step 5: Commit**

```bash
git add tests/
git commit -m "test: parametrize FLUX.2 Klein real-weight suites for klein-base-9b"
```

---

### Task 8: Pure-core / non-network tests pass end-to-end

- [ ] **Step 1: Run the fast test suite**

```bash
uv run pytest tests/ -v -m "not slow and not network"
```

Expected: all pass, no regressions from the variant addition. Address any failures before continuing.

- [ ] **Step 2: Run repo lint + typecheck**

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src/
```

Expected: all green.

- [ ] **Step 3: Commit nothing (this is a verification step)**

If any cleanup is needed, commit it with a chore message.

---

### Task 9: Docs — placeholders for tomorrow's numbers

**Files:**
- Modify: `README.md`, `CHANGELOG.md`, `docs/calibration.md`, `ROADMAP.md`

- [ ] **Step 1: README "Supported models" table**

Add a row for `flux2-klein-base-9b` right after `flux2-klein-base-4b`. Use a placeholder for the validation SSIM number — `<TBD: validation pass>` — and a footnote ³ pointing at the validation evidence path. Mark the row with a clear "v0.5.0" tag in the comment so we know to come back tomorrow to fill in real numbers.

- [ ] **Step 2: README "When to use mlx-teacache"**

Add klein-base-9b to the non-distilled FLUX.2 list alongside klein-base-4b.

- [ ] **Step 3: CHANGELOG.md v0.5.0 stub**

Add a top-of-file v0.5.0 section. Headline: variant addition + coefficient-reuse pattern. Leave the measured-numbers paragraph with `<TBD: validation SSIM + three-way bench>` placeholders.

- [ ] **Step 4: docs/calibration.md — coefficient-reuse section**

Add a short section ("Reusing coefficients across model sizes within an architecture family") explaining the v0.5.0 pattern: when it's appropriate (same family + same recipe + empirical validation), when it's not (cross-recipe or cross-architecture), with a forward reference to v0.5.0's validation evidence path.

- [ ] **Step 5: ROADMAP.md update**

Don't move the entry yet — it stays in "Active" until validation passes tomorrow. Leave the existing v0.5.0 entry as-is.

- [ ] **Step 6: Lint docs**

```bash
uv run ruff check README.md CHANGELOG.md docs/  # if ruff is configured for prose; otherwise skip
```

(Most ruff configs don't lint markdown; this is a no-op for safety.)

- [ ] **Step 7: Commit**

```bash
git add README.md CHANGELOG.md docs/calibration.md
git commit -m "docs(v0.5.0): scaffold klein-base-9b entries (placeholders for tomorrow's numbers)"
```

---

## **STOP HERE (today).**

Today's work is complete. The branch has:

- Variant detection + acceptance test
- Coefficient registry with intentional reuse + identity test
- API docstring + default-thresh wiring
- Calibration script unblocked
- Validation script committed (not run)
- Bench script wired (not run)
- Real-weight test suites parametrized (not run)
- Doc placeholders with `<TBD>` markers for tomorrow's measured numbers

**Tomorrow's tasks below are pre-staged but NOT executed today.**

---

## Phase 2 — Validation + bench + ship (tomorrow, generation)

### Task 10: Run the validation pass

**~1-2h on M1 Max. Run in main thread, NOT a subagent.**

- [ ] **Step 1: Confirm HF auth + model access**

```bash
hf auth whoami
hf download black-forest-labs/FLUX.2-klein-base-9B --revision main --include "*.safetensors" --dry-run 2>&1 | head -5
```

Expected: shows the user, and the dry-run lists weight files. If the model is gated and license isn't accepted, the dry-run fails — go accept on HF first.

- [ ] **Step 2: Run validation**

```bash
uv run python scripts/validate_klein_base_9b.py 2>&1 | tee /tmp/validate-klein-base-9b.log
```

Expected: `_artifacts/validation_klein_base_9b.json` produced. Exit code 0 (SSIM ≥ 0.95) is GO; exit 1 is STOP-and-recalibrate.

- [ ] **Step 3: Inspect output**

```bash
cat _artifacts/validation_klein_base_9b.json
```

Confirm SSIM, skip count, threshold.

- [ ] **Step 4: Commit the validation artifact**

```bash
git add _artifacts/validation_klein_base_9b.json
git commit -m "chore: commit klein-base-9b validation evidence (SSIM ≥ 0.95)"
```

---

### Task 11: Run the three-way bench

**~2-3h on M1 Max. Run in main thread.**

- [ ] **Step 1: Kick off**

```bash
uv run python scripts/bench_speedup.py --variant klein-base-9b --three-way 2>&1 | tee /tmp/bench-klein-base-9b.log
```

- [ ] **Step 2: Capture the report**

The bench should write its JSON report; commit it under `_artifacts/`. Record vanilla / no-gate / gated medians + skip counts + speedup factors.

- [ ] **Step 3: Commit**

```bash
git add _artifacts/<bench-report-filename>.json
git commit -m "chore: commit klein-base-9b three-way bench evidence"
```

---

### Task 12: Fill in docs with measured numbers

- [ ] **Step 1: Replace `<TBD>` placeholders** in README, CHANGELOG, ROADMAP with the actual validation SSIM and three-way speedup attribution.

- [ ] **Step 2: Move v0.5.0 from Active to Released in ROADMAP.md**

- [ ] **Step 3: Run humanizer over the new README + CHANGELOG prose** (the v0.5.0 entry is substantive new public-facing prose).

- [ ] **Step 4: Commit**

```bash
git add README.md CHANGELOG.md ROADMAP.md
git commit -m "docs(v0.5.0): fill in measured numbers and release notes"
```

---

### Task 13: PR + CI + STOP

- [ ] **Step 1: Push branch**

```bash
git push -u origin feature/v0.5.0-klein-base-9b
```

- [ ] **Step 2: Open PR**

```bash
gh pr create --title "v0.5.0: flux2-klein-base-9b support via coefficient reuse" --body "$(cat <<'EOF'
## Summary

Ships v0.5.0 adding `flux2-klein-base-9b` (non-distilled FLUX.2 Klein 9B, FLUX NC license) to the supported-variants list.

## What changed

[...measured numbers, validation SSIM, three-way bench attribution...]

## Test plan

- [ ] CI green (lint + format + mypy + test suite)
- [ ] Reviewer confirms validation evidence in `_artifacts/validation_klein_base_9b.json`
- [ ] Reviewer confirms three-way bench attribution distinguishes gating contribution from `mx.compile`-path avoidance
EOF
)"
```

- [ ] **Step 3: STOP**

Per the release-flow rule: do NOT call `gh pr merge`. Hand the PR link + summary to the user. The human merges on GitHub.

---

### Task 14: Post-merge — pull main + cut tag (user-authorized)

- [ ] **Step 1: After human merges**, pull main locally.

```bash
git checkout main && git pull --ff-only origin main
```

- [ ] **Step 2: Wait for explicit user authorization to push the tag.**

The tag push triggers PyPI publish via Trusted Publishing. This requires an explicit "push the tag" from the user — same protocol as v0.4.1.

- [ ] **Step 3: Tag + push (only after authorization)**

```bash
git tag v0.5.0 -m "v0.5.0: flux2-klein-base-9b support"
git push origin v0.5.0
```

- [ ] **Step 4: Confirm PyPI publish**

```bash
gh run watch <workflow-run-id>
# or check https://pypi.org/project/mlx-teacache/
```
