# v0.4.0 — `flux2-klein-base-4b` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `flux2-klein-base-4b` (Apache-2.0, non-distilled, 25-step calibration) as the fifth supported variant in mlx-teacache, scoped to `guidance=1.0` only. CFG-engaged caching is deferred to v0.4.1.

**Architecture:** Five additive change sites mirroring the v0.3 Klein 9B integration: replace a `_not_wired` placeholder in the calibration script, extend `detect.py`'s `Literal` + `_SUPPORTED` tuple + alias branch, add one row to the coefficient registry after calibration, add one branch to `scripts/bench_speedup.py`, and parametrize the FLUX.2 test fixtures with variant-aware generation kwargs (distilled stays at 8 steps; base-4b runs at 25). Plus a documentation bundle (the doc-clarity edits already in the working tree, plus v0.4-specific README/CHANGELOG/calibration.md/ROADMAP updates).

**Tech Stack:** Python 3.11+, `mflux>=0.17,<0.18`, MLX, `numpy.polyfit` (origin-constrained least-squares variant), pytest, ruff.

**Spec:** [`docs/superpowers/specs/2026-05-17-flux2-klein-base-4b-design.md`](../specs/2026-05-17-flux2-klein-base-4b-design.md). Spec audit: [`docs/superpowers/notes/2026-05-17-flux2-klein-base-4b-spec-audit.md`](../notes/2026-05-17-flux2-klein-base-4b-spec-audit.md). Plan audit (resolved): [`docs/superpowers/notes/2026-05-17-flux2-klein-base-4b-plan-audit.md`](../notes/2026-05-17-flux2-klein-base-4b-plan-audit.md).

---

## File map

| File | Responsibility | Action |
|---|---|---|
| `scripts/calibrate_flux2.py` | Calibration driver | Replace `_not_wired("v0.4.0")` placeholder for `klein-base-4b` with a real factory + 25-step config + output JSON path. |
| `scripts/_calibration_flux2_klein_base_4b.json` | base-4b calibration report (NEW) | Written by `--variant klein-base-4b` run. Coefficients + R² + raw `(x, y)` arrays. |
| `src/mlx_teacache/integrations/mflux/detect.py` | Variant detection | Extend `VariantId` Literal, extend `_SUPPORTED`, add alias branch in `identify_variant()`. |
| `src/mlx_teacache/api.py` | Public API | Update docstring at lines 141-146 to include the new variant. No runtime change. |
| `src/mlx_teacache/coefficients.py` | Coefficient registry | Add `_REGISTRY["flux2-klein-base-4b"]` entry with calibrated coefficients + `Provenance(source="builtin", ...)`. |
| `scripts/bench_speedup.py` | Reproducible bench | Add `klein-base-4b` to `--variant` choices, loader, and variant_config (25 steps, g=1.0). |
| `tests/test_detect.py` | Detect unit tests | Replace base-4b-rejected test with base-4b-accepted; keep base-9b-rejected. |
| `tests/test_coefficients.py` | Registry unit tests | Add base-4b assertion; extend parametrized supported-list test. |
| `tests/test_api.py` | Public API smoke (parity-marked) | Add Klein base-4b smoke test. |
| `tests/test_image_quality_flux2.py` | SSIM PR-gate (parity-marked) | Convert `_gen_kwargs_klein()` to variant-aware dispatch; extend `flux2_klein` fixture's params. |
| `tests/test_parity_flux2.py` | Parity oracle (parity-marked) | Same dispatch + fixture extension as image-quality tests. |
| `README.md` | User-facing docs | Add base-4b row to supported-variants table; add base-4b benchmarks row (g=1.0 label); add CFG-fallback note in Limitations. |
| `CHANGELOG.md` | Release notes | Add `## [0.4.0]` entry. |
| `docs/calibration.md` | Calibration procedure | Add base-4b row to "Built-in coefficient sources" table. |
| `ROADMAP.md` | Roadmap | Move v0.4.0 entry from "Active" to "Released"; v0.4.1 (CFG) is already in "Active" via the working-tree edit. |

---

## Preconditions (author machine)

Run these once before starting the task sequence. They are not steps in the plan because they don't change any tracked file; they're machine setup.

1. **Branch from main.** Local working tree currently has uncommitted doc-clarity edits (README, CHANGELOG, ROADMAP, postmortem coda) + the v0.4 spec file + the spec audit note. These will be committed as Task 1.

   ```bash
   git status --short
   # Expect: M README.md, M CHANGELOG.md, M ROADMAP.md, M docs/superpowers/notes/2026-05-16-...postmortem.md,
   #         ?? docs/superpowers/specs/2026-05-17-flux2-klein-base-4b-design.md,
   #         ?? docs/superpowers/notes/2026-05-17-flux2-klein-base-4b-spec-audit.md,
   #         (plus older v0.3 notes that won't be committed here)
   git checkout -b feature/v0.4.0-klein-base-4b
   ```

2. **Hugging Face auth + weight download.** base-4b weights are ~15 GB. The HF gated-access flow has already been completed in prior sessions; this is just the download.

   ```bash
   hf auth whoami   # verify still authenticated
   hf download black-forest-labs/FLUX.2-klein-base-4B
   ```

   This is overnight-safe. Run it in a separate terminal if local bandwidth is slow; the rest of the plan only needs it for Task 7 (calibration) and Task 10 (real-weight tests).

3. **CI gate to verify before any push:**

   ```bash
   uv run ruff check . && uv run ruff format --check . && uv run pytest tests/ -m "not parity and not slow and not benchmark and not network"
   ```

---

## Task 1: Commit doc-clarity working-tree edits

These edits cleanup v0.3 messaging around distilled-vs-non-distilled. They were prepared during the v0.4 brainstorming session (2026-05-17) and held in the working tree per the user's "batch with v0.4 PR" preference. Commit them first so they land cleanly in the v0.4 PR's history before any new code lands.

**Files:**
- Modify: `README.md` (already edited, ~10 line delta around lines 170-205)
- Modify: `CHANGELOG.md` (already edited, ~9 line delta in v0.3.0 entry)
- Modify: `ROADMAP.md` (already edited, ~41 line delta — restructured Active section + added v0.4.1 entry + strengthened Out-of-scope)
- Modify: `docs/superpowers/notes/2026-05-16-flux2-teacache-non-engagement-postmortem.md` (already edited, ~4 line coda)
- Create: `docs/superpowers/specs/2026-05-17-flux2-klein-base-4b-design.md` (already created)
- Create: `docs/superpowers/notes/2026-05-17-flux2-klein-base-4b-spec-audit.md` (already created)

- [ ] **Step 1: Verify working-tree state is as expected**

  ```bash
  git status --short
  git diff --stat README.md CHANGELOG.md ROADMAP.md docs/superpowers/notes/2026-05-16-flux2-teacache-non-engagement-postmortem.md
  ```

  Expect 4 modified files in the diff stat, plus the two untracked new files (spec + audit).

- [ ] **Step 2: Run local CI gate before committing**

  ```bash
  uv run ruff check . && uv run ruff format --check . && uv run pytest tests/ -m "not parity and not slow and not benchmark and not network"
  ```

  Expected: ruff green, 114 pure-core tests pass.

- [ ] **Step 3: Stage and commit the doc-clarity bundle**

  ```bash
  git add README.md CHANGELOG.md ROADMAP.md \
          docs/superpowers/notes/2026-05-16-flux2-teacache-non-engagement-postmortem.md \
          docs/superpowers/specs/2026-05-17-flux2-klein-base-4b-design.md \
          docs/superpowers/notes/2026-05-17-flux2-klein-base-4b-spec-audit.md

  git commit -m "$(cat <<'EOF'
  docs: clarify distilled-vs-non-distilled framing; add v0.4 spec + audit

  Cleans up v0.3 messaging across user-facing docs so it is unambiguous
  that distilled FLUX.2 Klein 4B + 9B at 4-8 step defaults are out of
  scope for algorithmic step-skipping by design. README's "If you want
  the gate to engage on Klein, bump rel_l1_thresh to 0.30 or higher"
  advice is removed; users are redirected to the v0.4 base-4b path
  instead. CHANGELOG, ROADMAP, and the 2026-05-16 postmortem get the
  same framing.

  ROADMAP additionally: (a) promotes flux2-klein-base-4b to be THE v0.4
  active item (was previously sub-bulleted next to a "FLUX.2 caching
  research" track that has now been dropped per the postmortem coda);
  (b) adds v0.4.1 (CFG per-branch caching for FLUX.2) as the next
  active item after base-4b; (c) strengthens the "Out of scope" section
  with explicit "no algorithmic step-skipping on distilled schedules"
  language so the decision is durable.

  Adds v0.4 spec + spec-audit response notes to the docs/superpowers
  tree for traceability.
  EOF
  )"
  ```

- [ ] **Step 4: Verify commit**

  ```bash
  git log --oneline -1
  git show --stat HEAD
  ```

  Expected: one new commit titled "docs: clarify distilled-vs-non-distilled framing; add v0.4 spec + audit" touching exactly the six files above.

---

## Task 2: Wire calibration script — replace `_not_wired` placeholder for base-4b

The calibration script's `_VARIANTS` dict already has `klein-base-4b` declared with `_not_wired("v0.4.0")` as a placeholder. Replace with a real factory + 25-step config + output JSON path.

**Files:**
- Modify: `scripts/calibrate_flux2.py:58-100` (add `_model_config_klein_base_4b` factory next to existing 4b/9b factories; update `_VARIANTS["klein-base-4b"]` entry)
- Test: `tests/test_calibrate_flux2.py` (or whichever file currently tests the calibration script's structure — check `tests/` for an existing calibration unit test first)

- [ ] **Step 1: Check whether a calibration-script unit test exists**

  ```bash
  ls tests/ | grep -i calibrat
  grep -rn "calibrate_flux2\|_VARIANTS\|_not_wired" tests/
  ```

  If a test file exists, modify it in steps 2-3 below. If not, skip steps 2-3 (the calibration script's runtime behavior is gated by the data-capture run in Task 7).

- [ ] **Step 2: If a calibration unit test exists, add a failing test for the wired base-4b variant**

  Add to whichever test file is appropriate (paste the test verbatim, replace path):

  ```python
  def test_klein_base_4b_variant_is_wired():
      """klein-base-4b should no longer raise NotImplementedError on factory call."""
      import importlib
      mod = importlib.import_module("scripts.calibrate_flux2")
      entry = mod._VARIANTS["klein-base-4b"]
      assert entry["num_inference_steps"] == 25
      assert entry["output_json"] == "_calibration_flux2_klein_base_4b.json"
      # Factory should NOT raise — but we don't actually call it here because
      # it loads mflux types lazily. Just assert it's not the _not_wired closure.
      assert entry["model_config_factory"].__name__ != "_raise"
  ```

  Run: `uv run pytest tests/test_calibrate_flux2.py::test_klein_base_4b_variant_is_wired -v` (adjust path).

  Expected: FAIL — entry still has placeholder values (`num_inference_steps=None`, `output_json=None`, factory is `_raise`).

- [ ] **Step 3: Implement the wiring**

  Edit `scripts/calibrate_flux2.py`. Below the existing `_model_config_klein_9b` factory (around line 67), add:

  ```python
  def _model_config_klein_base_4b() -> Any:
      from mflux.models.common.config.model_config import ModelConfig

      return ModelConfig.flux2_klein_base_4b()
  ```

  Then update the `_VARIANTS["klein-base-4b"]` entry. Replace the existing block:

  ```python
      "klein-base-4b": {
          "variant_id": "flux2-klein-base-4b",
          "model_config_factory": _not_wired("v0.4.0"),
          "num_inference_steps": None,
          "output_json": None,
      },
  ```

  with:

  ```python
      "klein-base-4b": {
          "variant_id": "flux2-klein-base-4b",
          "model_config_factory": _model_config_klein_base_4b,
          "num_inference_steps": 25,
          "output_json": "_calibration_flux2_klein_base_4b.json",
      },
  ```

  Also update the module docstring (lines 1-7) — remove the "wired in v0.4.0" claim for base-4b:

  Replace:

  ```python
  """Calibrate FLUX.2 polynomial coefficients for one variant.

  Run as: `uv run python scripts/calibrate_flux2.py --variant klein-4b`
          `uv run python scripts/calibrate_flux2.py --variant klein-9b`

  klein-base-4b and klein-base-9b are declared but raise NotImplementedError
  (wired in v0.4.0 and v0.5.0 respectively).
  ```

  With:

  ```python
  """Calibrate FLUX.2 polynomial coefficients for one variant.

  Run as: `uv run python scripts/calibrate_flux2.py --variant klein-4b`
          `uv run python scripts/calibrate_flux2.py --variant klein-9b`
          `uv run python scripts/calibrate_flux2.py --variant klein-base-4b --fit-mode origin`

  klein-base-9b is declared but raises NotImplementedError (wired in v0.5.0).
  ```

- [ ] **Step 4: Run the new unit test (or the full script module-load) to verify**

  If you added the unit test in step 2:
  ```bash
  uv run pytest tests/test_calibrate_flux2.py::test_klein_base_4b_variant_is_wired -v
  ```
  Expected: PASS.

  Either way, also verify the script still imports cleanly:
  ```bash
  uv run python -c "from scripts.calibrate_flux2 import _VARIANTS; print(_VARIANTS['klein-base-4b'])"
  ```
  Expected: prints a dict with `'num_inference_steps': 25` and a `_model_config_klein_base_4b` function (not `_raise`).

- [ ] **Step 5: Commit**

  ```bash
  git add scripts/calibrate_flux2.py tests/test_calibrate_flux2.py 2>/dev/null || git add scripts/calibrate_flux2.py
  git commit -m "feat(calibrate): wire klein-base-4b variant in calibration script

  Replaces the _not_wired(\"v0.4.0\") placeholder with a real
  _model_config_klein_base_4b factory + 25-step config + output JSON
  path. Module docstring updated to drop the 'wired in v0.4.0' claim
  for base-4b. klein-base-9b stays as _not_wired(\"v0.5.0\")."
  ```

---

## Task 3: Wire detect — Literal + `_SUPPORTED` + alias branch

The detect layer currently has three places that enumerate supported variants: the `VariantId = Literal[...]` type, the `_SUPPORTED` tuple, and a string-equality cascade inside `identify_variant()`'s `Flux2Klein` branch. All three must be updated; adding to `_SUPPORTED` alone is not enough (this was finding F2 in the spec audit).

**Files:**
- Modify: `src/mlx_teacache/integrations/mflux/detect.py:15-22` (Literal + tuple)
- Modify: `src/mlx_teacache/integrations/mflux/detect.py:75-84` (alias branch in `identify_variant`)
- Test: `tests/test_detect.py:68-71` (replace base-4b-rejected with base-4b-accepted)

- [ ] **Step 1: Update the failing test in `tests/test_detect.py`**

  Replace this block (currently around lines 68-71):

  ```python
  def test_flux2_klein_base_4b_rejected():
      with pytest.raises(IncompatibleModelError):
          identify_variant(_FakeFlux2Klein("flux2-klein-base-4b"))
  ```

  with:

  ```python
  def test_identify_flux2_klein_base_4b():
      assert identify_variant(_FakeFlux2Klein("flux2-klein-base-4b")) == "flux2-klein-base-4b"
  ```

  Keep `test_flux2_klein_base_9b_rejected` unchanged — base-9b is still v0.5.

- [ ] **Step 2: Run the modified test to verify it fails**

  ```bash
  uv run pytest tests/test_detect.py::test_identify_flux2_klein_base_4b -v
  ```

  Expected: FAIL — `identify_variant()` currently raises `IncompatibleModelError` because the `flux2-klein-base-4b` alias is not in the Flux2Klein branch's cascade.

- [ ] **Step 3: Update `detect.py`'s Literal type**

  Replace line 15:

  ```python
  VariantId = Literal["flux1-dev", "flux1-schnell", "flux2-klein-4b", "flux2-klein-9b"]
  ```

  with:

  ```python
  VariantId = Literal["flux1-dev", "flux1-schnell", "flux2-klein-4b", "flux2-klein-9b", "flux2-klein-base-4b"]
  ```

- [ ] **Step 4: Update `_SUPPORTED` tuple**

  Replace lines 17-22:

  ```python
  _SUPPORTED: tuple[str, ...] = (
      "flux1-dev",
      "flux1-schnell",
      "flux2-klein-4b",
      "flux2-klein-9b",
  )
  ```

  with:

  ```python
  _SUPPORTED: tuple[str, ...] = (
      "flux1-dev",
      "flux1-schnell",
      "flux2-klein-4b",
      "flux2-klein-9b",
      "flux2-klein-base-4b",
  )
  ```

- [ ] **Step 5: Add alias branch in `identify_variant()`**

  Inside the `if isinstance(flux, _Flux2KleinType):` block, after the `"flux2-klein-9b"` branch, add a `"flux2-klein-base-4b"` branch. Replace the block at lines 75-84:

  ```python
      if isinstance(flux, _Flux2KleinType):
          if "flux2-klein-4b" in aliases:
              return "flux2-klein-4b"
          if "flux2-klein-9b" in aliases:
              return "flux2-klein-9b"
          raise IncompatibleModelError(
              actual_type=actual_type,
              actual_model_name=model_name,
              supported=list(_SUPPORTED),
          )
  ```

  with:

  ```python
      if isinstance(flux, _Flux2KleinType):
          if "flux2-klein-4b" in aliases:
              return "flux2-klein-4b"
          if "flux2-klein-9b" in aliases:
              return "flux2-klein-9b"
          if "flux2-klein-base-4b" in aliases:
              return "flux2-klein-base-4b"
          raise IncompatibleModelError(
              actual_type=actual_type,
              actual_model_name=model_name,
              supported=list(_SUPPORTED),
          )
  ```

- [ ] **Step 6: Update the module-level docstring rejection list**

  At the top of `detect.py`, update the docstring (lines 1-7) to reflect that base-4b is now accepted:

  Replace:

  ```python
  """Identify which supported variant a flux instance is, or raise IncompatibleModelError.
  ```

  with itself (no change to that line) but in `identify_variant()`'s docstring (lines 46-50), replace:

  ```python
      """Return the variant_id for a supported mflux Flux1 / Flux2Klein instance.

      Raises IncompatibleModelError for unsupported model_name, unsupported
      Flux2Klein configuration (base-4b, base-9b variants), or any non-Flux type."""
  ```

  with:

  ```python
      """Return the variant_id for a supported mflux Flux1 / Flux2Klein instance.

      Raises IncompatibleModelError for unsupported model_name, unsupported
      Flux2Klein configuration (base-9b variant; base-9b lands in v0.5), or
      any non-Flux type."""
  ```

  Also update `tests/test_detect.py`'s module docstring (lines 5-8):

  Replace:

  ```python
  - Flux2Klein + flux2_klein_4b ⇒ flux2-klein-4b
  - Flux2Klein + flux2_klein_9b ⇒ flux2-klein-9b
  Rejects everything else (Klein base-4b, base-9b, unknown Flux1 aliases,
  non-Flux types) with IncompatibleModelError."""
  ```

  with:

  ```python
  - Flux2Klein + flux2_klein_4b ⇒ flux2-klein-4b
  - Flux2Klein + flux2_klein_9b ⇒ flux2-klein-9b
  - Flux2Klein + flux2_klein_base_4b ⇒ flux2-klein-base-4b
  Rejects everything else (Klein base-9b until v0.5, unknown Flux1 aliases,
  non-Flux types) with IncompatibleModelError."""
  ```

- [ ] **Step 7: Run the test to verify it passes**

  ```bash
  uv run pytest tests/test_detect.py -v
  ```

  Expected: all detect tests PASS. The previously-rejected case (`test_flux2_klein_base_4b_rejected`) was replaced with the accepted case (`test_identify_flux2_klein_base_4b`); `test_flux2_klein_base_9b_rejected` still passes (base-9b is still out of scope).

- [ ] **Step 8: Commit**

  ```bash
  git add src/mlx_teacache/integrations/mflux/detect.py tests/test_detect.py
  git commit -m "feat(detect): accept flux2-klein-base-4b alias

  Extends VariantId Literal, _SUPPORTED tuple, and identify_variant()'s
  Flux2Klein alias cascade to recognize 'flux2-klein-base-4b'. base-9b
  stays rejected until v0.5. Test docstring updated to match."
  ```

---

## Task 4: Update `api.py` docstring (no runtime change)

The `_predict` guard is already broadened to `startswith(\"flux2-\")`, but `apply_teacache()`'s docstring at `api.py:141-146` enumerates the four supported variants as a hard-coded comment. Add base-4b so the docstring matches reality.

**Files:**
- Modify: `src/mlx_teacache/api.py:141-148`

- [ ] **Step 1: Update the docstring**

  Replace lines 141-148:

  ```python
      """Enable TeaCache step-skipping on an mflux Flux1 or Flux2Klein instance.

      Supported variants (detected via flux.model_config.aliases):
        - flux1-dev, flux1-schnell
        - flux2-klein-4b, flux2-klein-9b

      See docs/superpowers/specs/2026-05-14-mlx-teacache-design.md §6.1 for the
      full docstring; this is the runtime entry point."""
  ```

  with:

  ```python
      """Enable TeaCache step-skipping on an mflux Flux1 or Flux2Klein instance.

      Supported variants (detected via flux.model_config.aliases):
        - flux1-dev, flux1-schnell
        - flux2-klein-4b, flux2-klein-9b (both distilled; gate does not engage
          at default threshold on the 4-8 step schedule — wall-clock benefit
          comes from mx.compile-path avoidance, see README "How the speedup
          happens")
        - flux2-klein-base-4b (non-distilled; calibrated at 25 steps; engages
          the polynomial gate at guidance=1.0. CFG / guidance > 1.0 falls back
          to vanilla mflux pending v0.4.1.)

      See docs/superpowers/specs/2026-05-14-mlx-teacache-design.md §6.1 for the
      full docstring; this is the runtime entry point."""
  ```

- [ ] **Step 2: Verify pure-core tests still pass (no runtime regression)**

  ```bash
  uv run pytest tests/ -m "not parity and not slow and not benchmark and not network" -v
  ```

  Expected: all 114 pure-core tests still pass.

- [ ] **Step 3: Commit**

  ```bash
  git add src/mlx_teacache/api.py
  git commit -m "docs(api): document flux2-klein-base-4b in apply_teacache docstring

  No runtime change. Updates the supported-variants comment in
  apply_teacache()'s docstring to include base-4b and to be explicit
  about (a) which variants get algorithmic step-skipping vs compile
  avoidance, and (b) that base-4b is currently g=1.0-only pending
  v0.4.1's CFG per-branch caching."
  ```

---

## Task 5: Wire `scripts/bench_speedup.py` for klein-base-4b

The bench script needs to accept `--variant klein-base-4b`, route it through `_load_flux2_klein`, and run at 25 steps + g=1.0. The variant label `g=1.0` should appear in the summary so the row in README's benchmarks table is unambiguous about the scope.

**Files:**
- Modify: `scripts/bench_speedup.py:52-95` (loader + variant_config)
- Modify: `scripts/bench_speedup.py:122-128` (argparse choices)

- [ ] **Step 1: Add klein-base-4b to the Klein loader**

  Replace lines 48-60:

  ```python
  def _load_flux2_klein(variant: str) -> Any:
      from mflux.models.common.config.model_config import ModelConfig
      from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

      if variant == "klein-4b":
          cfg = ModelConfig.flux2_klein_4b()
      elif variant == "klein-9b":
          cfg = ModelConfig.flux2_klein_9b()
      else:
          raise ValueError(f"unsupported klein variant: {variant!r}")
      flux = Flux2Klein(quantize=4, model_config=cfg)
      flux.freeze()
      return flux
  ```

  with:

  ```python
  def _load_flux2_klein(variant: str) -> Any:
      from mflux.models.common.config.model_config import ModelConfig
      from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

      if variant == "klein-4b":
          cfg = ModelConfig.flux2_klein_4b()
      elif variant == "klein-9b":
          cfg = ModelConfig.flux2_klein_9b()
      elif variant == "klein-base-4b":
          cfg = ModelConfig.flux2_klein_base_4b()
      else:
          raise ValueError(f"unsupported klein variant: {variant!r}")
      flux = Flux2Klein(quantize=4, model_config=cfg)
      flux.freeze()
      return flux
  ```

- [ ] **Step 2: Add klein-base-4b to `_variant_config`**

  Replace lines 76-95 (the body of `_variant_config`):

  ```python
  def _variant_config(variant: str) -> dict[str, Any]:
      if variant in ("klein-4b", "klein-9b"):
          return {
              "loader": _load_flux2_klein,
              "num_inference_steps": 8,
              "guidance": 1.0,
          }
      if variant == "flux1-dev":
          return {
              "loader": _load_flux1,
              "num_inference_steps": 25,
              "guidance": 3.5,
          }
      if variant == "flux1-schnell":
          return {
              "loader": _load_flux1,
              "num_inference_steps": 4,
              "guidance": 1.0,
          }
      raise ValueError(f"unknown variant: {variant!r}")
  ```

  with:

  ```python
  def _variant_config(variant: str) -> dict[str, Any]:
      if variant in ("klein-4b", "klein-9b"):
          return {
              "loader": _load_flux2_klein,
              "num_inference_steps": 8,
              "guidance": 1.0,
          }
      if variant == "klein-base-4b":
          # Non-distilled; calibrated at 25 steps. g=1.0 only — CFG falls
          # back to vanilla mflux pending v0.4.1 (per-branch caching).
          return {
              "loader": _load_flux2_klein,
              "num_inference_steps": 25,
              "guidance": 1.0,
          }
      if variant == "flux1-dev":
          return {
              "loader": _load_flux1,
              "num_inference_steps": 25,
              "guidance": 3.5,
          }
      if variant == "flux1-schnell":
          return {
              "loader": _load_flux1,
              "num_inference_steps": 4,
              "guidance": 1.0,
          }
      raise ValueError(f"unknown variant: {variant!r}")
  ```

- [ ] **Step 3: Update argparse choices**

  Replace lines 124-128:

  ```python
      parser.add_argument(
          "--variant",
          required=True,
          choices=["klein-4b", "klein-9b", "flux1-dev", "flux1-schnell"],
      )
  ```

  with:

  ```python
      parser.add_argument(
          "--variant",
          required=True,
          choices=["klein-4b", "klein-9b", "klein-base-4b", "flux1-dev", "flux1-schnell"],
      )
  ```

- [ ] **Step 4: Update the module docstring example list**

  Replace lines 17-21:

  ```python
  Run as:
    uv run python scripts/bench_speedup.py --variant klein-9b
    uv run python scripts/bench_speedup.py --variant klein-4b
    uv run python scripts/bench_speedup.py --variant flux1-dev
  ```

  with:

  ```python
  Run as:
    uv run python scripts/bench_speedup.py --variant klein-9b
    uv run python scripts/bench_speedup.py --variant klein-4b
    uv run python scripts/bench_speedup.py --variant klein-base-4b   # 25-step, g=1.0
    uv run python scripts/bench_speedup.py --variant flux1-dev
  ```

- [ ] **Step 5: Sanity-check the script parses arguments without loading mflux**

  Don't actually run the bench yet (weights need to be downloaded first; bench is Task 10's job). Just verify `--help` lists base-4b:

  ```bash
  uv run python scripts/bench_speedup.py --help | grep -A1 "variant"
  ```

  Expected: `klein-base-4b` appears in the choices list.

- [ ] **Step 6: Commit**

  ```bash
  git add scripts/bench_speedup.py
  git commit -m "feat(bench): support klein-base-4b variant in bench_speedup.py

  Adds klein-base-4b to the loader, variant config (25 steps, g=1.0),
  and argparse choices. Inline comment in _variant_config records the
  v0.4.0 g=1.0-only scope so the row in README's bench table is
  unambiguous about which guidance regime was measured."
  ```

---

## Task 6: Pre-flight — confirm weights are downloaded

Calibration (Task 7) and real-weight tests (Task 10) both need `black-forest-labs/FLUX.2-klein-base-4B` cached locally. This task verifies the download finished without blocking on it during a long-running task.

**Files:** none (verification only)

- [ ] **Step 1: Verify cache state**

  ```bash
  hf_cache=$(hf cache scan 2>/dev/null | grep -i "FLUX.2-klein-base-4B" || echo "not cached")
  echo "$hf_cache"
  du -sh ~/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-base-4B 2>/dev/null || echo "(cache dir not yet populated)"
  ```

  Expected: `~/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-base-4B` exists and is ~15 GB.

- [ ] **Step 2: If not cached, kick off download (long-running)**

  ```bash
  hf download black-forest-labs/FLUX.2-klein-base-4B 2>&1 | tee /tmp/hf-download-klein-base-4b.log
  ```

  Run as a foreground command from the user's terminal (don't background-dispatch this through a subagent — it's a multi-GB download and should be visible to the user). Re-run Step 1 after it finishes to verify the cache populated.

  No commit for this task — it's machine setup.

---

## Task 7: Run the calibration (long-running, data-capture task)

This is the calibration bench: 10 prompts × 25 steps × seed=42 on M1 Max at quantize=4, 512×512, g=1.0, origin-constrained polyfit. Expected ~8 hours on M1 Max. This task is **not TDD**; it produces a data artifact (the JSON report) that subsequent tasks consume.

Run in the main session via `Bash` with `run_in_background=true` per the global CLAUDE.md rule about heavy generations. Tee the log to `/tmp/` so the user can `tail -f` for live progress.

**Files:**
- Create: `scripts/_calibration_flux2_klein_base_4b.json` (written by the calibration script)

- [ ] **Step 1: Kick off the calibration**

  ```bash
  uv run python scripts/calibrate_flux2.py --variant klein-base-4b --fit-mode origin 2>&1 | tee /tmp/calibrate-klein-base-4b.log
  ```

  Use `run_in_background=true`. Tell the user the log path so they can `tail -f /tmp/calibrate-klein-base-4b.log`. Expected wall-clock: ~8 hours on M1 Max with thermal throttling. The script captures `(mod_in, body_out_concat)` per step across 10 prompts and fits an origin-constrained polynomial.

- [ ] **Step 2: Wait for the calibration to complete**

  Use `Monitor` or wait for the `run_in_background` task notification. **Do not poll** with sleep loops.

  When it completes, verify the output:

  ```bash
  ls -la scripts/_calibration_flux2_klein_base_4b.json
  python -c "import json; d = json.load(open('scripts/_calibration_flux2_klein_base_4b.json')); print('fit_mode:', d.get('fit_mode'), 'R^2:', d.get('fit_r_squared'), 'coeffs:', d.get('coefficients_c4_to_c0'), 'x range:', d.get('x_min'), d.get('x_max'), 'y range:', d.get('y_min'), d.get('y_max'))"
  ```

  Expected: file exists; `fit_mode = "origin"`; `coefficients_c4_to_c0` is a 5-tuple ending in `0.0`; `x_min`/`x_max` and `y_min`/`y_max` are floats; `fit_r_squared` is a float in (0, 1).

- [ ] **Step 3: Commit the calibration artifact**

  ```bash
  git add scripts/_calibration_flux2_klein_base_4b.json
  git commit -m "data: calibration report for flux2-klein-base-4b (25 steps, origin-constrained)

  10 prompts x 25 steps x seed=42 on M1 Max 32GB, mflux 0.17.5,
  quantize=4, 512x512, guidance=1.0, origin-constrained polyfit.
  Coefficients to be lifted into _REGISTRY in the next task."
  ```

---

## Task 8: Wire the coefficient registry from the calibration JSON

Read the calibration JSON, lift the coefficients + R² into `src/mlx_teacache/coefficients.py`, add the `_REGISTRY` entry and provenance, add the corresponding test assertion.

**Files:**
- Modify: `src/mlx_teacache/coefficients.py:99-143` (constants + `_REGISTRY`)
- Modify: `tests/test_coefficients.py:54-57` (extend parametrized supported-list test)
- Modify: `tests/test_coefficients.py` (add `test_load_builtin_flux2_klein_base_4b_has_dataset_and_metric`)

- [ ] **Step 1: Read the calibration JSON to get the exact coefficient values**

  ```bash
  python <<'PY'
  import json
  d = json.load(open("scripts/_calibration_flux2_klein_base_4b.json"))
  print("coefficients_c4_to_c0:", d["coefficients_c4_to_c0"])
  print("fit_r_squared:", d["fit_r_squared"])
  print("x_min,x_max:", d["x_min"], d["x_max"])
  print("y_min,y_max:", d["y_min"], d["y_max"])
  PY
  ```

  Record the 5-tuple and R² value — they're filled into the code in Step 3.

- [ ] **Step 2: Add a failing test for the new registry entry**

  In `tests/test_coefficients.py`, add a new test below `test_load_builtin_flux2_klein_9b_has_dataset_and_metric`:

  ```python
  def test_load_builtin_flux2_klein_base_4b_has_dataset_and_metric():
      coeffs, prov = load_builtin("flux2-klein-base-4b")
      assert len(coeffs) == 5
      assert all(math.isfinite(c) for c in coeffs)
      assert coeffs[-1] == 0.0  # origin-constrained: poly(0) = 0
      assert prov.source == "builtin"
      assert prov.revision is not None and prov.revision.startswith("in-repo-2026-05-")
      assert prov.calibration_dataset is not None
      assert "25 steps" in prov.calibration_dataset
      assert "guidance=1.0" in prov.calibration_dataset
      assert prov.fit_metric is not None
      assert prov.fit_metric_value is not None
      assert 0.0 < prov.fit_metric_value <= 1.0
      assert (prov.reference_url or "").endswith("calibrate_flux2.py")
  ```

  Also extend the parametrized list at line 54-57:

  Replace:

  ```python
  @pytest.mark.parametrize(
      "variant_id",
      ["flux1-dev", "flux1-schnell", "flux2-klein-4b", "flux2-klein-9b"],
  )
  ```

  with:

  ```python
  @pytest.mark.parametrize(
      "variant_id",
      ["flux1-dev", "flux1-schnell", "flux2-klein-4b", "flux2-klein-9b", "flux2-klein-base-4b"],
  )
  ```

- [ ] **Step 3: Run the test to verify it fails**

  ```bash
  uv run pytest tests/test_coefficients.py::test_load_builtin_flux2_klein_base_4b_has_dataset_and_metric -v
  ```

  Expected: FAIL — `load_builtin("flux2-klein-base-4b")` raises `CalibrationError` because the registry doesn't have an entry yet.

- [ ] **Step 4: Add the constant + registry entry**

  In `src/mlx_teacache/coefficients.py`, below `_FLUX2_KLEIN_9B_COEFFS` (around line 75-82), add:

  ```python
  # Origin-constrained polyfit, derived in-repo on 2026-05-<DD> from
  # flux2-klein-base-4B at 25-step schedule (non-distilled). Constants
  # filled in from scripts/_calibration_flux2_klein_base_4b.json's
  # coefficients_c4_to_c0 field. The trailing 0.0 reflects the origin
  # constraint (poly(0) = 0).
  _FLUX2_KLEIN_BASE_4B_COEFFS: tuple[float, float, float, float, float] = (
      <c4>, <c3>, <c2>, <c1>, 0.0,
  )
  ```

  Replace `<c4>`, `<c3>`, `<c2>`, `<c1>` with the actual numeric values from the JSON (recorded in Step 1).

  Then in the `_REGISTRY` dict (currently ending around line 143), add a new entry after the `flux2-klein-9b` entry. Insert immediately before the closing `}`:

  ```python
      "flux2-klein-base-4b": (
          _FLUX2_KLEIN_BASE_4B_COEFFS,
          Provenance(
              source="builtin",
              revision="in-repo-2026-05-<DD>-origin",
              calibration_dataset="10 prompts x 25 steps x seed=42, M1 Max 32GB, bf16, 512x512, guidance=1.0, origin-constrained polyfit",
              fit_metric="constrained-LSQ R^2 on consecutive-step (mod_in, body_out) rel-L1 pairs (poly(0)=0)",
              fit_metric_value=<r_squared>,
              reference_url="https://github.com/IonDen/mlx-teacache/blob/main/scripts/calibrate_flux2.py",
          ),
      ),
  ```

  Replace `<DD>` with the actual calibration date (read it off the calibration JSON's timestamp if recorded, or use today's date) and `<r_squared>` with the actual `fit_r_squared` value from the JSON.

- [ ] **Step 5: Run the tests to verify they pass**

  ```bash
  uv run pytest tests/test_coefficients.py -v
  ```

  Expected: all coefficient tests PASS — the new `test_load_builtin_flux2_klein_base_4b_has_dataset_and_metric` test passes, and the extended parametrized `test_every_supported_variant_has_builtin_coefficients` test runs against 5 variants instead of 4.

- [ ] **Step 6: Run the full pure-core CI gate**

  ```bash
  uv run ruff check . && uv run ruff format --check . && uv run pytest tests/ -m "not parity and not slow and not benchmark and not network"
  ```

  Expected: ruff green, all pure-core tests pass (now 116, up from 114: one new `test_load_builtin_flux2_klein_base_4b...` + one extra parametrized case).

- [ ] **Step 7: Commit**

  ```bash
  git add src/mlx_teacache/coefficients.py tests/test_coefficients.py
  git commit -m "feat(coefficients): register flux2-klein-base-4b

  Adds _FLUX2_KLEIN_BASE_4B_COEFFS constant (origin-constrained
  polyfit on 25-step calibration data from
  scripts/_calibration_flux2_klein_base_4b.json) and a corresponding
  _REGISTRY entry with Provenance(source='builtin'). Test coverage
  parametrized over the new variant id."
  ```

---

## Task 9: Variant-aware test kwargs + parametrize FLUX.2 tests for base-4b

The existing FLUX.2 image-quality + parity tests use a `_gen_kwargs_klein()` helper that hardcodes `num_inference_steps=8` (distilled default). base-4b runs at 25 steps. Convert the helper to a variant-aware dispatch and extend the `flux2_klein` fixture's params to include base-4b. This was finding F5 in the spec audit.

**Files:**
- Modify: `tests/test_image_quality_flux2.py:80-88` (rewrite `_gen_kwargs_klein` as variant-aware)
- Modify: `tests/test_image_quality_flux2.py:117-131` (extend fixture params + add base-4b config branch)
- Modify: `tests/test_parity_flux2.py` (same: locate `_gen_kwargs_klein` equivalent + fixture + extend)
- Modify: `tests/test_api.py:161-178` (add Klein base-4b smoke test alongside the 9B smoke test)

- [ ] **Step 1: Convert `_gen_kwargs_klein` to variant-aware dispatch in `test_image_quality_flux2.py`**

  Replace lines 80-88:

  ```python
  def _gen_kwargs_klein(prompt: str, *, guidance: float = 1.0) -> dict[str, Any]:
      return {
          "prompt": prompt,
          "seed": 42,
          "num_inference_steps": 8,
          "height": 512,
          "width": 512,
          "guidance": guidance,
      }
  ```

  with:

  ```python
  def _gen_kwargs_klein(prompt: str, *, variant_id: str = "flux2-klein-4b", guidance: float = 1.0) -> dict[str, Any]:
      """Generation kwargs for FLUX.2 Klein variants.

      Distilled Klein 4B / 9B use the 8-step default schedule (matches their
      runtime usage). base-4b uses the calibration-time 25-step schedule.
      Callers that need a different step count should override after this
      returns."""
      if variant_id in ("flux2-klein-4b", "flux2-klein-9b"):
          num_inference_steps = 8
      elif variant_id == "flux2-klein-base-4b":
          num_inference_steps = 25
      else:
          raise ValueError(f"unsupported variant_id for _gen_kwargs_klein: {variant_id!r}")
      return {
          "prompt": prompt,
          "seed": 42,
          "num_inference_steps": num_inference_steps,
          "height": 512,
          "width": 512,
          "guidance": guidance,
      }
  ```

- [ ] **Step 2: Update every call site of `_gen_kwargs_klein` in `test_image_quality_flux2.py`**

  ```bash
  grep -n "_gen_kwargs_klein" tests/test_image_quality_flux2.py
  ```

  At every call site, add the `variant_id=variant_id` keyword. Example replacements (you'll need to handle every grep hit):

  Replace:
  ```python
  kw = _gen_kwargs_klein(PR_TIME_PROMPT)
  ```
  with:
  ```python
  kw = _gen_kwargs_klein(PR_TIME_PROMPT, variant_id=variant_id)
  ```

  And replace:
  ```python
  kw = _gen_kwargs_klein(prompt)
  ```
  with:
  ```python
  kw = _gen_kwargs_klein(prompt, variant_id=variant_id)
  ```

  This requires the `variant_id` to be available in the test function scope — which it already is via the `flux2_klein` fixture (the fixture params ARE the variant ids; expose them by switching the fixture to also return the variant_id or by reading it from `request.param`). The cleanest fix is to make the fixture yield a tuple:

- [ ] **Step 3: Convert `flux2_klein` fixture to expose variant_id, extend params to include base-4b**

  Replace lines 117-131:

  ```python
  @pytest.fixture(scope="module", params=["flux2-klein-4b", "flux2-klein-9b"])
  def flux2_klein(request) -> Any:
      from mflux.models.common.config.model_config import ModelConfig
      from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

      variant_id = request.param
      if variant_id == "flux2-klein-4b":
          cfg = ModelConfig.flux2_klein_4b()
      elif variant_id == "flux2-klein-9b":
          cfg = ModelConfig.flux2_klein_9b()
      else:
          pytest.fail(f"unhandled variant_id={variant_id!r}")
      flux = Flux2Klein(quantize=4, model_config=cfg)
      flux.freeze()
      return flux
  ```

  with:

  ```python
  @pytest.fixture(scope="module", params=["flux2-klein-4b", "flux2-klein-9b", "flux2-klein-base-4b"])
  def flux2_klein(request) -> tuple[Any, str]:
      """Returns (flux instance, variant_id) so tests can pass variant_id to _gen_kwargs_klein."""
      from mflux.models.common.config.model_config import ModelConfig
      from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

      variant_id = request.param
      if variant_id == "flux2-klein-4b":
          cfg = ModelConfig.flux2_klein_4b()
      elif variant_id == "flux2-klein-9b":
          cfg = ModelConfig.flux2_klein_9b()
      elif variant_id == "flux2-klein-base-4b":
          cfg = ModelConfig.flux2_klein_base_4b()
      else:
          pytest.fail(f"unhandled variant_id={variant_id!r}")
      flux = Flux2Klein(quantize=4, model_config=cfg)
      flux.freeze()
      return flux, variant_id
  ```

  Then update every test function that takes `flux2_klein: Any` to unpack the tuple. Find them:

  ```bash
  grep -n "flux2_klein:" tests/test_image_quality_flux2.py
  ```

  At each test function signature, replace:

  ```python
  def test_xyz(flux2_klein: Any, ...) -> None:
      ...
      vanilla_latent = _capture(flux2_klein, **kw)
  ```

  with:

  ```python
  def test_xyz(flux2_klein: tuple[Any, str], ...) -> None:
      flux, variant_id = flux2_klein
      ...
      kw = _gen_kwargs_klein(prompt, variant_id=variant_id)  # or PR_TIME_PROMPT
      vanilla_latent = _capture(flux, **kw)
  ```

  (You'll need to replace every `flux2_klein` reference in the function body with `flux`, and add `variant_id=variant_id` to every `_gen_kwargs_klein` call.)

- [ ] **Step 4: Do the same for `tests/test_parity_flux2.py`**

  Locate the `_gen_kwargs_klein` and `flux2_klein` fixture in that file:

  ```bash
  grep -n "_gen_kwargs_klein\|^def flux2_klein\|@pytest.fixture" tests/test_parity_flux2.py
  ```

  Apply the same conversions: variant-aware `_gen_kwargs_klein`, fixture-returns-tuple, test functions unpack and propagate `variant_id`.

  Use the exact same code blocks as in Steps 1-3 of `test_image_quality_flux2.py`. (The two files have parallel structure; this is mechanical.)

- [ ] **Step 5: Add Klein base-4b smoke test to `tests/test_api.py`**

  After `test_apply_teacache_accepts_flux2_klein_9b` (currently ending at line 178), add:

  ```python
  @pytest.mark.parity
  def test_apply_teacache_accepts_flux2_klein_base_4b():
      """Smoke: apply_teacache returns a handle with the right variant_id on Klein base-4B.
      Catches api.py regressions in the variant_id Literal or the FLUX.2 _predict guard."""
      from mflux.models.common.config.model_config import ModelConfig
      from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

      from mlx_teacache import apply_teacache

      flux = Flux2Klein(quantize=4, model_config=ModelConfig.flux2_klein_base_4b())
      flux.freeze()
      handle = apply_teacache(flux)
      try:
          assert handle.variant_id == "flux2-klein-base-4b"
          assert len(handle.coefficients) == 5
          assert handle.provenance.source == "builtin"
      finally:
          handle.restore()
  ```

- [ ] **Step 6: Run the pure-core test suite to ensure no regressions**

  ```bash
  uv run ruff check . && uv run ruff format --check . && uv run pytest tests/ -m "not parity and not slow and not benchmark and not network"
  ```

  Expected: ruff green, all 116+ pure-core tests pass. The parity-marked tests added in Steps 1-5 won't run here — they need real weights.

- [ ] **Step 7: Commit**

  ```bash
  git add tests/test_image_quality_flux2.py tests/test_parity_flux2.py tests/test_api.py
  git commit -m "test(flux2): variant-aware _gen_kwargs_klein + base-4b parametrization

  Converts _gen_kwargs_klein() in test_image_quality_flux2.py and
  test_parity_flux2.py from a hardcoded 8-step (distilled) helper to a
  variant-aware dispatch: distilled Klein 4B/9B stay at 8 steps;
  flux2-klein-base-4b runs at 25 steps. flux2_klein fixture extended
  to return (flux, variant_id) so call sites propagate variant_id
  cleanly. test_api.py gains a parity-marked smoke test for base-4b.

  Resolves audit finding F5 (test plan was accidentally validating
  base-4b at the distilled 8-step schedule)."
  ```

---

## Task 10: Run real-weight tests — RELEASE GATE

This is the release-gate task. Run the parity oracle + SSIM PR-gate + bench against real `FLUX.2-klein-base-4B` weights at g=1.0. The bench's skip-count output is the load-bearing measurement for the release decision; if skip count is 0 at default threshold, route to Task 11 (contingency).

**Files:** none (validation only — no code changes)

- [ ] **Step 1: Run the parity PR-gate at threshold 0 for base-4b — narrow target**

  Target only the two PR-gate parity tests that run at `g=1.0` (the v0.4.0 scope). The full prompt-suite (`test_paired_parity_klein_full`, slow-marked) and the CFG-fallback test (`test_cfg_fallback_matches_vanilla` at `guidance=3.5`) are NOT part of the v0.4.0 release gate — CFG is explicitly deferred to v0.4.1.

  ```bash
  HF_TOKEN="$(hf auth token)" uv run pytest tests/test_parity_flux2.py::test_paired_parity_klein_pr_gate tests/test_parity_flux2.py::test_paired_parity_at_threshold_zero_klein_pr_gate -m parity -k "klein-base-4b" -v 2>&1 | tee /tmp/parity-base-4b.log
  ```

  Expected: cosine similarity ≥ `_FLUX2_COSINE_GATE` (defined as `0.97` in `tests/test_parity_flux2.py:63`; the measured value on existing FLUX.2 variants is ~0.99+, the gate is set 0.02 below to absorb prompt-to-prompt variance). If FAIL, do NOT proceed — investigate the forward path's eager-wrapper vs vanilla divergence on base-4b before doing anything else. CFG bit-exactness is observed in v0.4.1, not v0.4.0.

- [ ] **Step 2: Run the SSIM PR-gate for base-4b**

  ```bash
  HF_TOKEN="$(hf auth token)" uv run pytest tests/test_image_quality_flux2.py::test_default_threshold_ssim_klein_pr_gate -m parity -k "klein-base-4b" -v 2>&1 | tee /tmp/ssim-base-4b.log
  ```

  Expected: SSIM ≥ 0.85 on the red-apple prompt at default threshold. If FAIL, investigate the polynomial fit (it may be over-aggressive — consider whether the calibration captured enough prompt diversity).

- [ ] **Step 3: Run the bench at g=1.0**

  ```bash
  uv run python scripts/bench_speedup.py --variant klein-base-4b --report /tmp/bench-base-4b.json 2>&1 | tee /tmp/bench-base-4b.log
  ```

  Expected output sample:
  ```
  == Summary ==
    variant:          klein-base-4b
    num_inference_steps: 25
    guidance:         1.0
    reps:             3
    vanilla median:   <Xs>
    wrapper median:   <Ys>
    speedup (median): <Z>x
    skipped/computed: [<a>, <b>, <c>] / [<d>, <e>, <f>]
  ```

  Capture the median speedup and the per-rep skip counts — both go into the README's benchmarks row in Task 12.

- [ ] **Step 4: Evaluate the release gate**

  Decision rule from the spec:

  - **All three reps have `skipped_count >= 1` at default threshold?** → Proceed to Task 12 (docs).
  - **Any rep has `skipped_count == 0`?** → Proceed to Task 11 (0-skip contingency). Do NOT skip Task 11.

  Example check (read from the bench's JSON output):

  ```bash
  python <<'PY'
  import json
  d = json.load(open("/tmp/bench-base-4b.json"))
  skipped = d["skipped_counts"]
  print(f"skipped_counts: {skipped}")
  if all(s >= 1 for s in skipped):
      print("RELEASE GATE PASSED: proceed to Task 12")
  else:
      print("RELEASE GATE FAILED: route to Task 11 (0-skip contingency)")
  PY
  ```

- [ ] **Step 5: No commit — this is validation only. Continue to Task 11 or Task 12 based on Step 4's outcome.**

---

## Task 11: 0-skip contingency — STOP and escalate to human

This task runs ONLY IF Task 10's release gate failed (any rep produced `skipped_count == 0`). The spec lists three resolution paths; the plan does NOT pre-prescribe which one. The decision is made by the human after seeing the actual measurement data.

**Files:** none (decision point — no code changes in this task itself)

- [ ] **Step 1: Capture the failure context for the human**

  ```bash
  cat scripts/_calibration_flux2_klein_base_4b.json | python -c "
  import json, sys
  d = json.load(sys.stdin)
  print('=== Calibration data ===')
  print('fit_mode:', d.get('fit_mode'))
  print('coefficients:', d.get('coefficients_c4_to_c0'))
  print('R^2:', d.get('fit_r_squared'))
  print('x range:', d.get('x_min'), 'to', d.get('x_max'))
  print('y range (empirical):', d.get('y_min'), 'to', d.get('y_max'))
  "

  cat /tmp/bench-base-4b.json | python -c "
  import json, sys
  d = json.load(sys.stdin)
  print('=== Bench data ===')
  print('skipped_counts:', d['skipped_counts'])
  print('computed_counts:', d['computed_counts'])
  print('vanilla median:', d['vanilla_median'])
  print('wrapper median:', d['wrapper_median'])
  print('speedup:', d['speedup_median'])
  "
  ```

- [ ] **Step 2: Surface the data to the human with the three contingency options**

  Present the captured data (Step 1 output) along with the three paths from the spec:

  1. **Recalibrate.** If the calibration JSON's `y_min` is just slightly above `0.20` (e.g. `0.21 ≤ y_min ≤ 0.23`), an alternative origin-constrained fit with more prompts or a higher polynomial degree may produce coefficients whose minimum dips below `0.20`. Re-run Task 7 with the adjusted recipe.
  2. **Per-variant default threshold (API change, otherwise out of scope).** If `y_min >= 0.25` consistently, introduce a `default_thresh` field in the `Provenance` dataclass and look it up inside `apply_teacache`. Set base-4b's default to a value that produces ≥ 1 skip at SSIM ≥ 0.85. This is Approach B from the brainstorming, permitted only as a contingency.
  3. **Reframe v0.4.0 as structural-only release.** Drop the "first FLUX.2 variant where the polynomial gate engages" claim from README + CHANGELOG. Ship base-4b as supported with "0 skips at default; bump threshold at your own risk" framing.

  Wait for the human's decision. **Do not make this decision unilaterally** — each path has different downstream task implications.

- [ ] **Step 3: Execute the chosen contingency path**

  Whichever path the human picks, the next concrete sub-plan (re-run Task 7 with new prompts / add a `default_thresh` field to coefficients.py / rewrite the release framing) is determined post-decision. If Path 1 or 2 produces a passing bench, return to Task 10 Step 3 and verify the release gate.

- [ ] **Step 4: Once the contingency is resolved (skip count ≥ 1 at the chosen threshold, OR the release is reframed as structural), proceed to Task 12.**

---

## Task 12: v0.4 doc updates — branch by Task 10's release-gate outcome

Task 12 has **two mutually-exclusive branches**:

- **Task 12-engagement** runs if Task 10 Step 4 passed (skip count ≥ 1 across all 3 reps at default threshold). This is the expected path. Use the language "first FLUX.2 variant where the polynomial gate engages on its own (at g=1.0)."
- **Task 12-structural** runs if Task 10 routed to Task 11 (0-skip) AND the human chose contingency Path 3 (reframe as structural-only release). In this case the engagement-path language is forbidden — use the same "supported, gate does not engage at default; structural release only" framing as the v0.3 Klein 9B disclosure.

Pick exactly one of the two sub-tasks below based on Task 10/11's outcome.

---

### Task 12-engagement: v0.4 doc updates (engagement path)

Runs if Task 10's release gate passed at the default threshold. The doc-clarity edits already landed in Task 1's commit. This task adds the v0.4-specific content: a new variant row in the supported-variants table, a new benchmarks row from Task 10's bench, a CHANGELOG entry, a calibration.md row, and a ROADMAP update moving v0.4 to "Released."

**Files:**
- Modify: `README.md` (supported-variants table, benchmarks table, Limitations CFG-fallback note, optional quick-start tweak)
- Modify: `CHANGELOG.md` (add `## [0.4.0]` entry)
- Modify: `docs/calibration.md` (add base-4b row)
- Modify: `ROADMAP.md` (move v0.4.0 from "Active" to "Released")

- [ ] **Step 1: Add base-4b row to README's supported-variants table**

  Locate the supported-variants table in `README.md`. It currently lists `flux1-dev`, `flux1-schnell`, `flux2-klein-4b`, `flux2-klein-9b`. After the `flux2-klein-9b` row, add:

  ```markdown
  | `flux2-klein-base-4b`² | `Flux2Klein(model_config=ModelConfig.flux2_klein_base_4b())` | in-repo (25-step calibration, origin-constrained; see [`docs/calibration.md`](docs/calibration.md)) |
  ```

  Then add the footnote near the existing `¹` footnote:

  ```markdown
  ² `flux2-klein-base-4b` is the non-distilled FLUX.2 Klein 4B variant (Apache-2.0). TeaCache engages at `guidance=1.0` (gate runs at 25 steps, calibrated polynomial). At `guidance > 1.0` (CFG), the wrapper falls back to vanilla mflux — CFG-engaged caching lands in v0.4.1 (per-branch caching). The upstream BFL model card recommends `guidance_scale=4.0, num_inference_steps=50`; that recipe runs vanilla in v0.4.0.
  ```

- [ ] **Step 2: Add base-4b row to README's benchmarks table**

  Locate the benchmarks table (currently has FLUX.1-dev, klein-4b, klein-9b rows). After the `klein-9b` row, add a row with the measured numbers from Task 10's bench (the actual median wall-clock + skip counts from `/tmp/bench-base-4b.json`):

  ```markdown
  | `flux2-klein-base-4b`³ | 25 | <Xs> | <Ys> | <Z>× | **<a> / 25** | TeaCache step-skipping |
  ```

  Replace `<X>`, `<Y>`, `<Z>`, `<a>` with the actual numbers from the bench. Add the footnote:

  ```markdown
  ³ `flux2-klein-base-4b` at `guidance=1.0`. CFG (`guidance > 1.0`) falls back to vanilla mflux pending v0.4.1.
  ```

- [ ] **Step 3: Add CFG-fallback note to README Limitations**

  In the Limitations section (currently containing the distilled-out-of-scope bullet), add a separate bullet:

  ```markdown
  `flux2-klein-base-4b` runs TeaCache only at `guidance=1.0` in v0.4.0. At `guidance > 1.0` (CFG, e.g. the upstream-recommended `guidance_scale=4.0`), the wrapper falls back to vanilla mflux — caching is inactive. v0.4.1 will add per-branch CFG caching so the canonical upstream recipe (`guidance_scale=4.0, num_inference_steps=50`) is accelerated too.
  ```

- [ ] **Step 4: Add `## [0.4.0]` entry to CHANGELOG**

  Insert the new entry directly below the `## [Unreleased]` header, above the `## [0.3.0]` entry:

  ```markdown
  ## [0.4.0] — 2026-05-<DD>

  Adds `flux2-klein-base-4b` (Apache-2.0, non-distilled, 25-step calibration) as the first FLUX.2 variant where the polynomial gate engages on its own — at `guidance=1.0` only. CFG-engaged caching is deferred to v0.4.1.

  ### Added
  - **`flux2-klein-base-4b` support.** Apply TeaCache to mflux's `Flux2Klein` with `ModelConfig.flux2_klein_base_4b()`. Coefficients calibrated in-repo on M1 Max (10 prompts × 25 steps, origin-constrained polyfit). At `rel_l1_thresh=0.20` the polynomial gate skips <N>/25 steps (measured on the red-apple bench prompt at g=1.0), delivering ~<X>× wall-clock improvement over vanilla mflux. Output quality preserved (SSIM ≥ 0.85 PR-gate at default threshold).
  - New `--variant klein-base-4b` argument on `scripts/bench_speedup.py` (25 steps, g=1.0).
  - New `--variant klein-base-4b` argument on `scripts/calibrate_flux2.py` (replaces the v0.3 `_not_wired("v0.4.0")` placeholder).

  ### Changed
  - `apply_teacache()` accepts `Flux2Klein(model_config=ModelConfig.flux2_klein_base_4b())` instances. `detect.identify_variant()` returns `"flux2-klein-base-4b"`.
  - Test parametrization for FLUX.2 image-quality + parity tests extended with `klein-base-4b`; `_gen_kwargs_klein()` is now variant-aware (distilled Klein at 8 steps, base-4b at 25 steps).

  ### Scope notes
  - TeaCache on base-4b is engaged at `guidance=1.0` only. At `guidance > 1.0`, the wrapper records a `cfg-fallback` decision and runs vanilla mflux per the v0.1 design. The upstream BFL model card recommends `guidance_scale=4.0`; that recipe runs vanilla mflux speed in v0.4.0 and gets caching in v0.4.1 (per-branch caching for FLUX.2).
  - Distilled FLUX.2 Klein 4B + 9B remain out of scope for algorithmic step-skipping by design (already documented in v0.3.0).
  ```

  Replace `<DD>` with the release date, `<N>` and `<X>` with the actual bench numbers.

- [ ] **Step 5: Add base-4b row to `docs/calibration.md`**

  In the "Built-in coefficient sources" table, after the `flux2-klein-9b` row, add:

  ```markdown
  | `flux2-klein-base-4b` | Derived in-repo on 2026-05-<DD> from 10 prompts × **25 steps** × seed=42 on M1 Max 32GB, bf16, 512×512, guidance=1.0. **Origin-constrained** least-squares fit (forces `poly(0) = 0`); R² = <R>. First FLUX.2 variant where the polynomial gate is expected to engage at the package default `rel_l1_thresh=0.20`. | `_REGISTRY["flux2-klein-base-4b"]` in `coefficients.py`. Calibration script: `scripts/calibrate_flux2.py --variant klein-base-4b --fit-mode origin`. Full report: `scripts/_calibration_flux2_klein_base_4b.json`. |
  ```

  Replace `<DD>` and `<R>` with the actual values.

  Also add a new row to the "Producing new coefficients" section's command list:

  ```bash
  # Klein base-4B (origin-constrained polyfit; shipped since v0.4.0). Non-distilled
  # variant designed for 20-50 step generation; this calibration uses 25 steps.
  uv run python scripts/calibrate_flux2.py --variant klein-base-4b --fit-mode origin
  ```

  Insert below the existing klein-9b block, replacing the comment:

  ```bash
  # klein-base-4b and klein-base-9b are declared but raise
  # NotImplementedError until v0.4.0 and v0.5.0 respectively.
  ```

  with:

  ```bash
  # klein-base-9b is declared but raises NotImplementedError until v0.5.0.
  ```

- [ ] **Step 6: Update ROADMAP — move v0.4 to Released**

  In `ROADMAP.md`:

  Under the existing `## Released` section (currently lists v0.3.0, v0.1.x), add v0.4.0 as the newest entry. Add this as the first line of the Released list (above v0.3.0):

  ```markdown
  - **v0.4.0** — `flux2-klein-base-4b` (Apache-2.0, non-distilled, 25-step calibration). First FLUX.2 variant where the polynomial gate engages at the package default at `guidance=1.0`. CFG-engaged caching deferred to v0.4.1.
  ```

  Under the `## Active` section, the v0.4.0 entry should be removed (it's now released). The v0.4.1 entry (CFG per-branch caching) stays — it's now the next active item.

  Concretely: delete the entire `### v0.4.0: \`flux2-klein-base-4b\`` block from the Active section. v0.4.1 stays as the new first Active entry.

- [ ] **Step 7: Run the doc-clarity ruff + format check**

  ```bash
  uv run ruff check . && uv run ruff format --check .
  ```

  Expected: green (no Python code changed; this is just markdown).

- [ ] **Step 8: Commit**

  ```bash
  git add README.md CHANGELOG.md docs/calibration.md ROADMAP.md
  git commit -m "docs: v0.4.0 release entries — base-4b row + CHANGELOG + ROADMAP shuffle

  - README: new supported-variants row for flux2-klein-base-4b with
    CFG-fallback footnote; new benchmarks row at g=1.0; Limitations
    bullet explicitly calling out 'TeaCache on base-4b is g=1.0 only,
    CFG lands in v0.4.1'.
  - CHANGELOG: full v0.4.0 entry covering the variant, the bench
    measurements, and the scope/CFG-fallback story.
  - docs/calibration.md: new row in the built-in coefficient sources
    table; producer command added.
  - ROADMAP: v0.4.0 moved from Active to Released; v0.4.1 stays as the
    next Active item."
  ```

---

### Task 12-structural: v0.4 doc updates (structural-only path)

**Runs ONLY if Task 11 chose contingency Path 3 (reframe as structural-only).** Skip if Task 12-engagement ran. The "first FLUX.2 variant where the polynomial gate engages" language is forbidden in this branch — the gate did not engage at the default threshold, so the release is shipped as structural support without algorithmic step-skipping at default. This is the same shape as the v0.3 Klein 9B disclosure.

**Files:**
- Modify: `README.md` (supported-variants table, benchmarks table, Limitations CFG-fallback + 0-skip notes)
- Modify: `CHANGELOG.md` (add `## [0.4.0]` entry — structural-only framing)
- Modify: `docs/calibration.md` (add base-4b row — gate-does-not-engage framing)
- Modify: `ROADMAP.md` (move v0.4.0 from "Active" to "Released" with structural-only note)

- [ ] **Step 1: Add base-4b row to README's supported-variants table (structural framing)**

  After the `flux2-klein-9b` row, add:

  ```markdown
  | `flux2-klein-base-4b`² | `Flux2Klein(model_config=ModelConfig.flux2_klein_base_4b())` | in-repo (25-step calibration, origin-constrained; see [`docs/calibration.md`](docs/calibration.md)) |
  ```

  Footnote (structural framing):

  ```markdown
  ² `flux2-klein-base-4b` is the non-distilled FLUX.2 Klein 4B variant (Apache-2.0). At the package default `rel_l1_thresh=0.20` the polynomial gate produces 0 step-skips on the 25-step schedule (measured during v0.4.0 calibration); the wrapper still provides wall-clock improvement from `mx.compile`-path avoidance — same mechanism as distilled Klein 4B/9B. CFG (`guidance > 1.0`) also falls back to vanilla mflux pending v0.4.1. Whether a useful skip rate can be achieved on base-4b with a different threshold or calibration recipe is open research; see the v0.4.0 CHANGELOG for the measured behavior.
  ```

- [ ] **Step 2: Add base-4b row to README's benchmarks table (structural framing)**

  After the `klein-9b` row, add a row with measured numbers from Task 10's bench. Use `mx.compile` avoidance as the mechanism label, matching the distilled Klein 4B/9B rows:

  ```markdown
  | `flux2-klein-base-4b`³ | 25 | <Xs> | <Ys> | <Z>× | **0 / 25** | `mx.compile` avoidance only |
  ```

  Replace `<X>`, `<Y>`, `<Z>` with bench-measured values. Footnote:

  ```markdown
  ³ `flux2-klein-base-4b` at `guidance=1.0`. 0 skips at default threshold across 3 reps; the wall-clock improvement comes from `mx.compile`-path avoidance, same mechanism as Klein 4B/9B. See the v0.4.0 CHANGELOG and the calibration JSON for the empirical signal range. CFG (`guidance > 1.0`) also falls back to vanilla mflux pending v0.4.1.
  ```

- [ ] **Step 3: Add CFG-fallback + 0-skip notes to README Limitations**

  Add a separate bullet (structural framing):

  ```markdown
  `flux2-klein-base-4b` (v0.4.0) ships as structural support only: the polynomial gate produces 0 step-skips at the package default `rel_l1_thresh=0.20` on the calibrated 25-step schedule. Whether this is fixable (better calibration, higher threshold with quality bounds, different gate signal entirely) is open research — see the v0.4.0 CHANGELOG for the measured signal range. The wrapper still provides wall-clock improvement from `mx.compile`-path avoidance on chips where mflux compiles `_predict`. CFG (`guidance > 1.0`) also falls back to vanilla mflux pending v0.4.1.
  ```

- [ ] **Step 4: Add `## [0.4.0]` entry to CHANGELOG (structural framing)**

  Insert below `## [Unreleased]`:

  ```markdown
  ## [0.4.0] — 2026-05-<DD>

  Adds `flux2-klein-base-4b` (Apache-2.0, non-distilled, 25-step calibration) as the fifth supported variant. The polynomial gate produces 0 step-skips at the package default threshold on the calibrated 25-step schedule; the wrapper provides wall-clock improvement from `mx.compile`-path avoidance only, same mechanism as Klein 4B/9B. Whether a useful skip rate can be achieved on base-4b with a different threshold or calibration recipe is open research. CFG-engaged caching deferred to v0.4.1.

  ### Added
  - **`flux2-klein-base-4b` structural support.** Apply TeaCache to mflux's `Flux2Klein` with `ModelConfig.flux2_klein_base_4b()`. Coefficients calibrated in-repo on M1 Max (10 prompts × 25 steps, origin-constrained polyfit). At `rel_l1_thresh=0.20` the polynomial gate skips 0/25 steps; wall-clock improvement (~<X>× measured on M1 Max at g=1.0) comes from `mx.compile`-path avoidance. Output quality preserved (SSIM ≥ 0.85 PR-gate).
  - New `--variant klein-base-4b` argument on `scripts/bench_speedup.py` (25 steps, g=1.0).
  - New `--variant klein-base-4b` argument on `scripts/calibrate_flux2.py` (replaces the v0.3 `_not_wired("v0.4.0")` placeholder).

  ### Changed
  - `apply_teacache()` accepts `Flux2Klein(model_config=ModelConfig.flux2_klein_base_4b())` instances. `detect.identify_variant()` returns `"flux2-klein-base-4b"`.
  - Test parametrization for FLUX.2 image-quality + parity tests extended with `klein-base-4b`; `_gen_kwargs_klein()` is now variant-aware (distilled Klein at 8 steps, base-4b at 25 steps).

  ### Scope notes
  - TeaCache step-skipping does NOT engage on base-4b at the package default threshold. Empirical signal range on the v0.4 calibration is documented in `scripts/_calibration_flux2_klein_base_4b.json`. Whether a higher threshold or different calibration recipe can produce engagement with acceptable quality is an open question — out of scope for v0.4.0.
  - CFG / `guidance > 1.0` falls back to vanilla mflux per the v0.1 design; behavior unchanged from prior FLUX.2 variants. CFG caching is v0.4.1.
  - Distilled FLUX.2 Klein 4B + 9B remain out of scope for algorithmic step-skipping by design (already documented in v0.3.0).
  ```

- [ ] **Step 5: Add base-4b row to `docs/calibration.md` (structural framing)**

  In the "Built-in coefficient sources" table, after the `flux2-klein-9b` row, add:

  ```markdown
  | `flux2-klein-base-4b` | Derived in-repo on 2026-05-<DD> from 10 prompts × **25 steps** × seed=42 on M1 Max 32GB, bf16, 512×512, guidance=1.0. **Origin-constrained** least-squares fit (forces `poly(0) = 0`); R² = <R>. The polynomial gate produces 0 step-skips at the package default `rel_l1_thresh=0.20` on this calibration (empirical `y_min = <y_min>`, above the threshold). Wall-clock benefit on base-4b comes from `mx.compile`-path avoidance, same as Klein 4B/9B. Open research on whether engagement is achievable at a different threshold or with a different signal. | `_REGISTRY["flux2-klein-base-4b"]` in `coefficients.py`. Calibration script: `scripts/calibrate_flux2.py --variant klein-base-4b --fit-mode origin`. Full report: `scripts/_calibration_flux2_klein_base_4b.json`. |
  ```

  Same producer command addition as Task 12-engagement Step 5.

- [ ] **Step 6: Update ROADMAP — v0.4 to Released (structural framing)**

  Add v0.4.0 as the first line of the Released list:

  ```markdown
  - **v0.4.0** — `flux2-klein-base-4b` structural support (Apache-2.0, non-distilled, 25-step calibration). Gate does not engage at default threshold; wall-clock benefit is `mx.compile`-path avoidance only. CFG-engaged caching deferred to v0.4.1. Whether engagement on base-4b is achievable with a different calibration / threshold is open research.
  ```

  Remove the `### v0.4.0:` block from the Active section. v0.4.1 stays as the new first Active entry.

  **Additionally**, since this branch indicates the polynomial gate did not engage on base-4b at default, add a new "Deferred" entry under ROADMAP's appropriate section (probably "Future" or a new "Open research" subsection):

  ```markdown
  ### Open research: non-distilled FLUX.2 gate engagement

  v0.4.0 shipped `flux2-klein-base-4b` with 0 skips at default threshold. The polynomial gate is not engaging on a non-distilled FLUX.2 schedule, which was unexpected — NVIDIA's FLUX.2-dev blog shows engagement at threshold=0.05 on a 50-step schedule, and base-4b's non-distilled architecture should be similar enough that some engagement was likely. Open questions: (a) did our calibration set capture the wrong signal range? (b) is the polynomial functional form a bad fit for FLUX.2-family architectures (R² has been low across both 9B and base-4b)? (c) is the default threshold simply too low for FLUX.2? (d) would FBCache or DiCache work where the polynomial gate doesn't? No fixed release target.
  ```

- [ ] **Step 7: Run ruff/format**

  ```bash
  uv run ruff check . && uv run ruff format --check .
  ```

- [ ] **Step 8: Commit**

  ```bash
  git add README.md CHANGELOG.md docs/calibration.md ROADMAP.md
  git commit -m "docs: v0.4.0 release entries — structural-only framing (0 skips at default)

  Following Task 11's 0-skip contingency Path 3 (reframe as structural-
  only), v0.4.0 docs ship the same disclosure pattern v0.3 used for
  Klein 9B: variant is supported, polynomial gate does NOT engage at
  default threshold, wall-clock benefit comes from mx.compile-path
  avoidance only. CFG-engaged caching deferred to v0.4.1.

  ROADMAP gains an 'Open research: non-distilled FLUX.2 gate
  engagement' section documenting why this outcome was unexpected
  (NVIDIA's FLUX.2-dev blog shows engagement; our base-4b doesn't)
  and what questions are open for future investigation."
  ```

---

## Task 13: Open PR + CI

**Files:** none (PR creation only)

- [ ] **Step 1: Push the branch**

  ```bash
  git push -u origin feature/v0.4.0-klein-base-4b
  ```

- [ ] **Step 2: Open the PR**

  ```bash
  gh pr create --title "v0.4.0: flux2-klein-base-4b support (g=1.0 only; CFG in v0.4.1)" --body "$(cat <<'EOF'
  ## Summary
  - Adds `flux2-klein-base-4b` (Apache-2.0, non-distilled FLUX.2 Klein 4B) as the fifth supported variant.
  - Scoped to `guidance=1.0` only. CFG-engaged caching for FLUX.2 lands in v0.4.1 (per-branch caching).
  - Mirror of the v0.3 Klein 9B integration shape: calibration script + detect + registry + bench + tests + docs.
  - Bundles 4 doc-clarity edits prepared during 2026-05-17 brainstorming (README/CHANGELOG/ROADMAP/postmortem coda) so v0.4 ships with consistent distilled-vs-non-distilled messaging.

  ## Bench measurements (M1 Max, q4, 512×512, seed=42, g=1.0, 25 steps)
  - Vanilla median: <Xs>
  - Wrapper median: <Ys>
  - Median speedup: <Z>×
  - Skipped per rep: [<a>, <b>, <c>] / 25

  See `scripts/_calibration_flux2_klein_base_4b.json` for the calibration report and `/tmp/bench-base-4b.json` for the bench's full output.

  ## Test plan
  - [x] `uv run ruff check . && uv run ruff format --check .` green locally
  - [x] `uv run pytest tests/ -m "not parity and not slow and not benchmark and not network"` green locally (116+ pure-core tests)
  - [x] Parity oracle (`pytest tests/test_parity_flux2.py -m parity -k klein-base-4b`) green against real weights
  - [x] SSIM PR-gate (`pytest tests/test_image_quality_flux2.py::test_default_threshold_ssim_klein_pr_gate -m parity -k klein-base-4b`) green against real weights
  - [x] `scripts/bench_speedup.py --variant klein-base-4b` reports skip count ≥ 1 across all 3 reps at default threshold
  - [ ] CI green on PR (lint + typecheck + pure-core + mflux × 3 Python versions + coverage)

  ## Scope reminders
  - Distilled Klein 4B / 9B at 4-8 steps remain out of scope for algorithmic step-skipping by design (already shipped this messaging in v0.3.0).
  - `klein-base-9b` is deferred to v0.5.0 (FLUX Non-Commercial license + BFL safety filter).
  - CFG / `guidance > 1.0` falls back to vanilla mflux on every FLUX.2 variant in v0.4.0; CFG caching is v0.4.1.

  ## Spec + audit
  - Spec: `docs/superpowers/specs/2026-05-17-flux2-klein-base-4b-design.md`
  - Audit: `docs/superpowers/notes/2026-05-17-flux2-klein-base-4b-spec-audit.md` (all 6 findings resolved in spec; resolution table at the bottom of the spec)
  EOF
  )"
  ```

  Fill in `<X>`, `<Y>`, `<Z>`, `<a>`, `<b>`, `<c>` with Task 10's bench numbers.

- [ ] **Step 3: Wait for CI**

  Use `Monitor` or `gh run watch` — don't poll. Expected: all 7 CI jobs green (lint, typecheck, test-pure-core, test-mflux × 3 Python versions, coverage). test-parity is `workflow_dispatch`-only and skips on PR.

  ```bash
  gh pr checks <PR_NUMBER> --watch
  ```

---

## Task 14: Merge + tag v0.4.0

**Files:** none (release only)

- [ ] **Step 1: After CI green and human approval, merge the PR**

  ```bash
  gh pr merge <PR_NUMBER> --merge --delete-branch
  ```

- [ ] **Step 2: Fetch fresh main**

  ```bash
  git checkout main
  git pull origin main
  git log --oneline -3   # confirm merge commit is HEAD
  ```

- [ ] **Step 3: Cut the v0.4.0 tag from the merge commit**

  Get the merge commit SHA from Step 2's `git log`, then:

  ```bash
  git tag -a v0.4.0 <MERGE_SHA> -m "v0.4.0 — flux2-klein-base-4b support (g=1.0)

  - Adds flux2-klein-base-4b (Apache-2.0, non-distilled, 25-step
    calibration). First FLUX.2 variant where the polynomial gate
    engages at the package default at guidance=1.0.
  - CFG-engaged caching for FLUX.2 deferred to v0.4.1 (per-branch
    caching). guidance > 1.0 falls back to vanilla mflux per the v0.1
    design; behavior unchanged from prior FLUX.2 variants.
  - Bundles doc-clarity edits prepared during 2026-05-17
    brainstorming: README/CHANGELOG/ROADMAP/postmortem coda now
    consistently say distilled schedules are out of scope for
    algorithmic step-skipping by design.

  Denis Ineshin -- https://github.com/IonDen"
  ```

- [ ] **Step 4: Push the tag (triggers PyPI publish via release.yml)**

  ```bash
  git push origin v0.4.0
  ```

  This triggers the release workflow. Wait for the user's explicit "publish to PyPI" authorization before running this step — the sandbox classifier may block the tag push (as it did for v0.3.0); the human must approve.

- [ ] **Step 5: Monitor the release workflow**

  ```bash
  gh run list --workflow=release.yml --limit 1
  ```

  Watch the run via `gh run view <RUN_ID>`. Expected duration ~45 seconds (matching v0.3.0). Expected jobs: build, publish (PyPI Trusted Publishing), github-release — all green.

- [ ] **Step 6: Verify PyPI**

  ```bash
  pip index versions mlx-teacache 2>&1 | head -5
  ```

  Expected: `0.4.0` appears as the latest version.

---

## Self-review (post-plan-writing)

**Spec coverage:**

| Spec requirement | Plan task |
|---|---|
| Calibration script: replace `_not_wired` for base-4b | Task 2 |
| Coefficient registry: `_REGISTRY["flux2-klein-base-4b"]` with `Provenance(source="builtin", ...)` | Task 8 |
| Detect: `VariantId` Literal + `_SUPPORTED` + alias branch | Task 3 |
| API: docstring update at api.py:141-146 | Task 4 |
| Bench: new variant branch in `scripts/bench_speedup.py` | Task 5 |
| Tests: variant-aware `_gen_kwargs_klein` + fixture extension | Task 9 |
| Calibration JSON `_calibration_flux2_klein_base_4b.json` | Task 7 |
| PR-gate SSIM ≥ 0.85 release gate | Task 10 Step 2 |
| Skip count ≥ 1 release gate | Task 10 Step 4 |
| Cosine ≥ 0.99 parity release gate | Task 10 Step 1 |
| 0-skip contingency branch | Task 11 |
| README updates (variant row, bench row, CFG-fallback note) | Task 12 Steps 1-3 |
| CHANGELOG v0.4.0 entry | Task 12 Step 4 |
| `docs/calibration.md` row | Task 12 Step 5 |
| ROADMAP move v0.4.0 to Released | Task 12 Step 6 |
| Doc-clarity edits bundled | Task 1 |
| Tag v0.4.0 from merge commit | Task 14 Step 3 |

**Placeholder scan:** Plan uses `<DD>`, `<R>`, `<X>`, `<Y>`, `<Z>`, `<a>`, `<b>`, `<c>`, `<MERGE_SHA>`, `<PR_NUMBER>`, `<N>` as values to be filled in at runtime from measurements. These are not implementation-detail placeholders — they're explicit "fill in from the bench output" markers, each colocated with the exact data source. No TBD / TODO / "implement later" / "add appropriate error handling" / generic phrases.

**Type consistency:** `variant_id` strings used consistently as `"flux2-klein-base-4b"` (kebab-case) across detect/registry/tests/bench. `VariantId` Literal is the canonical source. The `_gen_kwargs_klein` signature, fixture return type `(flux, variant_id)`, and call-site unpacking are consistent across `test_image_quality_flux2.py` and `test_parity_flux2.py`.
