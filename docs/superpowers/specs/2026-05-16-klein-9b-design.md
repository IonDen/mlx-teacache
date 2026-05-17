# v0.3.0 — FLUX.2 Klein 9B support

**Release target:** v0.3.0
**Roadmap line:** ROADMAP.md "Deferred work → FLUX.2 Klein 9B variant support"
**Predecessor:** v0.2.0 (img2img + distilled-step warning)

## Goal

Extend mlx-teacache to support the `flux2-klein-9b` variant of mflux's `Flux2Klein` class, with fresh polynomial coefficients calibrated on the actual 9B transformer. Lay the variant-flag groundwork in the calibration script so v0.4 (`klein-base-4b`) and v0.5 (`klein-base-9b`) become small additive changes.

## Non-goals

- `flux2-klein-base-4b` (v0.4 release)
- `flux2-klein-base-9b` (v0.5 release)
- Any new TeaCache semantics, gate behavior, or stats fields
- mflux version bump (`>=0.17,<0.18` already includes `flux2_klein_9b`)
- Runtime license enforcement (docs-only call-out per brainstorm decision)

## Background

### Why split Klein 9B from base-* variants

The original v0.2.0 plan deferred Klein 9B "to v0.3.0" as a single bundle. During v0.3.0 brainstorming the user split the remaining Klein family across three releases (0.3 → 9B, 0.4 → base-4B, 0.5 → base-9B) to keep each release small and to validate the parameterized calibration path on one variant before doubling down.

### Why fresh coefficients per variant

The TeaCache polynomial captures the relationship between modulated block-0 input rel-L1 and transformer-body output rel-L1. Both signals are sensitive to:

- Transformer **width** (hidden dim) and **depth** (block count) — 9B is wider and deeper than 4B.
- The **noise schedule** the variant ships with — distilled Klein-* uses a short schedule (~8 steps); base-* uses long schedules (25–50 steps).

Reusing the 4B polynomial on 9B is *plausibly* fine for a distilled-to-distilled match but is a guess. The conservative path — and the one the user picked — is to fit a fresh polynomial per variant and let the SSIM gates validate the choice.

### Why drop `Img2ImgNotSupportedError` in v0.3.0

v0.2.0 deprecated it with a `DeprecationWarning` on construction and the CHANGELOG explicitly said "Removal planned for v0.3.0." Following through.

## Architecture

mlx-teacache's runtime is already variant-agnostic in lifecycle, gate, forward, and cache — those routes by `variant_id` string with no per-variant branching. The public `api.py` surface, however, still hard-codes the 4B-only support list in three places (the handle's `variant_id` `Literal`, the docstring, the FLUX.2 `_predict` defensive guard, and an `IncompatibleModelError` arm), so adding a variant is a **six-touch change**: api, detect, coefficients, calibration script, tests, docs.

```
flux instance
  └─ identify_variant(flux)  → "flux2-klein-9b"  (detect.py)
       └─ get_coefficients("flux2-klein-9b")  → (a4, a3, a2, a1, a0)  (coefficients.py)
            └─ flux2_forward_with_gate(...)  uses the same _predict-replacement
               integration that 4B uses; no per-variant branching in the hot
               path
```

Provenance is recorded in `_FLUX2_KLEIN_9B_COEFFS_PROVENANCE` next to the tuple, the same shape used for 4B.

## Components

### 1. Public API — `src/mlx_teacache/api.py`

The current code paths that hard-code 4B-only support all need to learn about 9B:

- `TeaCacheHandle.variant_id: Literal["flux1-dev", "flux1-schnell", "flux2-klein-4b"]` at `api.py:37` — add `"flux2-klein-9b"`. Import the `VariantId` alias from `detect.py` and reuse it here so the two cannot drift.
- The `apply_teacache` public docstring at `api.py:142-145` — extend the supported-variants list.
- The defensive `_predict` override guard at `api.py:194-200` — change `variant_id == "flux2-klein-4b"` to `variant_id.startswith("flux2-")` so 9B (and future FLUX.2 variants) get the same protection.
- The hard-coded `supported=["flux1-dev", "flux1-schnell", "flux2-klein-4b"]` in the `IncompatibleModelError` raise at `api.py:199` and `api.py:227` — replace with the `_SUPPORTED` tuple imported from `detect.py` so the supported list has one source of truth.

### 2. Variant registry — `src/mlx_teacache/integrations/mflux/detect.py`

- Add `"flux2-klein-9b"` to the `VariantId` `Literal` and to `_SUPPORTED`.
- In the `isinstance(flux, _Flux2KleinType)` branch, accept both `"flux2-klein-4b"` and `"flux2-klein-9b"` aliases. The `base-4b` / `base-9b` aliases continue to raise `IncompatibleModelError`.

### 3. Coefficients — `src/mlx_teacache/coefficients.py`

The existing layout stores coefficients and `Provenance` together as a tuple value inside the `_REGISTRY` dict (with `source: Literal["builtin", "user"]`), not as separate module-level constants. Match that shape for 9B:

```python
# Inside _REGISTRY (coefficients.py:73-105 region), add a third entry:
"flux2-klein-9b": (
    (a4, a3, a2, a1, a0),  # filled in from scripts/_calibration_flux2_klein_9b.json
    Provenance(
        source="builtin",  # matches the 4B entry's value
        revision="<short-sha of the v0.3.0 calibration commit>",
        calibration_dataset="10 prompts × 8 steps × seed=42, M1 Max 32GB, bf16, 512×512, guidance=1.0",
        fit_metric="R^2",
        fit_metric_value=<from JSON>,
        reference_url="https://github.com/IonDen/mlx-teacache/blob/main/scripts/calibrate_flux2.py",
    ),
),
```

Existing public symbols used by callers stay the same: `load_builtin(variant_id)` and the `_REGISTRY` dict. **Do not invent `get_coefficients()` or `_COEFF_REGISTRY`** — those names do not exist in the codebase.

In the same commit, also update the existing 4B entry's `reference_url` to point at the renamed `scripts/calibrate_flux2.py` (the script is renamed in component #4, so the URL would otherwise rot). The 4B coefficient tuple itself stays byte-for-byte identical.

The leading comment at `coefficients.py:43-49` that names `scripts/calibrate_flux2_klein.py` and `scripts/_calibration_flux2_klein.json` also gets updated to the new filenames.

### 4. Calibration script — rename + parameterize

Rename `scripts/calibrate_flux2_klein.py` → `scripts/calibrate_flux2.py`. Add an argparse `--variant` flag with the four declared Klein variants:

| `--variant` flag | mflux factory | Wired in v0.3.0? | Calibration `num_inference_steps` | Output JSON |
|---|---|---|---|---|
| `klein-4b` | `ModelConfig.flux2_klein_4b()` | Yes (rerun, identical inputs) | 8 | `scripts/_calibration_flux2_klein_4b.json` |
| `klein-9b` | `ModelConfig.flux2_klein_9b()` | Yes (new) | 8 | `scripts/_calibration_flux2_klein_9b.json` |
| `klein-base-4b` | `ModelConfig.flux2_klein_base_4b()` | Declared, raises `NotImplementedError("wired in v0.4.0")` | (n/a) | (n/a) |
| `klein-base-9b` | `ModelConfig.flux2_klein_base_9b()` | Declared, raises `NotImplementedError("wired in v0.5.0")` | (n/a) | (n/a) |

The existing output `scripts/_calibration_flux2_klein.json` is renamed to `_calibration_flux2_klein_4b.json` so the file layout is uniform. The 4B coefficients themselves stay byte-for-byte identical (we don't recalibrate 4B; the file rename is mechanical).

The prompt list, seed, guidance, and per-step capture wrapper are unchanged. Calibration prompts stay at 10.

**Why 8 steps, not the mflux default 4.** mflux 0.17.5's `Flux2Klein.generate_image` defaults `num_inference_steps=4`, and the BFL Hugging Face card for `FLUX.2-klein-9B` also shows 4-step inference. Calibrating at 4 steps is the *user-facing* default, but it is also the configuration the v0.2.0 `TeaCacheNoBenefitWarning` is designed to *block* — at 4 steps with the library's default `skip_first=1` + `skip_last=1`, `eligible = 4 - 2 = 2`, so `possible_skips = eligible - 1 = 1`. The polynomial would be fit on a degenerate ≤1-skip-opportunity dataset.

The 4B coefficients were fit at 8 steps for exactly this reason: TeaCache's value proposition only opens at schedules with at least a few skip opportunities. Keeping 9B's calibration at 8 steps matches that choice and keeps the 4B/9B fits on comparable footing.

The acceptance criteria do **not** assert quality at 4 steps. Users running Klein 9B at the mflux default of 4 steps continue to be guided by `TeaCacheNoBenefitWarning` (firing at this exact configuration) rather than by polynomial quality. The CHANGELOG and `docs/calibration.md` say this explicitly: "Klein 9B coefficients are calibrated at 8 steps; using them at `num_inference_steps < 8` produces a `TeaCacheNoBenefitWarning` rather than measurable benefit."

### 5. Tests

- `tests/test_detect.py`
  - Drop `test_flux2_klein_9b_rejected`.
  - Add `test_flux2_klein_9b_accepted` (mirroring the existing 4B-accepted test).
  - Keep `test_flux2_klein_base_4b_rejected` and add `test_flux2_klein_base_9b_rejected`.
- `tests/test_coefficients.py`
  - Add a parametrized check that `load_builtin(variant_id)` returns a length-5 tuple of finite floats for every variant in `_SUPPORTED`, including the new 9B entry.
  - Add a provenance round-trip assertion for the 9B entry (`Provenance.source == "builtin"`, `revision` non-empty, `reference_url` points at the renamed script).
- `tests/test_parity_flux2.py` (parity-marked, opt-in)
  - Add a `Flux2Klein(model_config=ModelConfig.flux2_klein_9b())` fixture branch parametrized by variant id.
  - Reuse the existing `image_strength=[0.0, 0.5, 0.7]` parametrization.
  - At `rel_l1_thresh=0.0`, assert cosine similarity ≥ 0.97 against vanilla (same oracle as 4B).
- `tests/test_image_quality_flux2.py` (parity-marked, opt-in)
  - Add a Klein 9B fixture branch parametrized by variant.
  - SSIM ≥ 0.85 on the PR-gate prompt at default threshold (matches 4B).
  - SSIM ≥ 0.80 on the 5-prompt suite at default threshold.
- `tests/test_api.py` (existing parity-marked file)
  - Add one positive smoke that `apply_teacache(Flux2Klein(model_config=ModelConfig.flux2_klein_9b()))` returns a handle and `handle.variant_id == "flux2-klein-9b"`. Catches the api.py `Literal` regression directly.

### 6. Img2ImgNotSupportedError removal

The error was deprecated in v0.2.0 with the explicit promise "Removal planned for v0.3.0." Following through:

- Delete the class definition (the entire `class Img2ImgNotSupportedError(...)` block, including its `__init__` `DeprecationWarning` emission) from `src/mlx_teacache/errors.py`.
- Remove the `Img2ImgNotSupportedError` import and `__all__` entry from `src/mlx_teacache/__init__.py:26-37` and `:45-64`.
- In `tests/test_errors.py`:
  - Remove the `Img2ImgNotSupportedError` import at `tests/test_errors.py:8-18`.
  - Remove its entry from the subclass-iteration list at `tests/test_errors.py:21-32`.
  - Delete the test function `test_img2img_not_supported_error_deprecated` at `tests/test_errors.py:107-111`.
- Add a small negative-import assertion to `test_errors.py`: `with pytest.raises(ImportError): from mlx_teacache.errors import Img2ImgNotSupportedError` and likewise for `from mlx_teacache import Img2ImgNotSupportedError`. Covers acceptance criterion 8 directly.
- CHANGELOG entry under `### Removed` for v0.3.0 with a one-line migration note ("Catch the underlying `IncompatibleModelError` or `InvalidStepWindowError` instead").

### 7. Docs

- `README.md`
  - Add a `flux2-klein-9b` row to the "Supported models" table.
  - Add a `## License obligations` section explaining the FLUX.2 Klein non-commercial terms and the BFL safety-filter obligation that flows with the weights. One short paragraph + a pointer to the official BFL license URL.
  - Note in the "Limitations" section that 9B inference on M1 Max 32GB at quantize=4 is memory-tight and may need the swap to assist on resolutions above 512².
- `docs/calibration.md`
  - Update the "Built-in coefficient sources" table to include the 9B row.
  - Update the "Producing new coefficients" code block to use the new parameterized script signature (`uv run python scripts/calibrate_flux2.py --variant klein-9b`).
- `CHANGELOG.md`
  - `## [0.3.0]` entry with `### Added` (Klein 9B variant + coefficients), `### Changed` (calibration script renamed + parameterized), `### Removed` (`Img2ImgNotSupportedError`), and a brief license-obligations note.

## Data flow

Unchanged from v0.2.0. The variant-id string flows from `identify_variant(flux)` into `apply_teacache`, which resolves the matching built-in polynomial **once** via `load_builtin(variant_id)` and stores `(coefficients, provenance)` on the handle. The runtime gate reads `handle.coefficients` per step; the `_REGISTRY` dict is not touched on the hot path. No per-variant branching in lifecycle or forward.

## Error handling

Unchanged. `IncompatibleModelError` already reports the supported list, which after v0.3.0 includes `flux2-klein-9b`. `base-4b` / `base-9b` continue to land in the existing rejection path until their releases.

## Testing strategy

- **Unit tests** (`test_detect.py`, `test_coefficients.py`, `test_errors.py`): run in the default pure-core CI suite. Fast.
- **Parity tests** (`test_parity_flux2.py`): opt-in `@pytest.mark.parity`; require the 9B weights and run only on developer machines / nightly CI with HF auth.
- **Image-quality tests** (`test_image_quality_flux2.py`): opt-in `@pytest.mark.parity` as well; SSIM oracle at default threshold (0.20) and 5-prompt suite.
- **Calibration itself**: not a test. The JSON report at `scripts/_calibration_flux2_klein_9b.json` is committed alongside the coefficient bump as authoring evidence, like the 4B report.

## Memory + time budget (M1 Max 32GB authoring machine)

- 9B weights at quantize=4: ~9 GB resident model memory + activation overhead.
- 9B calibration run: 10 prompts × 8 steps × ~25 s/step ≈ **~30 minutes wall-clock**. Single sitting; tee log to `/tmp/calibrate_klein_9b.log` per the heavy-generation feedback rule.
- 9B parity test pass: ~3 prompts × 2 thresholds × 3 image_strengths × ~3 minutes/prompt ≈ **~60 minutes**. Run nightly, not on every PR.
- 9B image-quality pass: 5 prompts × 1 threshold × ~3 minutes/prompt ≈ **~15 minutes**.

If the calibration or parity time budget proves wrong on the first 9B run, the script's prompt count is the easy knob to turn (drop to 5 prompts at the cost of fit confidence).

## Acceptance criteria

0. **Author-machine prerequisites for the calibration step:** `hf auth login` with a read-scoped token; gated access to `black-forest-labs/FLUX.2-klein-9B` accepted on the Hugging Face website; weights cached locally (32 GB, see MODELS.md). The calibration command in criterion 2 does not run without these.
1. `from mlx_teacache import apply_teacache; apply_teacache(Flux2Klein(model_config=ModelConfig.flux2_klein_9b()))` returns a handle and does not raise.
2. `scripts/calibrate_flux2.py --variant klein-9b` runs end-to-end, writes `_calibration_flux2_klein_9b.json`, and reports R² ≥ 0.50 (matches the 4B floor).
3. SSIM ≥ 0.85 on the PR-gate prompt at `num_inference_steps=8`, `rel_l1_thresh=0.20`.
4. SSIM ≥ 0.80 on the 5-prompt suite at `num_inference_steps=8`, `rel_l1_thresh=0.20`.
5. Cosine similarity ≥ 0.97 at `rel_l1_thresh=0.0` (parity oracle, `num_inference_steps=8`).
6. `pytest tests/ -m "not parity and not slow and not benchmark and not network"` stays green, with the new unit tests included.
7. `ruff check .` and `ruff format --check .` both clean.
8. `Img2ImgNotSupportedError` no longer importable from either `mlx_teacache` or `mlx_teacache.errors` (asserted by a negative-import test).
9. README + CHANGELOG + `docs/calibration.md` updated. Tag `v0.3.0` cuts the release.

Quality at the mflux default of `num_inference_steps=4` is explicitly out of scope — that configuration triggers `TeaCacheNoBenefitWarning` (introduced in v0.2.0) rather than the polynomial.

## Risks

- **Calibration may produce a polynomial that fails the SSIM floor.** Mitigation: the calibration script's `--variant` flag lets us rerun with adjusted prompt count or step budget; the SSIM tests are the gate.
- **9B inference may OOM on M1 Max 32GB at >512² resolution.** Mitigation: calibration and parity tests run at 512² only. Document the constraint in the README.
- **mflux upstream may bump the Klein 9B alias between releases.** Mitigation: `detect.py` already keys off `aliases`, not `model_name`; if upstream drops the alias we get a clean `IncompatibleModelError` rather than silent breakage.

## Out of scope (for completeness)

- Performance benchmarks on chips other than M1 Max. Community PRs welcome per the v0.2.0 chip-table protocol.
- Any change to img2img semantics.
- Any change to the `TeaCacheNoBenefitWarning` cadence.
- Anything in v0.4 or v0.5 scope.
