# FLUX.2 Klein base-4B plan audit

**Date:** 2026-05-17  
**Plan:** `docs/superpowers/plans/2026-05-17-flux2-klein-base-4b.md`  
**Scope:** Real plan issues only. Not a lint/TDD/syntax review.

## Findings

### 1. The real-weight parity release gate says `cosine >= 0.99`, but the planned test command still only enforces `0.97`

Task 10 defines the parity release gate as:

```bash
HF_TOKEN="$(hf auth token)" uv run pytest tests/test_parity_flux2.py -m parity -k "klein-base-4b" -v
```

and says the expected result is "cosine similarity >= 0.99" (`plan:1117-1123`). But the test file currently gates FLUX.2 parity at `0.97`:

- `tests/test_parity_flux2.py:57-63` says the measured value was ~0.99 and sets `_FLUX2_COSINE_GATE = 0.97`.
- The actual assertions in `tests/test_parity_flux2.py` use `_FLUX2_COSINE_GATE` or hard-coded `0.97`.

Impact: a base-4B run with cosine `0.971` would pass the planned release command while failing the plan's stated release gate. This is not a style issue; it means the automated evidence does not prove the claimed bar.

Fix: either change the plan's gate to `>= 0.97`, or add a base-4B-specific assertion/report that enforces `>= 0.99` before the release proceeds.

### 2. The Task 10 parity command lets later-release CFG scope leak into the v0.4 release gate

The plan scopes v0.4 to `guidance=1.0` and says Task 10 Step 1 is the parity oracle "at threshold 0 for base-4b" (`plan:1111-1123`). The command runs the entire `tests/test_parity_flux2.py` module filtered only by `-k "klein-base-4b"`.

After Task 9 adds the base-4B fixture parameter, that command will also select:

- the slow full-prompt parity test (`tests/test_parity_flux2.py:217-226`);
- `test_cfg_fallback_matches_vanilla` (`tests/test_parity_flux2.py:244-255`), which runs `guidance=3.5` and asserts `mx.array_equal(vb, w)`.

CFG-engaged caching is explicitly deferred to v0.4.1. This audit is not asking v0.4.0 to implement or solve CFG. The issue is that the v0.4.0 release command can fail on a `guidance=3.5` test even though the release is explicitly scoped to `guidance=1.0`.

Impact: the release gate can become both too broad and misleading. It may burn a lot of real-weight runtime on slow tests, then fail on CFG bit-exactness even though v0.4 explicitly defers CFG caching.

Fix: make Task 10 Step 1 target only the intended `g=1.0` threshold-zero PR gate. Keep CFG validation in the v0.4.1 plan or in a separate non-blocking observation for v0.4.0.

### 3. The 0-skip structural fallback still flows into docs that assume step-skipping engagement

Task 11 says that if `skipped_count == 0`, one allowed human decision is to reframe v0.4.0 as a structural-only release (`plan:1216-1218`). Step 4 then says to proceed to Task 12 once the release is reframed as structural (`plan:1222-1226`).

But Task 12's prescribed text still assumes step-skipping:

- README footnote: "TeaCache engages at `guidance=1.0`" (`plan:1250-1251`).
- README benchmark row mechanism: "TeaCache step-skipping" (`plan:1258-1260`).
- CHANGELOG intro: "first FLUX.2 variant where the polynomial gate engages" (`plan:1281-1286`).
- `docs/calibration.md` row: "expected to engage at the package default" (`plan:1303-1306`).

Impact: if the 0-skip contingency takes Path 3 and the executor then follows Task 12 literally, the plan will reintroduce the exact false-performance framing it is trying to prevent.

Fix: split Task 12 into two doc branches: engagement path vs structural-only path. The structural-only branch must remove the "first FLUX.2 variant where the gate engages" language and label the benchmark mechanism as non-engaging/structural.

## Validated Direction

- The plan resolves the main spec-audit issues: `guidance=1.0` scope, explicit detect alias branch, `Provenance(source="builtin")`, variant-aware 25-step test kwargs, and a zero-skip escalation gate.
- The base implementation shape is otherwise consistent with current code: calibration script placeholder, detect registry, coefficient registry, bench script, and FLUX.2 parity/SSIM fixtures are the right change sites.
