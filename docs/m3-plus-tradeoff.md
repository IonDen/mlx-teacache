# Apple Silicon compile tradeoff

## Why

mflux wraps `Flux2Klein._predict` in `mx.compile(predict)` on **most** Apple
Silicon chips. The exact gate (`mflux/utils/apple_silicon.py` +
`mflux/models/flux2/variants/txt2img/flux2_klein.py:278-281`):

```python
if AppleSiliconUtil.is_m1_or_m2():   # base + Pro M1/M2 (not Max/Ultra)
    return predict                    # eager
return mx.compile(predict)            # compiled
```

`is_m1_or_m2()` returns True when the chip brand-string is "Apple M1" or
"Apple M2" *and* contains neither "Max" nor "Ultra". So M1 Pro and M2 Pro
both fall on the eager side:

| Chip | Vanilla mflux `_predict` |
|---|---|
| Apple M1 (base), Apple M2 (base) | **eager** |
| M1 Pro, M2 Pro | **eager** |
| M1 Max / Ultra, M2 Max / Ultra | compiled |
| All M3* / M4* / M5* | compiled |

`mx.compile` traces the Python function once; subsequent calls reuse the compiled
graph. This means Python-side gating logic in our predict closure would NOT run
after step 1.

To keep TeaCache's gating live on every chip where mflux compiles, `mlx-teacache`
replaces `flux._predict` with an **uncompiled** eager-Python closure. Users on
those chips lose mflux's compile gain on this code path. The tradeoff: when the
gate actually engages we skip ~25% of steps, which more than compensates on
M1 Max / M1 Ultra / M2 Max / M2 Ultra (measured 1.46× on FLUX.1-dev / 25 steps
on M1 Max, 2026-05-31). The magnitude of the compile-loss tax grows on newer
hardware.

On chips that mflux already runs eager (base + Pro M1/M2), the wrapper does
not gain anything from compile avoidance — it only helps when the gate fires.
For FLUX.2 Klein at the distilled 4-8 step defaults the gate does not fire at
all, so Klein on M1/M2 base + Pro with mlx-teacache is approximately neutral or
slightly slower than vanilla.

## M5 specifically: Neural Accelerators

The M5 generation (October 2025+) adds dedicated matrix-multiplication hardware
("Neural Accelerators") to each GPU core, analogous to NVIDIA's tensor cores.
MLX dispatches onto them through Metal 4 `TensorOps`. **Neural Accelerators are
only available via the compiled (TensorOps) path** — our eager wrapper falls
back to MLX's general matmul kernels. Combined with Apple's claimed 4× AI compute
boost on M5 vs M4, the compile-loss tax may be large enough on M5 that the net
speedup approaches 1.0× (i.e., no speedup vs vanilla). Output correctness is
preserved on M5 — only value proposition shrinks. Requires macOS 26.2+ for MLX
to use Neural Accelerators.

References: [Apple ML Research — Exploring LLMs with MLX and the M5 GPU](https://machinelearning.apple.com/research/exploring-llms-mlx-m5), Apple M5 announcement (Oct 2025).

## What to measure on your hardware

```python
import time
import mlx.core as mx
from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
from mflux.models.common.config.model_config import ModelConfig
from mlx_teacache import apply_teacache

flux = Flux2Klein(quantize=4, model_config=ModelConfig.flux2_klein_4b())

ARGS = dict(seed=42, prompt="a red apple",
            num_inference_steps=25, height=512, width=512, guidance=1.0)

def time_it():
    start = time.perf_counter()
    flux.generate_image(**ARGS)
    mx.eval(mx.zeros(1))
    return time.perf_counter() - start

vanilla = time_it()
with apply_teacache(flux, rel_l1_thresh=0.25):
    teacache = time_it()
print(f"vanilla: {vanilla:.2f}s  teacache: {teacache:.2f}s  speedup: {vanilla/teacache:.2f}×")
```

If `vanilla/teacache > 1.0` on your hardware, mlx-teacache helps. Otherwise file an
issue with your timings.

The snippet above is a quick single-shot sanity check, not a submission-quality
measurement — it runs one vanilla and one wrapped generation back to back in the
same process, with no warmup and no repetition. For timings you want to report
in an issue or a PR, follow the protocol below.

## Benchmark protocol for community numbers

This is the standard we ask for when a submitted number is going to be compared
against the numbers already in this repo's README or `COMPARISON.md`. It applies
whether you use `scripts/bench_speedup.py` or your own timing harness.

- Report the environment: chip (`sysctl -n machdep.cpu.brand_string`), macOS
  version, Python version, and the installed `mlx`, `mflux`, and `mlx-teacache`
  versions, plus the dtype and quantization bits used for the run.
- Isolate each condition in its own process. Run vanilla and wrapped
  generations as separate process invocations, not back to back in one
  interpreter, so a compiled graph or a warm allocator from one condition
  cannot bias the other's timing.
- Discard a warmup repetition. The first generation in a process pays for
  Metal shader compilation and disk-cache population; discard its timing and
  measure only the repetitions after it.
- Time at least 3 repetitions per condition and report the median and the
  minimum. A single timing is noise: the median reflects the typical case,
  the minimum is the least affected by background load on the host.
- Record computed vs. skipped step counts alongside the timing, so a reader
  can tell whether an improvement comes from step-skipping or from
  `mx.compile` avoidance (see "Why" above); the two mechanisms are separate
  and a submitted number should attribute to the right one.
- Use one shared recipe across every condition being compared: identical
  prompt, seed, step count, guidance, and image dimensions. Changing any of
  these between the vanilla and wrapped runs invalidates the comparison.

`scripts/bench_speedup.py` in this repo follows the process-isolation and
environment-capture parts of this protocol: every (variant, condition,
repetition) triple runs in its own subprocess, and its report includes the
chip, OS, and package versions alongside the per-condition median. Use it, or
match its methodology, when submitting a number for a variant already in this
repo.

## v0.2+ plans

Investigate splitting `_predict` so the body-only computation can stay compiled
while gating runs in eager Python. Adds complexity; deferred until v0.1 is in
users' hands. On M5 specifically this is the only realistic path to keep the
Neural Accelerator fast path engaged — see ROADMAP.md "Compile-friendly gating"
for the design sketch.
