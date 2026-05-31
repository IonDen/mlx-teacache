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
all (see the v0.3.0 postmortem in `docs/superpowers/notes/`), so Klein on
M1/M2 base + Pro with mlx-teacache is approximately neutral or slightly slower
than vanilla.

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

## v0.2+ plans

Investigate splitting `_predict` so the body-only computation can stay compiled
while gating runs in eager Python. Adds complexity; deferred until v0.1 is in
users' hands. On M5 specifically this is the only realistic path to keep the
Neural Accelerator fast path engaged — see ROADMAP.md "Compile-friendly gating"
for the design sketch.
