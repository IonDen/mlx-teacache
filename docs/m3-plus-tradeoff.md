# M3+ silicon tradeoff

## Why

mflux's `Flux2Klein._predict` is wrapped in `mx.compile(predict)` on M3+ silicon
(see `mflux/models/flux2/variants/txt2img/flux2_klein.py:279`). `mx.compile` traces
the Python function once; subsequent calls reuse the compiled graph. This means
Python-side gating logic in our predict closure would NOT run after step 1.

To keep TeaCache's gating live on M3+, `mlx-teacache` replaces `flux._predict`
with an **uncompiled** eager-Python closure. M3+ users lose mflux's compile gain
on this code path.

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

## v0.2 plans

Investigate splitting `_predict` so the body-only computation can stay compiled
while gating runs in eager Python. Adds complexity; deferred until v0.1 is in users' hands.
