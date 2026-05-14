# Manual verification recipe

A smoke-test that proves a released version works on a fresh venv.

## Setup

```bash
python -m venv /tmp/mlx-teacache-verify
source /tmp/mlx-teacache-verify/bin/activate
pip install "mlx-teacache[mflux]==0.1.0"
```

## Verify (~2 min)

```python
from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
from mflux.models.common.config.model_config import ModelConfig
from mlx_teacache import apply_teacache

flux = Flux2Klein(quantize=4, model_config=ModelConfig.flux2_klein_4b())

# Threshold=0 must produce identical output to vanilla.
with apply_teacache(flux, rel_l1_thresh=0.0):
    r1 = flux.generate_image(prompt="a red apple", seed=42, num_inference_steps=10)
r2 = flux.generate_image(prompt="a red apple", seed=42, num_inference_steps=10)

import mlx.core as mx
assert mx.array_equal(r1.latents, r2.latents)

# Default threshold must skip at least one step.
with apply_teacache(flux, rel_l1_thresh=0.25) as h:
    flux.generate_image(prompt="a red apple", seed=42, num_inference_steps=25)
assert h.stats.skipped_count >= 1
print(f"OK — speedup_estimate={h.stats.speedup_estimate:.2f}×")
```

## Troubleshooting

- `IncompatibleModelError: mflux not installed` → reinstall with `[mflux]` extra.
- `AlreadyPatchedError` → call `handle.restore()` on the existing handle first.
- `MissingGenerationContextError` → call `handle.restore()` and reapply; do not register fault-prone before-loop callbacks after `apply_teacache()`.
