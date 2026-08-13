# Manual verification recipe

Quick smoke test for confirming a local `mlx-teacache` install is working.
This recipe verifies the latest PyPI release, matching the CHANGELOG's top
entry — pin the installed version explicitly:

```bash
uv pip install "mlx-teacache[mflux]==0.9.3"
```

## Shared capture helper

Both recipes below need a way to extract the final latent from a
`flux.generate_image()` call. mflux's returned `GeneratedImage` does NOT
carry latents, so we use a one-shot `after_loop` callback:

```python
class _LatentCapture:
    def __init__(self):
        self.latent = None
    def call_after_loop(self, seed, prompt, latents, config, **_):
        self.latent = latents


def capture(flux, **kwargs):
    cap = _LatentCapture()
    flux.callbacks.register(cap)
    try:
        flux.generate_image(**kwargs)
    finally:
        for attr in ("after_loop", "before_loop", "in_loop", "interrupt"):
            lst = getattr(flux.callbacks, attr, None)
            if isinstance(lst, list):
                for i in range(len(lst) - 1, -1, -1):
                    if lst[i] is cap:
                        del lst[i]
    return cap.latent
```

## FLUX.1-dev byte-exact threshold-zero smoke

At `rel_l1_thresh=0.0` the wrapper produces bit-exact latents matching
vanilla mflux:

```python
import mlx.core as mx
from mflux.models.flux.variants.txt2img.flux import Flux1
from mlx_teacache import apply_teacache

# (Paste the shared capture helper above before running this block.)

flux = Flux1.from_name("dev", quantize=4)
flux.freeze()
kwargs = dict(prompt="a red apple", num_inference_steps=25, seed=42,
              height=512, width=512, guidance=3.5)

vanilla = capture(flux, **kwargs)
with apply_teacache(flux, rel_l1_thresh=0.0):
    wrapper = capture(flux, **kwargs)

assert mx.array_equal(vanilla, wrapper), "FLUX.1 threshold-zero parity broken"
print("FLUX.1 threshold-zero parity: OK")
```

## FLUX.2 Klein 4B cosine smoke (not bit-exact)

FLUX.2 wrapper parity is cosine-based at threshold 0 (vanilla compile vs.
wrapper eager dispatch produces ~1 ULP per-element divergence that
compounds; cosine ≥ 0.99 measured). Use a relaxed oracle:

```python
import mlx.core as mx
import numpy as np
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
from mlx_teacache import apply_teacache

# (Re-use the shared capture helper from above.)

def cosine(a: mx.array, b: mx.array) -> float:
    a32 = np.asarray(a.astype(mx.float32)).reshape(-1)
    b32 = np.asarray(b.astype(mx.float32)).reshape(-1)
    return float(np.dot(a32, b32) / (np.linalg.norm(a32) * np.linalg.norm(b32)))


flux = Flux2Klein(quantize=4, model_config=ModelConfig.flux2_klein_4b())
flux.freeze()
kwargs = dict(prompt="a red apple", num_inference_steps=8, seed=42,
              height=512, width=512, guidance=1.0)
vanilla = capture(flux, **kwargs)
with apply_teacache(flux, rel_l1_thresh=0.0):
    wrapper = capture(flux, **kwargs)

score = cosine(vanilla, wrapper)
assert score >= 0.97, f"FLUX.2 wrapper cosine parity below 0.97 (got {score:.4f})"
print(f"FLUX.2 cosine parity: {score:.4f} (target ≥ 0.97)")
```

## What this catches

- Bit-exact regression in the FLUX.1 transformer proxy
- Catastrophic divergence in the FLUX.2 predict closure (cosine < 0.97)
- VAE decode breakage (the `flux.generate_image(...)` calls would raise)
- `TeaCacheNoBenefitWarning` firing unexpectedly at 25-step / 8-step config

## What this does NOT catch

- Quality regression at non-zero `rel_l1_thresh` (use the SSIM tests in
  `tests/test_image_quality_*.py` for that — they need real model weights
  and run as `@pytest.mark.parity`)
- Speedup regression (run the protocol in `docs/m3-plus-tradeoff.md`)

## Troubleshooting

- `IncompatibleModelError: mflux not installed` → reinstall with `[mflux]` extra.
- `AlreadyPatchedError` → call `handle.restore()` on the existing handle first.
- `MissingGenerationContextError` → call `handle.restore()` and reapply; do not
  register before-loop callbacks after `apply_teacache()`.
- `TeaCacheNoBenefitWarning` → check `num_inference_steps`, `skip_first_n_steps`,
  `skip_last_n_steps`, and (for img2img) `image_strength`; the effective window
  must satisfy `active_num_steps - skip_first - skip_last > 1`.
