"""Z-Image base variant (Tongyi-MAI/Z-Image) — self-contained TeaCache mini-kernel.

config.py + detect.py are mflux-free (eagerly imported by the registry).
integration.py imports mflux lazily and re-walks ZImageTransformer.__call__
with a TeaCache gate. No sibling-variant imports — this variant defines its
own internal handle and forward, depending only on the model-agnostic
_kernel/, the public handle, and the shared mflux lifecycle helpers.
"""
