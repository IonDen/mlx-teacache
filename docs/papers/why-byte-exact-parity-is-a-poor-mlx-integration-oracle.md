# Why byte-exact parity is a poor MLX integration oracle

*A case study from `mlx-teacache` and the test-oracle discipline that replaced a
bit-exactness gate*

Byte-exact fixtures looked like the safest way to test `mlx-teacache`. They were not.
One day before the library's first release in May 2026, its threshold-zero wrapper failed against
stored vanilla [mflux](https://github.com/filipstrand/mflux) output on **5 of 5** FLUX.1-dev
prompts. The same test passed on 5 of 5 FLUX.1-schnell prompts. Yet wrapper and vanilla matched
at all 25 denoising steps when compared inside one Python process.

That result supported the wrapper's math and exposed a bad oracle. The stored fixture captured the
script as well as the output. This paper explains the failure and separates four jobs: artifact
drift, no-op parity, numerical equivalence, and intentionally approximate output. Since v0.1.0,
each job has had its own test. The same problem can affect any integration that changes another
framework's numerical path on a lazy, GPU-dispatched runtime.

## Evidence scope

- The May 2026 divergence measurements are author-reported results from 14 and 15 May. The raw
  logs are not public, and nothing was rerun for this paper.
- Present-state claims about the test suite were checked against the public repository at v0.9.2.
- The incident ran on an M1 Max with 32 GB unified memory, using mflux 0.17.5, FLUX.1 at 4-bit
  quantization, 512×512 output, 25 steps, and fixed prompts and seeds.
- MLX source citations are pinned to the commit inspected during the investigation:
  [`046217b`](https://github.com/ml-explore/mlx/tree/046217bcae7347aa814665f39a8f0e404029ddb0).

## 1. The gate that failed

TeaCache ([arXiv:2411.19108](https://arxiv.org/abs/2411.19108)) skips diffusion transformer
steps when a calibrated polynomial predicts the body output will barely change. At
`rel_l1_thresh=0` the gate never skips: every step computes in full, and the wrapper should add
nothing but bookkeeping. The release documentation calls this the deliberate vanilla-equivalent
reference mode. The pre-release test used it as the correctness anchor. It generated final
latents with vanilla mflux and committed them as `.safetensors` fixtures. CI then asserted
`mx.array_equal` between each fixture and the threshold-zero wrapper output.

Trying to make that gate pass uncovered three unrelated integration bugs. Mflux 0.17 had changed
the field used for variant detection. The callback registry exposed methods where the wrapper
expected lists. A normalization layer also received an invalid keyword argument. All three were
fixed before release, so exact comparisons were useful during debugging.

It made a poor shipped oracle. After the real bugs were fixed, the dev-variant divergence remained.
Across five prompts, cosine similarity between the wrapper's final latent and the committed
fixture ranged from **0.105 to 0.256**. Maximum absolute differences ranged from **5.31 to 5.82**
on fp32 latents. A gate loose enough to accept a difference near 5.8 would no longer constrain
useful similarity.

The schnell fixtures still matched byte-for-byte. That inconsistency made the test more dangerous:
it could pass long enough to ship, then fail after an unrelated change.

## 2. Localizing the divergence: the fixture encoded the script

The investigation compared final-latent SHA-256 digests across single-variable setups. These are
the decisive author-reported rows from 14 May 2026:

| Setup (same model, prompt, seed, config) | Final latent digest |
|---|---|
| Original fixture-generation script (vanilla mflux, one after-loop callback) | `45471c34…` (the fixture) |
| Same script re-run days later | `45471c34…` (reproducible) |
| Wrapper at `rel_l1_thresh=0` | `0db096b2…` |
| Wrapper, forced skips vs. no skips at threshold 0 | both `0db096b2…` |
| **Vanilla mflux, no wrapper, callback class also defining an in-loop method** | `0db096b2…` |
| Vanilla vs. wrapper, same process, per-step comparison, all 25 steps | bit-exact |

The fifth row matters most. Vanilla mflux, with no TeaCache code on the model at all,
reproduced the *wrapper's* bytes once the registered callback object's class shape changed. This
shows that the digest was sensitive to Python-level process structure even without the wrapper's
math. It does not identify which MLX subsystem turned that structural change into different bytes.
The committed fixture had faithfully recorded one process configuration; the tested altered
setups recorded another.

Within one process, the result changed. Mflux evaluates the latents once per denoising step
([`mx.eval(latents)` in the generation loop, mflux 0.18.0](https://github.com/filipstrand/mflux/blob/v.0.18.0/src/mflux/models/flux/variants/txt2img/flux.py#L122)).
The diagnostic observed an exact wrapper/vanilla match at every boundary. In this case, the
same-process comparison controlled enough of the environment to make byte identity useful.

## 3. What MLX documents, and what it does not

MLX documentation does not say that MLX is nondeterministic, and this article does not make that
claim. The official material documents several relevant building blocks:

- **Lazy, dynamically built graphs.** "When you perform operations in MLX, no computation
  actually happens. Instead a compute graph is recorded," and it runs only on `eval()`
  ([lazy-evaluation docs](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html)).
  Graphs of unused outputs are still built.
- **Output-rooted evaluation.** `eval_impl` roots a synchronizer on the requested outputs and
  walks their *inputs*: depth-first collection, then a width-limited breadth-first execution
  tape ([`mlx/transforms.cpp` at the pinned commit](https://github.com/ml-explore/mlx/blob/046217bcae7347aa814665f39a8f0e404029ddb0/mlx/transforms.cpp#L74)).
- **Reference-count-gated buffer donation.** An MLX array "is really a node in a graph"
  ([`mlx/array.h`](https://github.com/ml-explore/mlx/blob/046217bcae7347aa814665f39a8f0e404029ddb0/mlx/array.h#L27)),
  and [`is_donatable()`](https://github.com/ml-explore/mlx/blob/046217bcae7347aa814665f39a8f0e404029ddb0/mlx/array.h#L294)
  permits buffer reuse only when the descriptor and data are uniquely owned. Other live
  references can therefore prevent an operation from reusing an input buffer.
- **Shape- and environment-dependent kernel dispatch.** The Metal matmul backend, for one,
  branches on shape heuristics and environment switches such as TF32 enablement
  ([`mlx/backend/metal/matmul.cpp`](https://github.com/ml-explore/mlx/blob/046217bcae7347aa814665f39a8f0e404029ddb0/mlx/backend/metal/matmul.cpp)).
- **PRNG keys control random draws, not reductions.** The
  [random-number docs](https://ml-explore.github.io/mlx/build/html/python/random.html) give
  deterministic keyed sampling; they promise nothing about kernel reduction order.

There is no MLX analogue of
[`torch.use_deterministic_algorithms()`](https://docs.pytorch.org/docs/stable/notes/randomness.html).
PyTorch ships such a switch. Its documentation still warns that reproducibility is not guaranteed
across releases, commits, or platforms. It also warns that identical seeds do not guarantee the
same result on CPU and GPU.

These blocks provide plausible routes to byte-level differences, not a root-cause proof for this
incident. Floating-point addition is not associative, so a changed kernel or reduction order can
change low bits. Graph shape and reference counts can affect scheduling, buffer reuse, and dispatch
conditions without changing the mathematical program. The May experiments did not isolate which,
if any, of those routes produced the digest shift. They established the narrower result that a
callback-class change correlated with different final bytes while paired same-process paths matched.
Once a numerical difference exists, each later denoising step can compound it through a nonlinear
transformer.

[Thinking Machines' batch-invariance work](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
documented that composition on server GPUs. Kernel reduction strategies changed with batch size,
and uncontrolled batch sizes turned 1,000 temperature-zero runs into 80 unique completions. The
MLX case here is an analogy, not evidence of the same mechanism:
script structure correlated with different bytes, and a committed fixture assumed that dependency
would remain stable. That is not a safe assumption for a correctness oracle.

This explanation has two limits. First, the *within-process* stability used by the replacement
test is an empirical regularity, not a documented contract. The current suite re-asserts it
whenever the manual parity job runs, and MLX promises it nowhere.

Second, the specific mechanism by which the wrapper's presence produced different bytes remains
unisolated. The leading hypothesis focused on cache tensors that the early threshold-zero wrapper
still built (`cached_residual = body_out − body_in`). Those tensors kept body and tail
intermediates alive, which could change donation eligibility at their boundary. The hypothesis
was concrete enough to act on. The project shipped a threshold-zero fast path that builds none
of those tensors. The result falsified donation as the dominant cause: divergence against the
committed fixtures did not shrink at all. The fast path stayed because it removes unused work.

The current
[`_kernel/gate.py`](https://github.com/IonDen/mlx-teacache/blob/v0.9.2/src/mlx_teacache/_kernel/gate.py)
still short-circuits when `rel_l1_thresh <= 0`, but the original causal account did not survive
its test. Buffer donation remains a
real, documented dispatch mechanism; what fell was the specific claim that these particular
tensors were the dominant lever in this divergence. This note therefore reports that claim as a
candidate that failed its first discriminating experiment, not as the explanation.

## 4. One assertion, four different questions

The v0.1.0 redesign began by separating four questions that the fixture assertion had combined.
Each question needs a different oracle. The public
[CHANGELOG](https://github.com/IonDen/mlx-teacache/blob/v0.9.2/CHANGELOG.md) records that change.

![Conceptual decision map connecting four MLX integration questions to their matching test oracles: artifact drift uses SHA-256 and pinned revisions; no-op parity uses paired same-process execution; changed numerical paths use byte equality or documented per-variant gates; intentional approximation uses SSIM plus a skip-count band](https://raw.githubusercontent.com/IonDen/mlx-teacache/main/docs/papers/diagrams/mlx-integration-oracle-selection.svg)

*Figure 1. Conceptual map of the four test questions and their matching oracles. The figure selects
a test boundary; it does not identify the unisolated cause of the cross-process byte difference,
and it adds no measurement beyond the evidence summarized in this paper. The editable
[PlantUML source](https://github.com/IonDen/mlx-teacache/blob/main/docs/papers/diagrams/mlx-integration-oracle-selection.puml)
is published with the SVG.*

### Did the committed artifact change?

That is drift detection, not correctness. The fixtures
stayed, demoted to fingerprints:
[`tests/test_fixtures_integrity.py`](https://github.com/IonDen/mlx-teacache/blob/v0.9.2/tests/test_fixtures_integrity.py)
SHA-256-checks the files on disk, and
[`tests/test_hf_revisions.py`](https://github.com/IonDen/mlx-teacache/blob/v0.9.2/tests/test_hf_revisions.py)
pins the upstream Hugging Face model revisions they were generated from. A failure there means
"something moved," never "the wrapper is wrong."

### Is the wrapper a mathematical no-op when configured as one?

This is the question the
byte-exact gate was trying to ask, and the tested FLUX.1 path has a byte-exact answer inside one
process. The shipped oracle is paired same-process parity
([`tests/test_parity_flux1.py`](https://github.com/IonDen/mlx-teacache/blob/v0.9.2/tests/test_parity_flux1.py)):
capture vanilla, apply the wrapper at threshold zero, then restore and capture vanilla again. The
test asserts `mx.array_equal(vanilla_before, wrapper)`,
`mx.array_equal(vanilla_before, vanilla_after)`, and zero recorded skips. The `vanilla_after`
control catches restore leakage (a callback, a proxy, a sentinel left behind).

A reverse-order
variant (wrapper first, then vanilla) checks whether running vanilla first warms the process into
a favorable state. Each paired case takes roughly 3× one generation. One prompt is the designated
fast parity case, and the five-prompt sweep carries a slow marker. Because the real weights require
gated access and credentials, the repository runs the
parity job manually as release-quality validation rather than on every pull request.

### Is the changed execution path numerically equivalent?

Every variant changes the execution path in some way. Whether byte identity remains stable is an
empirical question, so the answer differs by variant.

FLUX.1 swaps in a proxy transformer that re-walks the forward pass and places the gate between
body and tail. That re-walk measured bit-exact against vanilla within one process. FLUX.1 can
therefore keep the `mx.array_equal` oracle above.

FLUX.2 and Z-Image replace mflux's predict function with an eager closure so the gate can run
between body and tail. Mflux compiles the vanilla predict function on most hardware, but skips
compilation on base and Pro M1/M2 chips. The source-reported FLUX.2 result came from an M1 Max,
where mflux skips compilation. Compiled-versus-eager execution therefore cannot explain that
measurement. Only the changed forward walk is common to both hardware paths.

Qwen-Image does not use compilation. Mflux calls its transformer directly twice per step, while
the integration installs an eager proxy that re-walks `QwenTransformer.__call__` in stages. Its
threshold-zero test is not byte-exact. A separate first-step self-check measures cosine ≥ 0.999
between the re-walk and the unwrapped transformer.

The resulting oracles reflect this evidence. FLUX.2 recorded ULP-level per-step divergence and
about 0.99 full-generation cosine, so its gate is cosine ≥ 0.97
([`tests/test_parity_flux2.py`](https://github.com/IonDen/mlx-teacache/blob/v0.9.2/tests/test_parity_flux2.py)).
Qwen-Image and Z-Image use cosine ≥ 0.99 threshold-zero gates. Their published basis is a
first-step ≥ 0.999 faithful-port check, not a multi-prompt full-generation distribution
([Qwen-Image](https://github.com/IonDen/mlx-teacache/blob/v0.9.2/docs/variants/qwen-image.md),
[Z-Image](https://github.com/IonDen/mlx-teacache/blob/v0.9.2/docs/variants/z-image-base.md)).

The `vanilla_before` / `vanilla_after` restore control remains bit-exact across these variants.
Both vanilla runs use the same untouched mflux path in one process. Each comparison now uses the
strongest oracle that its boundary supports.

### Is intentionally approximate output acceptable?

At a positive threshold TeaCache is *supposed* to change the output. The test therefore checks
perceptual quality and behavior, not latent equality. It measures SSIM on VAE-decoded images
against a same-process vanilla baseline. It also requires 5 to 7 skips for FLUX.1-dev's
default-threshold bench recipe, so a dormant cache cannot pass by doing nothing. The dormancy
failure mode is documented in the
[companion note on the distilled-schedule gate](https://github.com/IonDen/mlx-teacache/blob/main/docs/papers/why-teacache-does-not-engage-on-short-distilled-schedules.md).

## 5. A discipline for calibrating tolerances

A defensible tolerance should be measured rather than negotiated upward after a failure. This is
the procedure the project aims to apply to each MLX integration:

1. **Run the paired comparison and record a distribution**, not a pass/fail. Record max-abs,
   mean-abs, max-relative (with an epsilon denominator), and cosine similarity across several
   prompts on the target hardware. For image models, also record SSIM after decode.
2. **Check that a separation exists.** A tolerance is only an oracle if measured noise and the
   smallest bug worth catching live on opposite sides of it. The May incident is the negative
   example: passing the fixture comparison would have needed an absolute tolerance above 5.8.
   Such a gate would no longer separate numerical drift from many meaningful failures, so the
   correct move was to change the comparison, not the constant.
3. **Set the gate just below the measured floor**, leaving margin only for known variance
   sources. The FLUX.2 cosine gate is 0.97 against a measured full-generation result of about
   0.99. Qwen-Image and Z-Image expose the limit of the current record: their 0.99 gates are
   supported by a ≥ 0.999 first-step faithful-port check, not by a published multi-prompt
   full-generation distribution. Those gates are useful regression guards, but they should remain
   provisional and be tightened or revised when that distribution is recorded.
4. **Commit the basis next to the constant.** A gate comment should state what was measured, when,
   and on what hardware, so the next person can distinguish "calibrated" from "provisional."

The alternative is to loosen a failing assertion until it turns green. That silently reclassifies
each newly accepted difference as noise, usually without measuring whether the test can still
catch the failures it was meant to catch.

## 6. How other caching projects test correctness

The project surveyed how other diffusion-caching layers validate correctness (survey of May 2026,
re-checked against the public repositories at drafting time):

- [ali-vilab/TeaCache](https://github.com/ali-vilab/TeaCache/blob/main/TeaCache4FLUX/teacache_flux.py),
  the reference implementation, patches `FluxTransformer2DModel.forward` at class level. It ships
  no parity tests; validation is visual side-by-side.
- [ComfyUI-TeaCache](https://github.com/welltop-cn/ComfyUI-TeaCache) integrates through ComfyUI's
  patch nodes; the May 2026 survey found no pytest-style correctness suite.
- Hugging Face Diffusers includes a stronger automated check. Its
  [`FirstBlockCacheTesterMixin`](https://github.com/huggingface/diffusers/blob/main/tests/pipelines/test_pipelines_common.py)
  runs a four-step dummy pipeline on CPU. It uses `np.allclose` with a default `atol=0.1` while
  caching is enabled and `atol=1e-4` after disabling it. The comparison is tolerant, runs in one
  process, and does not use a committed real-model artifact
  ([caching API docs](https://huggingface.co/docs/diffusers/api/cache)).

No surveyed project uses byte-exact comparison against committed real-model outputs as a
correctness gate. PyTorch's reproducibility notes disclaim cross-release and cross-platform
bitwise stability. Work on batch-invariant inference treats bitwise stability as an engineering
goal that requires purpose-built kernels, not as a default property. On MLX,
[`mlx-deterministic`](https://github.com/ProbioticFarmer/mlx-deterministic) provides
batch-invariant RMSNorm, matmul, attention, and softmax kernels. Its README reports measured
overhead of roughly **7%** for RMSNorm and **27 to 32%** for matmul. Its
scope is batch-size invariance for LLM inference rather than the script-structure sensitivity
observed here. It shows that bitwise determinism on Metal is possible when a project pays for it.
For a performance library such as `mlx-teacache`, that cost is hard to justify merely to preserve
one convenient assertion.

## 7. Claims and their status

| Claim | Status | Source |
|---|---|---|
| Threshold-zero wrapper vs. committed fixture: 5/5 dev failures, cosine 0.105 to 0.256, max_abs 5.31 to 5.82; 5/5 schnell passes | author-reported measurement from May 2026; raw logs not public | method and limitations summarized in sections 1 and 2 |
| Same-process wrapper vs. vanilla, per-step, 25 steps: bit-exact | author-reported May 2026 diagnostic; raw logs not public | method summarized in section 2 |
| Current FLUX.1 paired test asserts byte-exact final latents and zero skips | verified against the v0.9.2 public test | [`tests/test_parity_flux1.py`](https://github.com/IonDen/mlx-teacache/blob/v0.9.2/tests/test_parity_flux1.py) |
| Vanilla mflux with a changed callback shape reproduces the wrapper's digest, not the fixture's | author-reported single-variable experiment from May 2026; raw logs not public | method summarized in section 2 |
| Threshold-zero fast path did not reduce cross-process divergence | author-reported measurement that falsified the donation hypothesis as dominant cause | result summarized in section 3; fast path public in [`_kernel/gate.py`](https://github.com/IonDen/mlx-teacache/blob/v0.9.2/src/mlx_teacache/_kernel/gate.py) |
| FLUX.2 changed forward path: ULP-level divergence and about 0.99 full-generation cosine at threshold 0 on an M1 Max | source-reported measurement; because mflux skips compilation on this chip, it is not evidence for a compiled-vs-eager cause | [`tests/test_parity_flux2.py`](https://github.com/IonDen/mlx-teacache/blob/v0.9.2/tests/test_parity_flux2.py) |
| Qwen-Image/Z-Image threshold-zero gate is cosine ≥ 0.99; separate first-step re-walk check is ≥ 0.999 | verified as the documented public gate and faithful-port check; not a published multi-prompt full-generation floor | [variant pages](https://github.com/IonDen/mlx-teacache/tree/v0.9.2/docs/variants) |
| MLX documents lazy graphs, output-rooted eval, donation, and dispatch heuristics, but not nondeterminism; no deterministic-mode API found | verified against docs and pinned source at drafting time | links in section 3 |
| No surveyed caching project gates on committed real-model bytes | verified against public repositories at drafting time | links in section 6 |
| Root cause of the cross-process byte difference | unisolated; donation hypothesis failed its discriminating experiment | this note, section 3 |

## 8. Lessons

In short, a golden file can prove correctness only when the test controls every source of byte
changes. This file depended on the script that produced it. Schnell happened to match and dev did
not, but even an all-green result could have broken after a callback change or MLX upgrade.

Use separate tests for separate jobs. Hashes catch file drift. Paired runs test the wrapper's math.
If both runs share the same numerical path, compare bytes. If the path changes, use a measured
tolerance. When caching is enabled, check image quality and confirm that skips occurred.

The donation idea also shows how to handle an incomplete explanation. It predicted that removing
some live references would reduce the byte difference. That did not happen. The fast path remains
because it removes unused work, but the root cause is still unknown. The testing conclusion does
not depend on finding it.

## References and source notes

- MLX documentation and pinned source (commit
  [`046217b`](https://github.com/ml-explore/mlx/tree/046217bcae7347aa814665f39a8f0e404029ddb0),
  inspected 2026-05-14, links re-verified at drafting time):
  [lazy evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html),
  [random keys](https://ml-explore.github.io/mlx/build/html/python/random.html),
  [`transforms.cpp` (`eval_impl`)](https://github.com/ml-explore/mlx/blob/046217bcae7347aa814665f39a8f0e404029ddb0/mlx/transforms.cpp#L74),
  [`array.h` (graph node, `is_donatable`)](https://github.com/ml-explore/mlx/blob/046217bcae7347aa814665f39a8f0e404029ddb0/mlx/array.h#L294),
  [`backend/metal/matmul.cpp`](https://github.com/ml-explore/mlx/blob/046217bcae7347aa814665f39a8f0e404029ddb0/mlx/backend/metal/matmul.cpp).
- `mlx-teacache` public record: [CHANGELOG](https://github.com/IonDen/mlx-teacache/blob/v0.9.2/CHANGELOG.md)
  (v0.1.0 test-pyramid entry and later parity-suite revisions),
  [`tests/test_parity_flux1.py`](https://github.com/IonDen/mlx-teacache/blob/v0.9.2/tests/test_parity_flux1.py),
  [`tests/test_parity_flux2.py`](https://github.com/IonDen/mlx-teacache/blob/v0.9.2/tests/test_parity_flux2.py),
  [`tests/test_fixtures_integrity.py`](https://github.com/IonDen/mlx-teacache/blob/v0.9.2/tests/test_fixtures_integrity.py),
  [`tests/test_hf_revisions.py`](https://github.com/IonDen/mlx-teacache/blob/v0.9.2/tests/test_hf_revisions.py),
  and the [per-variant documentation](https://github.com/IonDen/mlx-teacache/tree/v0.9.2/docs/variants).
- mflux per-step evaluation boundary:
  [`flux.py` at v0.18.0](https://github.com/filipstrand/mflux/blob/v.0.18.0/src/mflux/models/flux/variants/txt2img/flux.py#L122).
- TeaCache: [arXiv:2411.19108](https://arxiv.org/abs/2411.19108);
  [ali-vilab/TeaCache reference implementation](https://github.com/ali-vilab/TeaCache/blob/main/TeaCache4FLUX/teacache_flux.py);
  [ComfyUI-TeaCache](https://github.com/welltop-cn/ComfyUI-TeaCache).
- Hugging Face Diffusers: [caching API](https://huggingface.co/docs/diffusers/api/cache);
  [`FirstBlockCacheTesterMixin`](https://github.com/huggingface/diffusers/blob/main/tests/pipelines/test_pipelines_common.py).
- Reproducibility prior art:
  [PyTorch randomness notes](https://docs.pytorch.org/docs/stable/notes/randomness.html);
  [Thinking Machines, "Defeating Nondeterminism in LLM Inference" (September 2025)](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/);
  [`mlx-deterministic` (batch-invariant Metal kernels)](https://github.com/ProbioticFarmer/mlx-deterministic).

---

*Prepared 2026-07-24. Last updated 2026-07-25. Denis Ineshin.*
