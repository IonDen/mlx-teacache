"""Recipe data for the comparison page: one prompt, one seed, six non-distilled models.

Pure (no mlx / mflux imports). Resolution drops only where memory forces it; the two tight rows carry a fallback that a
memory probe (``bench_comparison.py --probe``) decides. ``bench_report`` is the committed multi-rep bench report each
model block links for its multi-run speedup.
"""

import hashlib
from dataclasses import dataclass, replace

PROMPT = (
    "A beautiful young woman plays tennis on an outdoor hard court at sunset. She is caught just after a forehand, "
    "racket following through across her body, ponytail swinging, weight on her front foot. Her face shows focused, "
    "joyful determination: flushed cheeks, bright eyes, a light sheen of sweat on her forehead and temples. She wears "
    "a fitted white tennis dress with navy trim, a white visor, a terry wristband, small gold stud earrings, and white "
    "tennis shoes with navy laces. The green court's painted white lines lead back to a chain-link fence, the net, and "
    "silhouetted trees against an orange-pink sky. Low, warm sunlight rakes across the court, casting long soft "
    "shadows and a golden rim light on her hair. A yellow tennis ball hangs in the air just off the racket strings. "
    "The mood is cozy, warm and nostalgic. Photorealistic, full-frame camera, 85mm lens, shallow depth of field, "
    "natural skin texture, sharp focus on her face."
)
# Z-Image renders "flushed cheeks" and "sweat on her forehead" as red blotches and white specks on the face;
# this wording keeps the scene and asks for a softer glow, and the negative names the defects.
Z_IMAGE_PROMPT = (
    "A beautiful young woman plays tennis on an outdoor hard court at sunset. She is caught just after a forehand, "
    "racket following through across her body, ponytail swinging, weight on her front foot. Her face shows focused,"
    " joyful determination, with bright eyes, a natural warm glow on her cheeks and a faint dewy sheen on her skin."
    " She wears a fitted white tennis dress with navy trim, a white visor, a terry wristband, small gold stud "
    "earrings, and white tennis shoes with navy laces. The green court's painted white lines lead back to a chain-"
    "link fence, the net, and silhouetted trees against an orange-pink sky. Low, warm sunlight rakes across the "
    "court, casting long soft shadows and a golden rim light on her hair. A yellow tennis ball hangs in the air "
    "just off the racket strings. The mood is cozy, warm and nostalgic. Photorealistic, full-frame camera, 85mm "
    "lens, shallow depth of field, natural skin texture with fine pores, sharp focus on her face."
)
Z_IMAGE_NEGATIVE_PROMPT = (
    "blotchy skin, red patches on the cheeks, white specks on the skin, distorted face, asymmetric eyes, blurry, "
    "plastic skin, waxy skin, oversharpened, extra fingers, deformed hands, artifacts"
)
QWEN_PROMPT_SUFFIX = ", Ultra HD, 4K, cinematic composition."  # Alibaba's reference "positive magic"
SEED = 42
WORKING_SET_BYTES_DEFAULT = int(24.96 * 1024**3)  # M1 Max 32 GB max_recommended_working_set_size
MIN_HOST_FREE_PCT = 20.0
GUARD_LIMIT_BYTES = 25 * 1024**3  # mlx-guard's process-footprint kill threshold
FOOTPRINT_HEADROOM_BYTES = 1 * 1024**3  # footprint runs ~2-4 GiB above MLX's own peak; stay well clear


@dataclass(frozen=True, slots=True, kw_only=True)
class Recipe:
    slug: str
    variant_id: str
    display_name: str
    loader: str
    checkpoint: str
    decoder: str  # mlx-taef LivePreviewCallback variant
    bench_report: str  # committed multi-rep bench report (repo-relative)
    steps: int
    guidance: float
    quantize: int
    width: int
    height: int
    fallback: tuple[int, int] | None = None  # (width, height)
    free_encoders: bool = False
    wired_cap_gb: int = 22
    cache_gb: float = 2.0
    prompt: str | None = None  # per-model override; replaces the shared PROMPT when set
    negative_prompt: str | None = None
    rel_l1_thresh: float | None = (
        None  # TeaCache threshold for B; None leaves the variant's default in charge
    )


RECIPES: tuple[Recipe, ...] = (
    Recipe(
        slug="flux1-dev",
        variant_id="flux1-dev",
        display_name="FLUX.1 [dev]",
        loader="flux1-dev",
        checkpoint="black-forest-labs/FLUX.1-dev",
        decoder="taef1",
        bench_report="_artifacts/v0.10.0_bench_flux1_dev.json",
        steps=25,
        guidance=3.5,
        quantize=4,
        width=768,
        height=1024,
    ),
    Recipe(
        slug="flux1-krea-dev",
        variant_id="flux1-krea-dev",
        display_name="FLUX.1 Krea [dev]",
        loader="flux1-krea-dev",
        checkpoint="black-forest-labs/FLUX.1-Krea-dev",
        decoder="taef1",
        bench_report="_artifacts/v0.11.0_bench_krea_dev.json",
        steps=28,
        guidance=4.5,
        quantize=4,
        width=768,
        height=1024,
    ),
    Recipe(
        slug="klein-base-4b",
        variant_id="flux2-klein-base-4b",
        display_name="FLUX.2 [klein] base 4B",
        loader="klein-base-4b",
        checkpoint="black-forest-labs/FLUX.2-klein-base-4B",
        decoder="taef2",
        bench_report="_artifacts/v0.10.0_bench_klein_base_4b.json",
        steps=50,
        guidance=4.0,
        quantize=8,
        width=864,
        height=1152,
    ),
    Recipe(
        slug="z-image-base",
        variant_id="z-image-base",
        display_name="Z-Image",
        loader="z-image",
        checkpoint="Tongyi-MAI/Z-Image",
        decoder="zimage",
        bench_report="_artifacts/v0.10.0_bench_z_image.json",
        steps=50,
        guidance=4.0,
        quantize=8,
        width=864,
        height=1152,
        wired_cap_gb=24,
        prompt=Z_IMAGE_PROMPT,
        negative_prompt=Z_IMAGE_NEGATIVE_PROMPT,
    ),
    Recipe(
        slug="klein-base-9b",
        variant_id="flux2-klein-base-9b",
        display_name="FLUX.2 [klein] base 9B",
        loader="klein-base-9b",
        checkpoint="black-forest-labs/FLUX.2-klein-base-9B",
        decoder="taef2",
        bench_report="_artifacts/v0.10.0_bench_klein_base_9b.json",
        steps=50,
        guidance=4.0,
        quantize=8,
        width=864,
        height=1152,
        fallback=(576, 768),
        free_encoders=True,
    ),
    Recipe(
        slug="qwen-image",
        variant_id="qwen-image",
        display_name="Qwen-Image",
        loader="qwen-image-original",
        checkpoint="Qwen/Qwen-Image",
        decoder="qwen-image",
        bench_report="_artifacts/v0.11.0_bench_qwen_image.json",
        steps=50,
        guidance=4.0,
        quantize=4,
        width=672,
        height=896,
        fallback=(576, 768),
        free_encoders=True,
        wired_cap_gb=21,
        cache_gb=1.0,
    ),
)


def recipe_for(slug: str) -> Recipe:
    for recipe in RECIPES:
        if recipe.slug == slug:
            return recipe
    raise KeyError(f"no comparison recipe for slug {slug!r}; known: {[r.slug for r in RECIPES]}")


def prompt_for(recipe: Recipe) -> str:
    base = recipe.prompt if recipe.prompt is not None else PROMPT
    return base + QWEN_PROMPT_SUFFIX if recipe.slug == "qwen-image" else base


def prompt_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def with_resolution(recipe: Recipe, width: int, height: int) -> Recipe:
    return replace(recipe, width=width, height=height)


def apply_overrides(
    recipe: Recipe,
    *,
    width: int | None = None,
    height: int | None = None,
    guidance: float | None = None,
    quantize: int | None = None,
    prompt: str | None = None,
    negative_prompt: str | None = None,
    rel_l1_thresh: float | None = None,
) -> Recipe:
    """Replace only the given (non-None) fields; a quality probe's candidate settings over one recipe."""
    changes = {
        "width": width,
        "height": height,
        "guidance": guidance,
        "quantize": quantize,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "rel_l1_thresh": rel_l1_thresh,
    }
    given = {k: v for k, v in changes.items() if v is not None}
    return replace(recipe, **given) if given else recipe


def recipe_stamp(recipe: Recipe, *, versions: dict[str, str]) -> dict[str, object]:
    """Every field that changes what a chunk or probe measured, including memory budget fields.
    Git sha and mlx-teacache version are provenance only (an editable hatch-vcs install changes both on every
    commit). ``negative_prompt_sha256`` is emitted only when the recipe actually sets a negative, so every
    chunk and probe record written before this field existed keeps comparing equal (``check_chunk_stamp`` is
    an exact key-set match, not a superset match)."""
    stamp: dict[str, object] = {
        "slug": recipe.slug,
        "checkpoint": recipe.checkpoint,
        "decoder": recipe.decoder,
        "steps": recipe.steps,
        "guidance": recipe.guidance,
        "quantize": recipe.quantize,
        "width": recipe.width,
        "height": recipe.height,
        "free_encoders": recipe.free_encoders,
        "cache_gb": recipe.cache_gb,
        "wired_cap_gb": recipe.wired_cap_gb,
        "seed": SEED,
        "prompt_sha256": prompt_sha256(prompt_for(recipe)),
        **{f"version_{k}": val for k, val in sorted(versions.items())},
    }
    if recipe.negative_prompt:
        stamp["negative_prompt_sha256"] = prompt_sha256(recipe.negative_prompt)
    if recipe.rel_l1_thresh is not None:
        stamp["rel_l1_thresh"] = recipe.rel_l1_thresh
    return stamp


def teacache_kwargs(recipe: Recipe) -> dict[str, float]:
    """Keyword arguments for condition B's apply_teacache: the recipe's threshold override, if any."""
    return {} if recipe.rel_l1_thresh is None else {"rel_l1_thresh": recipe.rel_l1_thresh}


def probe_passes(
    *, active_peak_bytes: int, cache_limit_bytes: int, working_set_bytes: int, min_host_free_pct: float
) -> bool:
    """Active peak plus the whole cache pool under the working set, and the host keeps >= 20 % free (the harness
    kills background tasks at roughly 10-18 % free)."""
    return (
        active_peak_bytes + cache_limit_bytes < working_set_bytes and min_host_free_pct >= MIN_HOST_FREE_PCT
    )


def probe_record(
    recipe: Recipe,
    *,
    phase_peaks: dict[str, int],
    min_host_free_pct: float | None,
    working_set_bytes: int,
    versions: dict[str, str],
    aborted: str | None = None,
    peak_footprint_bytes: int | None = None,
) -> dict[str, object]:
    active = max(phase_peaks.values()) if phase_peaks else 0
    footprint_ok = (
        peak_footprint_bytes is not None
        and peak_footprint_bytes + FOOTPRINT_HEADROOM_BYTES < GUARD_LIMIT_BYTES
    )
    passed = (
        aborted is None
        and min_host_free_pct is not None
        and bool(phase_peaks)
        and footprint_ok
        and probe_passes(
            active_peak_bytes=active,
            cache_limit_bytes=int(recipe.cache_gb * 1024**3),
            working_set_bytes=working_set_bytes,
            min_host_free_pct=min_host_free_pct,
        )
    )
    return {
        "slug": recipe.slug,
        "width": recipe.width,
        "height": recipe.height,
        "pass": passed,
        "aborted": aborted,
        "active_peak_bytes": active,
        "peak_footprint_bytes": peak_footprint_bytes,
        "phase_peaks": dict(phase_peaks),
        "min_host_free_pct": min_host_free_pct,
        "working_set_bytes": working_set_bytes,
        "stamp": recipe_stamp(recipe, versions=versions),
    }


def resolve_resolution(
    recipe: Recipe, probes: list[dict[str, object]], *, versions: dict[str, str]
) -> Recipe:
    """Full size or the fallback, from probe records whose stamp matches today's recipe and versions."""
    if recipe.fallback is None:
        return recipe
    fallback = with_resolution(recipe, *recipe.fallback)

    def verdict(candidate: Recipe) -> bool | None:
        want = recipe_stamp(candidate, versions=versions)
        matching = [p for p in probes if p.get("stamp") == want]
        return None if not matching else bool(matching[-1].get("pass"))

    full_ok, fallback_ok = verdict(recipe), verdict(fallback)
    if full_ok is None:
        raise SystemExit(
            f"{recipe.slug}: no current probe at {recipe.width}x{recipe.height}; run "
            f"`bench_comparison.py --probe --only {recipe.slug}` first"
        )
    if full_ok:
        return recipe
    if fallback_ok is None:
        raise SystemExit(
            f"{recipe.slug}: full size failed its probe; run "
            f"`bench_comparison.py --probe --fallback --only {recipe.slug}`"
        )
    if fallback_ok:
        return fallback
    raise SystemExit(
        f"{recipe.slug}: both {recipe.width}x{recipe.height} and the fallback failed their probes; "
        "stop and report"
    )
