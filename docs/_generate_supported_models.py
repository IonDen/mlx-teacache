"""Generate the `## Supported models` table in README.md from the variant
registry. Run after adding or modifying a variant — the table goes between
the `<!-- SUPPORTED_MODELS_START -->` / `<!-- SUPPORTED_MODELS_END -->`
markers in README.md.

    uv run python docs/_generate_supported_models.py        # print to stdout
    uv run python docs/_generate_supported_models.py --write # update README.md in place
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from mlx_teacache.variants import _REGISTRY  # noqa: E402

START_MARKER = "<!-- SUPPORTED_MODELS_START -->"
END_MARKER = "<!-- SUPPORTED_MODELS_END -->"

# README order, by model family: families most used on Macs first (Hugging Face
# 30-day downloads of the MLX and mflux copies of the variants TeaCache speeds
# up, plus Google search interest over 90 days; checked 2026-10-22), the legacy
# FLUX.1 family last even though
# its official checkpoint still leads overall downloads. Inside a family, the
# variants TeaCache speeds up come before the distilled ones, each group by
# MLX-copy downloads. A new variant must be added here, or the generator
# refuses to run.
README_ORDER = (
    "qwen-image-2512",
    "qwen-image",
    "flux2-klein-base-9b",
    "flux2-klein-base-4b",
    "flux2-klein-4b",
    "flux2-klein-9b",
    "z-image-base",
    "flux1-dev",
    "flux1-krea-dev",
    "flux1-schnell",
)


def _ordered_ids(registry_ids: set[str], order: tuple[str, ...] = README_ORDER) -> list[str]:
    duplicates = sorted({v for v in order if order.count(v) > 1})
    missing = sorted(registry_ids - set(order))
    unknown = sorted(set(order) - registry_ids)
    if duplicates or missing or unknown:
        raise RuntimeError(
            "README_ORDER must list every registered variant exactly once: "
            f"duplicates={duplicates}, missing={missing}, unknown={unknown}"
        )
    return list(order)


def _recipe_summary(recipes: dict[str, dict[str, object]]) -> str:
    default = recipes.get("default", {})
    steps = default.get("num_inference_steps", "?")
    guidance = default.get("guidance", "?")
    return f"{steps} steps, g={guidance}"


def _build_table() -> str:
    header = "| Variant id | Display name | Distilled? | Default recipe | License |\n|---|---|---|---|---|"
    rows = []
    for variant_id in _ordered_ids(set(_REGISTRY)):
        meta = _REGISTRY[variant_id]["META"]
        distilled = "no" if meta["non_distilled"] else "yes"
        recipe = _recipe_summary(meta["recipes"])
        license_url = meta.get("license_url", "")
        license_cell = f"[{meta['license']}]({license_url})" if license_url else meta["license"]
        rows.append(
            f"| `{meta['variant_id']}` | {meta['display_name']} | {distilled} | {recipe} | {license_cell} |"
        )
    return header + "\n" + "\n".join(rows)


def _splice(readme: str, table: str) -> str:
    pattern = re.compile(
        re.escape(START_MARKER) + r".*?" + re.escape(END_MARKER),
        re.DOTALL,
    )
    block = f"{START_MARKER}\n{table}\n{END_MARKER}"
    if not pattern.search(readme):
        raise RuntimeError(
            f"README.md missing marker pair {START_MARKER!r}/{END_MARKER!r}. "
            "Insert them around the `## Supported models` table first."
        )
    return pattern.sub(block, readme)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Update README.md in place between the marker pair.",
    )
    args = parser.parse_args()

    table = _build_table()

    if not args.write:
        print(table)
        return

    readme_path = REPO_ROOT / "README.md"
    original = readme_path.read_text()
    updated = _splice(original, table)
    if updated == original:
        print("README.md already up to date.")
        return
    readme_path.write_text(updated)
    print(f"Updated {readme_path} between the marker pair.")


if __name__ == "__main__":
    main()
