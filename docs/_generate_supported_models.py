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


def _recipe_summary(recipes: dict) -> str:
    default = recipes.get("default", {})
    steps = default.get("num_inference_steps", "?")
    guidance = default.get("guidance", "?")
    return f"{steps} steps, g={guidance}"


def _build_table() -> str:
    header = (
        "| Variant id | Display name | Distilled? | Default recipe | License |\n"
        "|---|---|---|---|---|"
    )
    rows = []
    for variant_id in sorted(_REGISTRY):
        meta = _REGISTRY[variant_id]["META"]
        distilled = "no" if meta["non_distilled"] else "yes"
        recipe = _recipe_summary(meta["recipes"])
        license_url = meta.get("license_url", "")
        license_cell = (
            f"[{meta['license']}]({license_url})" if license_url else meta["license"]
        )
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
