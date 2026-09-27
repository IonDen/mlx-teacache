"""Critical call sites in the comparison harness's generation worker must appear exactly once, so a
hand-edit or a refactor can't silently duplicate or drop the wiring these tests exist to guard (style of
tests/test_scripts_memory_caps.py)."""

from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "bench_comparison.py"


def test_apply_teacache_passes_teacache_kwargs_exactly_once() -> None:
    # bug caught: a hand-edited call drops **teacache_kwargs(recipe), so a recipe's rel_l1_thresh override
    # (or a future teacache_kwargs field) silently never reaches condition B's apply_teacache.
    text = SCRIPT.read_text()
    assert text.count("apply_teacache(flux, **teacache_kwargs(recipe))") == 1


def test_generate_image_forwards_generate_kwargs_for_exactly_once() -> None:
    # bug caught: generate_image is called without **generate_kwargs_for(recipe), so Z-Image/Qwen's negative
    # prompt never reaches generate_image even though precompute_prompt already encoded it.
    text = SCRIPT.read_text()
    assert text.count("**generate_kwargs_for(recipe)") == 1


def test_quality_probe_worker_nests_output_by_slug_and_name_exactly_once() -> None:
    # bug caught: the quality-probe worker's path_slug stops matching quality_probe_dir's <slug>/<name>
    # nesting, so record.json and the images it points at end up in different directories.
    text = SCRIPT.read_text()
    assert text.count('path_slug=f"{recipe.slug}/{name}"') == 1
