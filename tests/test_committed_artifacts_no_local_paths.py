"""The committed JSON reports carry no absolute path from the machine that wrote them.

Scanned: every ``_artifacts/**/*.json`` bench report and every ``scripts/_calibration_*.json`` record.
The bench writers emit repo-relative paths (``tests/test_bench_stamps.py`` guards them); this module
guards the committed JSON itself, so an old report or a hand-edited one cannot ship a home directory.
Pure: reads the committed files (by glob, no git), imports nothing from MLX or mflux.
"""

import json
import re
from collections.abc import Iterator
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ARTIFACTS = _REPO_ROOT / "_artifacts"
_SCRIPTS = _REPO_ROOT / "scripts"

_LOCAL_PREFIXES = ("/Users/", "/home/", "/private/", "/var/folders/", "/tmp/", "/Volumes/", "~/")

# An absolute or home-relative path that starts at the beginning of the string or after a character that cannot
# continue a word, path or URL (space, quote, `=`, bracket, ...); it runs to the next space, quote or closing
# bracket. So `a/Users/b` and `https://host/home/x` yield nothing, `saved to /Users/x` yields `/Users/x`.
_PATH_CANDIDATE = re.compile(r"(?<![\w/.~-])(?:~/|/)[^\s'\"`<>|)\]},;]*")


def local_paths_in_string(value: str, repo_root: Path) -> list[str]:
    """Every path in ``value`` that names this machine: one under a home, temp, private or external-disk
    prefix (or ``~/``), or any absolute POSIX path inside the checkout."""
    root = repo_root.resolve()
    return [
        path
        for path in _PATH_CANDIDATE.findall(value)
        if path.startswith(_LOCAL_PREFIXES)
        or (path.startswith("/") and Path(path).resolve().is_relative_to(root))
    ]


def is_local_absolute_path(value: str, repo_root: Path) -> bool:
    """True when ``value`` holds a path that names this machine (see ``local_paths_in_string``)."""
    return bool(local_paths_in_string(value, repo_root))


def _pointer_token(key: str) -> str:
    return key.replace("~", "~0").replace("/", "~1")


def string_leaves(node: object, pointer: str = "") -> Iterator[tuple[str, str]]:
    """Yield ``(json_pointer, string)`` for every string in a parsed JSON document: each string value, and each
    object key (reported at the pointer of its member)."""
    if isinstance(node, str):
        yield pointer, node
    elif isinstance(node, dict):
        for key, child in node.items():
            child_pointer = f"{pointer}/{_pointer_token(str(key))}"
            yield child_pointer, str(key)
            yield from string_leaves(child, child_pointer)
    elif isinstance(node, list):
        for index, child in enumerate(node):
            yield from string_leaves(child, f"{pointer}/{index}")


def local_paths_in(document: object, repo_root: Path) -> list[tuple[str, str]]:
    """``(json_pointer, path)`` for every local path found in the document's strings."""
    return [(ptr, path) for ptr, s in string_leaves(document) for path in local_paths_in_string(s, repo_root)]


# --- the predicate -----------------------------------------------------------


# A checkout root outside every local prefix (it need not exist: resolve() is non-strict), so the
# inside-the-checkout rule is exercised on its own. A tmp_path root would sit under /private/ on macOS.
_FAKE_ROOT = Path("/srv/checkout")


def test_home_temp_and_private_prefixes_are_local() -> None:
    """Bug: dropping one prefix from the list lets that platform's home or temp dir through."""
    for value in (
        "/Users/someone/repo/tests/_artifacts/bench_images/flux1-dev",
        "/home/runner/work/repo/_artifacts/x",
        "/private/tmp/run/images",
        "/var/folders/ab/cd/T/images",
    ):
        assert is_local_absolute_path(value, _FAKE_ROOT), value


def test_absolute_path_inside_the_checkout_is_local() -> None:
    """Bug: only the prefix list checked, so a checkout under /opt or /srv leaks its location."""
    assert is_local_absolute_path("/srv/checkout/tests/_artifacts/bench_images", _FAKE_ROOT)
    assert is_local_absolute_path("/srv/checkout", _FAKE_ROOT)


def test_absolute_path_outside_the_checkout_with_no_local_prefix_is_not_flagged() -> None:
    """Bug: flagging every string that starts with a slash, or a string-prefix test on the root
    (``/srv/checkout-old`` is a sibling, not inside ``/srv/checkout``)."""
    assert not is_local_absolute_path("/opt/elsewhere/images", _FAKE_ROOT)
    assert not is_local_absolute_path("/srv/checkout-old/images", _FAKE_ROOT)


def test_relative_paths_and_plain_strings_are_not_flagged() -> None:
    """Bug: a substring match on ``Users/`` flagging the repo-relative form the writers emit."""
    for value in (
        "tests/_artifacts/bench_images/flux1-dev",
        "_artifacts/klein_base_vs_distilled/4b_q4",
        "black-forest-labs/FLUX.1-dev",
        "https://huggingface.co/Users/x",
        "",
    ):
        assert not is_local_absolute_path(value, _FAKE_ROOT), value


def test_tmp_external_volume_and_tilde_home_are_local() -> None:
    """Bug: dropping /tmp/, /Volumes/ or ~/ from the list (or resolving /tmp to /private/tmp only on macOS) lets a
    report written to a temp dir, an external disk or a home-relative path through."""
    for value in ("/tmp/run/images", "/Volumes/External/runs/images", "~/runs/images"):
        assert is_local_absolute_path(value, _FAKE_ROOT), value


def test_a_local_path_inside_a_longer_string_is_found() -> None:
    """Bug: only a string that starts with a local prefix is checked, so a path after a space, a quote, an equals
    sign or a bracket (a logged command, a message) ships."""
    for value, path in (
        ("saved to /Users/someone/x.png", "/Users/someone/x.png"),
        ("cmd='python /tmp/run.py'", "/tmp/run.py"),
        ('{"out": "/Volumes/ext/x"}', "/Volumes/ext/x"),
        ("--images=/home/runner/x", "/home/runner/x"),
        ("(see ~/notes/run.txt)", "~/notes/run.txt"),
        ("inside /srv/checkout/tests/_artifacts", "/srv/checkout/tests/_artifacts"),
    ):
        assert local_paths_in_string(value, _FAKE_ROOT) == [path], value


def test_a_local_prefix_inside_a_relative_path_or_url_is_not_flagged() -> None:
    """Bug: a bare substring search, which flags a relative path or URL that merely contains a local prefix."""
    for value in (
        "runs/tmp/images",
        "a/Users/b",
        "https://example.com/home/x",
        "https://huggingface.co/Volumes/x",
        "data~/x",
    ):
        assert local_paths_in_string(value, _FAKE_ROOT) == [], value


def test_walk_finds_a_path_in_a_dict_key() -> None:
    """Bug: the walk reads values only, so a report keyed by absolute image path ships it."""
    document = {"images": {"/Users/someone/x.png": 0.98, "rel/y.png": 0.97}}
    assert local_paths_in(document, _FAKE_ROOT) == [
        ("/images/~1Users~1someone~1x.png", "/Users/someone/x.png")
    ]


def test_walk_finds_strings_nested_in_dicts_and_lists() -> None:
    """Bug: the walk reading only top-level values, missing a path inside ``conditions[*]``."""
    document = {
        "speedup_median": 1.5,
        "conditions": [{"name": "teacache", "images": ["ok/rel", "/Users/someone/x.png"]}],
        "flag": True,
        "none": None,
    }
    assert local_paths_in(document, _FAKE_ROOT) == [("/conditions/0/images/1", "/Users/someone/x.png")]


# --- the committed reports ---------------------------------------------------


def _artifact_jsons() -> list[Path]:
    return sorted([*_ARTIFACTS.rglob("*.json"), *_SCRIPTS.glob("_calibration_*.json")])


def test_artifact_glob_sees_the_committed_reports() -> None:
    """Bug: a wrong directory or pattern makes the scan below pass on zero files."""
    names = {p.name for p in _artifact_jsons()}
    assert "v0.10.0_bench_flux1_dev.json" in names
    assert "klein_base_vs_distilled_4b_q4.json" in names


def test_artifact_glob_sees_the_committed_calibration_records() -> None:
    """Bug: the scan covers only _artifacts/, so a calibration record under scripts/ ships a local path."""
    names = {p.name for p in _artifact_jsons()}
    assert "_calibration_qwen.json" in names
    assert "_calibration_flux2_klein_4b.json" in names


def test_committed_artifacts_hold_no_absolute_local_path() -> None:
    """Bug: a report committed with ``str(images_dir)`` (absolute) instead of the repo-relative path."""
    offenders = [
        f"{path.relative_to(_REPO_ROOT).as_posix()}{ptr} = {value!r}"
        for path in _artifact_jsons()
        for ptr, value in local_paths_in(json.loads(path.read_text(encoding="utf-8")), _REPO_ROOT)
    ]
    assert offenders == [], "absolute local paths in committed reports:\n" + "\n".join(offenders)
