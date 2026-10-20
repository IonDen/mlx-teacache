"""Unit tests for the chunked/resumable sweep plumbing in
`scripts/sweep_threshold_qwen.py` — unit naming, run order (vanilla first),
resume (pending), and summary aggregation. Pure functions: no weights, no
generation.

Importing the script module pulls scikit-image (SSIM) at module top, so this runs
in the mflux lane (where scikit-image is installed).
"""

import json
import sys
from pathlib import Path

import pytest

# sweep_threshold_qwen imports PIL + scikit-image at module top; both live in the
# [mflux]/test extra and are absent in the pure-core CI env. Skip this module
# cleanly there (importorskip) instead of erroring collection on the import below.
pytest.importorskip("PIL")
pytest.importorskip("skimage")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import sweep_threshold_qwen as sw  # noqa: E402

pytestmark = pytest.mark.mflux


def test_threshold_name_three_decimals() -> None:
    assert sw._threshold_name(0.05) == "t0.050"
    assert sw._threshold_name(0.4) == "t0.400"


def test_units_vanilla_first() -> None:
    units = sw._units()
    assert units[0] == "vanilla"
    assert units[1:] == [sw._threshold_name(t) for t in sw.THRESHOLDS]
    assert len(units) == 1 + len(sw.THRESHOLDS)


def test_pending_units_preserves_run_order_and_skips_existing(tmp_path: Path) -> None:
    units = sw._units()
    (tmp_path / sw._chunk_filename("vanilla")).write_text("{}")
    (tmp_path / sw._chunk_filename(units[1])).write_text("{}")
    pending = sw._pending_units(tmp_path, units)
    assert "vanilla" not in pending
    assert units[1] not in pending
    assert pending == units[2:]  # remaining thresholds, order preserved


def test_pending_all_done_is_empty(tmp_path: Path) -> None:
    units = sw._units()
    for u in units:
        (tmp_path / sw._chunk_filename(u)).write_text("{}")
    assert sw._pending_units(tmp_path, units) == []


def test_build_summary_sorts_by_threshold_and_derives_speedup() -> None:
    chunks = [
        {"threshold": 0.25, "wrapper_seconds": 5.0, "skipped": 6, "computed": 44, "ssim_vs_vanilla": 0.99},
        {"threshold": 0.05, "wrapper_seconds": 8.0, "skipped": 1, "computed": 49, "ssim_vs_vanilla": 0.999},
    ]
    summary = sw._build_summary(chunks, vanilla_seconds=10.0)
    ts = summary["thresholds"]
    assert [r["threshold"] for r in ts] == [0.05, 0.25]  # ascending
    assert ts[0]["speedup_vs_vanilla_single_rep"] == 10.0 / 8.0
    assert ts[1]["speedup_vs_vanilla_single_rep"] == 10.0 / 5.0
    assert summary["vanilla_seconds"] == 10.0
    assert summary["signal"] == "A"
    assert summary["num_inference_steps"] == sw.STEPS


# --- v0.10.1: streak telemetry, threshold override, build label ---


def test_build_summary_carries_streak_telemetry_per_threshold() -> None:
    # bug caught: dropping skip_pattern / max_consecutive_skips when assembling the rows
    chunks = [
        {
            "threshold": 0.2,
            "wrapper_seconds": 5.0,
            "skipped": 6,
            "computed": 44,
            "ssim_vs_vanilla": 0.99,
            "skip_pattern": "CSSC",
            "max_consecutive_skips": 2,
        }
    ]
    row = sw._build_summary(chunks, vanilla_seconds=10.0)["thresholds"][0]
    assert row["skip_pattern"] == "CSSC"
    assert row["max_consecutive_skips"] == 2


def test_build_summary_tolerates_pre_telemetry_chunks() -> None:
    chunks = [
        {"threshold": 0.2, "wrapper_seconds": 5.0, "skipped": 6, "computed": 44, "ssim_vs_vanilla": 0.99}
    ]
    row = sw._build_summary(chunks, vanilla_seconds=10.0)["thresholds"][0]
    assert row["skip_pattern"] == ""
    assert row["max_consecutive_skips"] == 0


def test_units_honour_an_explicit_threshold_list() -> None:
    # bug caught: ignoring the CLI override and always sweeping the module constant
    assert sw._units([0.15, 0.3]) == ["vanilla", "t0.150", "t0.300"]


def test_build_summary_records_the_build() -> None:
    assert sw._build_summary([], vanilla_seconds=1.0, build="plain-q4")["build"] == "plain-q4"


def test_build_summary_records_whether_the_text_encoders_were_freed() -> None:
    # bug caught: a re-measure under MemorySaver silently compared against a run without it
    assert sw._build_summary([], vanilla_seconds=1.0, memory_saver=True)["memory_saver"] is True
    assert sw._build_summary([], vanilla_seconds=1.0)["memory_saver"] is False


def test_memory_saver_is_built_with_the_load_bearing_kwargs() -> None:
    """bug caught: dropping cache_limit_bytes=None (MemorySaver's default 1 GB branch
    overrides our cache cap, resets the peak counter and switches the VAE to tiled
    decode, changing the SSIM reference pixels) or keep_transformer=True."""
    seen: dict = dict()

    class _FakeSaver:
        def __init__(self, **kw):
            seen.update(kw)

    flux = object()
    sw._make_memory_saver(flux, _FakeSaver)
    assert seen == dict(model=flux, keep_transformer=True, cache_limit_bytes=None, num_seeds=1)


# --- --model: the sweep targets any Qwen checkpoint ---

_QWEN_2512 = "Qwen/Qwen-Image-2512"


def test_default_chunk_dir_of_the_original_checkpoint_is_unchanged() -> None:
    """Bug: adding --model moves the original checkpoint's chunks, so its committed-recipe sweeps stop resuming."""
    out = Path(sw.__file__).resolve().parent.parent / "tests" / "_artifacts" / "sweep_qwen"
    assert sw._default_chunk_dir(True) == out / "_chunks_plainq4"
    assert sw._default_chunk_dir(False) == out / "_chunks"
    assert sw._default_chunk_dir(True, "Qwen/Qwen-Image") == sw._default_chunk_dir(True)


@pytest.mark.parametrize("plain_q4", [True, False])
def test_default_chunk_dir_of_another_checkpoint_is_its_own_slug_folder(plain_q4: bool) -> None:
    """Bug: a 2512 sweep reuses the original's chunk dir, resumes on its vanilla/threshold chunks and mixes the two."""
    other = sw._default_chunk_dir(plain_q4, _QWEN_2512)
    assert other.name == "qwen-qwen-image-2512"
    assert other.parent == sw._default_chunk_dir(plain_q4)


def test_image_dir_of_another_checkpoint_never_shares_the_originals_pngs() -> None:
    """Bug: vanilla.png of one checkpoint is the SSIM reference of the other, or the PNGs overwrite each other."""
    assert sw._image_dir(True) == sw.OUT_DIR / "plainq4"
    assert sw._image_dir(False) == sw.OUT_DIR
    assert sw._image_dir(True, _QWEN_2512) == sw.OUT_DIR / "plainq4" / "qwen-qwen-image-2512"
    assert sw._image_dir(False, _QWEN_2512) == sw.OUT_DIR / "qwen-qwen-image-2512"


def test_build_summary_records_the_model() -> None:
    """Bug: a results file that does not say which checkpoint it measured."""
    assert sw._build_summary([], vanilla_seconds=1.0, model=_QWEN_2512)["model"] == _QWEN_2512
    assert sw._build_summary([], vanilla_seconds=1.0)["model"] == "Qwen/Qwen-Image"


def test_build_summary_names_the_variant_the_chunks_matched() -> None:
    """Bug: a 2512 results file labelled variant "qwen-image", so its numbers read as the original's."""
    chunk = {
        "threshold": 0.2,
        "wrapper_seconds": 5.0,
        "skipped": 6,
        "computed": 44,
        "ssim_vs_vanilla": 0.99,
        "variant": "qwen-image-2512",
    }
    assert sw._build_summary([chunk], vanilla_seconds=10.0)["variant"] == "qwen-image-2512"
    assert sw._build_summary([], vanilla_seconds=1.0)["variant"] == "qwen-image"


def _dry_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *argv: str) -> None:
    """Run the real orchestrator (and its dry-run worker subprocesses) with every default dir under tmp_path."""
    monkeypatch.setattr(sw, "OUT_DIR", tmp_path / "sweep_qwen")
    monkeypatch.setattr(sw, "CHUNK_DIR_DEFAULT", tmp_path / "sweep_qwen" / "_chunks")
    monkeypatch.setattr(sys, "argv", ["sweep_threshold_qwen.py", "--dry-run", *argv])
    sw.main()


def test_dry_run_for_another_checkpoint_writes_model_stamped_chunks_into_its_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Bug: --model is parsed but not passed to the workers, so the chunks (and results) carry the wrong checkpoint."""
    _dry_run(monkeypatch, tmp_path, "--model", _QWEN_2512, "--plain-q4", "--thresholds", "0.2")
    chunk_dir = sw._default_chunk_dir(True, _QWEN_2512)
    assert chunk_dir.parent == tmp_path / "sweep_qwen" / "_chunks_plainq4"
    for unit in ("vanilla", "t0.200"):
        chunk = json.loads((chunk_dir / sw._chunk_filename(unit)).read_text())
        assert chunk["model"] == _QWEN_2512
    results = json.loads((chunk_dir / "results_qwen_plainq4.json").read_text())
    assert results["model"] == _QWEN_2512
    assert not (tmp_path / "sweep_qwen" / "_chunks_plainq4" / sw._chunk_filename("vanilla")).exists()


def test_dry_run_default_model_is_the_original_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Bug: the default flips to another checkpoint, so a bare run loads the uncalibrated one."""
    _dry_run(monkeypatch, tmp_path, "--plain-q4", "--thresholds", "0.2")
    chunk_dir = tmp_path / "sweep_qwen" / "_chunks_plainq4"
    assert json.loads((chunk_dir / sw._chunk_filename("t0.200")).read_text())["model"] == "Qwen/Qwen-Image"


def test_orchestrator_refuses_chunks_from_another_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Bug: a shared --chunk-dir silently aggregates one checkpoint's vanilla with another's thresholds."""
    chunk_dir = tmp_path / "shared"
    _dry_run(monkeypatch, tmp_path, "--plain-q4", "--thresholds", "0.2", "--chunk-dir", str(chunk_dir))
    with pytest.raises(SystemExit, match="Qwen-Image-2512"):
        _dry_run(
            monkeypatch,
            tmp_path,
            "--model",
            _QWEN_2512,
            "--plain-q4",
            "--thresholds",
            "0.2",
            "--chunk-dir",
            str(chunk_dir),
        )


def test_real_run_aggregation_creates_the_per_model_results_folder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Bug: the canonical results JSON of a non-original checkpoint is written into a folder nobody created,
    so the aggregation of a finished 2512 sweep crashes with FileNotFoundError after all units ran."""
    monkeypatch.setattr(sw, "OUT_DIR", tmp_path / "sweep_qwen")
    monkeypatch.setattr(sw, "CHUNK_DIR_DEFAULT", tmp_path / "sweep_qwen" / "_chunks")
    chunk_dir = sw._default_chunk_dir(True, _QWEN_2512)
    chunk_dir.mkdir(parents=True)
    (chunk_dir / sw._chunk_filename("vanilla")).write_text(
        json.dumps({"unit": "vanilla", "model": _QWEN_2512, "vanilla_seconds": 10.0})
    )
    (chunk_dir / sw._chunk_filename("t0.200")).write_text(
        json.dumps(
            {
                "unit": "t0.200",
                "model": _QWEN_2512,
                "variant": "qwen-image-2512",
                "threshold": 0.2,
                "wrapper_seconds": 5.0,
                "skipped": 8,
                "computed": 42,
                "ssim_vs_vanilla": 0.99,
            }
        )
    )
    # Every unit already has its chunk and dry_run is False, so no worker starts and no model loads:
    # this walks the real-run path to the canonical results file.
    sw._run_orchestrator(
        chunk_dir=chunk_dir, dry_run=False, thresholds=[0.2], plain_q4=True, model=_QWEN_2512
    )
    results = json.loads(
        (tmp_path / "sweep_qwen" / "qwen-qwen-image-2512" / "results_qwen_plainq4.json").read_text()
    )
    assert results["model"] == _QWEN_2512
    assert results["variant"] == "qwen-image-2512"
