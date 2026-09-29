"""Every place that builds a QwenImage must pass the original checkpoint's config, never the alias.

On mflux 0.19+ ``ModelConfig.qwen_image()``, the ``qwen-image`` alias and QwenImage's own default config all
resolve to Qwen-Image-2512, which has no calibration and is ~58 GB. The rule is checked per construction site
on the parsed AST (scripts/*.py and the Qwen parity test), so a comment or docstring that mentions the call can
neither satisfy nor trip it, and a new script cannot slip past a hard-coded file list."""

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

# Helpers that resolve the original checkpoint (qwen_config only for a caller-chosen --model).
_ALLOWED_CONFIG_CALLS = {"qwen_original_config", "qwen_config"}
_ALIAS_NAMES = {"qwen-image", "qwen", "qwen-image-2512", "qwen-2512"}
# QwenImage construction sites today: bench_speedup, sweep_threshold_qwen, calibrate_qwen x2,
# _comparison_models, tests/test_parity_qwen. A floor, so a walker gone blind cannot pass vacuously.
_MIN_SITES = 6


def _callee(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _violations(source: str, *, in_scripts: bool) -> list[str]:
    """Return one message per construction site or alias reference that breaks the rule."""
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            if _callee(node) == "QwenImage":
                kw = next((k for k in node.keywords if k.arg == "model_config"), None)
                value = kw.value if kw else None
                ok = isinstance(value, ast.Call) and _callee(value) in _ALLOWED_CONFIG_CALLS
                if not ok:
                    found.append(f"line {node.lineno}: QwenImage without model_config=qwen_original_config()")
            if _callee(node) in {"from_name", "qwen_config"} and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and first.value in _ALIAS_NAMES:
                    found.append(f"line {node.lineno}: {_callee(node)}({first.value!r}) is the 2512 alias")
        if in_scripts and isinstance(node, ast.Attribute) and node.attr == "qwen_image":
            found.append(f"line {node.lineno}: reference to .qwen_image (2512 on mflux 0.19+)")
    return found


def _sites(source: str) -> int:
    return sum(isinstance(n, ast.Call) and _callee(n) == "QwenImage" for n in ast.walk(ast.parse(source)))


def test_rule_accepts_the_helper_and_ignores_comments() -> None:
    """Bug: the rule rejects the good form or reads comments as code, so it cannot guard anything."""
    good = "# ModelConfig.qwen_image() is 2512\nflux = QwenImage(quantize=4, model_config=qwen_original_config())\n"
    assert _violations(good, in_scripts=True) == []
    assert _violations("QwenImage(model_config=qwen_config(args.model))\n", in_scripts=True) == []


@pytest.mark.parametrize(
    "source",
    [
        "QwenImage(quantize=4)\n",  # default config is the 2512 alias
        "QwenImage(quantize=4, model_config=ModelConfig.qwen_image())\n",
        "q = ModelConfig.qwen_image\nQwenImage(model_config=q())\n",  # alias hidden behind a name
        "cfg = something()\nQwenImage(model_config=cfg)\n",  # unknown provenance
        "ModelConfig.from_name('qwen-image')\n",
        "ModelConfig.from_name('qwen-image-2512')\n",
        "QwenImage(model_config=qwen_config('qwen-image'))\n",  # the helper accepts any name; the alias is not one
    ],
)
def test_rule_rejects_every_way_back_to_the_alias(source: str) -> None:
    """Bug: a construction site drops model_config or routes the alias through a name and the rule stays green."""
    assert _violations(source, in_scripts=True) != []


def test_from_name_of_the_original_checkpoint_is_allowed() -> None:
    """Bug: the from_name rule flags the original checkpoint the helper itself resolves."""
    assert _violations("ModelConfig.from_name('Qwen/Qwen-Image')\n", in_scripts=True) == []


def test_every_qwen_construction_site_names_the_original_checkpoint() -> None:
    """Bug: any script or the parity test builds a QwenImage that loads the uncalibrated 2512 checkpoint."""
    files = sorted(SCRIPTS.glob("*.py"))
    parity = ROOT / "tests" / "test_parity_qwen.py"
    total = 0
    offenders: list[str] = []
    for path in [*files, parity]:
        source = path.read_text()
        total += _sites(source)
        offenders += [f"{path.name} {v}" for v in _violations(source, in_scripts=path.parent == SCRIPTS)]
    assert offenders == []
    assert total >= _MIN_SITES


@pytest.mark.mflux  # imports mflux; the AST tests above stay in the pure-core lane
@pytest.mark.parametrize("resolved", ["Qwen/Qwen-Image-2512", "Qwen/Qwen-Image-Edit"])
def test_helper_raises_when_the_name_resolves_elsewhere(
    monkeypatch: pytest.MonkeyPatch, resolved: str
) -> None:
    """Bug: the helper trusts from_name, so a mflux that resolves the name to another checkpoint loads it silently."""
    from types import SimpleNamespace

    import _qwen_config
    from mflux.models.common.config.model_config import ModelConfig

    fake = SimpleNamespace(model_name=resolved, aliases=["qwen-image"])
    monkeypatch.setattr(ModelConfig, "from_name", classmethod(lambda cls, *a, **k: fake))
    with pytest.raises(RuntimeError):
        _qwen_config.qwen_original_config()


@pytest.mark.mflux
def test_helper_raises_when_the_detector_alias_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bug: the helper checks only model_name, so a config without the "qwen-image" alias the detector needs passes."""
    from types import SimpleNamespace

    import _qwen_config
    from mflux.models.common.config.model_config import ModelConfig

    fake = SimpleNamespace(model_name="Qwen/Qwen-Image", aliases=["something-else"])
    monkeypatch.setattr(ModelConfig, "from_name", classmethod(lambda cls, *a, **k: fake))
    with pytest.raises(RuntimeError):
        _qwen_config.qwen_original_config()


@pytest.mark.mflux
def test_qwen_config_for_another_checkpoint_needs_only_the_qwen_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bug: --model 2512 is rejected by the original-only identity check, so users cannot calibrate their checkpoint;
    or a non-Qwen config is accepted."""
    from types import SimpleNamespace

    import _qwen_config
    from mflux.models.common.config.model_config import ModelConfig

    fake = SimpleNamespace(model_name="Qwen/Qwen-Image-2512", aliases=["qwen-image", "qwen-image-2512"])
    monkeypatch.setattr(ModelConfig, "from_name", classmethod(lambda cls, *a, **k: fake))
    assert _qwen_config.qwen_config("qwen-image-2512") is fake

    other = SimpleNamespace(model_name="black-forest-labs/FLUX.1-dev", aliases=["dev"])
    monkeypatch.setattr(ModelConfig, "from_name", classmethod(lambda cls, *a, **k: other))
    with pytest.raises(RuntimeError):
        _qwen_config.qwen_config("dev")
