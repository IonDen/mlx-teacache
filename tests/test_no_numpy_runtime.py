"""The runtime needs only mlx; numpy is a test-time dependency."""

import re
import subprocess
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

_PYPROJECT = Path(__file__).parent.parent / "pyproject.toml"

_BLOCK_NUMPY_THEN_IMPORT = (
    "import sys\n"
    "sys.modules['numpy'] = None\n"  # any `import numpy` now raises ImportError
    "import mlx_teacache, mlx_teacache.api, mlx_teacache.variants\n"
    "import mlx_teacache._kernel.gate, mlx_teacache._kernel.cache\n"
    "import mlx_teacache._kernel.stats, mlx_teacache._kernel.coefficients\n"
)


def test_core_imports_without_numpy():
    """Bug: a src module imports numpy at import time, so a mlx-only install fails on `import mlx_teacache`."""
    result = subprocess.run(
        [sys.executable, "-c", _BLOCK_NUMPY_THEN_IMPORT],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_numpy_is_not_a_runtime_dependency_but_is_a_test_dependency():
    """Bug: numpy is listed under [project].dependencies (forced on every install) or dropped from test-core."""
    data = tomllib.loads(_PYPROJECT.read_text())
    name = re.compile(r"^\s*numpy\b", re.IGNORECASE)
    assert not [d for d in data["project"]["dependencies"] if name.match(d)]
    assert [d for d in data["dependency-groups"]["test-core"] if isinstance(d, str) and name.match(d)]
