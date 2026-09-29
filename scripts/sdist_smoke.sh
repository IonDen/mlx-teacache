#!/usr/bin/env bash
# Assert the sdist is lean, free of gitignored-artifact references, and has no dead relative links.
# Usage: scripts/sdist_smoke.sh [path/to/sdist.tar.gz]   (builds one into a temp dir when no path is given)
set -euo pipefail

workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT

if [ "$#" -ge 1 ]; then
  tarball="$1"
else
  uv build --sdist --out-dir "$workdir/dist"
  tarball="$(find "$workdir/dist" -maxdepth 1 -name '*.tar.gz' -print -quit)"
fi
mkdir "$workdir/x"
tar -xzf "$tarball" -C "$workdir/x"
root="$(find "$workdir/x" -mindepth 1 -maxdepth 1 -type d -print -quit)"
archive_listing="$(tar -tf "$tarball")"

if grep -Eq '^[^/]+/tests/' <<<"$archive_listing"; then
  echo "FAIL: sdist still ships tests/"
  exit 1
fi

if grep -Eq '^[^/]+/docs/papers/' <<<"$archive_listing"; then
  echo "FAIL: sdist ships docs/papers/ (published on GitHub, not package docs)"
  exit 1
fi

if grep -Eq '^[^/]+/\.github/' <<<"$archive_listing"; then
  echo "FAIL: sdist ships .github/ (CI files, not package content)"
  exit 1
fi

if grep -Eq '^[^/]+/scripts/_verify' <<<"$archive_listing"; then
  echo "FAIL: sdist ships a local scripts/_verify* file"
  exit 1
fi

if grep -rn "tests/_artifacts/" "$root/README.md" "$root/CHANGELOG.md" "$root/docs" 2>/dev/null; then
  echo "FAIL: shipped doc cites gitignored tests/_artifacts/"
  exit 1
fi

# Every relative Markdown link in the shipped README and docs must resolve inside the archive.
python3 - "$root" <<'PY'
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
link = re.compile(r"\]\(\s*<?([^)\s>]+)")
files = [root / "README.md", *sorted((root / "docs").rglob("*.md"))]
dead = []
for md in files:
    for lineno, line in enumerate(md.read_text(encoding="utf-8").splitlines(), 1):
        for target in link.findall(line):
            if target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            path = target.split("#", 1)[0]
            if path and not (md.parent / path).resolve().exists():
                dead.append(f"{md.relative_to(root)}:{lineno}: {target}")
if dead:
    print("FAIL: relative links whose target is not in the sdist:")
    print("\n".join(dead))
    sys.exit(1)
PY

echo "OK: sdist is lean, free of gitignored-artifact references, and every relative link resolves"
