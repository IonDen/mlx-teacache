#!/usr/bin/env bash
# Assert the sdist is lean and free of gitignored-artifact references.
set -euo pipefail

workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT

uv build --sdist --out-dir "$workdir/dist"
tarball="$(find "$workdir/dist" -maxdepth 1 -name '*.tar.gz' -print -quit)"
tar -xzf "$tarball" -C "$workdir"
root="$(find "$workdir" -maxdepth 1 -type d -name 'mlx_teacache-*' -print -quit)"
archive_listing="$(tar -tf "$tarball")"

if grep -Eq '^[^/]+/tests/' <<<"$archive_listing"; then
  echo "FAIL: sdist still ships tests/"
  exit 1
fi

if grep -rn "tests/_artifacts/" "$root/README.md" "$root/CHANGELOG.md" "$root/docs" 2>/dev/null; then
  echo "FAIL: shipped doc cites gitignored tests/_artifacts/"
  exit 1
fi

echo "OK: sdist is lean and free of gitignored-artifact references"
