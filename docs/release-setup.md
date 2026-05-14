# One-time PyPI Trusted Publishing setup

Mirrors the procedure in `mlx-taef/docs/release-setup.md`. Done once by the
repo owner before the first release.

## Steps

1. Create GitHub repo `IonDen/mlx-teacache`, push initial code.
2. Go to https://pypi.org/manage/account/publishing/ → "Add a new publisher" → "Pending Publisher".
3. Fill in:
   - PyPI project name: `mlx-teacache`
   - Owner: `IonDen`
   - Repository name: `mlx-teacache`
   - Workflow filename: `release.yml`
   - Environment name: `pypi`
4. Create the `pypi` environment in GitHub Settings → Environments. No reviewers required for v0.x.
5. Push a tag `v0.1.0` to trigger the release workflow (Task 35).
