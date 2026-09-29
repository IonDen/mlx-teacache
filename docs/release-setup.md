# Releasing mlx-teacache

## One-time PyPI Trusted Publishing setup

The repo owner does this once, before the first release.

1. Create the GitHub repo `IonDen/mlx-teacache` and push the initial code.
2. Go to https://pypi.org/manage/account/publishing/ → "Add a new publisher" → "Pending Publisher".
3. Fill in:
   - PyPI project name: `mlx-teacache`
   - Owner: `IonDen`
   - Repository name: `mlx-teacache`
   - Workflow filename: `release.yml`
   - Environment name: `pypi`
4. Create the `pypi` environment in GitHub Settings → Environments. No reviewers required for v0.x.

## Cutting a release

The version comes from the git tag, so there is no version string to bump.

1. Merge the release pull request into `main`.
2. Wait for the CI workflow's run on the merge commit to finish green. That run includes
   `test-mflux-latest`, which installs the newest mflux inside the supported range.
3. Tag the merge commit by its SHA and push the tag. The push starts the release workflow.
   Use the SHA rather than `origin/main`: if anything else merged in the meantime, the tip of `main`
   is no longer the commit whose CI you checked. The merged pull request shows the SHA ("merged
   commit …"), and so does either command below.

   ```bash
   git fetch origin
   gh pr view <PR number> --json mergeCommit --jq .mergeCommit.oid   # or: git log --oneline origin/main
   git tag vX.Y.Z <merge-commit-sha>
   git push origin vX.Y.Z
   ```

The release workflow's `verify` job runs before anything reaches PyPI. It checks that the tagged
commit is on `main`, then looks up the CI workflow's push run for that commit. While that run is
missing or still going, the job polls every 30 seconds for up to 30 minutes and fails when the time
runs out. A run that finishes with anything other than success fails the job at once. GitHub API
errors are retried within the same 30 minutes. The PyPI upload and the GitHub Release only happen
after `verify` and the build have passed.

### When a new mflux release turns CI red

`test-mflux-latest` goes red when mflux publishes a new version inside the supported range whose
forward functions this library has not yet checked against its own copies. A commit whose CI run
failed cannot be released, and re-running CI on it installs the same new mflux and fails again.
Fix it on `main` instead: run the real-weights parity tests against the new mflux, add its
fingerprints to the version table in `tests/test_mflux_forward_drift.py`, merge that change, let CI
pass on the new merge commit, and tag that commit.
