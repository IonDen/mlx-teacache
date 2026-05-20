# Audit: v0.5.0 Klein base-9B review findings

Source: reviewer output in current review task
Branch: `feature/v0.5.0-klein-base-9b`
Date: 2026-05-19
Scope: Material release-blocking issues from the reviewer pass, verified against the local branch.

## Findings

### 1. 9B benchmark defaults to the known memory-unsafe same-process three-way path

Severity: Medium-High
Refs: `scripts/bench_speedup.py:228`, `scripts/bench_speedup.py:229`, `scripts/bench_speedup.py:230`

Evidence: `three_way` defaults to true for both `klein-base-4b` and `klein-base-9b` when `--three-way` is not explicitly set. For the 9B variant this runs warmup, vanilla, no-gate, and gated generations in one process. The reviewer notes the PR already documents this exact 9B three-way setup as not memory-safe on 32 GB, while subprocess isolation is deferred to v0.5.1. Because `--three-way` is `store_true` style, the documented default path also has no inverse flag to turn the default off.

Impact: The default benchmark command can send users on a path the release already knows is unsafe for 9B on 32 GB machines. That makes the documented validation workflow unreliable and may produce OOM/failure reports for the new variant.

Fix: Keep `klein-base-9b` two-way by default until the subprocess-per-rep harness exists, or make three-way an explicit opt-in for 9B with a clear memory warning.

### 2. README still contradicts the base-9B support claim and license guidance

Severity: Medium-High
Refs: `README.md:101`, `README.md:107`, `README.md:236`, `README.md:246`, `README.md:250`

Evidence: The supported-model table now lists `flux2-klein-base-9b` as available, and footnote 3 says it ships in v0.5.0. Later, the README still states that `flux2-klein-base-9b` is "not yet supported. Planned for v0.5.0". The License obligations section only names `flux2-klein-9b`, even though the supported-model row and footnote point base-9B users to that section.

Impact: Users following the README get conflicting release guidance for the new variant and incomplete license guidance for the exact non-commercial base-9B model being added.

Fix: Remove or rewrite the stale "not yet supported" paragraph, and update License obligations so it explicitly covers both `flux2-klein-9b` and `flux2-klein-base-9b` with the correct upstream model-card links and obligations.
