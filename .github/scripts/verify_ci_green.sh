#!/usr/bin/env bash
# Release gate: succeed only when the CI workflow's push run for $GITHUB_SHA has completed green.
# Polls while the run is absent or unfinished; fails at once on a finished run that is not a success.
# A failed `gh api` call (or an unreadable response) is logged as a warning and retried after the poll
# interval; the deadline still applies, and its message names the last error.
# Env: GITHUB_REPOSITORY, GITHUB_SHA, GH_TOKEN (for gh); optional VERIFY_POLL_SECONDS (default 30),
# VERIFY_TIMEOUT_SECONDS (default 1800).
set -euo pipefail

poll="${VERIFY_POLL_SECONDS:-30}"
timeout="${VERIFY_TIMEOUT_SECONDS:-1800}"
deadline=$((SECONDS + timeout))
endpoint="repos/${GITHUB_REPOSITORY}/actions/workflows/ci.yml/runs?head_sha=${GITHUB_SHA}&event=push"
err_file="$(mktemp)"
trap 'rm -f "$err_file"' EXIT

last="no response yet"
while true; do
  state=""
  if response="$(gh api "$endpoint" 2>"$err_file")"; then
    # Latest run for this commit; "absent" when CI has not started. A re-run keeps its run id, so
    # this reads the current status of that run's newest attempt.
    if ! state="$(jq -r '.workflow_runs | sort_by(.created_at) | last
                        | if . == null then "absent" else "\(.status) \(.conclusion // "none")" end' \
                    <<<"$response" 2>"$err_file")"; then
      state=""
    fi
  fi
  if [ -z "$state" ]; then
    error="$(tr '\n' ' ' <"$err_file" | sed 's/ *$//')"
    last="last error: ${error:-empty response from gh api}"
    echo "::warning::could not read the CI run for $GITHUB_SHA ($last)"
  fi
  case "$state" in
    "completed success")
      echo "CI run for $GITHUB_SHA completed successfully"
      exit 0
      ;;
    completed*)
      echo "::error::CI run for $GITHUB_SHA finished with conclusion '${state#completed }'; refusing to release"
      exit 1
      ;;
    absent)
      last="no CI push run for $GITHUB_SHA yet"
      ;;
    ?*)
      last="CI run for $GITHUB_SHA: $state"
      ;;
  esac
  if [ "$SECONDS" -ge "$deadline" ]; then
    echo "::error::timed out after ${timeout}s waiting for the CI run on $GITHUB_SHA ($last)"
    exit 1
  fi
  echo "$last; checking again in ${poll}s"
  sleep "$poll"
done
