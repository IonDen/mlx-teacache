# src/mlx_teacache/_kernel/decision_rule.py
"""The decision-rule revision and its history. Imports nothing, so reading the revision never loads MLX."""

# Revision of the decision rule: the mapping from a step's inputs to its compute or skip
# decision. Bump it on any change to gate_step decisions, to default-threshold resolution,
# to MAX_CONSECUTIVE_SKIPS, or to the skip-window defaults. It moves only in a minor or
# major release, gets a new DECISION_RULE_HISTORY entry naming that release, and the same
# change re-records tests/fixtures/decision_rule_golden.json.
DECISION_RULE_REVISION: int = 1
DECISION_RULE_HISTORY: dict[int, str] = {1: "0.16.0"}
