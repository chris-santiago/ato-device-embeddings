#!/usr/bin/env python3
"""
journal_log.py — Append a typed entry to .project-log/journal.jsonl

Usage:
    python3 .project-log/journal_log.py --type issue --description "..." --severity high --tags "a,b" --context "..."
    python3 .project-log/journal_log.py --type decision --description "..." --rationale "..." --alternatives "..."
    ... (see REQUIRED_FIELDS for all types)
"""

import argparse
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone


# --- Schema ---

REQUIRED_FIELDS = {
    "issue":       ["description", "severity"],
    "resolution":  ["description"],
    "decision":    ["description", "rationale"],
    "discovery":   ["description"],
    "hypothesis":  ["description"],
    "experiment":  ["description", "verdict"],
    "post_mortem": ["description", "what_failed", "root_cause"],
    "lesson":      ["description"],
    "summary":     ["description"],
    "checkpoint":  ["in_progress"],
    "git":         ["commit_hash", "message", "branch"],
}

OPTIONAL_FIELDS = {
    "issue":       ["tags", "context"],
    "resolution":  ["linked_issue_id", "approach"],
    "decision":    ["alternatives"],
    "discovery":   ["implications", "source"],
    "hypothesis":  ["expected_result", "metric"],
    "experiment":  ["linked_hypothesis_id", "metric", "result"],
    "post_mortem": ["contributing_factors", "lessons", "linked_issue_id"],
    "lesson":      ["context", "applies_to", "linked_id"],
    "summary":     ["key_decisions", "open_threads"],
    "checkpoint":  ["pending_decisions", "recently_completed", "key_context", "git_state", "open_threads"],
    "git":         ["files_changed", "diff_summary"],
}

VALID_SEVERITIES = {"low", "moderate", "high", "critical"}
VALID_VERDICTS   = {"confirmed", "refuted", "inconclusive"}

LIST_FIELDS = {"tags", "files_changed", "key_decisions", "open_threads"}


def get_repo_name():
    try:
        root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        return os.path.basename(root)
    except subprocess.CalledProcessError:
        return "unknown"


def get_journal_path():
    try:
        root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        return os.path.join(root, ".project-log", "journal.jsonl")
    except subprocess.CalledProcessError:
        sys.exit("ERROR: Not inside a git repository.")


def build_entry(args):
    entry_type = args.type

    if entry_type not in REQUIRED_FIELDS:
        sys.exit(f"ERROR: Unknown type '{entry_type}'. Valid types: {', '.join(sorted(REQUIRED_FIELDS))}")

    # Envelope
    entry = {
        "id":         str(uuid.uuid4()),
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "type":       entry_type,
        "project":    get_repo_name(),
        "session_id": os.environ.get("CLAUDE_SESSION_ID"),
    }

    # Collect all known fields for this type
    all_fields = REQUIRED_FIELDS[entry_type] + OPTIONAL_FIELDS.get(entry_type, [])

    for field in all_fields:
        raw = getattr(args, field.replace("-", "_"), None)
        if raw is None:
            continue
        if field in LIST_FIELDS:
            # Accept comma-separated string or already a list
            if isinstance(raw, str):
                entry[field] = [v.strip() for v in raw.split(",") if v.strip()]
            else:
                entry[field] = raw
        else:
            entry[field] = raw

    # Validate required fields
    missing = [f for f in REQUIRED_FIELDS[entry_type] if f not in entry]
    if missing:
        sys.exit(f"ERROR: Missing required fields for type '{entry_type}': {', '.join(missing)}")

    # Validate enums
    if "severity" in entry and entry["severity"] not in VALID_SEVERITIES:
        sys.exit(f"ERROR: Invalid severity '{entry['severity']}'. Must be one of: {', '.join(sorted(VALID_SEVERITIES))}")
    if "verdict" in entry and entry["verdict"] not in VALID_VERDICTS:
        sys.exit(f"ERROR: Invalid verdict '{entry['verdict']}'. Must be one of: {', '.join(sorted(VALID_VERDICTS))}")

    return entry


def append_entry(journal_path, entry):
    os.makedirs(os.path.dirname(journal_path), exist_ok=True)
    with open(journal_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Append a typed entry to the project journal.")
    parser.add_argument("--type", required=True, help="Entry type")

    # All possible fields across all types
    all_possible = set()
    for fields in list(REQUIRED_FIELDS.values()) + list(OPTIONAL_FIELDS.values()):
        all_possible.update(fields)

    for field in sorted(all_possible):
        # Register with hyphen flag (--in-progress) and underscore dest (in_progress)
        flag = field.replace("_", "-")
        parser.add_argument(f"--{flag}", dest=field, default=None)

    args = parser.parse_args()

    journal_path = get_journal_path()
    entry = build_entry(args)
    append_entry(journal_path, entry)

    print(f"Logged [{entry['type']}] {entry['id'][:8]} at {entry['timestamp'][:19]}Z")


if __name__ == "__main__":
    main()
