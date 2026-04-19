#!/usr/bin/env python3
"""
journal_query.py — Read and query .project-log/journal.jsonl

Usage:
    python3 .project-log/journal_query.py --latest-checkpoint
    python3 .project-log/journal_query.py --status
    python3 .project-log/journal_query.py --list issue [--since 7d]
    python3 .project-log/journal_query.py --list decision
    python3 .project-log/journal_query.py --unresolved-issues
    python3 .project-log/journal_query.py --entry <id-prefix>
    python3 .project-log/journal_query.py --recent 10
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from collections import defaultdict


# --- Helpers ---

def get_journal_path():
    try:
        root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        return os.path.join(root, ".project-log", "journal.jsonl")
    except subprocess.CalledProcessError:
        sys.exit("ERROR: Not inside a git repository.")


def load_entries(journal_path):
    if not os.path.exists(journal_path):
        sys.exit("ERROR: journal.jsonl not found. Run /journal-init to set up.")
    entries = []
    with open(journal_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"WARNING: Skipping malformed line {i}: {e}", file=sys.stderr)
    return entries


def parse_since(since_str):
    """Parse '7d', '24h', '30m' into a UTC datetime cutoff."""
    if since_str is None:
        return None
    unit = since_str[-1]
    try:
        value = int(since_str[:-1])
    except ValueError:
        sys.exit(f"ERROR: Invalid --since value '{since_str}'. Use format like 7d, 24h, 60m.")
    if unit == "d":
        delta = timedelta(days=value)
    elif unit == "h":
        delta = timedelta(hours=value)
    elif unit == "m":
        delta = timedelta(minutes=value)
    else:
        sys.exit(f"ERROR: Unknown time unit '{unit}'. Use d, h, or m.")
    return datetime.now(timezone.utc) - delta


def fmt_ts(ts_str):
    """Format ISO8601 timestamp to readable local-ish string."""
    try:
        dt = datetime.fromisoformat(ts_str)
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ts_str


def short_id(entry):
    return entry.get("id", "")[:8]


def divider(char="─", width=60):
    return char * width


def is_resolved(issue_id, resolved_prefixes):
    """Return True if any stored linked_issue_id is a prefix of the full issue UUID."""
    return any(issue_id.startswith(p) for p in resolved_prefixes if p)


# --- Query functions ---

def cmd_latest_checkpoint(entries):
    checkpoints = [e for e in entries if e.get("type") == "checkpoint"]
    if not checkpoints:
        print("No checkpoint found in journal.")
        return

    cp = checkpoints[-1]
    print(divider("═"))
    print(f"  LATEST CHECKPOINT  [{fmt_ts(cp['timestamp'])}]  id:{short_id(cp)}")
    print(divider("═"))

    fields = [
        ("In progress",         "in_progress"),
        ("Pending decisions",   "pending_decisions"),
        ("Recently completed",  "recently_completed"),
        ("Key context",         "key_context"),
        ("Git state",           "git_state"),
    ]
    for label, key in fields:
        val = cp.get(key)
        if val:
            print(f"\n{label}:")
            print(f"  {val}")

    threads = cp.get("open_threads", [])
    if threads:
        print("\nOpen threads:")
        for t in threads:
            print(f"  • {t}")

    print(divider())


def cmd_status(entries):
    if not entries:
        print("Journal is empty.")
        return

    session_id = os.environ.get("CLAUDE_SESSION_ID")
    session_entries = [e for e in entries if e.get("session_id") == session_id] if session_id else []

    # Counts by type for current session
    type_counts = defaultdict(int)
    for e in session_entries:
        type_counts[e["type"]] += 1

    # Unresolved issues
    resolved_prefixes = {e.get("linked_issue_id") for e in entries if e.get("type") == "resolution"}
    unresolved = [e for e in entries if e.get("type") == "issue" and not is_resolved(e["id"], resolved_prefixes)]

    # Latest checkpoint
    checkpoints = [e for e in entries if e.get("type") == "checkpoint"]
    last_cp = checkpoints[-1] if checkpoints else None

    # Last 5 git entries
    git_entries = [e for e in entries if e.get("type") == "git"][-5:]

    print(divider("═"))
    print("  JOURNAL STATUS")
    print(divider("═"))

    if last_cp:
        threads = last_cp.get("open_threads", [])
        print(f"\nLast checkpoint: {fmt_ts(last_cp['timestamp'])}  ({len(threads)} open thread{'s' if len(threads) != 1 else ''})")
        for t in threads:
            print(f"  • {t}")
    else:
        print("\nNo checkpoint yet.")

    print(f"\nThis session ({session_id[:8] if session_id else 'unknown'}):")
    if type_counts:
        for t, c in sorted(type_counts.items()):
            print(f"  {t:<16} {c}")
    else:
        print("  No entries yet.")

    print(f"\nUnresolved issues: {len(unresolved)}")
    for e in unresolved[-5:]:
        sev = e.get("severity", "?")
        print(f"  [{sev:<8}] {short_id(e)}  {e.get('description', '')[:60]}")

    if git_entries:
        print(f"\nRecent commits logged:")
        for e in reversed(git_entries):
            print(f"  {e.get('commit_hash','?')[:7]}  {e.get('message','')[:50]}")

    total = len(entries)
    print(f"\nTotal entries: {total}")
    print(divider())


def cmd_list(entries, entry_type, since_str):
    cutoff = parse_since(since_str)
    filtered = [e for e in entries if e.get("type") == entry_type]
    if cutoff:
        filtered = [e for e in filtered if datetime.fromisoformat(e["timestamp"]) >= cutoff]

    if not filtered:
        qualifier = f" in last {since_str}" if since_str else ""
        print(f"No {entry_type} entries found{qualifier}.")
        return

    print(divider("═"))
    print(f"  {entry_type.upper()} ENTRIES ({len(filtered)})")
    print(divider("═"))

    for e in filtered:
        print(f"\n[{short_id(e)}]  {fmt_ts(e['timestamp'])}")
        # Print type-specific summary fields
        for key in ["description", "severity", "verdict", "what_failed", "root_cause",
                    "rationale", "in_progress", "message", "context", "approach",
                    "implications", "source", "expected_result", "result",
                    "contributing_factors", "lessons", "applies_to"]:
            val = e.get(key)
            if val:
                print(f"  {key}: {val}")
        # List fields
        for key in ["tags", "open_threads", "key_decisions", "files_changed"]:
            val = e.get(key)
            if val:
                print(f"  {key}: {', '.join(val)}")
        # Linked IDs
        for key in ["linked_id", "linked_issue_id", "linked_hypothesis_id"]:
            val = e.get(key)
            if val:
                print(f"  {key}: {val[:8]}")

    print(divider())


def cmd_unresolved_issues(entries):
    resolved_prefixes = {e.get("linked_issue_id") for e in entries if e.get("type") == "resolution"}
    unresolved = [e for e in entries if e.get("type") == "issue" and not is_resolved(e["id"], resolved_prefixes)]

    if not unresolved:
        print("No unresolved issues.")
        return

    print(divider("═"))
    print(f"  UNRESOLVED ISSUES ({len(unresolved)})")
    print(divider("═"))
    for e in unresolved:
        sev = e.get("severity", "?")
        print(f"\n[{short_id(e)}]  {fmt_ts(e['timestamp'])}  severity:{sev}")
        print(f"  {e.get('description', '')}")
        ctx = e.get("context")
        if ctx:
            print(f"  context: {ctx}")
        tags = e.get("tags", [])
        if tags:
            print(f"  tags: {', '.join(tags)}")
    print(divider())


def cmd_recent(entries, n):
    if not entries:
        print("Journal is empty.")
        return

    recent = entries[-n:]
    print(divider("═"))
    print(f"  RECENT ENTRIES ({len(recent)} of {len(entries)} total)")
    print(divider("═"))

    for e in recent:
        etype = e.get("type", "?")
        print(f"\n[{short_id(e)}]  {fmt_ts(e['timestamp'])}  type:{etype}")
        for key in ["description", "severity", "verdict", "what_failed", "root_cause",
                    "rationale", "in_progress", "message", "context", "approach",
                    "implications", "source", "expected_result", "result",
                    "contributing_factors", "lessons", "applies_to"]:
            val = e.get(key)
            if val:
                print(f"  {key}: {val}")
        for key in ["tags", "open_threads", "key_decisions", "files_changed"]:
            val = e.get(key)
            if val:
                print(f"  {key}: {', '.join(val)}")
        for key in ["linked_id", "linked_issue_id", "linked_hypothesis_id"]:
            val = e.get(key)
            if val:
                print(f"  {key}: {val[:8]}")

    print(divider())


def cmd_entry(entries, id_prefix):
    matches = [e for e in entries if e.get("id", "").startswith(id_prefix)]
    if not matches:
        print(f"No entry found with id prefix '{id_prefix}'.")
        return
    if len(matches) > 1:
        print(f"Ambiguous prefix '{id_prefix}' matches {len(matches)} entries. Use more characters.")
        return
    e = matches[0]
    print(divider("═"))
    print(f"  ENTRY  [{e.get('type','?').upper()}]  {e.get('id')}")
    print(divider("═"))
    print(json.dumps(e, indent=2, ensure_ascii=False))
    print(divider())


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Query the project journal.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--latest-checkpoint", action="store_true")
    group.add_argument("--status",            action="store_true")
    group.add_argument("--list",              metavar="TYPE")
    group.add_argument("--unresolved-issues", action="store_true")
    group.add_argument("--entry",             metavar="ID_PREFIX")
    group.add_argument("--recent",            metavar="N", type=int,
                       help="Show the N most recent entries across all types")
    parser.add_argument("--since", default=None, help="Filter by recency: 7d, 24h, 60m")

    args = parser.parse_args()
    journal_path = get_journal_path()
    entries = load_entries(journal_path)

    if args.latest_checkpoint:
        cmd_latest_checkpoint(entries)
    elif args.status:
        cmd_status(entries)
    elif args.list:
        cmd_list(entries, args.list, args.since)
    elif args.unresolved_issues:
        cmd_unresolved_issues(entries)
    elif args.entry:
        cmd_entry(entries, args.entry)
    elif args.recent:
        cmd_recent(entries, args.recent)


if __name__ == "__main__":
    main()
