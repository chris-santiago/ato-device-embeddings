# log_entry.py
# /// script
# requires-python = ">=3.10"
# ///
"""
Structured INVESTIGATION_LOG.jsonl entry writer.
Enforces schema compliance, validates cat, auto-increments seq, auto-generates ts.
Usage: uv run log_entry.py --step 3 --cat subagent --action dispatch_critic --detail "..." [--artifact X] [--duration_s Y] [--meta '{"k":"v"}']
NEVER write log entries manually. Always use this script.
"""
import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path

ALLOWED_CATS = {'gate', 'write', 'read', 'subagent', 'exec', 'decision', 'debate', 'review', 'audit', 'workflow'}

parser = argparse.ArgumentParser()
parser.add_argument('--step', required=True)
parser.add_argument('--cat', required=True, choices=sorted(ALLOWED_CATS))
parser.add_argument('--action', required=True)
parser.add_argument('--detail', required=True)
parser.add_argument('--artifact', default=None)
parser.add_argument('--duration_s', type=float, default=None)
parser.add_argument('--meta', default='{}')
args = parser.parse_args()

try:
    meta = json.loads(args.meta)
except json.JSONDecodeError as e:
    print(f"ERROR: --meta must be valid JSON: {e}", file=sys.stderr)
    sys.exit(1)

log_file = Path('INVESTIGATION_LOG.jsonl')
seq = 1
if log_file.exists():
    lines = [l for l in log_file.read_text().splitlines() if l.strip()]
    if lines:
        try:
            seq = json.loads(lines[-1]).get('seq', 0) + 1
        except Exception:
            seq = len(lines) + 1

entry = {
    'ts': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    'step': args.step, 'seq': seq, 'cat': args.cat,
    'action': args.action, 'detail': args.detail,
    'artifact': args.artifact, 'duration_s': args.duration_s, 'meta': meta,
}
with open(log_file, 'a') as f:
    f.write(json.dumps(entry) + '\n')
print(f"[seq={seq}] {args.cat}/{args.action}: {args.detail}")
