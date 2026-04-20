"""Seed state/config_lineage.jsonl with the three launch entries.

Idempotent: if the file already exists and contains the pre-sprint or
6a entries, we re-use the existing file rather than overwriting. New
entries are appended by other tooling (deploy_6b.py) on actual deploys.

Schema for each JSONL line:
{
  "version_label": str,
  "status":         one of "historical" | "live" | "queued",
  "deployed_at":    ISO timestamp or null,
  "deployed_sha":   str or null,
  "key_changes":    list[str],
  "notes":          str (short, single line for dashboard rendering),
}

Writes atomically via tmp+rename.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List

HERE = Path(__file__).resolve().parent
ALGO = HERE.parent
REPO = ALGO.parent

STATE_DIRS = [REPO / "state", ALGO / "state"]


def _resolve_state_dir() -> Path:
    for d in STATE_DIRS:
        if d.exists():
            return d
    # default to algo/state and create
    STATE_DIRS[1].mkdir(parents=True, exist_ok=True)
    return STATE_DIRS[1]


LAUNCH_ENTRIES = [
    {
        "version_label": "pre-sprint",
        "status": "historical",
        "deployed_at": None,
        "deployed_sha": None,
        "key_changes": [
            "10x leverage",
            "no entry-side regime filters (F3 family off)",
            "no timeout_trail",
            "no E3 time-decayed SL",
        ],
        "notes": (
            "17 live trades Apr 12-19 2026: -22% compound, 29% WR, knife 35%. "
            "Trigger for sprint v1."
        ),
    },
    {
        "version_label": "6a observer",
        "status": "live",
        "deployed_at": "2026-04-19T22:26:17+00:00",
        "deployed_sha": "7c8d67f",
        "key_changes": [
            "audit writer enabled (state/daily_filter_audit.jsonl)",
            "shadow expectation + LiveDriftMonitor class deployed",
            "safety-valve auto-flip infrastructure ready",
            "sprint flags all default False (no behavior change yet)",
        ],
        "notes": (
            "Observer-only: bot still at pre-sprint 10x + no flags. "
            "Observation window 3-7 days before 6b promotion decision."
        ),
    },
    {
        "version_label": "6b PATH A",
        "status": "queued",
        "deployed_at": None,
        "deployed_sha": None,
        "key_changes": [
            "leverage 10x -> 5x",
            "timeout_trail_enabled True",
            "e3_time_decay_sl_enabled True",
            "f3c_enabled stays False (substrate disagreement per sprint v1.5 RED)",
        ],
        "notes": (
            "Queued for 2026-04-22 (Wed) ~06:00 Sofia, fallback 2026-04-23. "
            "Trigger: explicit user phrase PROMOTE PATH A + smoke-check gate pass."
        ),
    },
]


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def seed(force: bool = False) -> Path:
    state = _resolve_state_dir()
    lineage = state / "config_lineage.jsonl"

    if lineage.exists() and not force:
        # Idempotent: only seed if file is empty or missing the 3 labels.
        existing = [json.loads(ln) for ln in lineage.read_text().splitlines()
                    if ln.strip()]
        labels = {e.get("version_label") for e in existing}
        wanted = {"pre-sprint", "6a observer", "6b PATH A"}
        if wanted.issubset(labels):
            return lineage

    # Either file is missing, empty, or incomplete. Write fresh with the
    # three launch entries (plus any existing entries not already in
    # LAUNCH_ENTRIES, to avoid clobbering real deploy records).
    existing: List[dict] = []
    if lineage.exists():
        for ln in lineage.read_text().splitlines():
            ln = ln.strip()
            if not ln: continue
            try: existing.append(json.loads(ln))
            except json.JSONDecodeError: continue
    launch_labels = {e["version_label"] for e in LAUNCH_ENTRIES}
    extra = [e for e in existing if e.get("version_label") not in launch_labels]

    final = LAUNCH_ENTRIES + extra
    text = "\n".join(json.dumps(e) for e in final) + "\n"
    _atomic_write(lineage, text)
    return lineage


def append_deploy_entry(version_label: str, deployed_sha: str,
                         key_changes: List[str], notes: str) -> Path:
    """Called by deploy_6b.py after a successful deploy. Appends a new
    'live' entry and flips the previous 'live' entry to 'historical'."""
    state = _resolve_state_dir()
    lineage = state / "config_lineage.jsonl"
    if not lineage.exists():
        seed()

    entries = [json.loads(ln) for ln in lineage.read_text().splitlines()
               if ln.strip()]
    # Flip any existing 'live' or 'queued' entries with this label.
    now = datetime.now(timezone.utc).isoformat()
    for e in entries:
        if e.get("version_label") == version_label and e.get("status") in ("live", "queued"):
            e["status"] = "live"
            e["deployed_at"] = now
            e["deployed_sha"] = deployed_sha
    for e in entries:
        if e.get("status") == "live" and e.get("version_label") != version_label:
            e["status"] = "historical"

    # If label not present at all, append a fresh entry.
    if not any(e.get("version_label") == version_label for e in entries):
        entries.append({
            "version_label": version_label,
            "status": "live",
            "deployed_at": now,
            "deployed_sha": deployed_sha,
            "key_changes": key_changes,
            "notes": notes,
        })

    text = "\n".join(json.dumps(e) for e in entries) + "\n"
    _atomic_write(lineage, text)
    return lineage


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Seed config_lineage.jsonl")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing file with launch entries")
    args = ap.parse_args()
    p = seed(force=args.force)
    print(f"config_lineage.jsonl: {p}")
    for ln in p.read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)
            print(f"  [{r['status']:<10}] {r['version_label']:<14}  {r.get('notes','')[:80]}")
