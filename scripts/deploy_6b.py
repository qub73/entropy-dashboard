#!/usr/bin/env python3
"""
deploy_6b.py -- mechanical, idempotent, fail-fast 6b deploy for the
entropy_trader Pi service.

This script does NOT make the go/no-go decision. It executes the already-
approved PATH A deploy when explicitly invoked with the confirm phrase.
Everything else is pre-flight verification, atomic backup, and post-deploy
verify. Any abort condition leaves the Pi in its pre-deploy state.

Usage
-----
  python scripts/deploy_6b.py --dry-run
    Run every check, report what it WOULD do, exit without mutation.

  python scripts/deploy_6b.py --execute --confirm "PROMOTE PATH A"
    Full deploy. Confirm phrase is mandatory and exact-match.

  python scripts/deploy_6b.py --rollback --confirm "ROLLBACK TO PRE-6B"
    Restore files from the most recent pre_6b_backup_* directory and
    restart the service.

Design
------
- Repo-path-agnostic. Resolves paths relative to this file's location.
- Systemd service name resolved from $ENTROPY_SERVICE_NAME, default
  'entropy_trader.service'.
- Every file mutation is copy-then-rename (atomic on POSIX).
- On any exception during Phase 3, an automatic rollback is attempted.
- Exits with code 0 on success, non-zero on any abort.
"""
import argparse
import datetime as dt
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

# ------------ constants / paths ------------

REPO_ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = REPO_ROOT / "state"
LEGACY_STATE_DIR = REPO_ROOT / "algo" / "state"
CONFIG_FILE = REPO_ROOT / "algo" / "entropy_live_multi.py"
LEGACY_CONFIG_FILE = REPO_ROOT / "entropy_live_multi.py"

EXPECTED_COMMIT_SUBJECT = (
    "phase6b: PATH A promote -- timeout_trail + E3 + 5x; F3c stays off"
)
EXECUTE_CONFIRM = "PROMOTE PATH A"
ROLLBACK_CONFIRM = "ROLLBACK TO PRE-6B"
SERVICE_NAME_DEFAULT = "entropy_trader.service"

MAX_CLOCK_DRIFT_SEC = 30
BANNER_POLL_SEC = 120
AUDIT_POLL_SEC = 600
STARTUP_WAIT_SEC = 30


class AbortError(RuntimeError):
    """Raised when any pre-flight or verify check fails."""


# ------------ small helpers ------------

def _log(msg: str, level: str = "INFO") -> None:
    ts = dt.datetime.now(dt.timezone.utc).isoformat()
    print(f"[{ts}] [{level}] {msg}", flush=True)


def _resolve_state_dir() -> Path:
    """Pi layout uses state/ at repo root; dev layout uses algo/state/."""
    if STATE_DIR.exists():
        return STATE_DIR
    if LEGACY_STATE_DIR.exists():
        return LEGACY_STATE_DIR
    # Default to repo-root convention; create if missing.
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    return STATE_DIR


def _resolve_config_file() -> Path:
    """Pi has algo/entropy_live_multi.py; some layouts flatten it."""
    if CONFIG_FILE.exists():
        return CONFIG_FILE
    if LEGACY_CONFIG_FILE.exists():
        return LEGACY_CONFIG_FILE
    raise AbortError(
        f"entropy_live_multi.py not found at {CONFIG_FILE} or "
        f"{LEGACY_CONFIG_FILE}"
    )


def _service_name() -> str:
    return os.environ.get("ENTROPY_SERVICE_NAME", SERVICE_NAME_DEFAULT)


def _run(cmd: List[str], check: bool = True, timeout: int = 30,
         capture: bool = True) -> subprocess.CompletedProcess:
    _log(f"exec: {' '.join(cmd)}", "DEBUG")
    return subprocess.run(
        cmd, check=check, timeout=timeout,
        capture_output=capture, text=True,
    )


def _atomic_write(path: Path, data: str) -> None:
    """Write file atomically via tmp+rename."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(data)
    os.replace(tmp, path)


# ------------ phase 1: pre-flight ------------

def preflight(state_dir: Path, config_file: Path) -> dict:
    """Run every pre-flight check. Return a dict of verified facts.
    Raises AbortError on first failure."""
    facts: dict = {}

    # 1. git status clean
    try:
        res = _run(["git", "-C", str(REPO_ROOT), "status", "--porcelain"])
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        raise AbortError(f"git status failed: {e}")
    if res.stdout.strip():
        raise AbortError(
            f"working tree not clean:\n{res.stdout.strip()}"
        )
    facts["git_clean"] = True

    # 2. branch == main
    res = _run(["git", "-C", str(REPO_ROOT), "branch", "--show-current"])
    branch = res.stdout.strip()
    if branch != "main":
        raise AbortError(
            f"branch is '{branch}', expected 'main'. Deploy runs post-merge."
        )
    facts["branch"] = branch

    # 3. HEAD commit subject matches
    res = _run(["git", "-C", str(REPO_ROOT), "log", "-1", "--format=%s"])
    subject = res.stdout.strip()
    if subject != EXPECTED_COMMIT_SUBJECT:
        raise AbortError(
            f"HEAD subject is:\n  {subject!r}\nexpected:\n  "
            f"{EXPECTED_COMMIT_SUBJECT!r}"
        )
    facts["head_subject"] = subject
    res = _run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"])
    facts["head_sha"] = res.stdout.strip()

    # 4. position == None
    state_file = state_dir / "multi_trader_state.json"
    if not state_file.exists():
        raise AbortError(f"{state_file} not found")
    try:
        state = json.loads(state_file.read_text())
    except json.JSONDecodeError as e:
        raise AbortError(f"{state_file} is not valid JSON: {e}")
    pos = state.get("position")
    if pos is not None:
        raise AbortError(
            f"position is not None: {pos!r}. Cannot deploy mid-trade."
        )
    facts["position"] = None

    # 5. shadow_expectation.json exists + PATH A metadata
    shadow_file = state_dir / "shadow_expectation.json"
    if not shadow_file.exists():
        raise AbortError(f"{shadow_file} not found")
    try:
        shadow = json.loads(shadow_file.read_text())
    except json.JSONDecodeError as e:
        raise AbortError(f"{shadow_file} is not valid JSON: {e}")
    cell_name = shadow.get("cell_name", "")
    if "no F3c" not in cell_name:
        raise AbortError(
            f"shadow cell_name {cell_name!r} missing 'no F3c' marker"
        )
    if shadow.get("f3c_enabled") is not False:
        raise AbortError(
            f"shadow f3c_enabled is {shadow.get('f3c_enabled')!r}, "
            f"expected False"
        )
    data_source = shadow.get("data_source", "")
    if "Pi" not in data_source:
        raise AbortError(
            f"shadow data_source {data_source!r} missing 'Pi' marker"
        )
    facts["shadow_cell_name"] = cell_name
    facts["shadow_f3c_enabled"] = shadow.get("f3c_enabled")
    facts["shadow_data_source"] = data_source

    # 6. safety_valve.json MUST NOT exist
    valve = state_dir / "safety_valve.json"
    if valve.exists():
        raise AbortError(
            f"{valve} exists. Valve tripped. Investigate before deploy."
        )
    facts["safety_valve_absent"] = True

    # 7. clock drift < 30s (compare local clock against a trusted reference;
    #    we use `date -u` vs a NTP query if available, else skip the strict
    #    check and emit a warning -- the Pi should be running chronyd).
    skew_sec = _measure_clock_skew()
    facts["clock_skew_sec"] = skew_sec
    if skew_sec is not None and abs(skew_sec) > MAX_CLOCK_DRIFT_SEC:
        raise AbortError(
            f"clock drift {skew_sec:.1f}s exceeds +/-{MAX_CLOCK_DRIFT_SEC}s"
        )

    return facts


def _measure_clock_skew() -> Optional[float]:
    """Return clock skew seconds vs an NTP reference if available; else None
    (don't abort if no reference is reachable -- the Pi should run chronyd,
    and we don't want network failure to block deploy). A chronyc query
    is tried first; falls back to None on any error."""
    try:
        res = _run(["chronyc", "tracking"], check=False, timeout=5)
        if res.returncode != 0:
            return None
        for line in res.stdout.splitlines():
            if "System time" in line:
                parts = line.split()
                for i, p in enumerate(parts):
                    try:
                        v = float(p)
                        if i + 1 < len(parts) and parts[i+1].startswith("second"):
                            return v if "fast" in line.lower() else -v
                    except ValueError:
                        continue
        return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


# ------------ phase 2: backup ------------

def create_backup(state_dir: Path, config_file: Path) -> Path:
    """Copy critical state + config snapshot + git log into a timestamped
    backup dir. Record path in state/last_deploy_backup.txt."""
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = state_dir / f"pre_6b_backup_{ts}"
    backup.mkdir(parents=True, exist_ok=False)

    for name in ("shadow_expectation.json",
                 "shadow_expectation.v1.json",
                 "shadow_expectation.v1_with_f3c.json",
                 "live_drift_monitor.json",
                 "multi_trader_state.json",
                 "daily_filter_audit.jsonl",
                 "engine_state.json"):
        src = state_dir / name
        if src.exists():
            shutil.copy2(src, backup / name)

    res = _run(["git", "-C", str(REPO_ROOT), "log", "--oneline", "-20"])
    (backup / "git_head.txt").write_text(res.stdout)
    res = _run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"])
    (backup / "git_sha.txt").write_text(res.stdout)

    # Config snapshot: grep relevant flag lines from entropy_live_multi.py
    if config_file.exists():
        lines = config_file.read_text(errors="replace").splitlines()[:300]
        keep = [ln for ln in lines if any(
            k in ln for k in ("f3c_enabled", "timeout_trail_enabled",
                               "e3_time_decay_sl_enabled", "\"leverage\"")
        )]
        (backup / "config_snapshot.txt").write_text("\n".join(keep) + "\n")

    marker = state_dir / "last_deploy_backup.txt"
    _atomic_write(marker, str(backup))
    _log(f"backup created: {backup}")
    return backup


# ------------ phase 3: execute ------------

def execute_deploy(state_dir: Path) -> None:
    """Phase 3. Invoked only with --execute + correct confirm phrase, AND
    only after preflight + backup have succeeded."""
    # 10. rename v1 -> v1_with_f3c (idempotent)
    v1 = state_dir / "shadow_expectation.v1.json"
    v1_f3c = state_dir / "shadow_expectation.v1_with_f3c.json"
    if v1.exists() and not v1_f3c.exists():
        os.replace(v1, v1_f3c)
        _log(f"renamed {v1.name} -> {v1_f3c.name}")
    elif v1_f3c.exists() and v1.exists():
        # Both exist: keep the explicit _with_f3c version, drop the legacy.
        v1.unlink()
        _log(f"{v1_f3c.name} already present; removed legacy {v1.name}")
    else:
        _log(f"v1 rename: skipping (v1={v1.exists()} "
             f"v1_with_f3c={v1_f3c.exists()})")

    # 11. clear live_drift_monitor.json
    drift = state_dir / "live_drift_monitor.json"
    if drift.exists():
        drift.unlink()
        _log(f"cleared {drift.name}")
    else:
        _log(f"{drift.name} already absent")

    # 12. restart service
    svc = _service_name()
    _log(f"restarting systemd service: {svc}")
    _run(["systemctl", "restart", svc], timeout=STARTUP_WAIT_SEC)
    # poll is-active
    deadline = time.time() + STARTUP_WAIT_SEC
    active = False
    while time.time() < deadline:
        res = _run(["systemctl", "is-active", svc], check=False, timeout=5)
        if res.stdout.strip() == "active":
            active = True
            break
        time.sleep(1)
    if not active:
        raise AbortError(
            f"service {svc} did not reach 'active' within {STARTUP_WAIT_SEC}s"
        )
    _log(f"service active: {svc}")


def verify_startup_banner(logs_dir: Path) -> dict:
    """Poll the entropy_multi.log for up to BANNER_POLL_SEC looking for the
    startup banner that confirms PATH A flag state. Return the observed
    flag line(s) as a dict."""
    log_file = logs_dir / "entropy_multi.log"
    deadline = time.time() + BANNER_POLL_SEC
    want = {
        "f3c_enabled=False": False,
        "timeout_trail_enabled=True": False,
        "e3_time_decay_sl_enabled=True": False,
        "leverage=5": False,
    }
    observed_line: Optional[str] = None

    while time.time() < deadline:
        if log_file.exists():
            try:
                tail = log_file.read_text(errors="replace").splitlines()[-400:]
            except Exception:
                tail = []
            for ln in reversed(tail):
                if all(k.split("=")[0] in ln for k in want):
                    observed_line = ln
                    for k in want:
                        if k in ln:
                            want[k] = True
                    break
            if observed_line and all(want.values()):
                break
        time.sleep(2)

    missing = [k for k, ok in want.items() if not ok]
    if missing or observed_line is None:
        raise AbortError(
            f"startup banner mismatch. Expected all of {list(want)}, "
            f"missing: {missing}. Observed line: {observed_line!r}"
        )
    return {"banner_line": observed_line, "flags_ok": True}


def verify_first_audit_line(state_dir: Path) -> dict:
    """Poll daily_filter_audit.jsonl for up to AUDIT_POLL_SEC looking for
    the first new line after restart. If none (market dormant), that's OK
    -- we just log it. If any, confirm f3c_enabled=False."""
    audit_file = state_dir / "daily_filter_audit.jsonl"
    if not audit_file.exists():
        _log("audit file not present yet; market may be dormant. Skipping.")
        return {"first_audit_line": None, "market_dormant_ok": True}
    baseline_lines = audit_file.read_text(errors="replace").splitlines()
    deadline = time.time() + AUDIT_POLL_SEC

    while time.time() < deadline:
        now_lines = audit_file.read_text(errors="replace").splitlines()
        if len(now_lines) > len(baseline_lines):
            newline = now_lines[-1]
            try:
                rec = json.loads(newline)
            except json.JSONDecodeError:
                raise AbortError(
                    f"first new audit line is not valid JSON: {newline!r}"
                )
            if rec.get("f3c_enabled") is not False:
                raise AbortError(
                    f"first new audit line has f3c_enabled="
                    f"{rec.get('f3c_enabled')!r}, expected False"
                )
            return {"first_audit_line": newline, "market_dormant_ok": False}
        time.sleep(5)

    _log(f"no new audit line within {AUDIT_POLL_SEC}s. Market likely "
         "dormant. Not an abort.")
    return {"first_audit_line": None, "market_dormant_ok": True}


# ------------ phase 4: post-deploy artifact ------------

def write_deploy_record(state_dir: Path, facts: dict,
                         banner_result: dict, audit_result: dict,
                         backup_path: Path) -> Path:
    deploy_record = {
        "deployed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_sha": facts.get("head_sha"),
        "head_subject": facts.get("head_subject"),
        "shadow_cell_name": facts.get("shadow_cell_name"),
        "config_snapshot_line": banner_result.get("banner_line"),
        "first_audit_line": audit_result.get("first_audit_line"),
        "backup_path": str(backup_path),
        "service_name": _service_name(),
    }
    out = state_dir / "deploy_6b_complete.json"
    _atomic_write(out, json.dumps(deploy_record, indent=2))
    _log(f"deploy record: {out}")
    return out


# ------------ rollback ------------

def rollback(state_dir: Path) -> dict:
    """Restore state files from the most recent backup and restart service."""
    marker = state_dir / "last_deploy_backup.txt"
    if not marker.exists():
        raise AbortError(f"{marker} missing; no backup to roll back to")
    backup = Path(marker.read_text().strip())
    if not backup.exists():
        raise AbortError(f"backup dir {backup} not found")

    restored: List[str] = []
    for child in backup.iterdir():
        if not child.is_file():
            continue
        if child.name in ("git_head.txt", "git_sha.txt",
                          "config_snapshot.txt"):
            continue
        dst = state_dir / child.name
        shutil.copy2(child, dst)
        restored.append(child.name)

    svc = _service_name()
    _log(f"restarting service after rollback: {svc}")
    _run(["systemctl", "restart", svc], timeout=STARTUP_WAIT_SEC)
    deadline = time.time() + STARTUP_WAIT_SEC
    active = False
    while time.time() < deadline:
        r = _run(["systemctl", "is-active", svc], check=False, timeout=5)
        if r.stdout.strip() == "active":
            active = True
            break
        time.sleep(1)
    result = {
        "rolled_back_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "backup_path": str(backup),
        "files_restored": restored,
        "service_active": active,
    }
    rec = state_dir / "rollback_complete.json"
    _atomic_write(rec, json.dumps(result, indent=2))
    return result


# ------------ main ------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="6b mechanical deploy")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true",
                      help="pre-flight only; no mutation")
    mode.add_argument("--execute", action="store_true",
                      help="full deploy; requires --confirm")
    mode.add_argument("--rollback", action="store_true",
                      help="restore from last backup; requires --confirm")
    ap.add_argument("--confirm", default="",
                    help="exact confirmation phrase")
    args = ap.parse_args(argv)

    try:
        state_dir = _resolve_state_dir()
        config_file = _resolve_config_file()
    except AbortError as e:
        _log(f"ABORT: {e}", "ERROR"); return 2

    if args.execute and args.confirm != EXECUTE_CONFIRM:
        _log(f"ABORT: --execute requires --confirm {EXECUTE_CONFIRM!r}", "ERROR")
        return 2
    if args.rollback and args.confirm != ROLLBACK_CONFIRM:
        _log(f"ABORT: --rollback requires --confirm {ROLLBACK_CONFIRM!r}", "ERROR")
        return 2

    if args.rollback:
        try:
            out = rollback(state_dir)
            _log(f"ROLLBACK OK: restored {len(out['files_restored'])} files "
                 f"from {out['backup_path']}; service_active="
                 f"{out['service_active']}")
            return 0 if out["service_active"] else 1
        except AbortError as e:
            _log(f"ROLLBACK ABORT: {e}", "ERROR"); return 3

    # Dry-run or Execute -- both run preflight + backup first.
    try:
        facts = preflight(state_dir, config_file)
    except AbortError as e:
        _log(f"PREFLIGHT ABORT: {e}", "ERROR"); return 4

    _log("preflight OK:")
    for k, v in facts.items(): _log(f"  {k}: {v}")

    if args.dry_run:
        _log("DRY-RUN: would create backup, rename v1 shadow, clear drift, "
             "restart service, verify banner, verify first audit line, "
             "write deploy_6b_complete.json.")
        return 0

    # Execute path
    try:
        backup = create_backup(state_dir, config_file)
    except Exception as e:
        _log(f"BACKUP ABORT: {e}", "ERROR"); return 5

    try:
        execute_deploy(state_dir)
        banner = verify_startup_banner(REPO_ROOT / "algo" / "logs"
                                        if (REPO_ROOT / "algo" / "logs").exists()
                                        else REPO_ROOT / "logs")
        audit = verify_first_audit_line(state_dir)
    except Exception as e:
        _log(f"EXECUTE ABORT: {e} -- attempting rollback", "ERROR")
        try:
            rb = rollback(state_dir)
            _log(f"auto-rollback ok: restored {len(rb['files_restored'])} "
                 f"files; service_active={rb['service_active']}", "ERROR")
        except Exception as e2:
            _log(f"!!! AUTO-ROLLBACK FAILED: {e2} !!!", "ERROR")
        return 6

    record = write_deploy_record(state_dir, facts, banner, audit, backup)
    _log("=" * 60)
    _log("6b DEPLOY SUCCESS")
    _log(f"  git HEAD: {facts['head_sha']}")
    _log(f"  shadow:   {facts['shadow_cell_name']}")
    _log(f"  banner:   {banner['banner_line']}")
    _log(f"  backup:   {backup}")
    _log(f"  record:   {record}")
    _log("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
