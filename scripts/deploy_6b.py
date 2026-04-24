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
  python scripts/deploy_6b.py --health-check
    Local + Pi-reachability sanity checks; no state reads or mutations.

  python scripts/deploy_6b.py --dry-run [--mode=dev]
    Run every pre-flight check, report what it WOULD do, exit without
    mutation.

  python scripts/deploy_6b.py --execute --confirm "PROMOTE PATH A"
    Full deploy. Confirm phrase is mandatory and exact-match.

  python scripts/deploy_6b.py --rollback --confirm "ROLLBACK TO PRE-6B"
    Restore Pi state files from the most recent pre_6b_backup_* directory
    and restart the service.

Design
------
- --mode=dev (default): script runs on the dev workstation. LOCAL git
  pre-flight (clean tree, on main, HEAD subject matches PATH A) plus
  REMOTE state/service pre-flight via ssh against the Pi. Phase 2 deploy
  is rsync (config + shadow + lineage init) + ssh (rename, drift clear,
  systemctl restart, banner verify, audit verify). Backup, rollback,
  deploy-record write are all ssh-based against PI_STATE_DIR. The dev
  repo is the source of truth for what gets deployed; the Pi has no .git.
- --mode=pi: reserved, not implemented (would run the script on the Pi
  itself; currently raises NotImplementedError).
- Systemd service name resolved from $ENTROPY_SERVICE_NAME, default
  'entropy-trader.service'. Pre-flight cross-validates that the named
  unit actually is our bot (right WorkingDirectory, right ExecStart,
  right running-process cmdline) before any restart -- this guards
  against the 'env var points at a sibling trading system' class of
  error (see notes/pi_infrastructure_audit_2026_04_22.md).
- Pi connectivity: $PI_HOST_TARGET (default 'user@raspberry') resolves
  the ssh target. Default is the Tailscale MagicDNS hostname for
  remote access. Set to 'user@192.168.1.15' (or similar) when running
  from the home network without Tailscale, or to 'user@100.67.10.19'
  to use the raw Tailscale IP if MagicDNS is unavailable. ssh uses
  ControlMaster for connection multiplexing across the many calls in
  one deploy.
- Every Pi-side file mutation that overwrites is rsync (atomic on
  rename); state writes use 'cat > path' via ssh stdin (single fsync).
- On any exception during Phase 3 (execute path), an automatic
  rollback is attempted.
- Exits with code 0 on success, non-zero on any abort.
"""
import argparse
import datetime as dt
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ------------ constants / paths ------------

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_STATE_DIR_PRIMARY = REPO_ROOT / "state"
LOCAL_STATE_DIR_LEGACY = REPO_ROOT / "algo" / "state"
LOCAL_CONFIG_FILE_PRIMARY = REPO_ROOT / "algo" / "entropy_live_multi.py"
LOCAL_CONFIG_FILE_LEGACY = REPO_ROOT / "entropy_live_multi.py"

# Backwards-compat aliases (some tests still reference these).
STATE_DIR = LOCAL_STATE_DIR_PRIMARY
LEGACY_STATE_DIR = LOCAL_STATE_DIR_LEGACY
CONFIG_FILE = LOCAL_CONFIG_FILE_PRIMARY
LEGACY_CONFIG_FILE = LOCAL_CONFIG_FILE_LEGACY

# Pi-side absolute paths (script targets the Pi's filesystem via ssh).
PI_HOME = "/home/user/entropy_trader"
PI_STATE_DIR = f"{PI_HOME}/state"
PI_CONFIG_FILE = f"{PI_HOME}/entropy_live_multi.py"  # top-level on Pi
PI_LOG_FILE = f"{PI_HOME}/logs/entropy_multi.log"
PI_AUDIT_FILE = f"{PI_STATE_DIR}/daily_filter_audit.jsonl"
PI_LINEAGE_INIT = f"{PI_HOME}/algo/dashboard/config_lineage_init.py"
PI_LINEAGE_FILE = f"{PI_STATE_DIR}/config_lineage.jsonl"

EXPECTED_COMMIT_SUBJECT = (
    "phase6b: PATH A promote -- timeout_trail + E3 + 5x; F3c stays off"
)
EXECUTE_CONFIRM = "PROMOTE PATH A"
ROLLBACK_CONFIRM = "ROLLBACK TO PRE-6B"
SERVICE_NAME_DEFAULT = "entropy-trader.service"

# Pi ssh target. Default is Tailscale MagicDNS for remote access; can
# be overridden via $PI_HOST_TARGET.
PI_HOST_TARGET_DEFAULT = "user@raspberry"
PI_REACHABLE_TIMEOUT_SEC = 8

# Service-target invariants. The named unit must match all three or
# preflight aborts.
EXPECTED_WORKING_DIR = PI_HOME
EXPECTED_EXEC_MARKER = "entropy_live_multi.py"

MAX_CLOCK_DRIFT_SEC = 30
BANNER_POLL_SEC = 120
AUDIT_POLL_SEC = 600
STARTUP_WAIT_SEC = 30

# ssh options used for every remote call. ControlMaster multiplexes a
# single TCP connection across many ssh invocations within one deploy.
SSH_OPTS = [
    "-o", "ConnectTimeout=10",
    "-o", "BatchMode=yes",
    "-o", "ControlMaster=auto",
    "-o", "ControlPath=" + str(Path.home() / ".ssh" / "cm-%r@%h-%p"),
    "-o", "ControlPersist=60s",
]

# Banner verification: substrings that must appear together in the
# Pi log within BANNER_POLL_SEC after restart. Comes from the
# entropy_live_multi.py startup logger.info("Config: ...") block.
BANNER_REQUIRED_SUBSTRINGS = [
    '"leverage": 5',
    '"f3c_enabled": false',
    '"timeout_trail_enabled": true',
    '"e3_time_decay_sl_enabled": true',
    '"pst_max_wait_bars": 40',
]


class AbortError(RuntimeError):
    """Raised when any pre-flight or verify check fails."""


# ------------ small helpers ------------

def _log(msg: str, level: str = "INFO") -> None:
    ts = dt.datetime.now(dt.timezone.utc).isoformat()
    print(f"[{ts}] [{level}] {msg}", flush=True)


def _local_state_dir() -> Path:
    """Resolve the local (dev) state dir. Pi layout uses state/ at repo
    root; dev layout uses algo/state/."""
    if LOCAL_STATE_DIR_PRIMARY.exists():
        return LOCAL_STATE_DIR_PRIMARY
    if LOCAL_STATE_DIR_LEGACY.exists():
        return LOCAL_STATE_DIR_LEGACY
    LOCAL_STATE_DIR_PRIMARY.mkdir(parents=True, exist_ok=True)
    return LOCAL_STATE_DIR_PRIMARY


# Backwards-compat alias for tests that still call the old name.
_resolve_state_dir = _local_state_dir


def _local_config_file() -> Path:
    """Resolve the local entropy_live_multi.py to be rsync'd to Pi."""
    if LOCAL_CONFIG_FILE_PRIMARY.exists():
        return LOCAL_CONFIG_FILE_PRIMARY
    if LOCAL_CONFIG_FILE_LEGACY.exists():
        return LOCAL_CONFIG_FILE_LEGACY
    raise AbortError(
        f"entropy_live_multi.py not found at "
        f"{LOCAL_CONFIG_FILE_PRIMARY} or {LOCAL_CONFIG_FILE_LEGACY}"
    )


_resolve_config_file = _local_config_file


def _service_name() -> str:
    return os.environ.get("ENTROPY_SERVICE_NAME", SERVICE_NAME_DEFAULT)


def _pi_host_target() -> str:
    return os.environ.get("PI_HOST_TARGET", PI_HOST_TARGET_DEFAULT)


def _run(cmd: List[str], check: bool = True, timeout: int = 30,
         capture: bool = True) -> subprocess.CompletedProcess:
    _log(f"exec: {' '.join(cmd)}", "DEBUG")
    return subprocess.run(
        cmd, check=check, timeout=timeout,
        capture_output=capture, text=True,
    )


def _atomic_write(path: Path, data: str) -> None:
    """Write a local file atomically via tmp+rename."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(data)
    os.replace(tmp, path)


# ------------ ssh + rsync helpers ------------

def _ssh_cmd(remote: List[str], host: Optional[str] = None) -> List[str]:
    """Build the full ssh command list for a given remote command."""
    h = host or _pi_host_target()
    return ["ssh"] + SSH_OPTS + [h, "--"] + remote


def _ssh_run(remote: List[str], host: Optional[str] = None,
             check: bool = False, timeout: int = 30
             ) -> subprocess.CompletedProcess:
    """Run a command on the Pi via ssh. Returns CompletedProcess.
    By design: check defaults to False because most callers want to
    inspect rc + stderr to make a decision rather than throw."""
    return _run(_ssh_cmd(remote, host), check=check, timeout=timeout)


def _ssh_write_file(remote_path: str, content: str,
                    host: Optional[str] = None, timeout: int = 30) -> None:
    """Write content to a Pi file via ssh stdin (single round-trip)."""
    h = host or _pi_host_target()
    cmd = ["ssh"] + SSH_OPTS + [h, "--", "bash", "-c",
                                  f"cat > {remote_path}"]
    _log(f"exec: {' '.join(cmd)} (with stdin {len(content)} bytes)", "DEBUG")
    res = subprocess.run(
        cmd, input=content, text=True,
        capture_output=True, timeout=timeout,
    )
    if res.returncode != 0:
        raise AbortError(
            f"ssh write failed for {remote_path} on {h}: "
            f"rc={res.returncode}, stderr={(res.stderr or '').strip()!r}"
        )


def _pi_file_exists(remote_path: str) -> bool:
    return _ssh_run(["test", "-e", remote_path], timeout=10).returncode == 0


def _pi_read_file(remote_path: str) -> str:
    res = _ssh_run(["cat", remote_path], check=False, timeout=15)
    if res.returncode != 0:
        raise AbortError(
            f"could not read {remote_path} on Pi: "
            f"{(res.stderr or '').strip()!r}"
        )
    return res.stdout


def _rsync_to_pi(local_path: Path, remote_path: str,
                 host: Optional[str] = None) -> None:
    """rsync a local file to the Pi. Raises AbortError on failure."""
    h = host or _pi_host_target()
    ssh_str = "ssh " + " ".join(SSH_OPTS)
    cmd = [
        "rsync", "-az", "--checksum",
        "-e", ssh_str,
        str(local_path), f"{h}:{remote_path}",
    ]
    _log(f"exec: {' '.join(cmd)}", "DEBUG")
    res = subprocess.run(
        cmd, capture_output=True, text=True, timeout=60,
    )
    if res.returncode != 0:
        raise AbortError(
            f"rsync failed for {local_path} -> {h}:{remote_path}: "
            f"rc={res.returncode}, stderr={(res.stderr or '').strip()!r}"
        )


# ------------ service-target verification ------------

def _extract_unit_key(unit_text: str, key: str) -> Optional[str]:
    """Return the first `Key=Value` pair from the [Service] section, or
    None. Handles the standard `systemctl cat` output format where the
    unit file contents are preceded by a comment line with the path."""
    in_service = False
    for raw in unit_text.splitlines():
        line = raw.strip()
        if line.startswith("["):
            in_service = (line == "[Service]")
            continue
        if in_service and line.startswith(f"{key}="):
            return line.split("=", 1)[1].strip()
    return None


def _read_proc_cmdline(pid: int) -> str:
    """Read /proc/<pid>/cmdline on the Pi via ssh, with NULs translated
    to spaces. Broken out so tests can monkeypatch."""
    res = _ssh_run(
        ["bash", "-c", f"tr '\\0' ' ' < /proc/{pid}/cmdline"],
        check=False, timeout=10,
    )
    if res.returncode != 0:
        raise OSError(
            f"could not read /proc/{pid}/cmdline on Pi: "
            f"{(res.stderr or '').strip()!r}"
        )
    return res.stdout.strip()


def verify_service_targets_expected_bot(
        service_name: str) -> Tuple[bool, Optional[str]]:
    """Layered defense: confirm the named service is our entropy_trader
    bot and not a sibling trading system. Four checks, each naming both
    expected and observed values in the abort message:
        P1. systemctl is-active returns 'active'
        P2. unit WorkingDirectory matches EXPECTED_WORKING_DIR
        P3. unit ExecStart contains EXPECTED_EXEC_MARKER
        P4. running MainPID's /proc/<pid>/cmdline contains EXPECTED_EXEC_MARKER
    Returns (ok, error_msg). On ok=True, error_msg is None.

    All systemctl/proc reads are performed on the Pi via ssh in
    --mode=dev. Rationale: notes/pi_infrastructure_audit_2026_04_22.md.
    """
    # P1 -- active
    res = _ssh_run(["systemctl", "is-active", service_name],
                    check=False, timeout=10)
    actual_state = res.stdout.strip()
    if actual_state != "active":
        return False, (
            f"P1 FAILED: service {service_name!r} is not active. "
            f"systemctl is-active returned {actual_state!r} (expected "
            f"'active'). The env var may point at a stale/unrelated "
            f"unit -- refusing to proceed."
        )

    # P2, P3 -- parse unit file
    res = _ssh_run(["systemctl", "cat", service_name],
                    check=False, timeout=10)
    if res.returncode != 0:
        return False, (
            f"systemctl cat {service_name!r} failed (rc={res.returncode}): "
            f"{(res.stderr or '').strip()!r}"
        )
    unit_text = res.stdout

    wd = _extract_unit_key(unit_text, "WorkingDirectory")
    if wd != EXPECTED_WORKING_DIR:
        return False, (
            f"P2 FAILED: service {service_name!r} WorkingDirectory is "
            f"{wd!r}, expected {EXPECTED_WORKING_DIR!r}. This unit does "
            f"not look like our entropy_trader bot -- refusing to proceed."
        )

    execstart = _extract_unit_key(unit_text, "ExecStart") or ""
    if EXPECTED_EXEC_MARKER not in execstart:
        return False, (
            f"P3 FAILED: service {service_name!r} ExecStart is "
            f"{execstart!r}, expected to contain {EXPECTED_EXEC_MARKER!r}. "
            f"Refusing to proceed."
        )

    # P4 -- running process cmdline
    res = _ssh_run(["systemctl", "show", "-p", "MainPID", "--value",
                     service_name], check=False, timeout=10)
    pid_str = res.stdout.strip()
    try:
        pid = int(pid_str)
    except ValueError:
        return False, (
            f"P4 FAILED: systemctl show -p MainPID returned non-integer "
            f"{pid_str!r} for {service_name!r}."
        )
    if pid <= 0:
        return False, (
            f"P4 FAILED: service {service_name!r} has MainPID={pid} (no "
            f"running process). systemctl is-active said 'active' but "
            f"show disagrees -- state is inconsistent."
        )

    try:
        cmdline = _read_proc_cmdline(pid)
    except Exception as e:
        return False, (
            f"P4 FAILED: could not read /proc/{pid}/cmdline: "
            f"{type(e).__name__}: {e}"
        )
    if EXPECTED_EXEC_MARKER not in cmdline:
        return False, (
            f"P4 FAILED: running process PID {pid} cmdline is "
            f"{cmdline!r}, expected to contain {EXPECTED_EXEC_MARKER!r}. "
            f"The unit file matches but the live process does not -- "
            f"someone may have swapped the service. Refusing to proceed."
        )

    return True, None


# ------------ health-check (no state reads, no mutations) ------------

def health_check() -> Tuple[int, List[str], List[str]]:
    """Answer 'could this script be invoked right now?' Returns
    (exit_code, ok_messages, issue_messages). Exit 0 if all green."""
    ok: List[str] = []
    issues: List[str] = []

    # 1. env var
    svc = os.environ.get("ENTROPY_SERVICE_NAME", "")
    if svc:
        ok.append(f"ENTROPY_SERVICE_NAME={svc}")
    else:
        issues.append(
            f"ENTROPY_SERVICE_NAME not set (script would default to "
            f"{SERVICE_NAME_DEFAULT!r} which may not match the Pi's "
            f"actual unit name)"
        )

    # 2. git in PATH
    if shutil.which("git"):
        ok.append("git available in PATH")
    else:
        issues.append("git not found in PATH")

    # 3. python version
    if sys.version_info >= (3, 8):
        ok.append(f"python {sys.version.split()[0]} (>= 3.8)")
    else:
        issues.append(
            f"python {sys.version.split()[0]} is below required 3.8"
        )

    # 4. self-import (syntax)
    try:
        spec = importlib.util.spec_from_file_location(
            "deploy_6b_self", Path(__file__).resolve())
        if spec is None or spec.loader is None:
            issues.append("could not build import spec for self")
        else:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            ok.append("self-import clean (no syntax errors)")
    except Exception as e:
        issues.append(f"self-import raised {type(e).__name__}: {e}")

    # 5. state dir resolves and is writable
    try:
        state_dir = _local_state_dir()
        probe = state_dir / ".healthcheck_probe"
        probe.write_text("ok")
        probe.unlink()
        ok.append(f"state dir writable: {state_dir}")
    except Exception as e:
        issues.append(f"state dir not usable: {e}")

    # 6. tests file exists
    tests = REPO_ROOT / "tests" / "test_deploy_6b.py"
    if tests.exists():
        ok.append(f"tests present: {tests.relative_to(REPO_ROOT)}")
    else:
        issues.append(f"tests missing: {tests.relative_to(REPO_ROOT)}")

    # 7. Pi reachable via ssh (Tailscale or LAN). Default Tailscale.
    pi_host = _pi_host_target()
    try:
        res = _run(
            ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes",
             pi_host, "echo pi_reachable"],
            check=False, timeout=PI_REACHABLE_TIMEOUT_SEC,
        )
        if res.returncode == 0 and "pi_reachable" in (res.stdout or ""):
            ok.append(f"Pi reachable at {pi_host}")
        else:
            issues.append(
                f"PI unreachable at {pi_host} -- verify network/Tailscale "
                f"or set PI_HOST_TARGET env var to override "
                f"(rc={res.returncode}, stderr="
                f"{(res.stderr or '').strip()[:200]!r})"
            )
    except subprocess.TimeoutExpired:
        issues.append(
            f"PI unreachable at {pi_host} (ssh timeout > "
            f"{PI_REACHABLE_TIMEOUT_SEC}s) -- verify network/Tailscale "
            f"or set PI_HOST_TARGET env var to override"
        )
    except FileNotFoundError:
        issues.append(
            "ssh not found in PATH; cannot verify Pi reachability"
        )

    return (0 if not issues else 1), ok, issues


# ------------ phase 1: pre-flight ------------

def _measure_clock_skew_pi() -> Optional[float]:
    """Return clock skew (Pi - local) in seconds via a quick ssh date
    comparison. Returns None on failure (don't abort if Pi is briefly
    unreachable; the actual ssh-based phase ops will fail loudly later)."""
    try:
        local_before = time.time()
        res = _ssh_run(["date", "+%s.%N"], check=False, timeout=10)
        local_after = time.time()
        if res.returncode != 0:
            return None
        pi_t = float(res.stdout.strip())
        local_mid = (local_before + local_after) / 2.0
        return pi_t - local_mid
    except Exception:
        return None


# Backwards-compat alias for tests that monkeypatch the old name.
_measure_clock_skew = _measure_clock_skew_pi


def preflight() -> dict:
    """Run every pre-flight check. Return a dict of verified facts.
    Raises AbortError on first failure.

    LOCAL: git status, branch, HEAD subject, HEAD SHA, local shadow
    REMOTE: Pi position, valve absence, clock skew, P1..P4 service verify
    """
    facts: dict = {}

    # 1. git status clean (LOCAL)
    try:
        res = _run(["git", "-C", str(REPO_ROOT), "status", "--porcelain"])
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        raise AbortError(f"git status failed: {e}")
    if res.stdout.strip():
        raise AbortError(
            f"working tree not clean:\n{res.stdout.strip()}"
        )
    facts["git_clean"] = True

    # 2. branch == main (LOCAL)
    res = _run(["git", "-C", str(REPO_ROOT), "branch", "--show-current"])
    branch = res.stdout.strip()
    if branch != "main":
        raise AbortError(
            f"branch is '{branch}', expected 'main'. Deploy runs post-merge."
        )
    facts["branch"] = branch

    # 3. HEAD commit subject + SHA (LOCAL)
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

    # 4. Pi position == None (REMOTE via ssh)
    state_text = _pi_read_file(f"{PI_STATE_DIR}/multi_trader_state.json")
    try:
        state = json.loads(state_text)
    except json.JSONDecodeError as e:
        raise AbortError(
            f"Pi multi_trader_state.json is not valid JSON: {e}"
        )
    pos = state.get("position")
    if pos is not None:
        raise AbortError(
            f"Pi position is not None: {pos!r}. Cannot deploy mid-trade."
        )
    facts["position"] = None

    # 5. local shadow_expectation.json (the one we will rsync) -- PATH A metadata
    local_shadow = _local_state_dir() / "shadow_expectation.json"
    if not local_shadow.exists():
        raise AbortError(f"local shadow {local_shadow} not found")
    try:
        shadow = json.loads(local_shadow.read_text())
    except json.JSONDecodeError as e:
        raise AbortError(f"{local_shadow} is not valid JSON: {e}")
    cell_name = shadow.get("cell_name", "")
    if "no F3c" not in cell_name:
        raise AbortError(
            f"local shadow cell_name {cell_name!r} missing 'no F3c' marker"
        )
    if shadow.get("f3c_enabled") is not False:
        raise AbortError(
            f"local shadow f3c_enabled is {shadow.get('f3c_enabled')!r}, "
            f"expected False"
        )
    data_source = shadow.get("data_source", "")
    if "Pi" not in data_source:
        raise AbortError(
            f"local shadow data_source {data_source!r} missing 'Pi' marker"
        )
    facts["shadow_cell_name"] = cell_name
    facts["shadow_f3c_enabled"] = shadow.get("f3c_enabled")
    facts["shadow_data_source"] = data_source

    # 6. Pi safety_valve.json MUST NOT exist (REMOTE)
    if _pi_file_exists(f"{PI_STATE_DIR}/safety_valve.json"):
        raise AbortError(
            f"Pi {PI_STATE_DIR}/safety_valve.json exists. "
            f"Valve tripped. Investigate before deploy."
        )
    facts["safety_valve_absent"] = True

    # 7. clock drift Pi vs local
    skew_sec = _measure_clock_skew_pi()
    facts["clock_skew_sec"] = skew_sec
    if skew_sec is not None and abs(skew_sec) > MAX_CLOCK_DRIFT_SEC:
        raise AbortError(
            f"clock drift Pi-local is {skew_sec:.1f}s, exceeds "
            f"+/-{MAX_CLOCK_DRIFT_SEC}s"
        )

    # 8. service target verification (P1..P4) -- REMOTE
    svc = _service_name()
    ok, err = verify_service_targets_expected_bot(svc)
    if not ok:
        raise AbortError(err or f"service verification failed for {svc!r}")
    facts["service_verified"] = svc
    facts["service_working_dir"] = EXPECTED_WORKING_DIR
    facts["service_exec_marker"] = EXPECTED_EXEC_MARKER

    return facts


# ------------ phase 2: backup (Pi-side via ssh) ------------

def create_backup(facts: dict) -> str:
    """Create a timestamped pre_6b_backup_<iso>/ directory on the Pi,
    copy the critical state files into it, and write the marker.
    Returns the Pi-absolute backup path (string)."""
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_remote = f"{PI_STATE_DIR}/pre_6b_backup_{ts}"

    res = _ssh_run(["mkdir", "-p", backup_remote],
                    check=False, timeout=10)
    if res.returncode != 0:
        raise AbortError(
            f"mkdir {backup_remote} failed: "
            f"{(res.stderr or '').strip()!r}"
        )

    files_to_back_up = [
        "shadow_expectation.json",
        "shadow_expectation.v1.json",
        "shadow_expectation.v1_with_f3c.json",
        "live_drift_monitor.json",
        "multi_trader_state.json",
        "daily_filter_audit.jsonl",
        "engine_state.json",
        "config_lineage.jsonl",
    ]
    for name in files_to_back_up:
        # cp -p preserves mtime; suppress error if file absent.
        _ssh_run(
            ["bash", "-c",
             f"[ -e {PI_STATE_DIR}/{name} ] && "
             f"cp -p {PI_STATE_DIR}/{name} {backup_remote}/{name} || true"],
            check=False, timeout=15,
        )

    # git_head + git_sha snapshots written to Pi via ssh stdin.
    git_log_res = _run(
        ["git", "-C", str(REPO_ROOT), "log", "--oneline", "-20"])
    _ssh_write_file(f"{backup_remote}/git_head.txt", git_log_res.stdout)
    _ssh_write_file(f"{backup_remote}/git_sha.txt",
                     facts.get("head_sha", "") + "\n")

    # Update the marker so rollback can find this backup.
    _ssh_write_file(f"{PI_STATE_DIR}/last_deploy_backup.txt",
                     backup_remote + "\n")

    _log(f"backup created on Pi: {backup_remote}")
    return backup_remote


# ------------ phase 3: execute (rsync + ssh) ------------

def execute_deploy(facts: dict) -> None:
    """Phase 3. Push new config + shadow + lineage init via rsync,
    rename v1 shadow, clear drift monitor, restart service. Invoked
    only with --execute + correct confirm phrase, after preflight +
    backup have succeeded."""
    # 9. rsync the new entropy_live_multi.py to Pi (top-level location).
    local_config = _local_config_file()
    _rsync_to_pi(local_config, PI_CONFIG_FILE)
    _log(f"rsync'd {local_config} -> Pi {PI_CONFIG_FILE}")

    # 10. rsync shadow_expectation.json
    local_shadow = _local_state_dir() / "shadow_expectation.json"
    _rsync_to_pi(local_shadow, f"{PI_STATE_DIR}/shadow_expectation.json")
    _log("rsync'd shadow_expectation.json")

    # 11. rsync config_lineage_init.py to Pi (mkdir parent if needed).
    lineage_init = REPO_ROOT / "algo" / "dashboard" / "config_lineage_init.py"
    if lineage_init.exists():
        _ssh_run(["mkdir", "-p", f"{PI_HOME}/algo/dashboard"],
                 check=False, timeout=10)
        _rsync_to_pi(lineage_init, PI_LINEAGE_INIT)
        _log("rsync'd config_lineage_init.py")

    # 12. rename v1 -> v1_with_f3c (idempotent)
    v1 = f"{PI_STATE_DIR}/shadow_expectation.v1.json"
    v1f3c = f"{PI_STATE_DIR}/shadow_expectation.v1_with_f3c.json"
    _ssh_run(
        ["bash", "-c",
         f"if [ -e {v1} ] && [ ! -e {v1f3c} ]; then "
         f"  mv {v1} {v1f3c}; "
         f"elif [ -e {v1} ] && [ -e {v1f3c} ]; then "
         f"  rm {v1}; "
         f"fi"],
        check=False, timeout=10,
    )
    _log("v1 shadow rename: idempotent op done")

    # 13. clear live_drift_monitor.json
    _ssh_run(["rm", "-f", f"{PI_STATE_DIR}/live_drift_monitor.json"],
             check=False, timeout=10)
    _log("cleared live_drift_monitor.json")

    # 14. restart systemd service (passwordless sudo on Pi confirmed).
    svc = _service_name()
    _log(f"restarting Pi systemd service: {svc}")
    res = _ssh_run(["sudo", "-n", "systemctl", "restart", svc],
                    check=False, timeout=STARTUP_WAIT_SEC)
    if res.returncode != 0:
        raise AbortError(
            f"sudo systemctl restart {svc} failed: "
            f"{(res.stderr or '').strip()!r}"
        )
    # 15. poll is-active
    deadline = time.time() + STARTUP_WAIT_SEC
    active = False
    while time.time() < deadline:
        r = _ssh_run(["systemctl", "is-active", svc],
                     check=False, timeout=5)
        if r.stdout.strip() == "active":
            active = True
            break
        time.sleep(1)
    if not active:
        raise AbortError(
            f"service {svc} did not reach 'active' within "
            f"{STARTUP_WAIT_SEC}s"
        )
    _log(f"service active: {svc}")


def verify_startup_banner() -> dict:
    """Tail the Pi log via ssh, look for PATH A flag values in the
    startup banner. The bot logs the config in:
        Config: <SHARED_CONFIG indent=2 json>
        <pair>: <PAIRS[pair] json>
    around a 'Starting multi-pair entropy trader' marker. We search the
    last N lines for each required substring."""
    deadline = time.time() + BANNER_POLL_SEC
    matched: Dict[str, bool] = {k: False for k in BANNER_REQUIRED_SUBSTRINGS}
    last_blob = ""
    marker = "Starting multi-pair entropy trader"

    while time.time() < deadline:
        res = _ssh_run(["tail", "-n", "300", PI_LOG_FILE],
                        check=False, timeout=10)
        if res.returncode == 0:
            blob = res.stdout
            last_blob = blob
            # Scope to the most-recent startup if marker present.
            if marker in blob:
                idx = blob.rfind(marker)
                # Pull a window around the marker (the banner lines come
                # just before it; allow 4000 chars back / 1000 forward).
                start = max(0, idx - 4000)
                end = min(len(blob), idx + 1000)
                window = blob[start:end]
            else:
                window = blob
            for k in matched:
                if k in window:
                    matched[k] = True
            if all(matched.values()):
                break
        time.sleep(2)

    missing = [k for k, v in matched.items() if not v]
    if missing:
        raise AbortError(
            f"startup banner mismatch. Expected substrings: "
            f"{BANNER_REQUIRED_SUBSTRINGS}. Missing: {missing}. "
            f"Last log (tail 500 chars):\n{last_blob[-500:]!r}"
        )
    return {
        "banner_line": "all required substrings matched in startup window",
        "matched_substrings": list(matched.keys()),
        "flags_ok": True,
    }


def verify_first_audit_line() -> dict:
    """Poll the Pi audit file via ssh for a new line within
    AUDIT_POLL_SEC. If none (market dormant), that's OK -- we just log
    it. If any, confirm f3c_enabled=False."""
    res = _ssh_run(
        ["bash", "-c",
         f"[ -e {PI_AUDIT_FILE} ] && wc -l < {PI_AUDIT_FILE} || echo 0"],
        check=False, timeout=10,
    )
    try:
        baseline = int(res.stdout.strip())
    except ValueError:
        baseline = 0

    deadline = time.time() + AUDIT_POLL_SEC
    while time.time() < deadline:
        res = _ssh_run(
            ["bash", "-c",
             f"[ -e {PI_AUDIT_FILE} ] && wc -l < {PI_AUDIT_FILE} || echo 0"],
            check=False, timeout=10,
        )
        try:
            now = int(res.stdout.strip())
        except ValueError:
            now = 0
        if now > baseline:
            tail = _ssh_run(["tail", "-n", "1", PI_AUDIT_FILE],
                             check=False, timeout=10)
            newline = tail.stdout.strip()
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
            return {
                "first_audit_line": newline,
                "market_dormant_ok": False,
            }
        time.sleep(5)

    _log(f"no new audit line within {AUDIT_POLL_SEC}s. Market likely "
         "dormant. Not an abort.")
    return {"first_audit_line": None, "market_dormant_ok": True}


# ------------ phase 4: post-deploy artifact ------------

def write_deploy_record(facts: dict, banner_result: dict,
                          audit_result: dict, backup_remote: str) -> str:
    deploy_record = {
        "deployed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_sha": facts.get("head_sha"),
        "head_subject": facts.get("head_subject"),
        "shadow_cell_name": facts.get("shadow_cell_name"),
        "matched_substrings": banner_result.get("matched_substrings"),
        "first_audit_line": audit_result.get("first_audit_line"),
        "backup_path_pi": backup_remote,
        "service_name": _service_name(),
    }
    out_pi = f"{PI_STATE_DIR}/deploy_6b_complete.json"
    _ssh_write_file(out_pi, json.dumps(deploy_record, indent=2) + "\n")
    _log(f"deploy record written to Pi: {out_pi}")
    return out_pi


# ------------ lineage append ------------

PATH_A_LINEAGE_ENTRY = {
    "version_label": "6b PATH A",
    "key_changes": [
        "leverage 10x -> 5x",
        "timeout_trail_enabled True",
        "e3_time_decay_sl_enabled True",
        "pst_max_wait_bars 30 -> 40 (Candidate 3 fold-in)",
        "f3c_enabled stays False (substrate disagreement per sprint v1.5 RED)",
    ],
    "notes": (
        "Live post PATH A promote. F3c stays off pending sprint v2 "
        "native-L2 confirmation."
    ),
}


def append_lineage_on_success(facts: dict) -> Tuple[bool, str]:
    """Run lineage append LOCALLY (dev's algo/dashboard/config_lineage_init.py)
    then rsync the updated state/config_lineage.jsonl to the Pi.
    Non-fatal: any failure writes a LOUD error file and returns False."""
    try:
        init_path = REPO_ROOT / "algo" / "dashboard" / "config_lineage_init.py"
        if not init_path.exists():
            raise FileNotFoundError(f"{init_path} not found")
        spec = importlib.util.spec_from_file_location(
            "config_lineage_init", init_path)
        if spec is None or spec.loader is None:
            raise ImportError("cannot build import spec for lineage init")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        mod.append_deploy_entry(
            version_label=PATH_A_LINEAGE_ENTRY["version_label"],
            deployed_sha=facts.get("head_sha", "")[:7],
            key_changes=PATH_A_LINEAGE_ENTRY["key_changes"],
            notes=PATH_A_LINEAGE_ENTRY["notes"],
        )
        # rsync the updated lineage to Pi (best-effort).
        local_lineage = _local_state_dir() / "config_lineage.jsonl"
        if local_lineage.exists():
            try:
                _rsync_to_pi(local_lineage, PI_LINEAGE_FILE)
            except Exception as rerr:
                _log(f"lineage rsync to Pi failed (non-fatal): {rerr}",
                     "ERROR")
        return True, "lineage appended locally + rsync'd to Pi"
    except Exception as e:
        fail = {
            "failed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "exception": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
            "intended_entry": {
                **PATH_A_LINEAGE_ENTRY,
                "deployed_sha": facts.get("head_sha", "")[:7],
                "status": "live",
            },
            "recovery": (
                "Run: python -c 'import sys; sys.path.insert(0, "
                "\"algo/dashboard\"); import config_lineage_init as L; "
                "L.append_deploy_entry(**{intended fields})'"
            ),
        }
        try:
            out = _local_state_dir() / "lineage_append_failed.json"
            _atomic_write(out, json.dumps(fail, indent=2))
        except Exception:
            pass
        return False, f"{type(e).__name__}: {e}"


# ------------ rollback (Pi-side via ssh) ------------

def rollback() -> dict:
    """Restore Pi state files from the most recent Pi-side backup and
    restart the service."""
    marker_path = f"{PI_STATE_DIR}/last_deploy_backup.txt"
    if not _pi_file_exists(marker_path):
        raise AbortError(
            f"{marker_path} missing on Pi; no backup to roll back to"
        )
    backup_remote = _pi_read_file(marker_path).strip()
    if not backup_remote or not _pi_file_exists(backup_remote):
        raise AbortError(
            f"backup dir {backup_remote!r} not found on Pi"
        )

    # List files in the backup
    res = _ssh_run(["bash", "-c", f"ls -1 {backup_remote}"],
                    check=False, timeout=10)
    files = res.stdout.strip().splitlines()
    skip = {"git_head.txt", "git_sha.txt", "config_snapshot.txt"}
    restored: List[str] = []
    for name in files:
        if name in skip or not name:
            continue
        cp_res = _ssh_run(
            ["cp", "-p", f"{backup_remote}/{name}",
             f"{PI_STATE_DIR}/{name}"],
            check=False, timeout=15,
        )
        if cp_res.returncode == 0:
            restored.append(name)

    svc = _service_name()
    _log(f"restarting Pi service after rollback: {svc}")
    _ssh_run(["sudo", "-n", "systemctl", "restart", svc],
             check=False, timeout=STARTUP_WAIT_SEC)
    deadline = time.time() + STARTUP_WAIT_SEC
    active = False
    while time.time() < deadline:
        r = _ssh_run(["systemctl", "is-active", svc],
                     check=False, timeout=5)
        if r.stdout.strip() == "active":
            active = True
            break
        time.sleep(1)

    result = {
        "rolled_back_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "backup_path_pi": backup_remote,
        "files_restored": restored,
        "service_active": active,
    }
    _ssh_write_file(f"{PI_STATE_DIR}/rollback_complete.json",
                     json.dumps(result, indent=2) + "\n")
    return result


# ------------ main ------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="6b mechanical deploy")
    mode_group = ap.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--dry-run", action="store_true",
                              help="pre-flight only; no mutation")
    mode_group.add_argument("--execute", action="store_true",
                              help="full deploy; requires --confirm")
    mode_group.add_argument("--rollback", action="store_true",
                              help="restore from last backup; "
                                   "requires --confirm")
    mode_group.add_argument("--health-check", action="store_true",
                              help="env + syntax + state-dir + Pi-reach "
                                   "sanity check; no Pi-state reads")
    ap.add_argument("--confirm", default="",
                    help="exact confirmation phrase")
    ap.add_argument("--mode", choices=["dev", "pi"], default="dev",
                    help="dev (default): script runs on dev, Pi ops via "
                         "ssh+rsync. pi: reserved (NotImplementedError)")
    args = ap.parse_args(argv)

    if args.mode == "pi":
        _log("--mode=pi is reserved and not implemented in this build",
             "ERROR")
        return 2

    # Health-check runs stand-alone and exits before any pre-flight.
    if args.health_check:
        code, ok, issues = health_check()
        _log(f"HEALTH CHECK: {len(ok)} ok, {len(issues)} issues")
        for m in ok:     _log(f"  OK  - {m}")
        for m in issues: _log(f"  !!  - {m}", "ERROR")
        return code

    if args.execute and args.confirm != EXECUTE_CONFIRM:
        _log(f"ABORT: --execute requires --confirm "
             f"{EXECUTE_CONFIRM!r}", "ERROR")
        return 2
    if args.rollback and args.confirm != ROLLBACK_CONFIRM:
        _log(f"ABORT: --rollback requires --confirm "
             f"{ROLLBACK_CONFIRM!r}", "ERROR")
        return 2

    if args.rollback:
        try:
            out = rollback()
            _log(f"ROLLBACK OK: restored {len(out['files_restored'])} files "
                 f"from {out['backup_path_pi']}; service_active="
                 f"{out['service_active']}")
            return 0 if out["service_active"] else 1
        except AbortError as e:
            _log(f"ROLLBACK ABORT: {e}", "ERROR"); return 3

    # Dry-run or Execute -- both run preflight first.
    try:
        facts = preflight()
    except AbortError as e:
        _log(f"PREFLIGHT ABORT: {e}", "ERROR"); return 4

    _log("preflight OK:")
    for k, v in facts.items(): _log(f"  {k}: {v}")

    if args.dry_run:
        _log("DRY-RUN: would create Pi backup, rsync entropy_live_multi.py "
             "+ shadow_expectation.json + config_lineage_init.py, rename "
             "v1 shadow, clear drift, sudo systemctl restart, verify "
             "banner, verify first audit line, write Pi deploy_6b_complete.json.")
        return 0

    # Execute path
    try:
        backup_remote = create_backup(facts)
    except Exception as e:
        _log(f"BACKUP ABORT: {e}", "ERROR"); return 5

    try:
        execute_deploy(facts)
        banner = verify_startup_banner()
        audit = verify_first_audit_line()
    except Exception as e:
        _log(f"EXECUTE ABORT: {e} -- attempting rollback", "ERROR")
        try:
            rb = rollback()
            _log(f"auto-rollback ok: restored {len(rb['files_restored'])} "
                 f"files; service_active={rb['service_active']}", "ERROR")
        except Exception as e2:
            _log(f"!!! AUTO-ROLLBACK FAILED: {e2} !!!", "ERROR")
        return 6

    record = write_deploy_record(facts, banner, audit, backup_remote)

    # Lineage append -- non-fatal: deploy already succeeded.
    lineage_ok, lineage_msg = append_lineage_on_success(facts)
    if not lineage_ok:
        _log("=" * 60, "ERROR")
        _log("!!! DEPLOY SUCCEEDED BUT LINEAGE APPEND FAILED !!!", "ERROR")
        _log(f"    reason: {lineage_msg}", "ERROR")
        _log("    Dashboard will show stale 'queued' state.", "ERROR")
        _log(f"    See {_local_state_dir() / 'lineage_append_failed.json'} "
             "for intended entry and traceback.", "ERROR")
        _log("=" * 60, "ERROR")

    _log("=" * 60)
    _log("6b DEPLOY SUCCESS")
    _log(f"  git HEAD: {facts['head_sha']}")
    _log(f"  shadow:   {facts['shadow_cell_name']}")
    _log(f"  matched:  {banner.get('matched_substrings')}")
    _log(f"  backup:   {backup_remote}")
    _log(f"  record:   {record}")
    _log(f"  lineage:  {'ok' if lineage_ok else 'FAILED (non-fatal)'}")
    _log("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
