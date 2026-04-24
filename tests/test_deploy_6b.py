"""Tests for scripts/deploy_6b.py.

These run against a tmp-dir fake-repo fixture so they never touch the real
working tree. Subprocess + ssh helpers are monkey-patched to return scripted
responses. No network, no systemctl, no git mutation, no rsync.
"""
import json
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import deploy_6b as d6b  # noqa: E402


# ============ Common fakes / helpers ============

GOOD_UNIT_FILE = (
    "# /etc/systemd/system/entropy-trader.service\n"
    "[Unit]\n"
    "Description=Multi-Pair Entropy Trader (BTC+ETH, Kraken Futures 10x)\n"
    "\n"
    "[Service]\n"
    "Type=simple\n"
    "User=user\n"
    "WorkingDirectory=/home/user/entropy_trader\n"
    "ExecStart=/home/user/entropy_trader/venv/bin/python -u entropy_live_multi.py\n"
    "Restart=always\n"
    "\n"
    "[Install]\n"
    "WantedBy=multi-user.target\n"
)
GOOD_MAIN_PID = "227313"
GOOD_CMDLINE = (
    "/home/user/entropy_trader/venv/bin/python -u entropy_live_multi.py"
)
GOOD_PI_STATE = {"position": None, "trades": []}
GOOD_PI_SHADOW = {
    "cell_name": "timeout_trail+E3+5x (revised 6b, no F3c)",
    "f3c_enabled": False,
    "data_source": "Feb 18 - Apr 7 2026 Pi Kraken Futures ETH L2",
    "n_trades_total": 44,
    "buckets": {},
    "fallback_by_direction": {},
}


# ---- _ssh_run dispatch helpers ----

def make_ssh_run_dispatcher(
        unit_text=GOOD_UNIT_FILE, is_active="active",
        main_pid=GOOD_MAIN_PID, restart_ok=True,
        date_skew_sec=0.0):
    """Return a fake _ssh_run that recognizes the remote commands the
    deploy script issues and returns canned responses."""
    pi_now = [time.time() + date_skew_sec]

    def fake_ssh_run(remote, host=None, check=False, timeout=30):
        cmd = list(remote)
        # systemctl
        if cmd[:2] == ["systemctl", "is-active"]:
            return subprocess.CompletedProcess(
                cmd, 0, f"{is_active}\n", "")
        if cmd[:2] == ["systemctl", "cat"]:
            return subprocess.CompletedProcess(cmd, 0, unit_text, "")
        if cmd[:2] == ["systemctl", "show"] and "MainPID" in cmd:
            return subprocess.CompletedProcess(
                cmd, 0, f"{main_pid}\n", "")
        if cmd[:1] == ["sudo"] and cmd[1:4] == ["-n", "systemctl", "restart"]:
            return subprocess.CompletedProcess(
                cmd, 0 if restart_ok else 1,
                "", "" if restart_ok else "restart failed\n")
        # date
        if cmd == ["date", "+%s.%N"]:
            return subprocess.CompletedProcess(
                cmd, 0, f"{pi_now[0]:.6f}\n", "")
        # mkdir / cp / rm / mv / test / ls / cat / tail / bash -c ...
        return subprocess.CompletedProcess(cmd, 0, "", "")

    return fake_ssh_run


def install_happy_pi(monkeypatch, *,
                      pi_state=None, valve_exists=False,
                      audit_line_count_seq=None,
                      log_blob_with_banner=True,
                      shadow_for_local=None):
    """Patch all ssh helpers + the local-shadow file lookups to a
    canonical happy-path. Optional args tune the failure modes."""
    state = pi_state if pi_state is not None else GOOD_PI_STATE

    files: dict = {
        f"{d6b.PI_STATE_DIR}/multi_trader_state.json": json.dumps(state),
        f"{d6b.PI_STATE_DIR}/last_deploy_backup.txt": (
            f"{d6b.PI_STATE_DIR}/pre_6b_backup_TEST"),
    }
    if valve_exists:
        files[f"{d6b.PI_STATE_DIR}/safety_valve.json"] = "{}"

    def fake_pi_read_file(path):
        if path in files:
            return files[path]
        raise d6b.AbortError(f"fake_pi_read_file: {path} not in fixture")

    def fake_pi_file_exists(path):
        return path in files

    rsync_calls: list = []
    def fake_rsync(local_path, remote_path, host=None):
        rsync_calls.append((Path(local_path), remote_path))

    write_calls: list = []
    def fake_ssh_write(remote_path, content, host=None, timeout=30):
        write_calls.append((remote_path, content))
        files[remote_path] = content
        # When marker is written, register the backup dir as existing
        # so subsequent rollback finds it.
        if remote_path.endswith("last_deploy_backup.txt"):
            files[content.strip()] = "<dir>"

    monkeypatch.setattr(d6b, "_ssh_run",
                        make_ssh_run_dispatcher())
    monkeypatch.setattr(d6b, "_pi_read_file", fake_pi_read_file)
    monkeypatch.setattr(d6b, "_pi_file_exists", fake_pi_file_exists)
    monkeypatch.setattr(d6b, "_rsync_to_pi", fake_rsync)
    monkeypatch.setattr(d6b, "_ssh_write_file", fake_ssh_write)
    monkeypatch.setattr(d6b, "_read_proc_cmdline",
                        lambda pid: GOOD_CMDLINE)

    return {"files": files, "rsync_calls": rsync_calls,
            "write_calls": write_calls}


def install_local_git_ok(monkeypatch):
    """Patch _run to fake clean git state (status, branch=main, recent
    commits include PATH A subject). Local-only; not for ssh commands."""
    recent = (
        "merge: sprint v1 PATH A\n"
        + d6b.EXPECTED_COMMIT_SUBJECT + "\n"
        + "sprint v1: gitignore tune\n"
    )
    def fake_run(cmd, check=True, timeout=30, capture=True):
        if cmd[:3] == ["git", "-C", str(d6b.REPO_ROOT)]:
            rest = cmd[3:]
            if rest[:2] == ["status", "--porcelain"]:
                return subprocess.CompletedProcess(cmd, 0, "", "")
            if rest == ["branch", "--show-current"]:
                return subprocess.CompletedProcess(cmd, 0, "main\n", "")
            if rest[:2] == ["log", "-10"] and "--format=%s" in rest:
                return subprocess.CompletedProcess(cmd, 0, recent, "")
            if rest[:2] == ["log", "-1"] and "--format=%s" in rest:
                return subprocess.CompletedProcess(
                    cmd, 0, "merge: sprint v1 PATH A\n", "")
            if rest == ["rev-parse", "HEAD"]:
                return subprocess.CompletedProcess(
                    cmd, 0, "deadbeefdeadbeef\n", "")
            if rest[:2] == ["log", "--oneline"]:
                return subprocess.CompletedProcess(
                    cmd, 0, "deadbeef test head\n", "")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(d6b, "_run", fake_run)


# ---- fake_repo: tmp_path with state/ + shadow + tests/ ----

@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    (tmp_path / "state").mkdir()
    (tmp_path / "algo").mkdir()
    (tmp_path / "algo" / "logs").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_deploy_6b.py").write_text("# stub\n")

    (tmp_path / "algo" / "entropy_live_multi.py").write_text(textwrap.dedent("""
        PAIRS = {"ETH": {"f3c_enabled": False, "timeout_trail_enabled": True,
                          "e3_time_decay_sl_enabled": True,
                          "pst_max_wait_bars": 40}}
        SHARED_CONFIG = {"leverage": 5}
    """).strip())

    (tmp_path / "state" / "shadow_expectation.json").write_text(
        json.dumps(GOOD_PI_SHADOW))
    (tmp_path / "state" / "shadow_expectation.v1.json").write_text(
        json.dumps({"cell_name": "F3c+timeout_trail+E3+5x"}))

    monkeypatch.setattr(d6b, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(d6b, "LOCAL_STATE_DIR_PRIMARY", tmp_path / "state")
    monkeypatch.setattr(d6b, "LOCAL_STATE_DIR_LEGACY",
                        tmp_path / "algo" / "state")
    monkeypatch.setattr(d6b, "LOCAL_CONFIG_FILE_PRIMARY",
                        tmp_path / "algo" / "entropy_live_multi.py")
    monkeypatch.setattr(d6b, "LOCAL_CONFIG_FILE_LEGACY",
                        tmp_path / "entropy_live_multi.py")
    monkeypatch.setattr(d6b, "STATE_DIR", tmp_path / "state")

    return tmp_path


# ============ CLI / confirm-phrase tests ============

def test_execute_without_confirm_aborts(fake_repo):
    rc = d6b.main(["--execute"])
    assert rc != 0


def test_execute_wrong_confirm_aborts(fake_repo):
    rc = d6b.main(["--execute", "--confirm", "wrong phrase"])
    assert rc != 0


def test_rollback_without_confirm_aborts(fake_repo):
    rc = d6b.main(["--rollback"])
    assert rc != 0


def test_rollback_wrong_confirm_aborts(fake_repo):
    rc = d6b.main(["--rollback", "--confirm", "ROLLBACK NOW"])
    assert rc != 0


def test_mode_pi_not_implemented(fake_repo):
    """--mode=pi is reserved; should exit non-zero with explanatory log."""
    rc = d6b.main(["--dry-run", "--mode=pi"])
    assert rc != 0


# ============ Pre-flight failure modes ============

def test_preflight_dirty_git_aborts(fake_repo, monkeypatch):
    def fake_run(cmd, check=True, timeout=30, capture=True):
        if cmd[:3] == ["git", "-C", str(d6b.REPO_ROOT)] and \
           cmd[3:5] == ["status", "--porcelain"]:
            return subprocess.CompletedProcess(cmd, 0, " M file.py\n", "")
        return subprocess.CompletedProcess(cmd, 0, "main\n", "")
    monkeypatch.setattr(d6b, "_run", fake_run)
    with pytest.raises(d6b.AbortError, match="working tree not clean"):
        d6b.preflight()


def test_preflight_wrong_branch_aborts(fake_repo, monkeypatch):
    def fake_run(cmd, check=True, timeout=30, capture=True):
        rest = cmd[3:] if cmd[:3] == ["git", "-C", str(d6b.REPO_ROOT)] else []
        if rest[:2] == ["status", "--porcelain"]:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if rest == ["branch", "--show-current"]:
            return subprocess.CompletedProcess(cmd, 0, "sprint/x\n", "")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(d6b, "_run", fake_run)
    with pytest.raises(d6b.AbortError, match="branch is 'sprint/x'"):
        d6b.preflight()


def test_preflight_expected_subject_not_in_recent_aborts(fake_repo, monkeypatch):
    def fake_run(cmd, check=True, timeout=30, capture=True):
        rest = cmd[3:] if cmd[:3] == ["git", "-C", str(d6b.REPO_ROOT)] else []
        if rest[:2] == ["status", "--porcelain"]:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if rest == ["branch", "--show-current"]:
            return subprocess.CompletedProcess(cmd, 0, "main\n", "")
        if rest[:2] == ["log", "-10"] and "--format=%s" in rest:
            return subprocess.CompletedProcess(
                cmd, 0, "wrong subject\nanother wrong\n", "")
        if rest == ["rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(cmd, 0, "dead\n", "")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(d6b, "_run", fake_run)
    with pytest.raises(d6b.AbortError, match="not found in last 10 commits"):
        d6b.preflight()


def test_preflight_active_trade_aborts(fake_repo, monkeypatch):
    install_local_git_ok(monkeypatch)
    install_happy_pi(monkeypatch,
                       pi_state={"position": {"pair": "ETH", "direction": 1}})
    with pytest.raises(d6b.AbortError, match="position is not None"):
        d6b.preflight()


def test_preflight_wrong_shadow_aborts(fake_repo, monkeypatch):
    install_local_git_ok(monkeypatch)
    install_happy_pi(monkeypatch)
    # Overwrite local shadow to be bad.
    (fake_repo / "state" / "shadow_expectation.json").write_text(json.dumps({
        "cell_name": "F3c+timeout_trail+E3+5x",
        "f3c_enabled": True,
        "data_source": "Feb-Apr Pi",
    }))
    with pytest.raises(d6b.AbortError, match="no F3c"):
        d6b.preflight()


def test_preflight_valve_exists_aborts(fake_repo, monkeypatch):
    install_local_git_ok(monkeypatch)
    install_happy_pi(monkeypatch, valve_exists=True)
    with pytest.raises(d6b.AbortError, match="Valve tripped"):
        d6b.preflight()


# ============ P1..P4 service-target verification ============

def test_preflight_service_not_active_aborts(fake_repo, monkeypatch):
    install_local_git_ok(monkeypatch)
    install_happy_pi(monkeypatch)
    monkeypatch.setattr(
        d6b, "_ssh_run",
        make_ssh_run_dispatcher(is_active="inactive"))
    with pytest.raises(d6b.AbortError, match="P1 FAILED"):
        d6b.preflight()


def test_preflight_working_dir_mismatch_aborts(fake_repo, monkeypatch):
    install_local_git_ok(monkeypatch)
    install_happy_pi(monkeypatch)
    wrong_unit = GOOD_UNIT_FILE.replace(
        "WorkingDirectory=/home/user/entropy_trader",
        "WorkingDirectory=/home/user",
    ).replace(
        "ExecStart=/home/user/entropy_trader/venv/bin/python -u entropy_live_multi.py",
        "ExecStart=/home/user/hft/venv/bin/python -u -m hft.mr_live --live",
    )
    monkeypatch.setattr(
        d6b, "_ssh_run",
        make_ssh_run_dispatcher(unit_text=wrong_unit))
    with pytest.raises(d6b.AbortError, match="P2 FAILED"):
        d6b.preflight()


def test_preflight_execstart_mismatch_aborts(fake_repo, monkeypatch):
    install_local_git_ok(monkeypatch)
    install_happy_pi(monkeypatch)
    wrong_unit = GOOD_UNIT_FILE.replace(
        "ExecStart=/home/user/entropy_trader/venv/bin/python -u entropy_live_multi.py",
        "ExecStart=/home/user/entropy_trader/venv/bin/python -u other.py",
    )
    monkeypatch.setattr(
        d6b, "_ssh_run",
        make_ssh_run_dispatcher(unit_text=wrong_unit))
    with pytest.raises(d6b.AbortError, match="P3 FAILED"):
        d6b.preflight()


def test_preflight_running_pid_cmdline_mismatch_aborts(fake_repo, monkeypatch):
    install_local_git_ok(monkeypatch)
    install_happy_pi(monkeypatch)
    monkeypatch.setattr(
        d6b, "_read_proc_cmdline",
        lambda pid: "/usr/bin/python -m hft.mr_live --live",
    )
    with pytest.raises(d6b.AbortError, match="P4 FAILED"):
        d6b.preflight()


def test_preflight_all_checks_pass_when_service_is_correct(
        fake_repo, monkeypatch):
    install_local_git_ok(monkeypatch)
    install_happy_pi(monkeypatch)
    facts = d6b.preflight()
    assert facts["service_verified"] == d6b.SERVICE_NAME_DEFAULT
    assert facts["service_working_dir"] == d6b.EXPECTED_WORKING_DIR
    assert facts["service_exec_marker"] == d6b.EXPECTED_EXEC_MARKER
    assert facts["position"] is None
    assert facts["safety_valve_absent"] is True
    assert "no F3c" in facts["shadow_cell_name"]


# ============ Health check ============

def _mock_ssh_pi_reachable(monkeypatch, reachable=True):
    def fake_run(cmd, check=True, timeout=30, capture=True):
        if cmd[0] == "ssh":
            if reachable:
                return subprocess.CompletedProcess(
                    cmd, 0, "pi_reachable\n", "")
            return subprocess.CompletedProcess(
                cmd, 255, "", "ssh: connect: Connection timed out\n")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(d6b, "_run", fake_run)


def test_health_check_all_green(fake_repo, monkeypatch):
    monkeypatch.setenv("ENTROPY_SERVICE_NAME", "entropy-trader.service")
    _mock_ssh_pi_reachable(monkeypatch, reachable=True)
    code, ok, issues = d6b.health_check()
    assert code == 0, f"expected 0, got {code}; issues={issues}"
    assert len(issues) == 0
    assert any("ENTROPY_SERVICE_NAME=entropy-trader.service" in m for m in ok)
    assert any("Pi reachable at" in m for m in ok)


def test_health_check_missing_env_var(fake_repo, monkeypatch):
    monkeypatch.delenv("ENTROPY_SERVICE_NAME", raising=False)
    _mock_ssh_pi_reachable(monkeypatch, reachable=True)
    code, ok, issues = d6b.health_check()
    assert code == 1
    assert any("ENTROPY_SERVICE_NAME not set" in m for m in issues)


def test_health_check_pi_unreachable(fake_repo, monkeypatch):
    monkeypatch.setenv("ENTROPY_SERVICE_NAME", "entropy-trader.service")
    _mock_ssh_pi_reachable(monkeypatch, reachable=False)
    code, ok, issues = d6b.health_check()
    assert code == 1
    pi_issue = [m for m in issues if "PI unreachable" in m]
    assert len(pi_issue) == 1
    assert "PI_HOST_TARGET" in pi_issue[0]


def test_health_check_exits_fast(fake_repo, monkeypatch):
    monkeypatch.setenv("ENTROPY_SERVICE_NAME", "entropy-trader.service")
    _mock_ssh_pi_reachable(monkeypatch, reachable=True)
    t0 = time.time()
    rc = d6b.main(["--health-check"])
    assert time.time() - t0 < 3.0
    assert rc == 0


# ============ Backup (Pi-side ssh) ============

def test_backup_creates_files_on_pi_via_ssh(fake_repo, monkeypatch):
    install_local_git_ok(monkeypatch)
    pi = install_happy_pi(monkeypatch)
    facts = {"head_sha": "abc123"}
    backup_path = d6b.create_backup(facts)
    assert backup_path.startswith(d6b.PI_STATE_DIR + "/pre_6b_backup_")
    # Marker file written via ssh
    marker_writes = [w for w in pi["write_calls"]
                     if w[0] == f"{d6b.PI_STATE_DIR}/last_deploy_backup.txt"]
    assert len(marker_writes) == 1, f"expected one marker write, got {pi['write_calls']}"
    assert backup_path in marker_writes[0][1]
    # git_head + git_sha snapshots written via ssh
    git_writes = [w for w in pi["write_calls"]
                  if "git_head.txt" in w[0] or "git_sha.txt" in w[0]]
    assert len(git_writes) == 2


# ============ Rollback (Pi-side ssh) ============

def test_rollback_restores_files_via_ssh(fake_repo, monkeypatch):
    install_local_git_ok(monkeypatch)
    pi = install_happy_pi(monkeypatch)
    # Override the ls listing for the backup dir.
    backup_path = pi["files"][f"{d6b.PI_STATE_DIR}/last_deploy_backup.txt"]
    pi["files"][backup_path] = "exists"  # so _pi_file_exists is True

    captured = {"cp_calls": [], "ls_calls": []}
    base = d6b._ssh_run

    def patched_ssh_run(remote, host=None, check=False, timeout=30):
        cmd = list(remote)
        if cmd[:2] == ["bash", "-c"] and "ls -1" in (cmd[2] if len(cmd) > 2 else ""):
            captured["ls_calls"].append(cmd)
            return subprocess.CompletedProcess(
                cmd, 0,
                "shadow_expectation.json\nlive_drift_monitor.json\n"
                "multi_trader_state.json\ngit_head.txt\n", "")
        if cmd[:1] == ["cp"]:
            captured["cp_calls"].append(cmd)
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return base(remote, host=host, check=check, timeout=timeout)

    monkeypatch.setattr(d6b, "_ssh_run", patched_ssh_run)

    result = d6b.rollback()
    assert result["service_active"] is True
    # Should have skipped git_head.txt
    cp_targets = [c[2] for c in captured["cp_calls"]]
    assert any("shadow_expectation.json" in t for t in cp_targets)
    assert not any("git_head.txt" in t for t in cp_targets)
    assert "git_head.txt" not in result["files_restored"]


# ============ Execute path ============

def test_execute_dry_run_makes_no_pi_mutations(fake_repo, monkeypatch):
    install_local_git_ok(monkeypatch)
    pi = install_happy_pi(monkeypatch)
    rc = d6b.main(["--dry-run", "--mode=dev"])
    assert rc == 0
    # No rsync calls in dry-run.
    assert pi["rsync_calls"] == [], (
        f"dry-run made rsync calls: {pi['rsync_calls']}")
    # No file writes either.
    assert pi["write_calls"] == [], (
        f"dry-run made ssh writes: {pi['write_calls']}")


def test_execute_full_path_success(fake_repo, monkeypatch):
    install_local_git_ok(monkeypatch)
    pi = install_happy_pi(monkeypatch)
    # Override _ssh_run so banner-tail returns a banner blob containing
    # all required substrings, and the audit-line wc returns growing count.
    audit_seq = iter([5, 5, 5, 6])
    banner_blob = (
        "2026-04-24 12:00:00 INFO Config: {\n"
        '  "leverage": 5\n}\n'
        "2026-04-24 12:00:00 INFO   ETH: "
        '{"f3c_enabled": false, "timeout_trail_enabled": true, '
        '"e3_time_decay_sl_enabled": true, "pst_max_wait_bars": 40}\n'
        "2026-04-24 12:00:00 INFO Starting multi-pair entropy trader (PRIORITY mode)...\n"
    )
    audit_new = '{"f3c_enabled": false, "decision": "passed"}'

    def patched(remote, host=None, check=False, timeout=30):
        cmd = list(remote)
        if cmd[:2] == ["systemctl", "is-active"]:
            return subprocess.CompletedProcess(cmd, 0, "active\n", "")
        if cmd[:2] == ["systemctl", "cat"]:
            return subprocess.CompletedProcess(cmd, 0, GOOD_UNIT_FILE, "")
        if cmd[:2] == ["systemctl", "show"] and "MainPID" in cmd:
            return subprocess.CompletedProcess(
                cmd, 0, GOOD_MAIN_PID + "\n", "")
        if cmd[:1] == ["sudo"] and cmd[1:4] == ["-n", "systemctl", "restart"]:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if cmd == ["date", "+%s.%N"]:
            return subprocess.CompletedProcess(
                cmd, 0, f"{time.time():.6f}\n", "")
        if cmd[:1] == ["tail"] and "entropy_multi.log" in " ".join(cmd):
            return subprocess.CompletedProcess(cmd, 0, banner_blob, "")
        if cmd[:1] == ["tail"] and "daily_filter_audit" in " ".join(cmd):
            return subprocess.CompletedProcess(cmd, 0, audit_new + "\n", "")
        if cmd[:2] == ["bash", "-c"] and "wc -l" in cmd[2]:
            try:
                return subprocess.CompletedProcess(
                    cmd, 0, f"{next(audit_seq)}\n", "")
            except StopIteration:
                return subprocess.CompletedProcess(cmd, 0, "6\n", "")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(d6b, "_ssh_run", patched)
    monkeypatch.setattr(d6b, "BANNER_POLL_SEC", 5)
    monkeypatch.setattr(d6b, "AUDIT_POLL_SEC", 3)
    monkeypatch.setattr(d6b, "STARTUP_WAIT_SEC", 2)

    rc = d6b.main(["--execute", "--confirm", d6b.EXECUTE_CONFIRM])
    assert rc == 0, f"expected rc=0; rsync={pi['rsync_calls']}, writes={pi['write_calls']}"

    # entropy_live_multi.py rsync'd
    assert any(str(r[0]).endswith("entropy_live_multi.py")
               and r[1] == d6b.PI_CONFIG_FILE for r in pi["rsync_calls"])
    # shadow_expectation.json rsync'd
    assert any(str(r[0]).endswith("shadow_expectation.json")
               and r[1].endswith("shadow_expectation.json")
               for r in pi["rsync_calls"])


def test_execute_banner_mismatch_triggers_rollback(fake_repo, monkeypatch):
    install_local_git_ok(monkeypatch)
    pi = install_happy_pi(monkeypatch)
    # Banner blob never contains the required substrings -> banner mismatch.
    bad_blob = "2026-04-24 12:00:00 INFO some other line\n"

    def patched(remote, host=None, check=False, timeout=30):
        cmd = list(remote)
        if cmd[:2] == ["systemctl", "is-active"]:
            return subprocess.CompletedProcess(cmd, 0, "active\n", "")
        if cmd[:2] == ["systemctl", "cat"]:
            return subprocess.CompletedProcess(cmd, 0, GOOD_UNIT_FILE, "")
        if cmd[:2] == ["systemctl", "show"] and "MainPID" in cmd:
            return subprocess.CompletedProcess(
                cmd, 0, GOOD_MAIN_PID + "\n", "")
        if cmd[:1] == ["sudo"] and cmd[1:4] == ["-n", "systemctl", "restart"]:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if cmd == ["date", "+%s.%N"]:
            return subprocess.CompletedProcess(
                cmd, 0, f"{time.time():.6f}\n", "")
        if cmd[:1] == ["tail"] and "entropy_multi.log" in " ".join(cmd):
            return subprocess.CompletedProcess(cmd, 0, bad_blob, "")
        if cmd[:2] == ["bash", "-c"] and "ls -1" in cmd[2]:
            return subprocess.CompletedProcess(cmd, 0, "shadow_expectation.json\n", "")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(d6b, "_ssh_run", patched)
    monkeypatch.setattr(d6b, "BANNER_POLL_SEC", 1)
    monkeypatch.setattr(d6b, "AUDIT_POLL_SEC", 1)
    monkeypatch.setattr(d6b, "STARTUP_WAIT_SEC", 2)

    rc = d6b.main(["--execute", "--confirm", d6b.EXECUTE_CONFIRM])
    assert rc != 0
    # rollback should have written rollback_complete.json
    assert any(w[0] == f"{d6b.PI_STATE_DIR}/rollback_complete.json"
               for w in pi["write_calls"])


# ============ --mode=dev specific tests ============

def test_dev_mode_dry_run_no_ssh_mutation(fake_repo, monkeypatch):
    """--dry-run --mode=dev makes preflight ssh reads only; no rsync,
    no ssh-write, no Pi mutation."""
    install_local_git_ok(monkeypatch)
    pi = install_happy_pi(monkeypatch)
    rc = d6b.main(["--dry-run", "--mode=dev"])
    assert rc == 0
    assert pi["rsync_calls"] == []
    assert pi["write_calls"] == []


def test_dev_mode_preflight_pi_position_check_via_ssh(fake_repo, monkeypatch):
    """preflight reads multi_trader_state.json via _pi_read_file, not
    via local Path read; setting position non-None aborts."""
    install_local_git_ok(monkeypatch)
    install_happy_pi(monkeypatch,
                       pi_state={"position": {"pair": "ETH", "direction": -1}})
    with pytest.raises(d6b.AbortError, match="position is not None"):
        d6b.preflight()


def test_dev_mode_rsync_transfers_expected_files(fake_repo, monkeypatch):
    """execute_deploy rsyncs entropy_live_multi.py + shadow_expectation
    + (if present) config_lineage_init.py to the expected Pi paths."""
    install_local_git_ok(monkeypatch)
    pi = install_happy_pi(monkeypatch)
    # Provide a config_lineage_init.py so it gets included.
    (fake_repo / "algo" / "dashboard").mkdir()
    (fake_repo / "algo" / "dashboard" / "config_lineage_init.py").write_text(
        "def append_deploy_entry(**kw): pass\n")

    monkeypatch.setattr(d6b, "STARTUP_WAIT_SEC", 1)
    facts = {"head_sha": "deadbeef"}
    d6b.execute_deploy(facts)

    rsync_pairs = [(str(r[0]).split("\\")[-1].split("/")[-1], r[1])
                   for r in pi["rsync_calls"]]
    assert ("entropy_live_multi.py", d6b.PI_CONFIG_FILE) in rsync_pairs
    assert ("shadow_expectation.json",
            f"{d6b.PI_STATE_DIR}/shadow_expectation.json") in rsync_pairs
    assert ("config_lineage_init.py", d6b.PI_LINEAGE_INIT) in rsync_pairs


def test_dev_mode_service_restart_and_banner_verify(fake_repo, monkeypatch):
    """systemctl restart issued via sudo -n; is-active polled until
    'active'; banner verify runs."""
    install_local_git_ok(monkeypatch)
    install_happy_pi(monkeypatch)

    calls = {"restart_count": 0, "is_active_count": 0}

    def patched(remote, host=None, check=False, timeout=30):
        cmd = list(remote)
        if cmd[:1] == ["sudo"] and cmd[1:4] == ["-n", "systemctl", "restart"]:
            calls["restart_count"] += 1
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if cmd[:2] == ["systemctl", "is-active"]:
            calls["is_active_count"] += 1
            return subprocess.CompletedProcess(cmd, 0, "active\n", "")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(d6b, "_ssh_run", patched)
    monkeypatch.setattr(d6b, "STARTUP_WAIT_SEC", 1)
    facts = {"head_sha": "deadbeef"}
    d6b.execute_deploy(facts)

    assert calls["restart_count"] == 1
    assert calls["is_active_count"] >= 1


def test_dev_mode_rollback_via_ssh(fake_repo, monkeypatch):
    """rollback() uses ssh helpers exclusively -- no local Path reads
    of state files."""
    install_local_git_ok(monkeypatch)
    pi = install_happy_pi(monkeypatch)
    backup_path = pi["files"][f"{d6b.PI_STATE_DIR}/last_deploy_backup.txt"]
    pi["files"][backup_path] = "exists"

    captured = {"ls": False, "cp": 0, "restart": 0}

    def patched(remote, host=None, check=False, timeout=30):
        cmd = list(remote)
        if cmd[:2] == ["bash", "-c"] and "ls -1" in cmd[2]:
            captured["ls"] = True
            return subprocess.CompletedProcess(
                cmd, 0,
                "shadow_expectation.json\nmulti_trader_state.json\n", "")
        if cmd[:1] == ["cp"]:
            captured["cp"] += 1
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if cmd[:1] == ["sudo"] and cmd[1:4] == ["-n", "systemctl", "restart"]:
            captured["restart"] += 1
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if cmd[:2] == ["systemctl", "is-active"]:
            return subprocess.CompletedProcess(cmd, 0, "active\n", "")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(d6b, "_ssh_run", patched)
    monkeypatch.setattr(d6b, "STARTUP_WAIT_SEC", 1)
    result = d6b.rollback()
    assert result["service_active"] is True
    assert captured["ls"] is True
    assert captured["cp"] == 2
    assert captured["restart"] == 1


# ============ Lineage append (local + rsync) ============

def test_lineage_append_success_adds_entry(fake_repo, monkeypatch):
    """With the real config_lineage_init.py copied into the fake repo,
    append_lineage_on_success should append a live entry and rsync the
    file to Pi."""
    src = ROOT / "algo" / "dashboard" / "config_lineage_init.py"
    dst_dir = fake_repo / "algo" / "dashboard"
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst_dir / "config_lineage_init.py")

    lineage = fake_repo / "state" / "config_lineage.jsonl"
    lineage.write_text(json.dumps({
        "version_label": "6b PATH A", "status": "queued",
        "deployed_at": None, "deployed_sha": None,
        "key_changes": [], "notes": "queued",
    }) + "\n" + json.dumps({
        "version_label": "6a observer", "status": "live",
        "deployed_at": "2026-04-19T22:26:17+00:00",
        "deployed_sha": "7c8d67f",
        "key_changes": [], "notes": "observer",
    }) + "\n")

    rsync_calls: list = []
    def fake_rsync(local_path, remote_path, host=None):
        rsync_calls.append((Path(local_path), remote_path))
    monkeypatch.setattr(d6b, "_rsync_to_pi", fake_rsync)

    facts = {"head_sha": "abc1234567890abcdef"}
    ok, msg = d6b.append_lineage_on_success(facts)
    assert ok, f"append failed: {msg}"

    entries = [json.loads(ln) for ln in lineage.read_text().splitlines() if ln.strip()]
    six_b = [e for e in entries if e["version_label"] == "6b PATH A"]
    assert len(six_b) == 1
    assert six_b[0]["status"] == "live"
    assert six_b[0]["deployed_sha"] == "abc1234"
    six_a = [e for e in entries if e["version_label"] == "6a observer"]
    assert six_a[0]["status"] == "historical"

    # rsync to Pi was attempted
    assert any(str(r[0]).endswith("config_lineage.jsonl") and
               r[1] == d6b.PI_LINEAGE_FILE for r in rsync_calls)


def test_lineage_append_failure_writes_error_locally(fake_repo, monkeypatch):
    """With NO config_lineage_init.py in the fake repo, the append
    should fail gracefully, write lineage_append_failed.json locally,
    and return (False, msg). Deploy itself isn't run here -- this is
    a unit test of the lineage helper alone."""
    init = fake_repo / "algo" / "dashboard" / "config_lineage_init.py"
    assert not init.exists()

    facts = {"head_sha": "deadbeef"}
    ok, msg = d6b.append_lineage_on_success(facts)
    assert not ok
    fail = fake_repo / "state" / "lineage_append_failed.json"
    assert fail.exists()
    payload = json.loads(fail.read_text())
    assert "exception" in payload
    assert payload["intended_entry"]["version_label"] == "6b PATH A"
