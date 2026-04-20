"""Tests for scripts/deploy_6b.py.

These run against a tmp-dir fake-repo fixture so they never touch the real
working tree. Subprocess calls are monkey-patched to return scripted
responses. No network, no systemctl, no git mutation.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

# Make the script importable as a module even though it lives in scripts/.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import deploy_6b as d6b  # noqa: E402


# ---------- fixtures ----------

@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    """Build a tmp 'repo' with state/ and algo/ layouts matching the Pi
    after PATH A is in git HEAD and shadow_expectation.json is the revised
    no-F3c cell."""
    (tmp_path / "state").mkdir()
    (tmp_path / "algo").mkdir()
    (tmp_path / "algo" / "logs").mkdir()
    (tmp_path / "scripts").mkdir()

    # algo/entropy_live_multi.py with the right flag values (just the
    # relevant lines -- the script only greps).
    (tmp_path / "algo" / "entropy_live_multi.py").write_text(textwrap.dedent("""
        PAIRS = {"ETH": {"f3c_enabled": False, "timeout_trail_enabled": True,
                          "e3_time_decay_sl_enabled": True}}
        SHARED_CONFIG = {"leverage": 5}
    """).strip())

    # state files at PATH-A-ready state
    (tmp_path / "state" / "multi_trader_state.json").write_text(json.dumps({
        "position": None, "trades": []
    }))
    (tmp_path / "state" / "shadow_expectation.json").write_text(json.dumps({
        "cell_name": "timeout_trail+E3+5x (revised 6b, no F3c)",
        "f3c_enabled": False,
        "data_source": "Feb 18 - Apr 7 2026 Pi Kraken Futures ETH L2",
        "generated_at": "2026-04-20T18:52:40+00:00",
        "n_trades_total": 44,
        "buckets": {}, "fallback_by_direction": {},
    }))
    (tmp_path / "state" / "shadow_expectation.v1.json").write_text(json.dumps({
        "cell_name": "F3c+timeout_trail+E3+5x"
    }))
    # drift + audit preexist; rollback needs something to restore
    (tmp_path / "state" / "live_drift_monitor.json").write_text(json.dumps({
        "z_scores": []
    }))

    # Monkey-patch the module-level path constants.
    monkeypatch.setattr(d6b, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(d6b, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(d6b, "LEGACY_STATE_DIR", tmp_path / "algo" / "state")
    monkeypatch.setattr(d6b, "CONFIG_FILE", tmp_path / "algo" / "entropy_live_multi.py")
    monkeypatch.setattr(d6b, "LEGACY_CONFIG_FILE", tmp_path / "entropy_live_multi.py")

    return tmp_path


@pytest.fixture
def mock_git_ok(monkeypatch):
    """_run returns clean git status, 'main' branch, PATH-A HEAD subject."""
    calls = []
    def fake_run(cmd, check=True, timeout=30, capture=True):
        calls.append(cmd)
        stdout = ""
        rc = 0
        if cmd[:3] == ["git", "-C", str(d6b.REPO_ROOT)]:
            rest = cmd[3:]
            if rest[:2] == ["status", "--porcelain"]:
                stdout = ""
            elif rest == ["branch", "--show-current"]:
                stdout = "main\n"
            elif rest[:2] == ["log", "-1"] and "--format=%s" in rest:
                stdout = d6b.EXPECTED_COMMIT_SUBJECT + "\n"
            elif rest == ["rev-parse", "HEAD"]:
                stdout = "deadbeefdeadbeef\n"
            elif rest[:2] == ["log", "--oneline"]:
                stdout = "deadbeef test head\n"
        elif cmd[0] == "systemctl":
            stdout = "active\n"
        elif cmd[0] == "chronyc":
            # no chronyc in test -> simulate absence
            raise FileNotFoundError("chronyc not found in test env")
        return subprocess.CompletedProcess(cmd, rc, stdout, "")

    monkeypatch.setattr(d6b, "_run", fake_run)
    return calls


# ---------- dry-run safety ----------

def test_dry_run_no_mutation(fake_repo, mock_git_ok, capsys):
    """--dry-run must not create backup or modify any state."""
    snap = {p.name: p.read_bytes() for p in (fake_repo / "state").iterdir()
            if p.is_file()}
    rc = d6b.main(["--dry-run"])
    assert rc == 0
    # Nothing in state/ should have changed.
    for p in (fake_repo / "state").iterdir():
        if p.is_file():
            assert p.read_bytes() == snap[p.name], f"{p.name} mutated in dry-run"
    # No backup dir created.
    assert not any((fake_repo / "state").glob("pre_6b_backup_*"))


# ---------- confirm phrase gate ----------

def test_execute_without_confirm_aborts(fake_repo, mock_git_ok):
    rc = d6b.main(["--execute"])
    assert rc != 0  # abort


def test_execute_wrong_confirm_aborts(fake_repo, mock_git_ok):
    rc = d6b.main(["--execute", "--confirm", "wrong phrase"])
    assert rc != 0


def test_rollback_without_confirm_aborts(fake_repo, mock_git_ok):
    rc = d6b.main(["--rollback"])
    assert rc != 0


def test_rollback_wrong_confirm_aborts(fake_repo, mock_git_ok):
    rc = d6b.main(["--rollback", "--confirm", "ROLLBACK NOW"])
    assert rc != 0


# ---------- pre-flight abort cases ----------

def test_preflight_dirty_git_aborts(fake_repo, monkeypatch):
    def fake_run(cmd, check=True, timeout=30, capture=True):
        if cmd[:3] == ["git", "-C", str(d6b.REPO_ROOT)] and \
           cmd[3:5] == ["status", "--porcelain"]:
            return subprocess.CompletedProcess(cmd, 0, " M some_file.py\n", "")
        if cmd[0] == "chronyc":
            raise FileNotFoundError()
        return subprocess.CompletedProcess(cmd, 0, "main\n", "")
    monkeypatch.setattr(d6b, "_run", fake_run)
    with pytest.raises(d6b.AbortError, match="working tree not clean"):
        d6b.preflight(fake_repo / "state", fake_repo / "algo" / "entropy_live_multi.py")


def test_preflight_wrong_branch_aborts(fake_repo, monkeypatch):
    def fake_run(cmd, check=True, timeout=30, capture=True):
        if cmd[3:5] == ["status", "--porcelain"]:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if cmd[3:] == ["branch", "--show-current"]:
            return subprocess.CompletedProcess(cmd, 0, "sprint/x\n", "")
        if cmd[0] == "chronyc": raise FileNotFoundError()
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(d6b, "_run", fake_run)
    with pytest.raises(d6b.AbortError, match="branch is 'sprint/x'"):
        d6b.preflight(fake_repo / "state", fake_repo / "algo" / "entropy_live_multi.py")


def test_preflight_wrong_head_subject_aborts(fake_repo, monkeypatch):
    def fake_run(cmd, check=True, timeout=30, capture=True):
        if cmd[3:5] == ["status", "--porcelain"]:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if cmd[3:] == ["branch", "--show-current"]:
            return subprocess.CompletedProcess(cmd, 0, "main\n", "")
        if cmd[3:5] == ["log", "-1"] and "--format=%s" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "some other commit\n", "")
        if cmd[3:] == ["rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(cmd, 0, "dead\n", "")
        if cmd[0] == "chronyc": raise FileNotFoundError()
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(d6b, "_run", fake_run)
    with pytest.raises(d6b.AbortError, match="HEAD subject is"):
        d6b.preflight(fake_repo / "state", fake_repo / "algo" / "entropy_live_multi.py")


def test_preflight_active_trade_aborts(fake_repo, mock_git_ok):
    (fake_repo / "state" / "multi_trader_state.json").write_text(json.dumps({
        "position": {"pair": "ETH", "direction": 1}
    }))
    with pytest.raises(d6b.AbortError, match="position is not None"):
        d6b.preflight(fake_repo / "state", fake_repo / "algo" / "entropy_live_multi.py")


def test_preflight_wrong_shadow_aborts(fake_repo, mock_git_ok):
    (fake_repo / "state" / "shadow_expectation.json").write_text(json.dumps({
        "cell_name": "F3c+timeout_trail+E3+5x",
        "f3c_enabled": True,
        "data_source": "Feb-Apr Pi",
    }))
    with pytest.raises(d6b.AbortError, match="no F3c"):
        d6b.preflight(fake_repo / "state", fake_repo / "algo" / "entropy_live_multi.py")


def test_preflight_valve_exists_aborts(fake_repo, mock_git_ok):
    (fake_repo / "state" / "safety_valve.json").write_text("{}")
    with pytest.raises(d6b.AbortError, match="Valve tripped"):
        d6b.preflight(fake_repo / "state", fake_repo / "algo" / "entropy_live_multi.py")


# ---------- backup ----------

def test_backup_creates_expected_files(fake_repo, mock_git_ok):
    backup = d6b.create_backup(fake_repo / "state",
                                fake_repo / "algo" / "entropy_live_multi.py")
    assert backup.exists()
    assert (backup / "shadow_expectation.json").exists()
    assert (backup / "shadow_expectation.v1.json").exists()
    assert (backup / "live_drift_monitor.json").exists()
    assert (backup / "multi_trader_state.json").exists()
    assert (backup / "git_head.txt").exists()
    assert (backup / "git_sha.txt").exists()
    assert (backup / "config_snapshot.txt").exists()
    # marker file points at the backup
    marker = (fake_repo / "state" / "last_deploy_backup.txt").read_text().strip()
    assert marker == str(backup)


# ---------- rollback ----------

def test_rollback_restores_files(fake_repo, mock_git_ok):
    # First make a backup
    backup = d6b.create_backup(fake_repo / "state",
                                fake_repo / "algo" / "entropy_live_multi.py")
    # Mutate state after backup (simulate post-deploy state)
    (fake_repo / "state" / "shadow_expectation.json").write_text("MUTATED")
    (fake_repo / "state" / "live_drift_monitor.json").unlink()

    result = d6b.rollback(fake_repo / "state")
    assert result["service_active"] is True
    assert "shadow_expectation.json" in result["files_restored"]
    # shadow should be restored from backup, not "MUTATED"
    restored = (fake_repo / "state" / "shadow_expectation.json").read_text()
    assert restored != "MUTATED"
    assert "no F3c" in restored


# ---------- execute-path integration ----------

def test_execute_full_path_success(fake_repo, mock_git_ok, monkeypatch):
    """End-to-end: dry-run first to prove preflight passes, then full
    execute. Startup banner is faked by pre-writing a matching line to the
    log file before execute_deploy runs."""
    # Pre-seed the log with a matching banner line so verify_startup_banner
    # returns immediately.
    banner = ("2026-04-22 06:00:01 INFO ETH: "
              "f3c_enabled=False timeout_trail_enabled=True "
              "e3_time_decay_sl_enabled=True leverage=5x h_thresh=0.4352")
    (fake_repo / "algo" / "logs" / "entropy_multi.log").write_text(banner + "\n")

    # Shorten poll timers for test speed.
    monkeypatch.setattr(d6b, "BANNER_POLL_SEC", 2)
    monkeypatch.setattr(d6b, "AUDIT_POLL_SEC", 1)
    monkeypatch.setattr(d6b, "STARTUP_WAIT_SEC", 2)

    rc = d6b.main(["--execute", "--confirm", d6b.EXECUTE_CONFIRM])
    assert rc == 0

    # Post-deploy record written
    rec = json.loads((fake_repo / "state" / "deploy_6b_complete.json").read_text())
    assert rec["git_sha"] == "deadbeefdeadbeef"
    assert "no F3c" in rec["shadow_cell_name"]

    # v1 rename happened
    assert (fake_repo / "state" / "shadow_expectation.v1_with_f3c.json").exists()
    assert not (fake_repo / "state" / "shadow_expectation.v1.json").exists()

    # drift monitor cleared
    assert not (fake_repo / "state" / "live_drift_monitor.json").exists()


def test_execute_banner_mismatch_triggers_rollback(fake_repo, mock_git_ok,
                                                     monkeypatch):
    """If startup banner never matches, execute should trigger rollback
    and exit non-zero."""
    # Seed a WRONG banner (leverage=10x instead of 5x).
    bad_banner = ("2026-04-22 06:00:01 INFO ETH: "
                  "f3c_enabled=False timeout_trail_enabled=True "
                  "e3_time_decay_sl_enabled=True leverage=10x")
    (fake_repo / "algo" / "logs" / "entropy_multi.log").write_text(bad_banner + "\n")

    monkeypatch.setattr(d6b, "BANNER_POLL_SEC", 1)
    monkeypatch.setattr(d6b, "AUDIT_POLL_SEC", 1)
    monkeypatch.setattr(d6b, "STARTUP_WAIT_SEC", 2)

    rc = d6b.main(["--execute", "--confirm", d6b.EXECUTE_CONFIRM])
    assert rc != 0

    # Rollback record should exist
    assert (fake_repo / "state" / "rollback_complete.json").exists()


# ---------- health-check ----------

def test_health_check_all_green(fake_repo, monkeypatch):
    """With ENTROPY_SERVICE_NAME set + state writable + tests present,
    exit 0 and no issues reported."""
    # Copy tests file into fake_repo so the 'tests present' check passes.
    (fake_repo / "tests").mkdir(exist_ok=True)
    (fake_repo / "tests" / "test_deploy_6b.py").write_text("# stub\n")
    monkeypatch.setenv("ENTROPY_SERVICE_NAME", "mr-trading.service")
    code, ok, issues = d6b.health_check()
    assert code == 0, f"expected 0, got {code}; issues={issues}"
    assert len(issues) == 0
    # Spot-check that env var was observed
    assert any("ENTROPY_SERVICE_NAME=mr-trading.service" in m for m in ok)


def test_health_check_missing_env_var(fake_repo, monkeypatch):
    (fake_repo / "tests").mkdir(exist_ok=True)
    (fake_repo / "tests" / "test_deploy_6b.py").write_text("# stub\n")
    monkeypatch.delenv("ENTROPY_SERVICE_NAME", raising=False)
    code, ok, issues = d6b.health_check()
    assert code == 1
    assert any("ENTROPY_SERVICE_NAME not set" in m for m in issues)


def test_health_check_exits_fast(fake_repo, monkeypatch):
    """Sanity: run under 5 seconds (we give it 3 for safety)."""
    import time
    (fake_repo / "tests").mkdir(exist_ok=True)
    (fake_repo / "tests" / "test_deploy_6b.py").write_text("# stub\n")
    monkeypatch.setenv("ENTROPY_SERVICE_NAME", "mr-trading.service")
    t0 = time.time()
    rc = d6b.main(["--health-check"])
    assert time.time() - t0 < 3.0
    assert rc == 0


# ---------- lineage append integration ----------

def test_lineage_append_success_adds_entry(fake_repo, mock_git_ok):
    """With the real config_lineage_init.py copied into the fake repo,
    append_lineage_on_success should add a live 6b entry to the lineage
    file."""
    # Seed the init module + lineage state in the fake repo
    src = ROOT / "algo" / "dashboard" / "config_lineage_init.py"
    dst_dir = fake_repo / "algo" / "dashboard"
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst_dir / "config_lineage_init.py")

    # Pre-seed the lineage file with a queued 6b entry so the append
    # flips queued -> live.
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

    facts = {"head_sha": "abc1234567890abcdef"}
    ok, msg = d6b.append_lineage_on_success(fake_repo / "state", facts)
    assert ok, f"append failed: {msg}"

    entries = [json.loads(ln) for ln in lineage.read_text().splitlines() if ln.strip()]
    six_b = [e for e in entries if e["version_label"] == "6b PATH A"]
    assert len(six_b) == 1
    assert six_b[0]["status"] == "live"
    assert six_b[0]["deployed_sha"] == "abc1234"
    # previous 'live' entry (6a) flipped to 'historical'
    six_a = [e for e in entries if e["version_label"] == "6a observer"]
    assert six_a[0]["status"] == "historical"


def test_lineage_append_failure_writes_error_but_deploy_succeeds(
        fake_repo, mock_git_ok, monkeypatch):
    """If the init module is MISSING, append_lineage_on_success must
    return (False, msg) AND write state/lineage_append_failed.json, AND
    the overall --execute return code must still be 0."""
    # Ensure the init file is NOT present (fake_repo already has no
    # algo/dashboard/ by default).
    init = fake_repo / "algo" / "dashboard" / "config_lineage_init.py"
    assert not init.exists()

    # Seed banner so the execute path gets past verify_startup_banner.
    banner = ("2026-04-22 06:00:01 INFO ETH: "
              "f3c_enabled=False timeout_trail_enabled=True "
              "e3_time_decay_sl_enabled=True leverage=5x")
    (fake_repo / "algo" / "logs" / "entropy_multi.log").write_text(banner + "\n")
    monkeypatch.setattr(d6b, "BANNER_POLL_SEC", 2)
    monkeypatch.setattr(d6b, "AUDIT_POLL_SEC", 1)
    monkeypatch.setattr(d6b, "STARTUP_WAIT_SEC", 2)

    rc = d6b.main(["--execute", "--confirm", d6b.EXECUTE_CONFIRM])
    # Deploy itself succeeded; lineage append is non-fatal.
    assert rc == 0

    # Deploy record written
    assert (fake_repo / "state" / "deploy_6b_complete.json").exists()

    # Lineage-failure marker written with traceback + intended entry
    fail = fake_repo / "state" / "lineage_append_failed.json"
    assert fail.exists()
    payload = json.loads(fail.read_text())
    assert "exception" in payload
    assert "traceback" in payload
    assert payload["intended_entry"]["version_label"] == "6b PATH A"
    assert payload["intended_entry"]["status"] == "live"
