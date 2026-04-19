"""Unit tests for LiveDriftMonitor."""
import json
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "algo"))
from live_drift_monitor import LiveDriftMonitor  # noqa: E402


SHADOW_FIXTURE = {
    "cell_name": "test",
    "generated_at": "2026-01-01T00:00:00+00:00",
    "buckets": {
        "long_asia":      {"count": 10, "mean_pnl_bps": 50.0, "std_bps": 100.0},
        "long_europe":    {"count":  3, "mean_pnl_bps": 30.0, "std_bps":  80.0},  # <5 -> fallback
        "long_americas":  {"count":  7, "mean_pnl_bps": 20.0, "std_bps":  60.0},
        "short_asia":     {"count":  2, "mean_pnl_bps":-10.0, "std_bps":  70.0},  # <5 -> fallback
        "short_europe":   {"count":  8, "mean_pnl_bps": 40.0, "std_bps":  90.0},
        "short_americas": {"count":  6, "mean_pnl_bps": 15.0, "std_bps":  75.0},
    },
    "fallback_by_direction": {
        "long":  {"count": 20, "mean_pnl_bps": 40.0, "std_bps": 100.0},
        "short": {"count": 16, "mean_pnl_bps": 20.0, "std_bps":  80.0},
    },
}


@pytest.fixture
def state_dir(tmp_path):
    d = tmp_path / "state"
    d.mkdir()
    (d / "shadow_expectation.json").write_text(json.dumps(SHADOW_FIXTURE))
    return d


@pytest.fixture
def empty_state_dir(tmp_path):
    """state dir without a shadow_expectation.json file."""
    d = tmp_path / "state_empty"
    d.mkdir()
    return d


def mk(pnl, direction=1, iso="2026-04-20T03:00:00+00:00", oid=None):
    return {"pnl_bps": pnl, "direction": direction, "time": iso,
            "order_id": oid or f"t{pnl}_{direction}_{iso[11:13]}"}


# ---- bucket classification ----

def test_bucket_sessions_long():
    boundaries = [
        ("2026-04-20T00:00:00+00:00", "asia"),
        ("2026-04-20T03:00:00+00:00", "asia"),
        ("2026-04-20T07:59:00+00:00", "asia"),
        ("2026-04-20T08:00:00+00:00", "europe"),
        ("2026-04-20T12:00:00+00:00", "europe"),
        ("2026-04-20T15:59:00+00:00", "europe"),
        ("2026-04-20T16:00:00+00:00", "americas"),
        ("2026-04-20T23:59:00+00:00", "americas"),
    ]
    for iso, expected in boundaries:
        primary, fb = LiveDriftMonitor.bucket_for(1, iso)
        assert primary == f"long_{expected}", (iso, primary)
        assert fb == "long"


def test_bucket_short_direction():
    primary, fb = LiveDriftMonitor.bucket_for(-1, "2026-04-20T09:00:00+00:00")
    assert primary == "short_europe"
    assert fb == "short"


def test_bucket_handles_offset_timezones():
    # +03:00 5am == 02:00 UTC => asia
    primary, _ = LiveDriftMonitor.bucket_for(1, "2026-04-20T05:00:00+03:00")
    assert primary == "long_asia"


# ---- z-score computation ----

def test_z_primary_bucket(state_dir):
    m = LiveDriftMonitor(state_dir)
    # long_asia: mean=50, std=100; pnl=150 => z = 1.0
    r = m.check_trade(mk(150, 1, "2026-04-20T03:00:00+00:00"))
    assert r["z"] == pytest.approx(1.0)
    assert r["bucket"] == "long_asia"
    assert r["fired"] is False
    assert r["reason"] is None


def test_z_fallback_when_primary_sparse(state_dir):
    # long_europe has count=3, falls back to long (mean=40, std=100); pnl=40 => z=0
    m = LiveDriftMonitor(state_dir)
    r = m.check_trade(mk(40, 1, "2026-04-20T10:00:00+00:00"))
    assert r["z"] == pytest.approx(0.0)
    assert "fallback" in r["bucket"]
    assert r["bucket"].startswith("long")


def test_z_short_fallback(state_dir):
    # short_asia has count=2, falls back to short (mean=20, std=80); pnl=-60 => z=-1
    m = LiveDriftMonitor(state_dir)
    r = m.check_trade(mk(-60, -1, "2026-04-20T04:00:00+00:00"))
    assert r["z"] == pytest.approx(-1.0)
    assert r["bucket"].startswith("short")


# ---- window management ----

def test_window_rolls_at_window_size(state_dir):
    m = LiveDriftMonitor(state_dir, window_size=10)
    for i in range(15):
        m.check_trade(mk(50, 1, "2026-04-20T03:00:00+00:00", oid=f"t{i}"))
    assert m.window_size_current() == 10


def test_window_not_full_no_trigger(state_dir):
    m = LiveDriftMonitor(state_dir)
    # 9 trades at z=-3.5 each; cum=-31.5 would trigger if window was full
    for i in range(9):
        r = m.check_trade(mk(-300, 1, "2026-04-20T03:00:00+00:00", oid=f"t{i}"))
        assert r["fired"] is False
    assert m.should_auto_disable() is False
    assert not (state_dir / "safety_valve.json").exists()


# ---- trigger condition ----

def test_valve_fires_when_cum_z_below_threshold(state_dir):
    m = LiveDriftMonitor(state_dir)
    # long_asia mean=50, std=100; pnl=-50 -> z=-1.0 each
    # cum over 10 = -10; threshold ~ -6.32 -> fire on the 10th
    for i in range(9):
        r = m.check_trade(mk(-50, 1, "2026-04-20T03:00:00+00:00", oid=f"t{i}"))
        assert r["fired"] is False
    r = m.check_trade(mk(-50, 1, "2026-04-20T03:00:00+00:00", oid="t9"))
    assert r["fired"] is True
    assert r["cum_z"] == pytest.approx(-10.0)
    valve_path = state_dir / "safety_valve.json"
    assert valve_path.exists()
    marker = json.loads(valve_path.read_text())
    assert marker["cumulative_z"] == pytest.approx(-10.0)
    assert len(marker["z_scores_at_trip"]) == 10
    assert "f3c_enabled" in marker["disabled_flags"]
    assert marker["leverage_reverted_to"] == 10


def test_valve_idempotent(state_dir):
    m = LiveDriftMonitor(state_dir)
    for i in range(10):
        m.check_trade(mk(-50, 1, "2026-04-20T03:00:00+00:00", oid=f"t{i}"))
    marker_before = (state_dir / "safety_valve.json").read_text()
    mtime_before = (state_dir / "safety_valve.json").stat().st_mtime
    # Fresh instance sees the marker; subsequent checks must not re-write
    m2 = LiveDriftMonitor(state_dir)
    r1 = m2.check_trade(mk(-80, 1, "2026-04-20T03:00:00+00:00", oid="post1"))
    r2 = m2.check_trade(mk(+80, 1, "2026-04-20T03:00:00+00:00", oid="post2"))
    assert r1["fired"] is False and r1["reason"] == "already_disabled"
    assert r2["fired"] is False and r2["reason"] == "already_disabled"
    marker_after = (state_dir / "safety_valve.json").read_text()
    mtime_after  = (state_dir / "safety_valve.json").stat().st_mtime
    assert marker_before == marker_after
    assert mtime_before == mtime_after


def test_positive_drift_does_not_trigger(state_dir):
    m = LiveDriftMonitor(state_dir)
    # pnl=+250 on long_asia (mean=50, std=100) -> z=+2 each
    for i in range(10):
        r = m.check_trade(mk(250, 1, "2026-04-20T03:00:00+00:00", oid=f"t{i}"))
        assert r["fired"] is False
    assert not (state_dir / "safety_valve.json").exists()
    assert m.cumulative_z() == pytest.approx(20.0)


def test_custom_threshold(state_dir):
    # Tighter threshold should trigger earlier
    m = LiveDriftMonitor(state_dir, window_size=10, threshold=-3.0)
    # z=-0.5 each; cum=-5.0 after 10; -5 < -3 -> fire
    for i in range(9):
        r = m.check_trade(mk(0, 1, "2026-04-20T03:00:00+00:00", oid=f"t{i}"))
        # long_asia: pnl=0 -> z = (0-50)/100 = -0.5
        assert r["fired"] is False
    r = m.check_trade(mk(0, 1, "2026-04-20T03:00:00+00:00", oid="t9"))
    assert r["fired"] is True
    assert r["cum_z"] == pytest.approx(-5.0)


# ---- persistence ----

def test_restore_window_across_instances(state_dir):
    m1 = LiveDriftMonitor(state_dir)
    for i in range(5):
        m1.check_trade(mk(100, 1, "2026-04-20T03:00:00+00:00", oid=f"t{i}"))
    # pnl=100 => z=(100-50)/100 = 0.5 each
    m2 = LiveDriftMonitor(state_dir)
    assert m2.window_size_current() == 5
    assert m2.cumulative_z() == pytest.approx(2.5)


def test_reset_window(state_dir):
    m = LiveDriftMonitor(state_dir)
    for i in range(5):
        m.check_trade(mk(100, 1, "2026-04-20T03:00:00+00:00", oid=f"t{i}"))
    m.reset_window()
    assert m.window_size_current() == 0
    assert not (state_dir / "live_drift_monitor.json").exists()


def test_clear_valve(state_dir):
    m = LiveDriftMonitor(state_dir)
    for i in range(10):
        m.check_trade(mk(-50, 1, "2026-04-20T03:00:00+00:00", oid=f"t{i}"))
    assert (state_dir / "safety_valve.json").exists()
    m.clear_valve()
    assert not (state_dir / "safety_valve.json").exists()


# ---- edge cases ----

def test_no_shadow_file_is_dormant(empty_state_dir):
    m = LiveDriftMonitor(empty_state_dir)
    r = m.check_trade(mk(50, 1))
    assert r["fired"] is False
    assert r["reason"] == "no_shadow_expectation"


def test_malformed_trade_returns_reason(state_dir):
    m = LiveDriftMonitor(state_dir)
    r = m.check_trade({"pnl_bps": 100})  # missing direction and time
    assert r["fired"] is False
    assert r["reason"] == "malformed_trade"


def test_default_threshold_is_neg_2_sqrt_n():
    assert LiveDriftMonitor.default_threshold(10) == pytest.approx(-6.3245553, abs=1e-5)
    assert LiveDriftMonitor.default_threshold(25) == pytest.approx(-10.0, abs=1e-5)
