"""Throwaway simulation: instantiate LiveDriftMonitor against the real
shadow_expectation.json (in a temp dir so we do not touch state/),
feed 10 synthetic trades chosen to drive cum_z below -6.32, verify
idempotency, print everything, clean up.

After this script exits:
  - state/safety_valve.json  -- must NOT exist
  - state/live_drift_monitor.json  -- must NOT exist
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "algo"))
from live_drift_monitor import LiveDriftMonitor


REAL_SHADOW = ROOT / "algo" / "state" / "shadow_expectation.json"
REAL_STATE_DIR = ROOT / "algo" / "state"


def main():
    if not REAL_SHADOW.exists():
        print(f"ERROR: {REAL_SHADOW} not found. Run shadow_expectation_generator "
              f"first (without --dry-run).", file=sys.stderr)
        sys.exit(1)

    shadow = json.loads(REAL_SHADOW.read_text())
    print(f"Using shadow_expectation from: {REAL_SHADOW}")
    print(f"  Generated: {shadow.get('generated_at', '?')}")
    print(f"  Total trades in expectation: {shadow.get('n_trades_total', '?')}\n")

    # ---- Run simulation in an isolated temp state dir ----
    with tempfile.TemporaryDirectory() as td:
        tmp_state = Path(td) / "state"
        tmp_state.mkdir()
        (tmp_state / "shadow_expectation.json").write_text(
            REAL_SHADOW.read_text())

        m = LiveDriftMonitor(tmp_state)
        threshold = m.threshold
        print(f"Monitor:")
        print(f"  window_size   = {m.window_size}")
        print(f"  threshold     = {threshold:.4f} (= -2*sqrt({m.window_size}))")

        # Resolve which bucket we will drive. Use long at asia hours.
        # long bucket's shadow stats -- pull from fallback since long_asia
        # may have count<5 on a 21-day baseline.
        stats, used_bucket = m._resolve_stats(1, "2026-04-20T03:00:00+00:00")
        if stats is None:
            print("ERROR: could not resolve stats for long + asia")
            sys.exit(2)
        mean_bps = stats["mean_pnl_bps"]; std_bps = stats["std_bps"]
        print(f"\nResolved bucket for drive: {used_bucket}")
        print(f"  mean_pnl_bps = {mean_bps:+.2f}")
        print(f"  std_bps      = {std_bps:.2f}\n")

        # Target each trade at z = -1.0 so cum_z over 10 trades = -10 < -6.32.
        # Solve pnl_bps for z = -1.0: pnl = mean - std.
        target_z = -1.0
        pnl_bps  = mean_bps + target_z * std_bps
        print(f"Synthetic trade pnl_bps: {pnl_bps:+.2f} "
              f"(targeting z = {target_z:.2f} each)\n")

        # Feed 10 synthetic trades
        print(f"{'#':>2}  {'z':>7}  {'cum_z':>8}  {'fired':>6}")
        for i in range(10):
            trade = {
                "pnl_bps": pnl_bps,
                "direction": 1,
                "time": f"2026-04-20T0{(i % 8):d}:00:00+00:00",  # all asia
                "order_id": f"sim-{i:02d}",
            }
            r = m.check_trade(trade)
            z = r["z"]; cum_z = r["cum_z"]; fired = r["fired"]
            print(f"{i+1:>2}  {z:>+7.2f}  {cum_z:>+8.2f}  {str(fired):>6}")

        # Verify state after 10 trades
        valve_path = tmp_state / "safety_valve.json"
        assert valve_path.exists(), "Valve should have been written on trade 10"
        marker = json.loads(valve_path.read_text())
        print(f"\nValve tripped:")
        print(f"  tripped_at    = {marker['auto_disabled_at']}")
        print(f"  cumulative_z  = {marker['cumulative_z']:.2f}")
        print(f"  threshold     = {marker['threshold']:.2f}")
        print(f"  n z-scores    = {len(marker['z_scores_at_trip'])}")
        print(f"  flags_disabled= {marker['disabled_flags']}")
        print(f"  leverage_to   = {marker['leverage_reverted_to']}")

        # Idempotency: simulate bot restart (new instance) + another trade
        print(f"\nIdempotency check:")
        mtime_before = valve_path.stat().st_mtime
        marker_text_before = valve_path.read_text()
        m2 = LiveDriftMonitor(tmp_state)
        assert m2.is_disabled() is True, "New instance should see existing valve"
        r_post1 = m2.check_trade({
            "pnl_bps": -200, "direction": 1,
            "time": "2026-04-20T04:00:00+00:00",
            "order_id": "post-disable-1",
        })
        r_post2 = m2.check_trade({
            "pnl_bps": +200, "direction": 1,
            "time": "2026-04-20T05:00:00+00:00",
            "order_id": "post-disable-2",
        })
        print(f"  after-disable trade 1 -> fired={r_post1['fired']}, "
              f"reason={r_post1['reason']}")
        print(f"  after-disable trade 2 -> fired={r_post2['fired']}, "
              f"reason={r_post2['reason']}")
        mtime_after = valve_path.stat().st_mtime
        marker_text_after = valve_path.read_text()
        assert r_post1["fired"] is False and r_post1["reason"] == "already_disabled"
        assert r_post2["fired"] is False and r_post2["reason"] == "already_disabled"
        assert marker_text_before == marker_text_after, "Marker was rewritten!"
        assert mtime_before == mtime_after, "Marker file was touched!"
        print(f"  marker bytes unchanged = {marker_text_before == marker_text_after}")
        print(f"  mtime unchanged        = {mtime_before == mtime_after}")

        # Temp dir auto-cleans when context exits

    # Confirm we did not pollute the real state/ directory
    real_valve = REAL_STATE_DIR / "safety_valve.json"
    real_drift = REAL_STATE_DIR / "live_drift_monitor.json"
    if real_valve.exists():
        print(f"\nCLEANUP WARNING: {real_valve} exists; unlinking")
        real_valve.unlink()
    if real_drift.exists():
        print(f"CLEANUP WARNING: {real_drift} exists; unlinking")
        real_drift.unlink()
    print(f"\nPost-run check of {REAL_STATE_DIR}:")
    print(f"  safety_valve.json exists       : {real_valve.exists()}")
    print(f"  live_drift_monitor.json exists : {real_drift.exists()}")
    print("\nAll assertions passed.")


if __name__ == "__main__":
    main()
