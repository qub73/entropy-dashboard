#!/usr/bin/env python3
"""
Publish bot status to a JSON file for the dashboard.
Run via cron every minute on the Pi.
Reads the bot's state file + latest log lines, writes status.json.
Optionally pushes to a GitHub gist for remote access.
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime, timezone

STATE_DIR = Path(__file__).parent / "state"
LOG_DIR = Path(__file__).parent / "logs"
# Use local_run.log if it exists (local mode), else service.log (Pi)
LOG_FILE = LOG_DIR / "local_run.log" if (LOG_DIR / "local_run.log").exists() else LOG_DIR / "service.log"
OUTPUT = Path(__file__).parent / "status.json"


def get_latest_status_from_log():
    """Parse last STATUS line from log."""
    if not LOG_FILE.exists():
        return {}
    try:
        # Read last ~100KB of file to find recent STATUS line
        with open(LOG_FILE, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 100_000))
            chunk = f.read().decode("utf-8", errors="ignore")
        lines = [l for l in chunk.split("\n") if "STATUS: " in l]
        if lines:
            last = lines[-1]
            idx = last.find('STATUS: ')
            if idx >= 0:
                return json.loads(last[idx + 8:])
    except Exception as e:
        print(f"Log read error: {e}")
    return {}


def get_service_uptime():
    """Get systemd service uptime."""
    try:
        result = subprocess.run(
            ["systemctl", "show", "entropy-trader", "--property=ActiveEnterTimestamp"],
            capture_output=True, text=True, timeout=5)
        line = result.stdout.strip()
        if "=" in line:
            ts_str = line.split("=", 1)[1].strip()
            if ts_str:
                # Parse systemd timestamp
                from email.utils import parsedate_to_datetime
                try:
                    start = datetime.fromisoformat(ts_str.replace(" ", "T").split(".")[0])
                    delta = datetime.now() - start
                    hours = int(delta.total_seconds() // 3600)
                    mins = int((delta.total_seconds() % 3600) // 60)
                    return f"{hours}h {mins}m"
                except Exception:
                    return ts_str
    except Exception:
        pass
    return "--"


def load_full_history():
    """Read every trade ever recorded from the permanent append-only log."""
    hist_file = STATE_DIR / "trade_history.jsonl"
    if not hist_file.exists():
        return []
    out = []
    try:
        with open(hist_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    except Exception:
        pass
    return out


def compute_cumulative(history):
    if not history:
        return {"trades": 0, "wins": 0, "losses": 0, "win_rate": 0,
                "compound_return_bps_lev": 0, "compound_return_pct": 0,
                "sum_pnl_bps_lev": 0, "mean_bps_lev": 0,
                "best_bps_lev": 0, "worst_bps_lev": 0,
                "max_dd_pct": 0,
                "tp": 0, "sl": 0, "timeout": 0, "other": 0,
                "long_trades": 0, "long_wr": 0,
                "short_trades": 0, "short_wr": 0,
                "since": ""}
    pnls = [t.get("pnl_bps_leveraged", 0) for t in history]
    wins = sum(1 for p in pnls if p > 0)
    losses = sum(1 for p in pnls if p < 0)
    reasons = {"tp": 0, "sl": 0, "timeout": 0, "other": 0, "manual_close": 0}
    for t in history:
        r = t.get("reason", "other")
        reasons[r] = reasons.get(r, 0) + 1
    longs = [t for t in history if t.get("direction") == 1]
    shorts = [t for t in history if t.get("direction") == -1]
    lw = sum(1 for t in longs if t.get("pnl_bps_leveraged", 0) > 0)
    sw = sum(1 for t in shorts if t.get("pnl_bps_leveraged", 0) > 0)

    # Compound equity curve (treating each pnl_bps_leveraged as % equity return)
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for p in pnls:
        equity *= (1.0 + p / 10000.0)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
    compound_ret = equity - 1.0  # fractional return

    return {
        "trades": len(history),
        "wins": wins, "losses": losses,
        "win_rate": wins / len(history) if history else 0,
        # Compound return -- correct way to express cumulative P&L
        "compound_return_bps_lev": compound_ret * 10000,
        "compound_return_pct": compound_ret * 100,
        "max_dd_pct": max_dd * 100,
        # Arithmetic sum -- kept for backwards compat, but not a true return
        "sum_pnl_bps_lev": sum(pnls),
        "mean_bps_lev": sum(pnls) / len(pnls) if pnls else 0,
        "best_bps_lev": max(pnls) if pnls else 0,
        "worst_bps_lev": min(pnls) if pnls else 0,
        "tp": reasons.get("tp", 0),
        "sl": reasons.get("sl", 0),
        "timeout": reasons.get("timeout", 0),
        "manual_close": reasons.get("manual_close", 0),
        "other": reasons.get("other", 0),
        "long_trades": len(longs),
        "long_wr": lw / len(longs) if longs else 0,
        "short_trades": len(shorts),
        "short_wr": sw / len(shorts) if shorts else 0,
        "since": history[0].get("time", ""),
    }


def build_status():
    """Build full status JSON."""
    # Read state file
    state_file = STATE_DIR / "multi_trader_state.json"
    state = {}
    if state_file.exists():
        with open(state_file) as f:
            state = json.load(f)

    # Full history (cumulative, never truncated)
    full_history = load_full_history()
    cumulative = compute_cumulative(full_history)

    # Get latest STATUS from log
    log_status = get_latest_status_from_log()

    # Merge
    status = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "last_update": time.time(),
        "uptime": get_service_uptime(),
        "position": state.get("position"),
        "trades": full_history[-50:],  # show last 50 on dashboard (persistent)
        "total_trades": len(full_history),
        "daily_pnl": state.get("daily_pnl", 0),
        "cumulative": cumulative,
        "signals": log_status.get("signals", {}),
        # BTC field intentionally omitted (BTC pair disabled)
        "ETH": log_status.get("ETH", {}),
    }

    # Get equity from account if possible
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).parent.parent / ".env")
        from kraken.futures import User
        u = User(
            key=os.getenv("KRAKEN_API_FUTURES_KEY"),
            secret=os.getenv("KRAKEN_API_FUTURES_SECRET"))
        accts = u.get_wallets()
        flex = accts.get("accounts", {}).get("flex", {})
        status["equity"] = flex.get("portfolioValue", 0)
        status["available_margin"] = flex.get("availableMargin", 0)
    except Exception:
        pass

    return status


def push_to_gist(status_json, gist_id=None):
    """Push status.json to a GitHub gist (requires gh CLI authenticated)."""
    if not gist_id:
        gist_id = os.getenv("GIST_ID", "")
    if not gist_id:
        return

    try:
        # Write temp file
        tmp = "/tmp/entropy_status.json"
        with open(tmp, "w") as f:
            json.dump(status_json, f, indent=2)

        subprocess.run(
            ["gh", "gist", "edit", gist_id, "-f", "status.json", tmp],
            capture_output=True, text=True, timeout=15)
    except Exception:
        pass


if __name__ == "__main__":
    status = build_status()

    # Write local file
    with open(OUTPUT, "w") as f:
        json.dump(status, f, indent=2)

    # Push to gist if configured
    gist_id = os.getenv("GIST_ID", "")
    if gist_id:
        push_to_gist(status, gist_id)

    print(json.dumps(status, indent=2)[:300])
