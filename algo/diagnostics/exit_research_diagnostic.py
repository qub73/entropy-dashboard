"""Classify every live trade into the three failure modes:
  (a) exit-too-soon: trade exited in our favor, but price continued
      trending in the same direction past the exit (left money on table)
  (b) exit-at-bottom: trade exited at a loss (SL or pst_floored or
      E3-tightened SL), but price recovered in our favor within N bars
  (c) stuck-in-stale: trade held long (>=180 bars) and exited
      flat-to-negative, with no meaningful MFE (peak<+30bps)
Uses 1-min Binance OHLC (cached from earlier ablation runs)."""
from __future__ import annotations
import datetime as dt, json, sys, time, urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

TRADES_FILE = Path("C:/tmp/6a_smoke_trades.jsonl")
CACHE_DIR = Path("C:/tmp/ohlc_cache"); CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT = Path("algo/reports/exit_diagnostic.json")

FORWARD_MIN = 180   # look 3h past exit
RECOVERY_BARS = 120 # (b) threshold: price reverses within 120 min
STALL_MIN_HOLD = 180
STALL_MAX_PEAK_BPS = 30


def fetch_binance_1m(start_ms, end_ms, symbol="ETHUSDT"):
    cache = CACHE_DIR / f"binance_{symbol}_{start_ms}_{end_ms}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    url = (f"https://api.binance.com/api/v3/klines?symbol={symbol}"
           f"&interval=1m&startTime={start_ms}&endTime={end_ms}&limit=1000")
    req = urllib.request.Request(url, headers={"User-Agent": "exit-diag"})
    for a in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                d = json.loads(r.read())
            break
        except Exception:
            if a == 2: raise
            time.sleep(2 * (a+1))
    cache.write_text(json.dumps(d))
    return d


def get_bars(entry_iso, hold_min, forward_min=FORWARD_MIN):
    entry_dt = dt.datetime.fromisoformat(entry_iso.replace("Z","+00:00"))
    start_ms = int((entry_dt - dt.timedelta(minutes=15)).timestamp() * 1000)
    end_ms = int((entry_dt + dt.timedelta(minutes=hold_min + forward_min))
                 .timestamp() * 1000)
    all_bars = []
    cursor = start_ms
    while cursor < end_ms:
        chunk_end = min(cursor + 60*60*1000*16, end_ms)
        chunk = fetch_binance_1m(cursor, chunk_end)
        if not chunk: break
        all_bars.extend(chunk)
        last = int(chunk[-1][0])
        if last >= end_ms: break
        cursor = last + 60000
        time.sleep(0.15)
    bars = []
    for b in all_bars:
        bars.append({"ts": int(b[0])//1000, "o":float(b[1]), "h":float(b[2]),
                     "l":float(b[3]), "c":float(b[4])})
    uniq = {}
    for x in bars: uniq[x["ts"]] = x
    return sorted(uniq.values(), key=lambda x: x["ts"])


def classify_trade(t, bars):
    direction = int(t["direction"])
    entry_price = float(t["entry_price"])
    exit_price = float(t["exit_price"])
    pnl_bps = float(t["pnl_bps"])
    hold_min = float(t["hold_min"])
    exit_dt = dt.datetime.fromisoformat(t["time"].replace("Z","+00:00"))
    entry_dt = exit_dt - dt.timedelta(minutes=hold_min)
    entry_ts = int(entry_dt.timestamp())
    exit_ts = int(exit_dt.timestamp())

    # Find peak bps during hold + max favorable after exit within forward window
    mfe_during_hold = -1e9; mae_during_hold = 1e9
    for b in bars:
        if b["ts"] < entry_ts or b["ts"] > exit_ts: continue
        if direction == 1:
            fav = (b["h"]/entry_price - 1) * 10000
            adv = (b["l"]/entry_price - 1) * 10000
        else:
            fav = -((b["l"]/entry_price - 1) * 10000)
            adv = -((b["h"]/entry_price - 1) * 10000)
        mfe_during_hold = max(mfe_during_hold, fav)
        mae_during_hold = min(mae_during_hold, adv)

    # Forward path past exit
    forward_fav = -1e9
    forward_reverse_within_120 = False
    forward_fav_at_120 = None
    for b in bars:
        if b["ts"] <= exit_ts: continue
        if b["ts"] > exit_ts + FORWARD_MIN*60: break
        if direction == 1:
            fav_from_exit = (b["h"]/exit_price - 1) * 10000
            fav_from_entry = (b["h"]/entry_price - 1) * 10000
        else:
            fav_from_exit = -((b["l"]/exit_price - 1) * 10000)
            fav_from_entry = -((b["l"]/entry_price - 1) * 10000)
        forward_fav = max(forward_fav, fav_from_exit)
        # For (b): if exit was a loss, check whether price recovered BEYOND entry
        if pnl_bps <= 0:
            minutes_since_exit = (b["ts"] - exit_ts) / 60
            if minutes_since_exit <= RECOVERY_BARS and fav_from_entry > 20:
                forward_reverse_within_120 = True
                if forward_fav_at_120 is None or fav_from_entry > forward_fav_at_120:
                    forward_fav_at_120 = fav_from_entry

    # Classify
    modes = []
    # (a) exit-too-soon: trade profitable, but price continued favorably
    # after exit by >= 50 bps. Signals tp_trail / trail_stop fired early.
    if pnl_bps > 20 and forward_fav >= 50:
        modes.append("a_too_soon")
    # (b) exit-at-bottom: trade losing, price recovered past entry within 120m
    if pnl_bps <= 0 and forward_reverse_within_120:
        modes.append("b_at_bottom")
    # (c) stuck-in-stale: held >=180min, flat-to-negative outcome, peak<+30
    if hold_min >= STALL_MIN_HOLD and pnl_bps <= 30 and mfe_during_hold < STALL_MAX_PEAK_BPS:
        modes.append("c_stale")
    if not modes:
        modes.append("clean")

    return {
        "order_id": t.get("order_id",""),
        "entry_ts_utc": entry_dt.isoformat(),
        "direction": direction,
        "hold_min": hold_min,
        "reason_live": t["reason"],
        "pnl_bps_live": pnl_bps,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "mfe_during_hold_bps": round(mfe_during_hold, 1),
        "mae_during_hold_bps": round(mae_during_hold, 1),
        "forward_fav_from_exit_bps": round(forward_fav, 1),
        "forward_recovery_to_entry": forward_reverse_within_120,
        "forward_fav_at_recovery_bps": round(forward_fav_at_120, 1) if forward_fav_at_120 else None,
        "failure_modes": modes,
    }


def main():
    trades = [json.loads(l) for l in TRADES_FILE.open()]
    print(f"Classifying {len(trades)} trades...")
    rows = []
    for i, t in enumerate(trades, 1):
        exit_dt = dt.datetime.fromisoformat(t["time"].replace("Z","+00:00"))
        entry_dt = exit_dt - dt.timedelta(minutes=float(t["hold_min"]))
        try:
            bars = get_bars(entry_dt.isoformat(), t["hold_min"])
        except Exception as e:
            print(f"  [{i}] fetch failed: {e}")
            continue
        if len(bars) < 10:
            print(f"  [{i}] only {len(bars)} bars, skip")
            continue
        row = classify_trade(t, bars)
        rows.append(row)
        print(f"  [{i:>2}/{len(trades)}] {row['direction']:+d} "
              f"hold={row['hold_min']:.0f}m  live={row['reason_live']:<12} "
              f"pnl={row['pnl_bps_live']:+6.1f}  MFE_hold={row['mfe_during_hold_bps']:+5.1f}  "
              f"fwd_fav_from_exit={row['forward_fav_from_exit_bps']:+5.1f}  "
              f"recover={row['forward_recovery_to_entry']:<5}  modes={row['failure_modes']}")

    # Aggregate
    counts = Counter()
    for r in rows:
        for m in r["failure_modes"]: counts[m] += 1
    net_modes = Counter()
    for r in rows:
        key = "+".join(r["failure_modes"])
        net_modes[key] += 1

    # Money-on-the-table estimates
    a_loss = sum(r["forward_fav_from_exit_bps"] for r in rows if "a_too_soon" in r["failure_modes"])
    b_missed = sum(r["forward_fav_at_recovery_bps"] or 0
                    for r in rows if "b_at_bottom" in r["failure_modes"])
    c_realized_loss = sum(r["pnl_bps_live"] for r in rows if "c_stale" in r["failure_modes"])

    summary = {
        "n_trades": len(rows),
        "mode_counts": dict(counts),
        "combined_mode_freq": dict(net_modes),
        "money_on_the_table_estimates_bps": {
            "a_forward_fav_missed": round(a_loss, 1),
            "b_recovery_missed": round(b_missed, 1),
            "c_slow_bleed_realized": round(c_realized_loss, 1),
        },
        "trades": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2, default=str))

    print(f"\n=== summary ===")
    print(f"  (a) exit-too-soon:  {counts['a_too_soon']} trades, "
          f"~{a_loss:.0f}bps left on table")
    print(f"  (b) exit-at-bottom: {counts['b_at_bottom']} trades, "
          f"~{b_missed:.0f}bps missed recovery")
    print(f"  (c) stuck-stale:    {counts['c_stale']} trades, "
          f"{c_realized_loss:.0f}bps realized from this mode")
    print(f"  clean (no mode):    {counts.get('clean',0)} trades")
    print(f"\ncombined-mode frequency:")
    for k, v in sorted(net_modes.items(), key=lambda x:-x[1]):
        print(f"  {k}: {v}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
