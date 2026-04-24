"""
D1 -- Regime-gate replay on the 17 live ETH trades.

For each live entry, we need >=150 min of 1-min close bars ending at
entry to compute EMA(30), EMA(150), ATR(30).

Data strategy (per the prompt):
  1. Prefer state/trade_charts.json if its stored bars extend far enough
     back from entry. Stored captures only cover ~1 h pre-entry so this
     rarely suffices by itself.
  2. Fall back to Kraken public OHLC live. 1-min series covers ~12 h
     back from now; 15-min covers ~7.5 days. For trades older than 12 h
     we use 15-min bars and compute EMAs/ATR in window-equivalent units:
         30 min   -> EMA_2 on 15-min bars
        150 min   -> EMA_10 on 15-min bars
         ATR_30m  -> ATR_2 on 15-min bars
  3. If neither path yields at least 10 pre-entry bars at the chosen
     interval, mark the trade "insufficient data" and list it
     separately (not counted in the block tally).

Outputs:
  - reports/d1_regime_gate_replay.json
  - reports/d1_regime_gate_replay.csv
"""
import csv, json, os, sys, time, urllib.request
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

HERE = Path(__file__).resolve().parent
ALGO = HERE.parent
ROOT = ALGO.parent

HIST = ROOT / "reports_out" / "trade_history.jsonl"
CHARTS = ROOT / "reports_out" / "trade_charts.json"
OUT_JSON = ALGO / "reports" / "d1_regime_gate_replay.json"
OUT_CSV = ALGO / "reports" / "d1_regime_gate_replay.csv"


def ema(arr, n):
    """Standard EMA; returns full series same length as arr."""
    a = np.asarray(arr, dtype=float)
    if len(a) == 0: return a
    alpha = 2.0 / (n + 1.0)
    out = np.empty_like(a)
    out[0] = a[0]
    for i in range(1, len(a)):
        out[i] = alpha * a[i] + (1 - alpha) * out[i-1]
    return out


def atr(high, low, close, n):
    """ATR using Wilder smoothing; returns same length as inputs."""
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    n_bars = len(h)
    if n_bars == 0: return np.zeros(0)
    tr = np.zeros(n_bars)
    tr[0] = h[0] - l[0]
    for i in range(1, n_bars):
        tr[i] = max(h[i] - l[i],
                    abs(h[i] - c[i-1]),
                    abs(l[i] - c[i-1]))
    out = np.zeros(n_bars)
    # initial simple average
    if n_bars <= n:
        return np.cumsum(tr) / np.maximum(np.arange(1, n_bars+1), 1)
    out[:n] = tr[:n].mean()
    for i in range(n, n_bars):
        out[i] = (out[i-1] * (n - 1) + tr[i]) / n
    return out


def fetch_kraken(interval_min, since_sec, max_attempts=4):
    url = (f"https://api.kraken.com/0/public/OHLC?"
           f"pair=ETHUSD&interval={interval_min}&since={since_sec}")
    delay = 2.0
    for _ in range(max_attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            resp = urllib.request.urlopen(req, timeout=20)
            data = json.loads(resp.read())
            err = data.get("error")
            if err and any("Too many" in str(e) for e in err):
                time.sleep(delay); delay = min(delay*2, 20); continue
            if err:
                return None
            for k, v in data["result"].items():
                if k != "last":
                    return v
            return None
        except Exception:
            time.sleep(delay); delay = min(delay*2, 20)
    return None


def load_trades():
    rows = []
    for line in HIST.read_text().splitlines():
        line = line.strip()
        if line: rows.append(json.loads(line))
    return rows


def pick_interval_and_fetch(entry_ts_s, now_s):
    """Return (interval_min, window_bars) with >=10 bars before entry.
    Tries 1-min (covers last 12 h), then 5-min (2.5 d), then 15-min (7.5 d)."""
    age_h = (now_s - entry_ts_s) / 3600
    candidates = []
    if age_h < 11:    candidates.append(1)
    if age_h < 55:    candidates.append(5)
    if age_h < 170:   candidates.append(15)
    if not candidates: candidates = [60]  # 30-day range
    for interval in candidates:
        # need at least 150 min of history + some buffer
        need_bars = max(15, int(150 / interval) + 2)
        since = int(entry_ts_s - (need_bars + 30) * interval * 60)
        bars = fetch_kraken(interval, since)
        time.sleep(1.1)
        if not bars: continue
        # bars is list of [ts, o, h, l, c, vwap, vol, count]
        pre = [b for b in bars if int(b[0]) <= entry_ts_s]
        if len(pre) >= min(10, need_bars - 5):
            return interval, bars
    return None, None


def compute_regime(interval_min, bars, entry_ts_s):
    """Compute EMA fast / slow and ATR at the bar containing entry."""
    # pre = bars with ts <= entry_ts
    pre = [b for b in bars if int(b[0]) <= entry_ts_s]
    if len(pre) < 3:
        return None
    closes = np.asarray([float(b[4]) for b in pre])
    highs  = np.asarray([float(b[2]) for b in pre])
    lows   = np.asarray([float(b[3]) for b in pre])
    # Window equivalents
    n_fast = max(2, round(30  / interval_min))
    n_slow = max(3, round(150 / interval_min))
    n_atr  = max(2, round(30  / interval_min))
    ema_f = ema(closes, n_fast)[-1]
    ema_s = ema(closes, n_slow)[-1]
    atr_v = atr(highs, lows, closes, n_atr)[-1]
    # Convert ATR (price units) to bps of entry price for comparison
    entry_px = closes[-1]
    atr_bps = (atr_v / entry_px) * 10000 if entry_px > 0 else 0
    return {
        "interval_min": interval_min,
        "n_pre_bars": len(pre),
        "n_fast": n_fast, "n_slow": n_slow, "n_atr": n_atr,
        "ema_fast": float(ema_f),
        "ema_slow": float(ema_s),
        "ema_diff": float(ema_f - ema_s),
        "ema_diff_normalized": float((ema_f - ema_s) / atr_v) if atr_v > 0 else None,
        "atr": float(atr_v),
        "atr_bps": float(atr_bps),
    }


def main():
    trades = load_trades()
    print(f"Loaded {len(trades)} live trades", flush=True)

    now_s = time.time()
    rows = []
    for i, t in enumerate(trades, 1):
        exit_ts = datetime.fromisoformat(t["time"]).timestamp()
        entry_ts = exit_ts - t["hold_min"] * 60
        print(f"[{i}/{len(trades)}] {datetime.fromtimestamp(entry_ts, tz=timezone.utc):%m-%d %H:%M} "
              f"age={(now_s-entry_ts)/3600:.1f}h", flush=True)
        interval, bars = pick_interval_and_fetch(entry_ts, now_s)
        if not bars:
            print("    insufficient data", flush=True)
            rows.append({
                "trade_idx": i,
                "entry_ts": int(entry_ts),
                "entry_iso": datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                "direction": t["direction"],
                "pnl_bps_lev": t["pnl_bps_leveraged"],
                "reason": t["reason"],
                "status": "insufficient_data",
            })
            continue
        reg = compute_regime(interval, bars, entry_ts)
        if reg is None:
            rows.append({
                "trade_idx": i, "entry_ts": int(entry_ts),
                "direction": t["direction"],
                "pnl_bps_lev": t["pnl_bps_leveraged"], "reason": t["reason"],
                "status": "insufficient_data",
            })
            continue
        # Apply the Phase 2 F3 rule (symmetric for shorts; our trades are all LONG)
        d = t["direction"]
        if d == 1:
            blocked = reg["ema_fast"] < (reg["ema_slow"] - 0.5 * reg["atr"])
        else:
            blocked = reg["ema_fast"] > (reg["ema_slow"] + 0.5 * reg["atr"])
        rows.append({
            "trade_idx": i,
            "entry_ts": int(entry_ts),
            "entry_iso": datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
            "direction": d, "dir_label": "LONG" if d == 1 else "SHORT",
            "pnl_bps_gross": t["pnl_bps"],
            "pnl_bps_lev": t["pnl_bps_leveraged"],
            "reason": t["reason"],
            "hold_min": t["hold_min"],
            "status": "ok",
            "blocked_by_gate": bool(blocked),
            **reg,
        })

    # --- Summarize ---
    ok = [r for r in rows if r.get("status") == "ok"]
    insuff = [r for r in rows if r.get("status") == "insufficient_data"]
    blocked = [r for r in ok if r["blocked_by_gate"]]
    not_blocked = [r for r in ok if not r["blocked_by_gate"]]
    blocked_wins = [r for r in blocked if r["pnl_bps_lev"] > 0]
    blocked_loss = [r for r in blocked if r["pnl_bps_lev"] <= 0]
    nb_wins = [r for r in not_blocked if r["pnl_bps_lev"] > 0]
    nb_loss = [r for r in not_blocked if r["pnl_bps_lev"] <= 0]
    bps_saved_if_skipped_losers = sum(-r["pnl_bps_lev"] for r in blocked_loss)
    bps_lost_if_skipped_winners = sum(r["pnl_bps_lev"] for r in blocked_wins)
    net_impact = bps_saved_if_skipped_losers - bps_lost_if_skipped_winners

    summary = {
        "n_total": len(trades),
        "n_with_data": len(ok),
        "n_insufficient_data": len(insuff),
        "n_blocked_by_gate": len(blocked),
        "n_not_blocked": len(not_blocked),
        "blocked_wins_n": len(blocked_wins),
        "blocked_losses_n": len(blocked_loss),
        "not_blocked_wins_n": len(nb_wins),
        "not_blocked_losses_n": len(nb_loss),
        "bps_saved_if_blocked_losers_skipped":  float(bps_saved_if_skipped_losers),
        "bps_lost_if_blocked_winners_skipped":  float(bps_lost_if_skipped_winners),
        "net_bps_at_10x_if_gate_enforced":      float(net_impact),
        "decision_band": (
            "F3 mandatory-on" if len(blocked) >= 10
            else ("F3 toggleable (in 4-9 band)" if len(blocked) >= 4
                  else "F3 weaker than hypothesized (<4 blocks)")
        ),
    }

    out = {"summary": summary, "trades": rows,
           "generated_utc": datetime.now(timezone.utc).isoformat()}
    OUT_JSON.write_text(json.dumps(out, indent=2, default=str))

    # CSV
    with open(OUT_CSV, "w", newline="") as f:
        fields = ["trade_idx", "entry_iso", "dir_label", "blocked_by_gate",
                 "ema_fast", "ema_slow", "ema_diff", "atr", "atr_bps",
                 "ema_diff_normalized", "pnl_bps_gross", "pnl_bps_lev",
                 "reason", "hold_min", "interval_min", "status"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows: w.writerow(r)

    # Console
    print("\n=== SUMMARY ===")
    for k, v in summary.items(): print(f"  {k}: {v}")
    print(f"\nWrote {OUT_JSON}")
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
