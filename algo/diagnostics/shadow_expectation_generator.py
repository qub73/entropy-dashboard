"""
Shadow expectation generator (Phase 6 pre-promote item A).

Runs the promoted Phase-5 config against the Feb 18 - Apr 7 Pi baseline
and emits a lookup table of expected PnL statistics per trade bucket.
The live auto-flip monitor (Item B) uses this file to score each live
trade's z against its shadow expectation.

Promoted config:
  - F3c entry gate (threshold +0.3)
  - timeout_trail (post_signal_trail on timeout-at-loss)
  - E3 time-decayed SL (tighten to -25 at 60m if peak < +50)
  - 5x leverage (*note*: pnl_bps is leverage-invariant; leverage only
    affects USD scaling, so shadow expectation is reported in pnl_bps
    to work across any leverage)

Buckets: (direction, session_utc) where session_utc is one of
  asia   (00:00-07:59 UTC)
  europe (08:00-15:59 UTC)
  americas (16:00-23:59 UTC)

This gives 6 primary buckets plus 2 fallback buckets (by direction only)
for sparse primary cells.

Output: state/shadow_expectation.json
"""
import json, os, sys
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

HERE = Path(__file__).resolve().parent
ALGO = HERE.parent
sys.path.insert(0, str(ALGO))

from ob_entropy import (
    NUM_STATES_OB, rolling_entropy_ob, load_orderbook_range,
    compute_ob_features, classify_ob_states,
)
from kaggle_ob_trainer import resample_pi_to_1min
from upgrade_backtest import make_features, candidate_signals

STATE_DIR = ALGO / "state"
OUT = STATE_DIR / "shadow_expectation.json"

H_THRESH = 0.4352
CORE_PARAMS = {"imb_min": 0.05, "spread_max": 20, "ret_low": 20, "ret_high": 80}
PST_WIDTH_BPS = 20
PST_HARD_FLOOR = -60
PST_MAX_WAIT = 30


def ema_series(arr, n):
    a = np.asarray(arr, dtype=float)
    alpha = 2.0 / (n + 1.0)
    out = np.empty_like(a); out[0] = a[0]
    for i in range(1, len(a)):
        out[i] = alpha * a[i] + (1 - alpha) * out[i-1]
    return out


def wilder_atr(highs, lows, closes, n):
    h = np.asarray(highs); l = np.asarray(lows); c = np.asarray(closes)
    tr = np.zeros(len(h)); tr[0] = h[0] - l[0]
    for i in range(1, len(h)):
        tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
    out = np.zeros(len(h))
    if len(h) <= n:
        return np.cumsum(tr) / np.maximum(np.arange(1, len(h)+1), 1)
    out[:n] = tr[:n].mean()
    for i in range(n, len(h)):
        out[i] = (out[i-1]*(n-1) + tr[i]) / n
    return out


def session_from_ts(ts_utc):
    """Classify a UTC timestamp into asia|europe|americas by hour-of-day."""
    if isinstance(ts_utc, (int, float)):
        dt = datetime.fromtimestamp(ts_utc, tz=timezone.utc)
    else:
        dt = ts_utc
    h = dt.hour
    if 0 <= h < 8: return "asia"
    if 8 <= h < 16: return "europe"
    return "americas"


def simulate_promoted(feats, ts_ms):
    """Simulate the promoted Phase-5 config on Pi data, emit per-trade records.

    Records include pnl_bps (leverage-invariant), direction, and entry-time
    session so the caller can bucket them. `ts_ms` is an array of unix ms
    timestamps aligned with bars.
    """
    n = feats['n']
    mids = feats['mid']; highs = feats['high']; lows = feats['low']
    atr_bps_arr = feats['atr_bps']; imb = feats['imb5']
    dH_5 = feats['dH_5']; ret_60 = feats['ret_60']
    ret_150 = np.zeros(n); ret_150[150:] = (mids[150:] / mids[:-150] - 1) * 10000
    # F3c helpers
    atr_30 = wilder_atr(highs, lows, mids, 30)
    ef = ema_series(mids, 30); es = ema_series(mids, 150)

    sl = 50; tp = 200; trail_after = 150; trail_bps = 50
    knife_bps = 50; ext_cap = 100
    f3c_threshold = 0.3
    # E3
    e3_after_bars = 60; e3_peak_threshold = 50; e3_tightened_sl = 25
    cands = set(candidate_signals(feats, H_THRESH, CORE_PARAMS))

    in_trade = False
    entry_idx = entry_price = direction = 0
    peak_pnl = 0.0
    tp_trailing = trailing_active = pst_active = False
    pst_peak_bps = 0.0; pst_entry_bar = 0
    sl_current = -sl
    e3_tightened = False
    trades = []

    for i in range(n):
        if in_trade:
            d = direction
            if d == 1:
                worst_bps = (lows[i]/entry_price-1)*10000
                best_bps  = (highs[i]/entry_price-1)*10000
            else:
                worst_bps = -(highs[i]/entry_price-1)*10000
                best_bps  = -(lows[i]/entry_price-1)*10000
            curr_bps = d * (mids[i]/entry_price - 1) * 10000
            peak_pnl = max(peak_pnl, best_bps)
            exit_reason = exit_pnl = None

            if worst_bps <= sl_current:
                exit_reason = 'sl'; exit_pnl = sl_current

            if exit_reason is None and not tp_trailing and peak_pnl >= trail_after:
                tp_trailing = True
            if exit_reason is None and tp_trailing:
                floor = max(peak_pnl - trail_bps, sl_current)
                if worst_bps <= floor:
                    exit_reason = 'tp_trail'; exit_pnl = max(floor, curr_bps)

            if exit_reason is None and trailing_active:
                tw = 2.0 * atr_bps_arr[i] if atr_bps_arr[i] > 0 else 50
                floor = max(peak_pnl - tw, sl_current)
                if worst_bps <= floor:
                    exit_reason = 'trail_stop'; exit_pnl = max(floor, curr_bps)
                elif best_bps >= tp:
                    exit_reason = 'tp'; exit_pnl = tp

            if exit_reason is None and pst_active:
                pst_peak_bps = max(pst_peak_bps, best_bps)
                bars_in_trail = i - pst_entry_bar
                floor_bps = max(pst_peak_bps - PST_WIDTH_BPS, sl_current)
                if worst_bps <= floor_bps:
                    exit_reason = 'pst_trail'; exit_pnl = max(floor_bps, curr_bps)
                elif worst_bps <= PST_HARD_FLOOR:
                    exit_reason = 'pst_floored'; exit_pnl = PST_HARD_FLOOR
                elif bars_in_trail >= PST_MAX_WAIT:
                    exit_reason = 'pst_timeout'; exit_pnl = curr_bps

            if (exit_reason is None and not tp_trailing and not pst_active
                and not trailing_active and best_bps >= tp):
                exit_reason = 'tp'; exit_pnl = tp

            # E3 time-decayed SL
            if (exit_reason is None and not e3_tightened
                and (i - entry_idx) >= e3_after_bars and peak_pnl < e3_peak_threshold):
                sl_current = -e3_tightened_sl
                e3_tightened = True

            # Timeout -> ATR trail or post_signal_trail
            if (exit_reason is None and (i - entry_idx) >= 240
                and not trailing_active and not pst_active):
                if curr_bps > 0:
                    trailing_active = True
                else:
                    pst_active = True
                    pst_entry_bar = i
                    pst_peak_bps = best_bps

            if exit_reason:
                entry_ts_ms = (int(ts_ms[entry_idx])
                               if entry_idx < len(ts_ms) else None)
                entry_ts_s = entry_ts_ms / 1000 if entry_ts_ms else None
                trades.append({
                    "direction": direction,
                    "pnl_bps": exit_pnl,
                    "reason": exit_reason,
                    "peak_bps": peak_pnl,
                    "hold_bars": i - entry_idx,
                    "entry_bar": entry_idx,
                    "entry_ts": (datetime.fromtimestamp(entry_ts_s, tz=timezone.utc).isoformat()
                                 if entry_ts_s is not None else None),
                    "session": session_from_ts(entry_ts_s) if entry_ts_s is not None else None,
                })
                in_trade = False
                tp_trailing = False; trailing_active = False; pst_active = False
                e3_tightened = False
            continue

        # entry side
        if i not in cands: continue
        if not np.isnan(dH_5[i]) and dH_5[i] >= 0: continue
        d = 1 if imb[i] > 0 else -1
        if knife_bps and d == 1 and ret_60[i] < -knife_bps: continue
        if knife_bps and d == -1 and ret_60[i] > knife_bps: continue
        if ext_cap is not None:
            r150 = ret_150[i]
            if d == 1 and r150 > ext_cap: continue
            if d == -1 and r150 < -ext_cap: continue
        # F3c
        if atr_30[i] > 0:
            norm = (ef[i] - es[i]) / atr_30[i]
        else:
            norm = 0.0
        if d == 1 and norm < f3c_threshold: continue
        if d == -1 and norm > -f3c_threshold: continue

        in_trade = True; entry_idx = i; entry_price = mids[i]
        direction = d; peak_pnl = 0
        tp_trailing = False; trailing_active = False; pst_active = False
        sl_current = -sl
        e3_tightened = False

    return trades


def summarize_bucket(trades_in_bucket):
    if not trades_in_bucket:
        return {"count": 0, "mean_pnl_bps": 0.0, "std_bps": 0.0}
    vals = np.asarray([t["pnl_bps"] for t in trades_in_bucket], dtype=float)
    return {
        "count": int(len(vals)),
        "mean_pnl_bps": float(vals.mean()),
        "std_bps": float(vals.std(ddof=1)) if len(vals) > 1 else float(vals.std()),
        "p25_bps": float(np.percentile(vals, 25)),
        "p50_bps": float(np.percentile(vals, 50)),
        "p75_bps": float(np.percentile(vals, 75)),
        "min_bps": float(vals.min()),
        "max_bps": float(vals.max()),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="print bucket table without writing shadow_expectation.json")
    args = parser.parse_args()

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading ETH Pi orderbook (Feb 18 - Apr 7)...", flush=True)
    df = load_orderbook_range("data/orderbook_pi", "ob_PF_ETHUSD")
    df_1m = resample_pi_to_1min(df)
    df_1m = compute_ob_features(df_1m)
    df_1m = classify_ob_states(df_1m, window=60)
    ent = rolling_entropy_ob(df_1m['state_ob'].values, NUM_STATES_OB, 30)
    feats = make_features(df_1m, ent)
    # Timestamps per bar (ms since epoch)
    if 'ts_ms' in df_1m.columns:
        ts_ms = df_1m['ts_ms'].values
    else:
        # Fallback: synthesize from RangeIndex assuming 1-min bars
        ts_ms = np.zeros(feats['n'], dtype=np.int64)
    print(f"  {feats['n']} bars, {feats['n']/1440:.1f} days", flush=True)

    trades = simulate_promoted(feats, ts_ms)
    print(f"  {len(trades)} trades simulated", flush=True)

    # Bucket
    buckets = {}   # "long_asia" -> list
    by_direction = {"long": [], "short": []}
    skipped_no_session = 0
    for t in trades:
        sess = t.get("session")
        if sess is None:
            skipped_no_session += 1
            continue
        d = "long" if t["direction"] == 1 else "short"
        key = f"{d}_{sess}"
        buckets.setdefault(key, []).append(t)
        by_direction[d].append(t)

    bucket_summaries = {k: summarize_bucket(v) for k, v in buckets.items()}
    fallback = {k: summarize_bucket(v) for k, v in by_direction.items()}

    out = {
        "cell_name": "F3c+timeout_trail+E3+5x",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_source": "Feb 18 - Apr 7 2026 Pi Kraken Futures ETH L2",
        "n_trades_total": len(trades),
        "skipped_no_session": skipped_no_session,
        "buckets": bucket_summaries,
        "fallback_by_direction": fallback,
        "notes": (
            "pnl_bps is leverage-invariant; live auto-flip monitor should "
            "score the realized pnl_bps (gross price move, not leveraged) "
            "against these expectations. Buckets = (direction, UTC session). "
            "If a live trade's bucket has count<5, fall back to the "
            "direction-only fallback_by_direction entry."
        ),
    }
    if args.dry_run:
        print(f"\n[DRY RUN] skipping write to {OUT}")
    else:
        OUT.write_text(json.dumps(out, indent=2, default=str))

    print(f"\nTotal trades processed: {len(trades)}")
    if skipped_no_session:
        print(f"Skipped (no session): {skipped_no_session}")
    print("\nBucket summary (n<5 -> fallback):")
    print(f"  {'bucket':<18} {'n':>3} {'mean':>9} {'std':>8}  use")
    fallback_buckets = []
    for k, s in sorted(bucket_summaries.items()):
        uses_fallback = s['count'] < 5
        if uses_fallback: fallback_buckets.append(k)
        print(f"  {k:<18} {s['count']:>3} {s['mean_pnl_bps']:>+8.1f} "
              f"{s['std_bps']:>7.1f}  {'FALLBACK' if uses_fallback else 'primary'}")
    print("\nDirection-only fallback:")
    for k, s in sorted(fallback.items()):
        print(f"  {k:<18} {s['count']:>3} {s['mean_pnl_bps']:>+8.1f} "
              f"{s['std_bps']:>7.1f}")
    if fallback_buckets:
        print(f"\n{len(fallback_buckets)} of {len(bucket_summaries)} primary "
              f"buckets below n=5 threshold and will use fallback: "
              f"{', '.join(fallback_buckets)}")
    if not args.dry_run:
        print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
