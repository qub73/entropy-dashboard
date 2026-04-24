"""
Sprint v1.5 Step 4 -- live-window reconstruction.

For each ETH live trade whose entry falls on a clean day (2026-04-13 .. -15),
reconstruct the Kraken-native L2 feature state at the entry bar and report
what F3c (threshold +/-0.3) would have said: BLOCK or ALLOW.

Compare to the actual live trade direction + reason + pnl_bps.

Not a decision input for 6b. Retrospective only.

Output: reports/sprint_v15_live_window_reconstruction.json
"""
import json, sys, time
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ALGO = HERE.parent
sys.path.insert(0, str(ALGO))
sys.path.insert(0, str(HERE))

from ob_entropy import (
    NUM_STATES_OB, rolling_entropy_ob,
    compute_ob_features, classify_ob_states,
)
from kaggle_ob_trainer import resample_pi_to_1min
from upgrade_backtest import make_features
from phase2_filter_ablation_v2 import build_extra_features
from kraken_hf_loader import load_kraken_hf_day, clean_days_from_report

OUT = ALGO / "reports" / "sprint_v15_live_window_reconstruction.json"
TRADE_HISTORY = ALGO / "state" / "trade_history.jsonl"

EVAL_RANGE = ("2026-03-18", "2026-04-15")
F3C_THRESHOLD = 0.3
ENTROPY_WINDOW_BARS = 30


def load_eval_slice():
    dates = clean_days_from_report(date_range=EVAL_RANGE)
    parts = []
    for d in dates:
        df = load_kraken_hf_day(d, verbose=False)
        if df.empty: continue
        parts.append(df)
    df_all = (pd.concat(parts, ignore_index=True)
              .sort_values("ts_ms").reset_index(drop=True))
    df_1m = resample_pi_to_1min(df_all)
    df_1m = compute_ob_features(df_1m)
    df_1m = classify_ob_states(df_1m, window=60)
    H = rolling_entropy_ob(df_1m["state_ob"].values, NUM_STATES_OB,
                           ENTROPY_WINDOW_BARS)
    feats = make_features(df_1m, H)
    volumes = (df_1m["bid_depth_5"].values + df_1m["ask_depth_5"].values
               if "bid_depth_5" in df_1m.columns else None)
    extra = build_extra_features(feats, volumes)
    return df_1m, feats, extra


def load_trades():
    if not TRADE_HISTORY.exists():
        raise RuntimeError(f"trade history missing: {TRADE_HISTORY}")
    trades = []
    with open(TRADE_HISTORY) as f:
        for ln in f:
            ln = ln.strip()
            if ln: trades.append(json.loads(ln))
    return trades


def find_nearest_bar(ts_target_ms, ts_ms_array):
    idx = int(np.searchsorted(ts_ms_array, ts_target_ms))
    if idx == 0: return 0
    if idx >= len(ts_ms_array): return len(ts_ms_array) - 1
    before = ts_ms_array[idx - 1]; after = ts_ms_array[idx]
    return idx if (after - ts_target_ms) < (ts_target_ms - before) else idx - 1


def reconstruct_trade(tr, df_1m, feats, extra, clean_days_set):
    exit_dt = datetime.fromisoformat(tr["time"].replace("Z", "+00:00"))
    hold_min = float(tr.get("hold_min", 0))
    entry_dt = exit_dt - timedelta(minutes=hold_min)
    entry_ms = int(entry_dt.timestamp() * 1000)
    entry_date = entry_dt.strftime("%Y-%m-%d")
    exit_date = exit_dt.strftime("%Y-%m-%d")

    overlap = "none"
    if entry_date in clean_days_set and exit_date in clean_days_set:
        overlap = "wholly"
    elif exit_date in clean_days_set or entry_date in clean_days_set:
        overlap = "partial"

    ts_array = feats["mid"]  # placeholder; real ts_ms needed
    # df_1m has ts_ms column
    ts_ms_array = df_1m["ts_ms"].values
    idx = find_nearest_bar(entry_ms, ts_ms_array)
    bar_ts_ms = int(ts_ms_array[idx])
    bar_dt = datetime.fromtimestamp(bar_ts_ms / 1000, tz=entry_dt.tzinfo)
    gap_s = abs((bar_dt - entry_dt).total_seconds())

    imb5 = float(feats["imb5"][idx])
    spread_bps = float(feats["spread_bps"][idx])
    mid = float(feats["mid"][idx])
    ema_fast = float(extra["ema_fast"][idx])
    ema_slow = float(extra["ema_slow"][idx])
    atr_30 = float(extra["atr_30"][idx])
    H_entry = feats["H"][idx]
    norm = (ema_fast - ema_slow) / atr_30 if atr_30 > 0 else 0.0

    direction = int(tr.get("direction", 0))
    # F3c: block if LONG with norm < 0.3, block if SHORT with norm > -0.3
    if direction == 1:
        f3c_block = bool(norm < F3C_THRESHOLD)
    elif direction == -1:
        f3c_block = bool(norm > -F3C_THRESHOLD)
    else:
        f3c_block = None

    return {
        "order_id": tr.get("order_id", ""),
        "pair": tr.get("pair", ""),
        "direction": direction,
        "direction_label": {1: "LONG", -1: "SHORT"}.get(direction, "?"),
        "entry_time_utc": entry_dt.isoformat(),
        "exit_time_utc": tr["time"],
        "entry_date": entry_date,
        "exit_date": exit_date,
        "clean_overlap": overlap,
        "hold_min": hold_min,
        "entry_price_live": float(tr.get("entry_price", 0)),
        "exit_price_live": float(tr.get("exit_price", 0)),
        "pnl_bps_live": float(tr.get("pnl_bps", 0)),
        "reason_live": tr.get("reason", ""),
        "kraken_native_reconstruction": {
            "bar_idx": idx,
            "bar_ts_utc": bar_dt.isoformat(),
            "time_gap_s_to_entry": round(gap_s, 1),
            "mid": round(mid, 2),
            "spread_bps": round(spread_bps, 3),
            "imbalance_5": round(imb5, 4),
            "H": None if (H_entry is None or
                          (isinstance(H_entry, float) and np.isnan(H_entry)))
                 else float(H_entry),
            "ema_fast": round(ema_fast, 2),
            "ema_slow": round(ema_slow, 2),
            "atr_30": round(atr_30, 2),
            "ema_diff_norm": round(float(norm), 4),
            "f3c_threshold": F3C_THRESHOLD,
            "f3c_would_block": f3c_block,
        },
    }


def main():
    print("[STEP 4] Live-window reconstruction from Kraken-native L2\n")
    trades = load_trades()
    print(f"  loaded {len(trades)} live trades")
    for i, tr in enumerate(trades, 1):
        print(f"    {i}. pair={tr['pair']} dir={tr['direction']:+d} "
              f"exit={tr['time']} hold_min={tr['hold_min']:.1f} "
              f"reason={tr['reason']} pnl={tr['pnl_bps']:+.1f}bps")

    clean_days = clean_days_from_report()
    clean_days_set = set(clean_days)

    print("\n  loading EVAL Kraken-native slice...", flush=True)
    t0 = time.time()
    df_1m, feats, extra = load_eval_slice()
    print(f"  loaded {feats['n']:,} bars  ({time.time()-t0:.0f}s)")

    records = []
    for i, tr in enumerate(trades, 1):
        rec = reconstruct_trade(tr, df_1m, feats, extra, clean_days_set)
        rec["live_trade_num_in_log"] = i
        records.append(rec)

    # Summary
    print("\n  per-trade F3c reconstruction:")
    print(f"  {'#':>2}  {'dir':<5} {'overlap':<8} {'entry_date':<11} "
          f"{'norm':>7}  {'F3c':<8}  {'live_pnl_bps':>12}  {'live_reason':<12}")
    for r in records:
        rc = r["kraken_native_reconstruction"]
        f3c_str = "BLOCK" if rc["f3c_would_block"] else "ALLOW"
        print(f"  {r['live_trade_num_in_log']:>2}  "
              f"{r['direction_label']:<5} {r['clean_overlap']:<8} "
              f"{r['entry_date']:<11} "
              f"{rc['ema_diff_norm']:>7.3f}  {f3c_str:<8}  "
              f"{r['pnl_bps_live']:>+12.1f}  "
              f"{r['reason_live']:<12}")

    # Aggregate: for trades with wholly-clean overlap, how many would F3c block?
    wholly = [r for r in records if r["clean_overlap"] == "wholly"]
    partial = [r for r in records if r["clean_overlap"] == "partial"]
    n_block_wholly = sum(1 for r in wholly
                         if r["kraken_native_reconstruction"]["f3c_would_block"])
    n_block_partial = sum(1 for r in partial
                          if r["kraken_native_reconstruction"]["f3c_would_block"])
    # Would-block analysis: what fraction of winning vs losing trades would F3c have blocked?
    would_block_winners = sum(1 for r in records
                              if r["pnl_bps_live"] > 0 and
                              r["kraken_native_reconstruction"]["f3c_would_block"])
    would_block_losers = sum(1 for r in records
                             if r["pnl_bps_live"] <= 0 and
                             r["kraken_native_reconstruction"]["f3c_would_block"])

    out = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat() + "Z",
        "sprint": "v1.5 step 4 -- live-window reconstruction",
        "venue": "Kraken-native L2 (Abraxasccs HF)",
        "pair": "ETH/USD",
        "trade_history_file": str(TRADE_HISTORY),
        "f3c_threshold": F3C_THRESHOLD,
        "n_trades_total": len(records),
        "n_wholly_clean_overlap": len(wholly),
        "n_partial_clean_overlap": len(partial),
        "f3c_would_block_wholly": n_block_wholly,
        "f3c_would_block_partial": n_block_partial,
        "winners_total": sum(1 for r in records if r["pnl_bps_live"] > 0),
        "losers_total": sum(1 for r in records if r["pnl_bps_live"] <= 0),
        "f3c_would_block_winners": would_block_winners,
        "f3c_would_block_losers": would_block_losers,
        "trades": records,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))

    print(f"\n  wholly-clean overlap: {len(wholly)}, "
          f"F3c would block: {n_block_wholly}")
    print(f"  partial overlap:      {len(partial)}, "
          f"F3c would block: {n_block_partial}")
    print(f"  winners blocked by F3c: {would_block_winners} / "
          f"{sum(1 for r in records if r['pnl_bps_live'] > 0)}")
    print(f"  losers blocked by F3c:  {would_block_losers} / "
          f"{sum(1 for r in records if r['pnl_bps_live'] <= 0)}")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
