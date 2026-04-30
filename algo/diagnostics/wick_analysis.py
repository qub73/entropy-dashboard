"""Wick-vs-true-SL analysis for post-deploy trades.

For each post-deploy trade, fetch 1-min OHLC from Kraken Futures public API
covering the trade window, then determine whether the SL exit was triggered
by an intra-bar wick that snapped back (where bar_low/high crossed -50 bps
but bar_close was much closer to entry) vs a genuine adverse move.

A "wick exit" is defined as: SL triggered (worst_bps <= -50) AND realized
pnl_bps (close-fill) is < |25| bps from entry. The gap between the SL
trigger and the recorded pnl reveals how much the price snapped back
between the trigger bar and the close-order fill.
"""
import json
import urllib.request
import urllib.parse
import time
import datetime as dt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
TRADE_HISTORY = ROOT / "algo" / "state" / "pi_pull_2026_04_30" / "trade_history.jsonl"
DEPLOY_TS = dt.datetime(2026, 4, 24, 13, 29, 39, tzinfo=dt.timezone.utc).timestamp()

OUT_DIR = ROOT / "algo" / "state" / "pi_pull_2026_04_30" / "wick_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_kraken_futures_1m(start_unix: int, end_unix: int) -> list:
    """Returns list of [ts_ms, o, h, l, c, v] for PF_ETHUSD 1-min bars.

    The API returns up to 2000 candles per call starting at `from`. Paginate
    by walking `from` forward until the latest returned ts >= end_unix.
    """
    bars = []
    cur_from = start_unix
    iter_count = 0
    while cur_from < end_unix and iter_count < 50:
        iter_count += 1
        url = (
            f"https://futures.kraken.com/api/charts/v1/trade/PF_ETHUSD/1m"
            f"?from={cur_from}&to={end_unix}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "wick-analysis"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())
        candles = data.get("candles", [])
        more = data.get("more_candles", False)
        if not candles:
            break
        chunk = []
        for c in candles:
            chunk.append([
                int(c["time"]),
                float(c["open"]),
                float(c["high"]),
                float(c["low"]),
                float(c["close"]),
                float(c.get("volume", 0)),
            ])
        chunk.sort(key=lambda b: b[0])
        bars.extend(chunk)
        last_ts = chunk[-1][0] // 1000
        # if API says no more candles, or we've reached end window, stop
        if not more or last_ts >= end_unix - 60:
            break
        cur_from = last_ts + 60
        time.sleep(0.3)
    # de-dup by ts
    seen = {}
    for b in bars:
        seen[b[0]] = b
    out = sorted(seen.values(), key=lambda b: b[0])
    return out


def analyze_trade(trade: dict, bars: list, sl_default: int = 50, sl_e3: int = 25) -> dict:
    """Determine wick severity for a trade.

    Returns dict with: trigger_bar_idx, exit_bar_idx, max_low_bps_for_long_or_high_bps_for_short,
    pnl_bps_at_close (computed from charted bars), pnl_bps_recorded (from log),
    wick_classification.
    """
    entry_px = trade["entry_price"]
    direction = trade["direction"]
    exit_iso = trade["time"]
    exit_unix_ms = int(dt.datetime.fromisoformat(exit_iso).timestamp() * 1000)
    entry_unix_ms = exit_unix_ms - int(trade["hold_min"] * 60 * 1000)

    # Identify in-trade bars (entry_ts <= bar_ts <= exit_ts), give 1-min slack
    in_trade = [b for b in bars if entry_unix_ms - 60_000 <= b[0] <= exit_unix_ms + 60_000]
    if not in_trade:
        return {"error": "no in-trade bars"}

    def bps_from_entry(px: float) -> float:
        return direction * (px / entry_px - 1.0) * 10000

    rows = []
    worst_seen = 0.0  # most negative
    best_seen = 0.0
    for b in in_trade:
        ts, o, h, l, c, v = b
        worst_bps_bar = bps_from_entry(l) if direction == 1 else bps_from_entry(h)
        best_bps_bar = bps_from_entry(h) if direction == 1 else bps_from_entry(l)
        close_bps = bps_from_entry(c)
        worst_seen = min(worst_seen, worst_bps_bar)
        best_seen = max(best_seen, best_bps_bar)
        rows.append({
            "ts": ts,
            "iso": dt.datetime.fromtimestamp(ts / 1000, tz=dt.timezone.utc).isoformat(),
            "o_bps": round(bps_from_entry(o), 2),
            "h_bps": round(bps_from_entry(h), 2),
            "l_bps": round(bps_from_entry(l), 2),
            "c_bps": round(bps_from_entry(c), 2),
            "worst_bps": round(worst_bps_bar, 2),
            "best_bps": round(best_bps_bar, 2),
        })

    # Find the bar where SL would have triggered (worst_bps <= -sl_default first time)
    # Track elapsed minutes to know if E3 might have applied
    sl_trigger_bar = None
    sl_distance_used = sl_default
    elapsed_min = 0
    e3_fired_bar = None
    for i, b in enumerate(in_trade):
        elapsed_min = (b[0] - entry_unix_ms) / 60000.0
        # E3: at >= 60 min, if peak < 50, tighten sl to 25
        if elapsed_min >= 60 and not e3_fired_bar:
            # peak so far
            pk = max(bps_from_entry(bb[2] if direction == 1 else bb[3]) * direction
                     for bb in in_trade[: i + 1])
            # Recompute peak the right way
            pk = 0.0
            for bb in in_trade[: i + 1]:
                bb_high_bps = bps_from_entry(bb[2]) if direction == 1 else -bps_from_entry(bb[3])
                # for long, high is best; for short, low is best
                if direction == 1:
                    cand = bps_from_entry(bb[2])
                else:
                    cand = -bps_from_entry(bb[3])
                pk = max(pk, cand)
            if pk < 50:
                e3_fired_bar = i
                sl_distance_used = sl_e3

        worst_bps_bar = bps_from_entry(b[3]) if direction == 1 else -bps_from_entry(b[2])
        # for long: worst = (low/entry-1)*10000  (negative if loss)
        # for short: worst = -(high/entry-1)*10000 (negative if loss)
        if direction == 1:
            worst_bps_bar = (b[3] / entry_px - 1) * 10000
        else:
            worst_bps_bar = -(b[2] / entry_px - 1) * 10000

        if worst_bps_bar <= -sl_distance_used and sl_trigger_bar is None:
            sl_trigger_bar = i
            break

    # Determine: at the SL trigger bar, what was bar close vs bar low/high?
    classification = "no_sl_trigger_in_data"
    sl_close_bps = None
    if sl_trigger_bar is not None:
        b = in_trade[sl_trigger_bar]
        # close pnl in bps for direction
        sl_close_bps = bps_from_entry(b[4])
        # Wick test: if the bar's worst_bps was <= -sl_distance_used but close bps > -25,
        # this is a wick.
        worst_at_trigger = (b[3] / entry_px - 1) * 10000 if direction == 1 else -(b[2] / entry_px - 1) * 10000
        if sl_close_bps > -25 and worst_at_trigger <= -sl_distance_used:
            classification = "wick"
        elif sl_close_bps > worst_at_trigger + 15:
            classification = "soft_wick"
        else:
            classification = "true_sl"

    return {
        "order_id": trade["order_id"],
        "entry_iso": dt.datetime.fromtimestamp(entry_unix_ms / 1000, tz=dt.timezone.utc).isoformat(),
        "exit_iso": exit_iso,
        "direction": direction,
        "entry_price": entry_px,
        "exit_price_recorded": trade["exit_price"],
        "pnl_bps_recorded": trade["pnl_bps"],
        "hold_min": trade["hold_min"],
        "n_in_trade_bars": len(in_trade),
        "worst_bps_seen": round(worst_seen, 2),
        "best_bps_seen": round(best_seen, 2),
        "sl_distance_used_at_trigger": sl_distance_used,
        "e3_fired_bar_idx": e3_fired_bar,
        "sl_trigger_bar_idx": sl_trigger_bar,
        "sl_trigger_close_bps": round(sl_close_bps, 2) if sl_close_bps is not None else None,
        "classification": classification,
        "first_3_bars": rows[:3],
        "last_3_bars": rows[-3:],
    }


def main():
    trades = [json.loads(l) for l in open(TRADE_HISTORY)]
    post_deploy = [t for t in trades
                   if dt.datetime.fromisoformat(t["time"]).timestamp() > DEPLOY_TS]

    print(f"Post-deploy trades: {len(post_deploy)}")

    # Fetch one big window covering all post-deploy trades + slack
    earliest = min(dt.datetime.fromisoformat(t["time"]).timestamp() - t["hold_min"] * 60
                   for t in post_deploy)
    latest = max(dt.datetime.fromisoformat(t["time"]).timestamp() for t in post_deploy)
    pad = 60 * 60  # 1h padding
    start_unix = int(earliest - pad)
    end_unix = int(latest + pad)
    print(f"Fetching Kraken Futures 1m bars {start_unix} to {end_unix} "
          f"({(end_unix-start_unix)/3600:.1f}h span)")

    bars = fetch_kraken_futures_1m(start_unix, end_unix)
    print(f"Fetched {len(bars)} 1m bars")
    (OUT_DIR / "kraken_futures_1m_post_deploy.json").write_text(json.dumps(bars))

    results = []
    for t in post_deploy:
        r = analyze_trade(t, bars)
        # always tag with order_id for downstream
        r.setdefault("order_id", t["order_id"])
        results.append(r)
        if "error" in r:
            print(f"\n--- {t['order_id'][:8]}  ERROR: {r['error']}  ---")
            continue
        print(f"\n--- {t['order_id'][:8]} ({r['direction']:+d}) "
              f"hold={r['hold_min']:.0f}min recorded={r['pnl_bps_recorded']:.1f}bps ---")
        print(f"  worst_seen={r['worst_bps_seen']}  best_seen={r['best_bps_seen']}  "
              f"e3_fired_bar={r['e3_fired_bar_idx']}  "
              f"sl_trigger_bar={r['sl_trigger_bar_idx']}  "
              f"sl_trigger_close={r['sl_trigger_close_bps']}  "
              f"=> {r['classification']}")

    # Summary
    by_class = {}
    for r in results:
        c = r.get("classification", "error")
        by_class.setdefault(c, []).append(r["order_id"][:8])
    print("\n=== Summary ===")
    for k, v in by_class.items():
        print(f"  {k}: {len(v)}  {v}")

    (OUT_DIR / "wick_analysis_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {OUT_DIR}/wick_analysis_results.json")


if __name__ == "__main__":
    main()
