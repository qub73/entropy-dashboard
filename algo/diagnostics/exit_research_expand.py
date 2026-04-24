"""Expand the exit-research sweep: run ALL 151 cells on Pi IS (not just
top-8 from Kraken-native), and replay them against the 19 live trades
to see which cells would have best handled the three failure modes.
Loads cached sweep results."""
from __future__ import annotations
import datetime as dt
import json, sys, time, urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np

HERE = Path(__file__).resolve().parent
ALGO = HERE.parent
sys.path.insert(0, str(ALGO)); sys.path.insert(0, str(HERE))

from phase4_sizing_sim import load_pi
from exit_research_engine import ExitParams, run_exit_variant

IN = ALGO / "reports" / "exit_research_sweep.json"
OUT = ALGO / "reports" / "exit_research_expand.json"

# Live-trade replay params (19 live trades)
TRADES_FILE = Path("C:/tmp/6a_smoke_trades.jsonl")
CACHE_DIR = Path("C:/tmp/ohlc_cache")
LIVE_SL_LIVE = 50; LIVE_TP = 200; LIVE_FEE = 5; LIVE_LEV = 10
# Each live trade was at 10x (pre-sprint); compute equity impact per config.
PST_WIDTH_DEFAULT = 20; PST_FLOOR_DEFAULT = -60; PST_MAX_WAIT_DEFAULT = 30


def fetch_binance_1m(start_ms, end_ms):
    cache = CACHE_DIR / f"binance_ETHUSDT_{start_ms}_{end_ms}.json"
    if cache.exists(): return json.loads(cache.read_text())
    url = (f"https://api.binance.com/api/v3/klines?symbol=ETHUSDT"
           f"&interval=1m&startTime={start_ms}&endTime={end_ms}&limit=1000")
    req = urllib.request.Request(url, headers={"User-Agent": "live-replay"})
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.loads(r.read())
    cache.write_text(json.dumps(d))
    return d


def get_bars(entry_dt, hold_min, forward_min=60):
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
        time.sleep(0.1)
    bars = []
    for b in all_bars:
        bars.append({"ts": int(b[0])//1000, "o":float(b[1]), "h":float(b[2]),
                     "l":float(b[3]), "c":float(b[4])})
    uniq = {}
    for x in bars: uniq[x["ts"]] = x
    return sorted(uniq.values(), key=lambda x: x["ts"])


def compute_atr_bps(bars, window=14):
    n = len(bars)
    if n == 0: return []
    tr = [bars[0]["h"] - bars[0]["l"]]
    for i in range(1, n):
        tr.append(max(bars[i]["h"]-bars[i]["l"],
                       abs(bars[i]["h"]-bars[i-1]["c"]),
                       abs(bars[i]["l"]-bars[i-1]["c"])))
    atr = [0.0]*n
    if n <= window:
        s = 0
        for i, v in enumerate(tr):
            s += v; atr[i] = s/(i+1)
    else:
        atr[:window] = [sum(tr[:window])/window]*window
        for i in range(window, n):
            atr[i] = (atr[i-1]*(window-1) + tr[i]) / window
    return [atr[i]/bars[i]["c"]*10000 if bars[i]["c"] > 0 else 0 for i in range(n)]


def replay_live_trade(p: ExitParams, bars: List[dict], direction: int,
                       entry_ts: int, entry_price: float) -> Dict:
    """Replay one live trade under params p. Uses 1m Binance bars."""
    entry_idx = None
    for i, b in enumerate(bars):
        if b["ts"] >= entry_ts - 61:
            entry_idx = i; break
    if entry_idx is None:
        return {"reason": "no_entry_bar", "pnl_bps": 0.0}

    atr_bps = compute_atr_bps(bars, 14)
    peak = 0.0
    tp_trailing = trailing_active = pst_active = False
    pst_peak = 0.0; pst_entry_bar = 0
    sl_current = -p.sl_bps
    e3_tightened = False
    hard_cap = entry_idx + 600

    for j in range(entry_idx, min(len(bars), hard_cap+1)):
        b = bars[j]; rel = j - entry_idx
        if direction == 1:
            worst = (b["l"]/entry_price - 1) * 10000
            best  = (b["h"]/entry_price - 1) * 10000
        else:
            worst = -((b["h"]/entry_price - 1) * 10000)
            best  = -((b["l"]/entry_price - 1) * 10000)
        curr = direction * (b["c"]/entry_price - 1) * 10000
        peak = max(peak, best)

        if j == entry_idx: continue
        # E3
        if p.e3_enabled and not e3_tightened and rel >= p.e3_after_bars \
           and peak < p.e3_peak_threshold_bps:
            sl_current = -p.e3_tightened_sl_bps
            e3_tightened = True
        # SL
        if worst <= sl_current and not pst_active:
            return {"reason": "sl", "pnl_bps": sl_current, "peak": peak,
                     "hold_bars": rel}
        # tp_trail activate
        if not tp_trailing and peak >= p.trail_after:
            tp_trailing = True
        if tp_trailing and not pst_active:
            if p.exit_mode == "standard":
                floor = max(peak - p.trail_bps, sl_current)
            elif p.exit_mode == "atr_trail":
                aw = p.atr_trail_mult * atr_bps[j] if atr_bps[j] > 0 else 50
                floor = max(peak - aw, sl_current)
            elif p.exit_mode == "dual_stage":
                tw = p.trail_bps_tight if peak < p.dual_stage_pivot else p.trail_bps_loose
                floor = max(peak - tw, sl_current)
            else:
                floor = max(peak - p.trail_bps, sl_current)
            if worst <= floor:
                return {"reason": "tp_trail", "pnl_bps": max(floor, curr),
                         "peak": peak, "hold_bars": rel}
        # post-timeout paths
        if trailing_active and not pst_active:
            tw = p.atr_trail_mult * atr_bps[j] if atr_bps[j] > 0 else 50
            floor = max(peak - tw, sl_current)
            if worst <= floor:
                return {"reason": "trail_stop", "pnl_bps": max(floor, curr),
                         "peak": peak, "hold_bars": rel}
            elif best >= p.tp_bps:
                return {"reason": "tp", "pnl_bps": p.tp_bps,
                         "peak": peak, "hold_bars": rel}
        if pst_active:
            pst_peak = max(pst_peak, best)
            floor = max(pst_peak - p.pst_width_bps, sl_current)
            bars_pst = j - pst_entry_bar
            if worst <= floor:
                return {"reason": "pst_trail", "pnl_bps": max(floor, curr),
                         "peak": peak, "hold_bars": rel}
            elif worst <= p.pst_hard_floor_bps:
                return {"reason": "pst_floored", "pnl_bps": p.pst_hard_floor_bps,
                         "peak": peak, "hold_bars": rel}
            elif bars_pst >= p.pst_max_wait_bars:
                return {"reason": "pst_timeout", "pnl_bps": curr,
                         "peak": peak, "hold_bars": rel}
        # Fixed TP
        if (not tp_trailing and not pst_active and not trailing_active
            and best >= p.tp_bps):
            return {"reason": "tp", "pnl_bps": p.tp_bps, "peak": peak,
                     "hold_bars": rel}
        # Timeout
        if rel >= p.timeout_bars and not trailing_active and not pst_active:
            if curr > 0: trailing_active = True
            else: pst_active = True; pst_entry_bar = j; pst_peak = best

    # Force exit at hard cap
    j = min(len(bars) - 1, hard_cap)
    b = bars[j]; rel = j - entry_idx
    curr = direction * (b["c"]/entry_price - 1) * 10000
    return {"reason": "hard_cap", "pnl_bps": curr, "peak": peak, "hold_bars": rel}


def replay_all_live(p: ExitParams, live_data) -> Dict:
    results = []
    for t, bars in live_data:
        exit_dt = dt.datetime.fromisoformat(t["time"].replace("Z","+00:00"))
        entry_dt = exit_dt - dt.timedelta(minutes=float(t["hold_min"]))
        r = replay_live_trade(p, bars, int(t["direction"]),
                                int(entry_dt.timestamp()),
                                float(t["entry_price"]))
        r["live_pnl_bps"] = float(t["pnl_bps"])
        r["live_reason"] = t["reason"]
        results.append(r)

    bps = [r["pnl_bps"] - 2*LIVE_FEE for r in results]
    wins = [b for b in bps if b > 0]; losses = [b for b in bps if b <= 0]
    # Compound at 10x (what live was at)
    eq = 10000.0
    EQ_SCALE = 0.9 * LIVE_LEV / 10000
    for b in bps: eq *= (1 + EQ_SCALE * b)
    ret = (eq - 10000) / 100.0
    # Compare against live actual pnl
    live_bps = [float(t["pnl_bps"]) - 2*LIVE_FEE for t, _ in live_data]
    eq_live = 10000.0
    for b in live_bps: eq_live *= (1 + EQ_SCALE * b)
    ret_live = (eq_live - 10000) / 100.0
    return {
        "n_trades": len(results),
        "compound_ret_10x_pct": float(ret),
        "compound_ret_live_actual_pct": float(ret_live),
        "ret_delta_vs_live": float(ret - ret_live),
        "win_rate": len(wins)/len(results) if results else 0,
        "mean_pnl_bps": float(np.mean(bps)) if bps else 0.0,
        "avg_winner_bps": float(np.mean(wins)) if wins else 0.0,
        "avg_loser_bps": float(np.mean(losses)) if losses else 0.0,
        "trades": results,
    }


def main():
    t0 = time.time()
    data = json.loads(IN.read_text())
    cells = data["cells"]
    print(f"loaded {len(cells)} cells from {IN.name}")

    # Load Pi IS
    print("loading Pi IS...")
    pi_feats, pi_extra = load_pi()

    # Run all cells on Pi IS
    print(f"[pi_is] running all {len(cells)} cells...")
    t1 = time.time()
    for i, c in enumerate(cells, 1):
        if "pi_is" in c: continue  # already run for top cells
        p = ExitParams(**c["params"])
        r = run_exit_variant(pi_feats, pi_extra, p)
        c["pi_is"] = {k: v for k, v in r.items() if k != "trades"}
        if i % 25 == 0 or i == len(cells):
            print(f"  [{i:>3}/{len(cells)}] elapsed {time.time()-t1:.0f}s")

    # Fetch live-trade bars once
    print("\nfetching 19 live-trade OHLC windows...")
    trades = [json.loads(l) for l in TRADES_FILE.open()]
    live_data = []
    for i, t in enumerate(trades, 1):
        exit_dt = dt.datetime.fromisoformat(t["time"].replace("Z","+00:00"))
        entry_dt = exit_dt - dt.timedelta(minutes=float(t["hold_min"]))
        bars = get_bars(entry_dt, t["hold_min"], forward_min=60)
        if len(bars) < 20:
            print(f"  [{i}] skip, {len(bars)} bars")
            continue
        live_data.append((t, bars))
    print(f"  loaded {len(live_data)}/{len(trades)} trade windows")

    print(f"\n[live] replaying all {len(cells)} cells on {len(live_data)} trades...")
    for i, c in enumerate(cells, 1):
        p = ExitParams(**c["params"])
        c["live"] = replay_all_live(p, live_data)
        if i % 25 == 0 or i == len(cells):
            print(f"  [{i:>3}/{len(cells)}] elapsed {time.time()-t0:.0f}s")

    # Ranking: by each substrate and combined
    def score(c, sub):
        m = c.get(sub, {})
        ret_key = ("compound_ret_10x_pct" if sub == "live"
                   else "compound_ret_pct")
        return (m.get(ret_key, -1e9)
                - 0.5 * m.get("max_dd_pct", 0))

    print("\n--- top 10 by Pi IS ---")
    pi_ranked = sorted(cells, key=lambda c: score(c, "pi_is"), reverse=True)[:10]
    for c in pi_ranked:
        pi = c["pi_is"]; kn = c["kraken_native"]; live = c["live"]
        print(f"  {c['label']:<50}  PI:{pi['compound_ret_pct']:+7.2f}%/{pi['max_dd_pct']:5.2f}%  "
              f"KN:{kn['compound_ret_pct']:+7.2f}%/{kn['max_dd_pct']:5.2f}%  "
              f"LIVE@10x:{live['compound_ret_10x_pct']:+7.2f}%")

    print("\n--- top 10 by LIVE 19-trade @10x ---")
    live_ranked = sorted(cells, key=lambda c: score(c, "live"), reverse=True)[:10]
    for c in live_ranked:
        pi = c["pi_is"]; kn = c["kraken_native"]; live = c["live"]
        print(f"  {c['label']:<50}  LIVE:{live['compound_ret_10x_pct']:+7.2f}%  "
              f"PI:{pi['compound_ret_pct']:+7.2f}%/{pi['max_dd_pct']:5.2f}%  "
              f"KN:{kn['compound_ret_pct']:+7.2f}%/{kn['max_dd_pct']:5.2f}%")

    print("\n--- combined rank (Pi ret + KN ret + LIVE ret) ---")
    comb = sorted(cells, key=lambda c: (
        c["pi_is"]["compound_ret_pct"] +
        c["kraken_native"]["compound_ret_pct"] +
        c["live"]["compound_ret_10x_pct"]
    ), reverse=True)[:10]
    for c in comb:
        pi = c["pi_is"]; kn = c["kraken_native"]; live = c["live"]
        total = (pi["compound_ret_pct"] + kn["compound_ret_pct"]
                 + live["compound_ret_10x_pct"])
        print(f"  {c['label']:<50}  sum={total:+7.1f}  "
              f"PI:{pi['compound_ret_pct']:+7.2f}% "
              f"KN:{kn['compound_ret_pct']:+7.2f}% "
              f"LIVE:{live['compound_ret_10x_pct']:+7.2f}%")

    # Write out (strip per-trade arrays from cells for size)
    slim_cells = []
    for c in cells:
        cc = dict(c)
        if "live" in cc and "trades" in cc["live"]:
            cc["live"] = {k: v for k, v in cc["live"].items() if k != "trades"}
        slim_cells.append(cc)
    data["cells"] = slim_cells
    data["expanded_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    data["elapsed_expand_sec"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(data, indent=2, default=str))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
