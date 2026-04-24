"""
Timeout-reduction ablation for sprint v1 analysis.

Three substrates:
  1. 17 pre-sprint live trades (Apr 12-19 2026) at original params @10x.
     Per-bar reconstruction uses Kraken spot 1m OHLC (spot == futures for
     ETH to <5 bps; documented caveat).
  2. Pi IS 21.3-day baseline (Feb 18 - Apr 7) via the phase4 engine.
  3. Kaggle 60d OOS via the phase4 engine.

Variants: T_90, T_120, T_150, T_180, T_240 (baseline).

At the shorter timeout bar:
  - If trade already exited (SL/TP/tp_trail): unchanged
  - If still open and in profit: enter ATR-trail from that bar
    (trail width = 2 * ATR_14_bps, floor = entry +/- sl). Continue until
    trail fires or hard cap at original 240.
  - If still open and in loss: exit at that bar's close (original
    "exit-at-curr" timeout semantics, not post_signal_trail).

Outputs:
  algo/reports/timeout_reduction_ablation.json
  notes/timeout_reduction_findings.md  (generated separately)
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
ALGO = HERE.parent
REPO = ALGO.parent
sys.path.insert(0, str(ALGO))
sys.path.insert(0, str(HERE))

OUT = ALGO / "reports" / "timeout_reduction_ablation.json"

# Live-trade reconstruction
TRADES_FILE = Path("C:/tmp/6a_smoke_trades.jsonl")  # already pulled from Pi
DEPLOY_6A_UTC = dt.datetime(2026, 4, 19, 22, 26, 17, tzinfo=dt.timezone.utc)
CACHE_DIR = Path("C:/tmp/ohlc_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Pre-sprint live params
SL_LIVE = 50.0      # bps
TP_LIVE = 200.0     # bps
TRAIL_AFTER = 150.0 # bps (tp_trail activation threshold)
TRAIL_BPS = 50.0    # bps (tp_trail width)
FEE_BPS = 5.0       # per side
LEVERAGE_LIVE = 10
ATR_WINDOW = 14

TIMEOUT_VARIANTS = [90, 120, 150, 180, 240]


# ---------------- Kraken 1m OHLC fetch ----------------

def fetch_binance_1m_ohlc(start_ms: int, end_ms: int,
                            symbol: str = "ETHUSDT") -> List[list]:
    """Returns list of kline [open_ts_ms, o, h, l, c, volume, ...] for
    1-min bars in [start_ms, end_ms]. Cached locally. Binance is used
    because Kraken spot public OHLC only serves ~12h of 1-min history."""
    cache_file = CACHE_DIR / f"binance_{symbol}_{start_ms}_{end_ms}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    url = (f"https://api.binance.com/api/v3/klines?symbol={symbol}"
           f"&interval=1m&startTime={start_ms}&endTime={end_ms}&limit=1000")
    req = urllib.request.Request(url, headers={"User-Agent": "timeout-ablation"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read())
            break
        except Exception as e:
            if attempt == 2: raise
            time.sleep(2 * (attempt + 1))
    cache_file.write_text(json.dumps(data))
    return data


def get_bars_for_trade(entry_ts_iso: str, hold_min: float,
                        buffer_min: int = 15) -> List[dict]:
    """Fetch 1-min OHLC covering [entry-buffer, exit+buffer]. Returns
    list of {ts, o, h, l, c} dicts sorted by ts."""
    entry_dt = dt.datetime.fromisoformat(entry_ts_iso.replace("Z","+00:00"))
    start_ms = int((entry_dt - dt.timedelta(minutes=buffer_min)).timestamp() * 1000)
    end_ms   = int((entry_dt + dt.timedelta(minutes=hold_min + buffer_min))
                   .timestamp() * 1000)
    all_bars: List[list] = []
    cursor = start_ms
    while cursor < end_ms:
        chunk_end = min(cursor + 60 * 60 * 1000 * 16, end_ms)  # up to 16h per request
        chunk = fetch_binance_1m_ohlc(cursor, chunk_end)
        if not chunk: break
        all_bars.extend(chunk)
        last_open = int(chunk[-1][0])
        if last_open >= end_ms or len(chunk) < 1:
            break
        cursor = last_open + 60000
        time.sleep(0.2)  # gentle

    bars = []
    for b in all_bars:
        ts_sec = int(b[0]) // 1000
        bars.append({"ts": ts_sec, "o": float(b[1]), "h": float(b[2]),
                      "l": float(b[3]), "c": float(b[4])})
    uniq: Dict[int, dict] = {}
    for x in bars: uniq[x["ts"]] = x
    return sorted(uniq.values(), key=lambda x: x["ts"])


# ---------------- simulation helpers ----------------

def compute_atr_bps(bars: List[dict], window: int = ATR_WINDOW) -> List[float]:
    """Wilder ATR in bps for each bar. Index 0 uses range only; ramps up."""
    n = len(bars)
    if n == 0: return []
    tr = [0.0] * n
    tr[0] = bars[0]["h"] - bars[0]["l"]
    for i in range(1, n):
        hl = bars[i]["h"] - bars[i]["l"]
        hc = abs(bars[i]["h"] - bars[i-1]["c"])
        lc = abs(bars[i]["l"] - bars[i-1]["c"])
        tr[i] = max(hl, hc, lc)
    atr = [0.0] * n
    if n <= window:
        # just cumulative mean
        s = 0.0
        for i, v in enumerate(tr):
            s += v
            atr[i] = s / (i + 1)
    else:
        atr[:window] = [sum(tr[:window]) / window] * window
        for i in range(window, n):
            atr[i] = (atr[i-1] * (window - 1) + tr[i]) / window
    # convert to bps on close
    atr_bps = [atr[i] / bars[i]["c"] * 10000 if bars[i]["c"] > 0 else 0
               for i in range(n)]
    return atr_bps


def simulate_single_trade(bars: List[dict], direction: int, entry_ts_sec: int,
                           entry_price: float, sl_bps: float, tp_bps: float,
                           timeout_bars: int,
                           trail_after_bps: float = TRAIL_AFTER,
                           trail_bps: float = TRAIL_BPS) -> Dict:
    """Replay a single trade over per-minute OHLC. Returns dict with
    exit_bar_idx, exit_price, pnl_bps (pre-leverage, fee-adjusted
    round-trip), reason, mfe_bps, hold_bars.

    Logic mirrors the engine:
      - SL check: if low/high crosses entry +/- sl/10000 -> stop
      - tp_trail: once peak_pnl >= trail_after, switch to trailing mode
        with trail_bps behind peak
      - TP: if best >= tp (and not in trail mode) -> exit at tp
      - Timeout: at timeout_bars from entry:
          in profit -> enter ATR trail (2x atr_bps) until trail hit
          in loss -> exit at close
      - Hard cap at 240 bars beyond trail activation (no infinite holds).
    """
    # Find entry bar index
    entry_idx = None
    for i, b in enumerate(bars):
        if b["ts"] >= entry_ts_sec - 61:
            entry_idx = i; break
    if entry_idx is None:
        return {"exit_bar_idx": None, "reason": "no_entry_bar"}

    atr_bps = compute_atr_bps(bars)
    peak_pnl = 0.0
    tp_trailing = False
    trailing_active = False  # post-timeout ATR trail
    # Hard cap must sit above 240 so the post-timeout ATR trail has room
    # to play out. Live bot has no fixed cap; we use 10h to avoid runaway.
    hard_cap_idx = entry_idx + 600
    n = len(bars)

    mfe = 0.0
    mae = 0.0
    exit_idx = None
    exit_price = None
    reason = None
    exit_pnl_bps = None

    for j in range(entry_idx, min(n, hard_cap_idx + 1)):
        b = bars[j]
        rel = j - entry_idx
        if direction == 1:
            worst = (b["l"] / entry_price - 1) * 10000
            best  = (b["h"] / entry_price - 1) * 10000
        else:
            worst = -((b["h"] / entry_price - 1) * 10000)
            best  = -((b["l"] / entry_price - 1) * 10000)
        curr = direction * (b["c"] / entry_price - 1) * 10000
        peak_pnl = max(peak_pnl, best)
        mfe = max(mfe, best); mae = min(mae, worst)

        if j == entry_idx:
            # Can still hit SL/TP intra-bar on entry bar; skip only if the
            # entry fill is the close (conservative).
            continue

        # SL first
        if worst <= -sl_bps and not trailing_active:
            reason = "sl"; exit_pnl_bps = -sl_bps
            exit_idx = j; exit_price = entry_price * (1 + direction * (-sl_bps)/10000)
            break

        # tp_trail activation
        if not tp_trailing and peak_pnl >= trail_after_bps:
            tp_trailing = True

        if tp_trailing:
            floor = max(peak_pnl - trail_bps, -sl_bps)
            if worst <= floor:
                reason = "tp_trail"; exit_pnl_bps = max(floor, curr)
                exit_idx = j
                exit_price = entry_price * (1 + direction * exit_pnl_bps/10000)
                break
            continue

        # Post-timeout ATR trail mode
        if trailing_active:
            tw = 2.0 * atr_bps[j] if atr_bps[j] > 0 else 50.0
            floor = max(peak_pnl - tw, -sl_bps)
            if worst <= floor:
                reason = "trail_stop"; exit_pnl_bps = max(floor, curr)
                exit_idx = j
                exit_price = entry_price * (1 + direction * exit_pnl_bps/10000)
                break
            elif best >= tp_bps:
                reason = "tp"; exit_pnl_bps = tp_bps
                exit_idx = j
                exit_price = entry_price * (1 + direction * tp_bps/10000)
                break
            continue

        # Fixed TP (non-trailing path, shouldn't reach here often because
        # peak>=150 already flipped to tp_trailing)
        if best >= tp_bps:
            reason = "tp"; exit_pnl_bps = tp_bps
            exit_idx = j
            exit_price = entry_price * (1 + direction * tp_bps/10000)
            break

        # Timeout at configured bar
        if rel >= timeout_bars and not trailing_active:
            if curr > 0:
                trailing_active = True  # continue as ATR trail
            else:
                reason = "timeout"; exit_pnl_bps = curr
                exit_idx = j; exit_price = b["c"]
                break

    if exit_idx is None:
        # Ran past hard cap without any exit. Force exit at last available
        # bar close.
        j = min(n - 1, hard_cap_idx)
        exit_idx = j
        bar = bars[j]
        exit_price = bar["c"]
        curr = direction * (bar["c"] / entry_price - 1) * 10000
        reason = "hard_cap"; exit_pnl_bps = curr

    # Apply round-trip fees on pnl_bps
    net_pnl_bps = exit_pnl_bps - 2 * FEE_BPS
    return {
        "exit_bar_idx": exit_idx,
        "hold_bars": exit_idx - entry_idx,
        "exit_price": exit_price,
        "reason": reason,
        "pnl_bps_gross": exit_pnl_bps,
        "pnl_bps_net": net_pnl_bps,
        "mfe_bps": mfe,
        "mae_bps": mae,
    }


# ---------------- metrics ----------------

def metrics_from_trades(results: List[dict], leverage: int = LEVERAGE_LIVE) -> Dict:
    """Compute compound-return, DD, win rate, avg win/loss, knife rate,
    exit-reason distribution."""
    if not results:
        return {}
    bps = [r["pnl_bps_net"] for r in results]
    # Compound at leverage with 90% margin: equity *= 1 + 0.9*lev*(bps-fee_offset)/10000
    EQ_SCALE = 0.9 * leverage / 10000  # for pre-net-bps you'd subtract fees too
    equity = 10000.0
    eq = [equity]
    for p in bps:
        # pnl_bps_net is already fee-adjusted round-trip; equity multiplier:
        equity *= (1 + EQ_SCALE * p)
        eq.append(max(equity, 0))
    eq_arr = np.array(eq)
    peak = np.maximum.accumulate(eq_arr)
    dd = float(((peak - eq_arr) / peak * 100).max())
    ret = (equity - 10000) / 100.0
    wins = [p for p in bps if p > 0]; losses = [p for p in bps if p <= 0]
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    knife = sum(1 for r in results if r["pnl_bps_net"] <= 0 and r["mfe_bps"] < 20)
    reasons = {}
    for r in results:
        reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    return {
        "n_trades": len(results),
        "compound_ret_pct": ret,
        "max_dd_pct": dd,
        "win_rate": len(wins) / len(results),
        "avg_winner_bps": avg_win,
        "avg_loser_bps": avg_loss,
        "knife_rate": knife / len(results),
        "knife_catchers": knife,
        "mean_pnl_bps": float(np.mean(bps)),
        "reason_distribution": reasons,
    }


# ---------------- live-trade driver ----------------

def run_live_ablation() -> Dict:
    """Reconstruct 17 pre-sprint trades under each timeout variant."""
    if not TRADES_FILE.exists():
        raise RuntimeError(f"trade history not at {TRADES_FILE} (pull from Pi first)")
    all_trades = [json.loads(l) for l in TRADES_FILE.open()]
    pre = [t for t in all_trades if
           dt.datetime.fromisoformat(t["time"].replace("Z","+00:00")) <= DEPLOY_6A_UTC]
    print(f"[live] 17 pre-sprint trades: {len(pre)}", flush=True)

    # Fetch per-trade bars
    per_trade = []
    for i, t in enumerate(pre, 1):
        exit_dt = dt.datetime.fromisoformat(t["time"].replace("Z","+00:00"))
        hold = float(t["hold_min"])
        entry_dt = exit_dt - dt.timedelta(minutes=hold)
        print(f"  [{i:>2}/{len(pre)}] {entry_dt.isoformat()}  "
              f"hold={hold:.0f}m  reason={t['reason']}", flush=True)
        try:
            bars = get_bars_for_trade(entry_dt.isoformat(), hold, buffer_min=15)
        except Exception as e:
            print(f"    skip: {e}", flush=True)
            continue
        if len(bars) < 30:
            print(f"    skip: only {len(bars)} bars", flush=True)
            continue
        per_trade.append({"trade": t, "entry_dt": entry_dt, "bars": bars})

    print(f"\n[live] fetched OHLC for {len(per_trade)}/{len(pre)} trades",
          flush=True)

    # Simulate per variant
    variant_results: Dict[str, List[dict]] = {f"T_{T}": [] for T in TIMEOUT_VARIANTS}
    per_trade_rows: List[dict] = []

    for pt in per_trade:
        t = pt["trade"]
        entry_dt = pt["entry_dt"]
        bars = pt["bars"]
        row = {
            "order_id": t.get("order_id"),
            "direction": t["direction"],
            "entry_time_utc": entry_dt.isoformat(),
            "live_pnl_bps": float(t["pnl_bps"]),
            "live_reason": t["reason"],
            "live_hold_min": float(t["hold_min"]),
            "variants": {},
        }
        for T in TIMEOUT_VARIANTS:
            res = simulate_single_trade(
                bars=bars, direction=int(t["direction"]),
                entry_ts_sec=int(entry_dt.timestamp()),
                entry_price=float(t["entry_price"]),
                sl_bps=SL_LIVE, tp_bps=TP_LIVE, timeout_bars=T,
            )
            variant_results[f"T_{T}"].append(res)
            row["variants"][f"T_{T}"] = res
        per_trade_rows.append(row)

    agg = {k: metrics_from_trades(v, leverage=LEVERAGE_LIVE)
           for k, v in variant_results.items()}

    # Bucket cross-tab (use T_240 as the reference)
    buckets = {"<30": [], "30-90": [], "90-180": [], ">180": []}
    for row in per_trade_rows:
        T240 = row["variants"]["T_240"]
        if T240.get("exit_bar_idx") is None: continue
        h = T240["hold_bars"]
        if h < 30: buckets["<30"].append(row)
        elif h < 90: buckets["30-90"].append(row)
        elif h < 180: buckets["90-180"].append(row)
        else: buckets[">180"].append(row)

    bucket_report: Dict = {}
    for bname, rows in buckets.items():
        entry = {"n_trades": len(rows), "by_variant": {}}
        for T in TIMEOUT_VARIANTS:
            if T == 240: continue
            changed = []
            for r in rows:
                t240 = r["variants"]["T_240"]; tvar = r["variants"][f"T_{T}"]
                delta = tvar["pnl_bps_net"] - t240["pnl_bps_net"]
                if abs(delta) > 0.1:
                    changed.append({
                        "order_id": r["order_id"],
                        "delta_pnl_bps": round(delta, 2),
                        "t240_reason": t240["reason"],
                        "t_var_reason": tvar["reason"],
                    })
            entry["by_variant"][f"T_{T}"] = {
                "n_changed": len(changed),
                "sum_delta_bps": round(sum(c["delta_pnl_bps"] for c in changed), 2),
                "changes": changed,
            }
        bucket_report[bname] = entry

    return {
        "n_pre_sprint_live_trades": len(pre),
        "n_reconstructed": len(per_trade),
        "params": {"sl_bps": SL_LIVE, "tp_bps": TP_LIVE,
                    "trail_after_bps": TRAIL_AFTER, "trail_bps": TRAIL_BPS,
                    "fee_bps_per_side": FEE_BPS,
                    "leverage_live": LEVERAGE_LIVE},
        "metrics_by_variant": agg,
        "bucket_cross_tab": bucket_report,
        "per_trade": per_trade_rows,
    }


# ---------------- Pi/Kaggle driver ----------------

def run_phase4_with_timeout(substrate: str, timeout_bars: int) -> Dict:
    """Re-run the phase4-style engine with configurable timeout.
    substrate in {'pi', 'kaggle'}.
    Uses revised-6b cell (F3c OFF, timeout_trail ON, E3 ON, lev 5x)."""
    from phase4_sizing_sim import load_pi, load_kaggle, build_extra_features
    from status_2026_04_21_replay import run_cell_nof3c

    # We need a timeout-parameterized version of run_cell_nof3c. The
    # original hard-codes 240; monkey-patch by exec'ing a local copy.
    import importlib
    import status_2026_04_21_replay as mod
    orig = mod.run_cell_nof3c

    def patched(feats, extra, sl, tp, knife_bps, ext_cap_bps,
                 leverage, e3_overlay):
        # Copy of original with timeout_bars replacing hard-coded 240.
        n = feats['n']
        mids = feats['mid']; highs = feats['high']; lows = feats['low']
        atr_bps_arr = feats['atr_bps']; imb = feats['imb5']
        dH_5 = feats['dH_5']; ret_60 = feats['ret_60']
        ret_150 = np.zeros(n); ret_150[150:] = (mids[150:] / mids[:-150] - 1) * 10000
        from phase4_sizing_sim import (
            CORE_PARAMS, PST_WIDTH_BPS, PST_HARD_FLOOR, PST_MAX_WAIT, H_THRESH,
        )
        from upgrade_backtest import candidate_signals
        cands = set(candidate_signals(feats, H_THRESH, CORE_PARAMS))
        trail_after, trail_bps = 150, 50
        equity = 10000.0; eq_curve = [equity]; in_trade=False
        entry_idx=entry_price=direction=0; notional=peak_pnl=0.0
        initial_notional=0.0
        tp_trailing=trailing_active=pst_active=False
        pst_peak_bps=0.0; pst_entry_bar=0; sl_current=-sl; e3_tightened=False
        trades=[]
        for i in range(n):
            if in_trade:
                d=direction
                if d==1:
                    worst=(lows[i]/entry_price-1)*10000
                    best=(highs[i]/entry_price-1)*10000
                else:
                    worst=-(highs[i]/entry_price-1)*10000
                    best=-(lows[i]/entry_price-1)*10000
                curr=d*(mids[i]/entry_price-1)*10000
                peak_pnl=max(peak_pnl,best); exit_reason=exit_pnl=None
                if worst<=sl_current:
                    exit_reason='sl'; exit_pnl=sl_current
                if exit_reason is None and not tp_trailing and peak_pnl>=trail_after:
                    tp_trailing=True
                if exit_reason is None and tp_trailing:
                    floor=max(peak_pnl-trail_bps,sl_current)
                    if worst<=floor:
                        exit_reason='tp_trail'; exit_pnl=max(floor,curr)
                if exit_reason is None and trailing_active:
                    tw=2.0*atr_bps_arr[i] if atr_bps_arr[i]>0 else 50
                    floor=max(peak_pnl-tw,sl_current)
                    if worst<=floor:
                        exit_reason='trail_stop'; exit_pnl=max(floor,curr)
                    elif best>=tp:
                        exit_reason='tp'; exit_pnl=tp
                if exit_reason is None and pst_active:
                    b=i-pst_entry_bar; pst_peak_bps=max(pst_peak_bps,best)
                    floor_b=max(pst_peak_bps-PST_WIDTH_BPS,sl_current)
                    if worst<=floor_b:
                        exit_reason='pst_trail'; exit_pnl=max(floor_b,curr)
                    elif worst<=PST_HARD_FLOOR:
                        exit_reason='pst_floored'; exit_pnl=PST_HARD_FLOOR
                    elif b>=PST_MAX_WAIT:
                        exit_reason='pst_timeout'; exit_pnl=curr
                if (exit_reason is None and not tp_trailing and not pst_active
                    and not trailing_active and best>=tp):
                    exit_reason='tp'; exit_pnl=tp
                if (exit_reason is None and e3_overlay and not e3_tightened
                    and (i-entry_idx)>=60 and peak_pnl<50):
                    sl_current=-25; e3_tightened=True
                # timeout at configurable bar count
                if (exit_reason is None and (i-entry_idx)>=timeout_bars
                    and not trailing_active and not pst_active):
                    if curr>0: trailing_active=True
                    else: pst_active=True; pst_entry_bar=i; pst_peak_bps=best
                if exit_reason:
                    fee=notional*FEE_BPS/10000
                    realized=(exit_pnl/10000)*notional-fee; equity+=realized
                    pos_pnl_bps=realized/initial_notional*10000 if initial_notional>0 else exit_pnl
                    trades.append({'pnl_bps':pos_pnl_bps,'direction':direction,
                                   'peak_bps':peak_pnl,'reason':exit_reason})
                    in_trade=False; tp_trailing=False; trailing_active=False
                    pst_active=False; e3_tightened=False
                eq_curve.append(equity); continue
            eq_curve.append(equity)
            if i not in cands or equity<=0: continue
            if not np.isnan(dH_5[i]) and dH_5[i]>=0: continue
            d=1 if imb[i]>0 else -1
            if knife_bps and d==1 and ret_60[i]<-knife_bps: continue
            if knife_bps and d==-1 and ret_60[i]>knife_bps: continue
            if ext_cap_bps is not None:
                r150=ret_150[i]
                if d==1 and r150>ext_cap_bps: continue
                if d==-1 and r150<-ext_cap_bps: continue
            margin=equity*0.90; notional=margin*leverage; initial_notional=notional
            equity-=notional*FEE_BPS/10000
            in_trade=True; entry_idx=i; entry_price=mids[i]; direction=d; peak_pnl=0
            tp_trailing=False; trailing_active=False; pst_active=False
            e3_tightened=False; sl_current=-sl
        nt=len(trades)
        eq_arr=np.asarray(eq_curve)
        pk=np.maximum.accumulate(eq_arr)
        dd=float(((pk-eq_arr)/pk*100).max()) if len(eq_arr)>1 else 0.0
        ret=(equity-10000)/100.0 if nt>0 else 0.0
        bps_list=[t['pnl_bps'] for t in trades]
        wins=[p for p in bps_list if p>0]; losses=[p for p in bps_list if p<=0]
        knife=sum(1 for t in trades if t['pnl_bps']<=0 and t['peak_bps']<20)
        reasons={}
        for t in trades: reasons[t['reason']]=reasons.get(t['reason'],0)+1
        return {'n_trades':nt,'compound_ret_pct':ret,'max_dd_pct':dd,
                'win_rate':len(wins)/nt if nt else 0.0,
                'avg_winner_bps':float(np.mean(wins)) if wins else 0.0,
                'avg_loser_bps':float(np.mean(losses)) if losses else 0.0,
                'knife_rate':knife/nt if nt else 0.0,'knife_catchers':knife,
                'mean_pnl_bps':float(np.mean(bps_list)) if bps_list else 0.0,
                'reason_distribution':reasons}
    # Run loader
    if substrate == "pi":
        feats, extra = load_pi()
        params = {"sl":50,"tp":200,"knife_bps":50,"ext_cap_bps":100}
    elif substrate == "kaggle":
        feats, extra = load_kaggle(60)
        params = {"sl":50,"tp":150,"knife_bps":100,"ext_cap_bps":None}
    else:
        raise ValueError(substrate)
    return patched(feats, extra, **params, leverage=5, e3_overlay=True)


# ---------------- main ----------------

def main():
    t0 = time.time()
    out: Dict = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "timeout_variants_bars": TIMEOUT_VARIANTS,
    }

    # --- LIVE 17-trade retrospective ---
    print("=" * 60)
    print("SUBSTRATE 1: 17 pre-sprint live trades (Kraken 1m OHLC)")
    print("=" * 60, flush=True)
    try:
        live = run_live_ablation()
        out["live_retrospective"] = live
    except Exception as e:
        out["live_retrospective"] = {"error": str(e)}
        print(f"LIVE FAILED: {e}", flush=True)

    # --- Pi IS substrate ---
    print("\n" + "=" * 60)
    print("SUBSTRATE 2: Pi IS 21.3d at phase4 engine, revised 6b config")
    print("=" * 60, flush=True)
    out["pi_is_by_variant"] = {}
    try:
        for T in TIMEOUT_VARIANTS:
            print(f"  T={T}...", flush=True)
            m = run_phase4_with_timeout("pi", T)
            out["pi_is_by_variant"][f"T_{T}"] = m
            print(f"    n={m['n_trades']}  ret={m['compound_ret_pct']:+.2f}%  "
                  f"DD={m['max_dd_pct']:.2f}%  WR={m['win_rate']*100:.1f}%", flush=True)
    except Exception as e:
        out["pi_is_by_variant"] = {"error": str(e)}
        print(f"PI FAILED: {e}", flush=True)

    # --- Kaggle OOS substrate ---
    print("\n" + "=" * 60)
    print("SUBSTRATE 3: Kaggle 60d OOS at phase4 engine, revised 6b config")
    print("=" * 60, flush=True)
    out["kaggle_oos_by_variant"] = {}
    try:
        for T in TIMEOUT_VARIANTS:
            print(f"  T={T}...", flush=True)
            m = run_phase4_with_timeout("kaggle", T)
            out["kaggle_oos_by_variant"][f"T_{T}"] = m
            print(f"    n={m['n_trades']}  ret={m['compound_ret_pct']:+.2f}%  "
                  f"DD={m['max_dd_pct']:.2f}%  WR={m['win_rate']*100:.1f}%", flush=True)
    except Exception as e:
        out["kaggle_oos_by_variant"] = {"error": str(e)}
        print(f"KAGGLE FAILED: {e}", flush=True)

    out["elapsed_sec"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {OUT}  ({out['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
