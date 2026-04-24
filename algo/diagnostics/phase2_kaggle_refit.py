"""
Kaggle-fit baseline parameters for Phase 2 OOS fairness.

Splits the 1-year Kaggle ETH/USDT dataset into:
   refit window  = first 60 days of data (separate from eval)
   (gap)         = skipped middle ~245 days to avoid regime overlap
   eval window   = last 60 days (same window as v1 OOS)

Grid-searches (sl, tp, knife_bps, ext_cap_bps) on the refit window
against the current live-style engine. H_thresh stays at 0.4352
(out-of-scope per the original prompt guardrails). Picks by Calmar.

Writes reports/phase2_kaggle_refit.json with:
  - refit window coverage
  - eval window coverage
  - full grid results
  - selected parameters
Phase 2 v2 then reads this file to parameterize Kaggle OOS runs.
"""
import json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ALGO = HERE.parent
sys.path.insert(0, str(ALGO))

from ob_entropy import (
    NUM_STATES_OB, rolling_entropy_ob,
    compute_ob_features, classify_ob_states,
)
from kaggle_ob_trainer import (
    resample_pi_to_1min,
    load_kaggle_csv, parse_kaggle_to_ob_features,
)
from upgrade_backtest import make_features, candidate_signals

OUT = ALGO / "reports" / "phase2_kaggle_refit.json"
H_THRESH = 0.4352
CORE_PARAMS = {"imb_min": 0.05, "spread_max": 20, "ret_low": 20, "ret_high": 80}


def run_engine(feats, sl, tp, knife_bps, ext_cap_bps,
               trail_after=150, trail_bps=50):
    """Same engine as production. Returns summary dict."""
    n = feats['n']
    mids = feats['mid']; highs = feats['high']; lows = feats['low']
    atr_bps = feats['atr_bps']; imb = feats['imb5']
    dH_5 = feats['dH_5']; ret_60 = feats['ret_60']
    ret_150 = np.zeros(n)
    ret_150[150:] = (mids[150:] / mids[:-150] - 1) * 10000
    cands = set(candidate_signals(feats, H_THRESH, CORE_PARAMS))

    equity = 10000.0; in_trade = False
    entry_idx = entry_price = direction = 0
    notional = peak_pnl = 0.0
    tp_trailing = trailing_active = False
    trades = []
    for i in range(n):
        if in_trade:
            d = direction
            if d == 1:
                worst = (lows[i]/entry_price-1)*10000
                best  = (highs[i]/entry_price-1)*10000
            else:
                worst = -(highs[i]/entry_price-1)*10000
                best  = -(lows[i]/entry_price-1)*10000
            curr = d * (mids[i]/entry_price - 1) * 10000
            peak_pnl = max(peak_pnl, best)
            exit_reason = exit_pnl = None
            if not tp_trailing and peak_pnl >= trail_after:
                tp_trailing = True
            if tp_trailing:
                floor = max(peak_pnl - trail_bps, -sl)
                if worst <= floor:
                    exit_reason = 'tp_trail'; exit_pnl = max(floor, curr)
            elif trailing_active:
                tw = 2.0 * atr_bps[i] if atr_bps[i] > 0 else 50
                floor = max(peak_pnl - tw, -sl)
                if worst <= floor:
                    exit_reason = 'trail_stop'; exit_pnl = max(floor, curr)
                elif best >= tp:
                    exit_reason = 'tp'; exit_pnl = tp
            else:
                if worst <= -sl:
                    exit_reason = 'sl'; exit_pnl = -sl
                elif best >= tp:
                    exit_reason = 'tp'; exit_pnl = tp
                elif (i - entry_idx) >= 240:
                    if curr > 0: trailing_active = True
                    else: exit_reason = 'timeout'; exit_pnl = curr
            if exit_reason:
                fee = notional*5.0/10000
                realized = (exit_pnl/10000)*notional - fee
                equity += realized
                trades.append({"pnl_bps": exit_pnl, "pnl_usd": realized,
                               "direction": direction, "reason": exit_reason,
                               "peak_bps": peak_pnl})
                in_trade = False
                tp_trailing = False; trailing_active = False
        else:
            if i not in cands or equity <= 0: continue
            if not np.isnan(dH_5[i]) and dH_5[i] >= 0: continue
            d = 1 if imb[i] > 0 else -1
            if knife_bps and d == 1 and ret_60[i] < -knife_bps: continue
            if knife_bps and d == -1 and ret_60[i] > knife_bps: continue
            if ext_cap_bps is not None:
                r150 = ret_150[i]
                if d == 1 and r150 > ext_cap_bps: continue
                if d == -1 and r150 < -ext_cap_bps: continue
            margin = equity*0.90; notional = margin*10
            equity -= notional*5.0/10000
            in_trade = True; entry_idx = i; entry_price = mids[i]
            direction = d; peak_pnl = 0
            tp_trailing = False; trailing_active = False

    if in_trade:
        curr = direction*(mids[-1]/entry_price-1)*10000
        fee = notional*5.0/10000
        realized = (curr/10000)*notional - fee
        equity += realized
        trades.append({"pnl_bps": curr, "pnl_usd": realized,
                       "direction": direction, "reason": "end",
                       "peak_bps": peak_pnl})

    nt = len(trades)
    if nt == 0:
        return {"trades": 0, "compound_ret_pct": 0, "max_dd": 0,
                "win_rate": 0, "pf": 0, "calmar": 0,
                "knife_catchers": 0, "knife_rate": 0}
    pnls = [t["pnl_usd"] for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    pk = 10000; cum = 10000; mdd = 0
    for p in pnls:
        cum += p; pk = max(pk, cum)
        mdd = max(mdd, (pk-cum)/pk*100)
    ret = (equity - 10000)/10000*100
    bps = [t["pnl_bps"] for t in trades]
    win_bps = sum(b for b in bps if b > 0)
    loss_bps = sum(b for b in bps if b <= 0)
    pf = abs(win_bps/loss_bps) if loss_bps != 0 else float("inf")
    knife = sum(1 for t in trades
                if t["pnl_bps"] <= 0 and t["peak_bps"] < 20)
    return {
        "trades": nt, "wins": wins, "win_rate": wins/nt,
        "compound_ret_pct": ret, "max_dd": mdd,
        "calmar": ret/mdd if mdd > 0 else (ret if ret != 0 else 0),
        "pf": pf,
        "knife_catchers": knife, "knife_rate": knife/nt,
    }


def build_features_for_slice(df_slice):
    df_1m = resample_pi_to_1min(df_slice)
    df_1m = compute_ob_features(df_1m)
    df_1m = classify_ob_states(df_1m, window=60)
    ent = rolling_entropy_ob(df_1m['state_ob'].values, NUM_STATES_OB, 30)
    feats = make_features(df_1m, ent)
    return feats, df_1m.index[0], df_1m.index[-1]


def main():
    print("Loading Kaggle ETH/USDT (full year)...", flush=True)
    raw = load_kaggle_csv("data/kaggle/ETH_USDT.csv")
    df_all = parse_kaggle_to_ob_features(raw)
    df_all = compute_ob_features(df_all)
    df_all_1m = resample_pi_to_1min(df_all)
    total_bars = len(df_all_1m)
    total_days = total_bars / 1440
    print(f"  {total_bars} bars, {total_days:.1f} days", flush=True)

    # Split: first 60 days = refit, last 60 days = eval. Large gap between.
    bars_60d = 60 * 1440
    if total_bars < 2 * bars_60d:
        raise SystemExit(f"Not enough Kaggle data: need 120d, have {total_days:.1f}d")

    refit_slice = df_all_1m.iloc[:bars_60d].copy()
    eval_slice = df_all_1m.iloc[-bars_60d:].copy()
    try:
        gap_days = (eval_slice.index[0] - refit_slice.index[-1]).total_seconds() / 86400
        refit_start = str(refit_slice.index[0]); refit_end = str(refit_slice.index[-1])
        eval_start  = str(eval_slice.index[0]);  eval_end  = str(eval_slice.index[-1])
    except Exception:
        # integer index fallback: infer from bar count
        gap_days = (total_bars - 2 * bars_60d) / 1440
        refit_start = f"bar {0}"; refit_end = f"bar {bars_60d-1}"
        eval_start = f"bar {total_bars - bars_60d}"; eval_end = f"bar {total_bars-1}"
    print(f"  Refit window: {refit_start} -> {refit_end} ({len(refit_slice)} bars)")
    print(f"  Eval  window: {eval_start} -> {eval_end} ({len(eval_slice)} bars)")
    print(f"  Gap between: {gap_days:.0f} days", flush=True)

    # Build features on refit slice (re-run feature pipeline on sliced df)
    refit_slice = compute_ob_features(refit_slice)
    refit_slice = classify_ob_states(refit_slice, window=60)
    ent = rolling_entropy_ob(refit_slice['state_ob'].values, NUM_STATES_OB, 30)
    feats_refit = make_features(refit_slice, ent)

    # Grid search -- keep it bounded
    sl_grid    = [35, 50, 65, 80]
    tp_grid    = [150, 200]
    knife_grid = [None, 50, 100]
    ext_grid   = [None, 100, 150]

    grid = []
    total = len(sl_grid)*len(tp_grid)*len(knife_grid)*len(ext_grid)
    print(f"\nGrid search ({total} cells) on refit window...", flush=True)
    for sl in sl_grid:
        for tp in tp_grid:
            if tp <= sl: continue
            for knife in knife_grid:
                for ext in ext_grid:
                    r = run_engine(feats_refit, sl, tp, knife, ext)
                    cell = {
                        "sl": sl, "tp": tp, "knife_bps": knife,
                        "ext_cap_bps": ext,
                        **r,
                    }
                    grid.append(cell)

    # Live config as reference
    live = run_engine(feats_refit, 50, 200, 50, 100)

    # Select by Calmar; require trades >= 10 to avoid single-lucky-trade fits
    eligible = [g for g in grid if g["trades"] >= 10 and g["pf"] > 1.0]
    if not eligible:
        print("  WARN: no cell had >=10 trades and PF>1. Falling back to all.")
        eligible = grid
    eligible.sort(key=lambda g: g["calmar"], reverse=True)
    chosen = eligible[0]

    # Evaluate chosen on eval slice too (transparency only; Phase 2 v2
    # will re-evaluate all filter variants there)
    eval_slice = compute_ob_features(eval_slice)
    eval_slice = classify_ob_states(eval_slice, window=60)
    ent_eval = rolling_entropy_ob(eval_slice['state_ob'].values, NUM_STATES_OB, 30)
    feats_eval = make_features(eval_slice, ent_eval)
    chosen_on_eval = run_engine(feats_eval, chosen["sl"], chosen["tp"],
                                 chosen["knife_bps"], chosen["ext_cap_bps"])
    live_on_eval = run_engine(feats_eval, 50, 200, 50, 100)

    out = {
        "refit_window": {
            "start": refit_start, "end": refit_end,
            "n_bars": len(refit_slice),
        },
        "eval_window": {
            "start": eval_start, "end": eval_end,
            "n_bars": len(eval_slice),
        },
        "gap_days_between": gap_days,
        "h_thresh_used": H_THRESH,
        "grid_size": total,
        "selected_refit_baseline": chosen,
        "live_config_on_refit_window": live,
        "selected_on_eval_window": chosen_on_eval,
        "live_config_on_eval_window": live_on_eval,
        "top_5_grid_cells_by_calmar": eligible[:5],
        "full_grid": grid,
    }
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n=== REFIT RESULT ===")
    print(f"Live config on refit window: trades={live['trades']}, "
          f"ret={live['compound_ret_pct']:+.1f}%, DD={live['max_dd']:.1f}%, "
          f"PF={live['pf']:.2f}, Calmar={live['calmar']:.2f}")
    print(f"Selected refit baseline: "
          f"sl={chosen['sl']}, tp={chosen['tp']}, "
          f"knife={chosen['knife_bps']}, ext={chosen['ext_cap_bps']}")
    print(f"  on refit window: trades={chosen['trades']}, "
          f"ret={chosen['compound_ret_pct']:+.1f}%, DD={chosen['max_dd']:.1f}%, "
          f"PF={chosen['pf']:.2f}, Calmar={chosen['calmar']:.2f}")
    print(f"\nTransparency check (selected on eval window):")
    print(f"  trades={chosen_on_eval['trades']}, "
          f"ret={chosen_on_eval['compound_ret_pct']:+.1f}%, "
          f"DD={chosen_on_eval['max_dd']:.1f}%, PF={chosen_on_eval['pf']:.2f}")
    print(f"Transparency check (live config on eval window):")
    print(f"  trades={live_on_eval['trades']}, "
          f"ret={live_on_eval['compound_ret_pct']:+.1f}%, "
          f"DD={live_on_eval['max_dd']:.1f}%, PF={live_on_eval['pf']:.2f}")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
