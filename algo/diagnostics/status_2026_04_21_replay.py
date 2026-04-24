"""
One-off read-only replay helper for the 2026-04-21 status audit.

Replays the revised 6b config (timeout_trail ON, E3 ON, F3c OFF, leverage 5x)
on the 21-day Pi Feb-Mar baseline and on the Kaggle 60d OOS slice, then
reports n_trades, compound_ret_pct, max_dd, sharpe_daily, profit_factor,
knife_rate, win_rate. Also runs a 1x leverage variant per substrate so we
can separate path-of-trades DD from leverage scaling.

No files written. Prints JSON to stdout for the status doc to consume.
"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ALGO = HERE.parent
sys.path.insert(0, str(ALGO))
sys.path.insert(0, str(HERE))

from phase4_sizing_sim import (
    load_pi, load_kaggle, build_extra_features,
    CORE_PARAMS, PST_WIDTH_BPS, PST_HARD_FLOOR, PST_MAX_WAIT, H_THRESH,
)
from upgrade_backtest import candidate_signals


def run_cell_nof3c(feats, extra, sl, tp, knife_bps, ext_cap_bps,
                    leverage, e3_overlay):
    """Copy of phase4.run_cell with F3c gate removed."""
    n = feats['n']
    mids = feats['mid']; highs = feats['high']; lows = feats['low']
    atr_bps_arr = feats['atr_bps']; imb = feats['imb5']
    dH_5 = feats['dH_5']; ret_60 = feats['ret_60']
    ret_150 = np.zeros(n); ret_150[150:] = (mids[150:] / mids[:-150] - 1) * 10000

    cands = set(candidate_signals(feats, H_THRESH, CORE_PARAMS))
    trail_after, trail_bps = 150, 50
    equity = 10000.0
    equity_curve = [equity]
    in_trade = False
    entry_idx = entry_price = direction = 0
    notional = peak_pnl = 0.0
    initial_notional = 0.0
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
                bars_in_trail = i - pst_entry_bar
                pst_peak_bps = max(pst_peak_bps, best_bps)
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
            if (exit_reason is None and e3_overlay and not e3_tightened
                and (i - entry_idx) >= 60 and peak_pnl < 50):
                sl_current = -25
                e3_tightened = True
            if exit_reason is None and (i - entry_idx) >= 240 \
               and not trailing_active and not pst_active:
                if curr_bps > 0:
                    trailing_active = True
                else:
                    pst_active = True; pst_entry_bar = i
                    pst_peak_bps = best_bps

            if exit_reason:
                fee = notional * 5.0 / 10000
                realized = (exit_pnl / 10000) * notional - fee
                equity += realized
                pos_pnl_bps = (realized / initial_notional * 10000
                               if initial_notional > 0 else exit_pnl)
                trades.append({"direction": direction, "pnl_bps": pos_pnl_bps,
                                "pnl_usd": realized, "reason": exit_reason,
                                "peak_bps": peak_pnl})
                in_trade = False
                tp_trailing = False; trailing_active = False
                pst_active = False; e3_tightened = False
            equity_curve.append(equity)
            continue

        equity_curve.append(equity)
        if i not in cands or equity <= 0: continue
        if not np.isnan(dH_5[i]) and dH_5[i] >= 0: continue
        d = 1 if imb[i] > 0 else -1
        if knife_bps and d == 1 and ret_60[i] < -knife_bps: continue
        if knife_bps and d == -1 and ret_60[i] > knife_bps: continue
        if ext_cap_bps is not None:
            r150 = ret_150[i]
            if d == 1 and r150 > ext_cap_bps: continue
            if d == -1 and r150 < -ext_cap_bps: continue

        margin = equity * 0.90
        notional = margin * leverage
        initial_notional = notional
        equity -= notional * 5.0 / 10000
        in_trade = True; entry_idx = i; entry_price = mids[i]
        direction = d; peak_pnl = 0
        tp_trailing = False; trailing_active = False; pst_active = False
        e3_tightened = False
        sl_current = -sl

    if in_trade:
        curr = direction * (mids[-1]/entry_price - 1) * 10000
        fee = notional * 5.0 / 10000
        realized = (curr/10000) * notional - fee
        equity += realized
        pos_pnl_bps = (realized / initial_notional * 10000
                       if initial_notional > 0 else curr)
        trades.append({"direction": direction, "pnl_bps": pos_pnl_bps,
                        "pnl_usd": realized, "reason": "end",
                        "peak_bps": peak_pnl})
        equity_curve.append(equity)

    nt = len(trades)
    eq_arr = np.asarray(equity_curve)
    if len(eq_arr) > 1:
        pk = np.maximum.accumulate(eq_arr)
        dd = float(((pk - eq_arr) / pk * 100).max())
    else:
        dd = 0.0
    ret = (equity - 10000)/10000*100 if nt > 0 else 0.0
    pnls_usd = [t["pnl_usd"] for t in trades]
    wins = sum(1 for p in pnls_usd if p > 0)
    bps = [t["pnl_bps"] for t in trades]
    win_bps = sum(b for b in bps if b > 0)
    loss_bps = sum(b for b in bps if b <= 0)
    pf = (abs(win_bps/loss_bps) if loss_bps != 0
          else (float("inf") if win_bps > 0 else 0.0))
    knife = sum(1 for t in trades if t["pnl_bps"] <= 0 and t["peak_bps"] < 20)

    bars_per_day = 1440
    if len(eq_arr) > bars_per_day:
        eod_idx = np.arange(bars_per_day, len(eq_arr), bars_per_day)
        eod_eq = eq_arr[eod_idx]
        daily_ret = np.diff(eod_eq) / eod_eq[:-1]
        sharpe = ((daily_ret.mean() / daily_ret.std()) * np.sqrt(365)
                  if len(daily_ret) > 1 and daily_ret.std() > 0 else 0.0)
    else:
        sharpe = 0.0

    return {
        "trades": nt, "wins": wins,
        "win_rate": (wins/nt if nt > 0 else 0.0),
        "compound_ret_pct": ret, "max_dd": dd, "pf": pf,
        "sharpe_daily": sharpe,
        "knife_rate": (knife/nt if nt > 0 else 0.0),
        "knife_catchers": knife,
        "trades_long": sum(1 for t in trades if t["direction"] == 1),
        "trades_short": sum(1 for t in trades if t["direction"] == -1),
        "dd_method": "max peak-to-trough on per-bar compound equity curve (includes open-trade floating PnL)",
    }


def main():
    print("Loading Pi 21-day baseline...", flush=True)
    pi_feats, pi_extra = load_pi()
    print(f"  {pi_feats['n']:,} bars ({pi_feats['n']/1440:.1f} days)", flush=True)

    print("\nLoading Kaggle 60d OOS...", flush=True)
    kg_feats, kg_extra = load_kaggle(60)
    print(f"  {kg_feats['n']:,} bars", flush=True)

    # Pi params: live defaults
    pi_params = {"sl": 50, "tp": 200, "knife_bps": 50, "ext_cap_bps": 100}
    # Kaggle v2 refit params
    kg_params = {"sl": 50, "tp": 150, "knife_bps": 100, "ext_cap_bps": None}

    all_results = {"params": {"pi": pi_params, "kaggle": kg_params}}

    for substrate, feats, extra, params in [
        ("pi", pi_feats, pi_extra, pi_params),
        ("kaggle", kg_feats, kg_extra, kg_params),
    ]:
        print(f"\n=== {substrate.upper()} -- no F3c, 5x ===")
        r5 = run_cell_nof3c(feats, extra, **params, leverage=5, e3_overlay=True)
        for k, v in r5.items():
            if isinstance(v, float): print(f"  {k}: {v:.4f}")
            else:                    print(f"  {k}: {v}")
        all_results[f"{substrate}_5x_noF3c"] = r5

        if r5["max_dd"] > 20.0:
            print(f"  DD>20% -> running 1x sanity variant")
            r1 = run_cell_nof3c(feats, extra, **params, leverage=1, e3_overlay=True)
            for k, v in r1.items():
                if isinstance(v, float): print(f"    {k}: {v:.4f}")
                else:                    print(f"    {k}: {v}")
            all_results[f"{substrate}_1x_noF3c_sanity"] = r1

    print("\n--- JSON RESULT ---")
    print(json.dumps(all_results, indent=2, default=str))


if __name__ == "__main__":
    main()
