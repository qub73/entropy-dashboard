"""
Sprint v1.5 close-out helper -- compute revised-6b OOS numbers for the
dashboard tagline. The original +103% @5x claim came from Phase 4's
lev5x_ewoff_e3on cell, which had F3c baked in. Revised 6b drops F3c.

This re-runs that exact cell (lev=5, entropy_weighted=OFF, e3_overlay=ON,
timeout_trail always ON) on Kaggle ETH/USDT 60d OOS with F3c disabled.

Output: printed to stdout + reports/sprint_v15_revised_6b_shadow.json
"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ALGO = HERE.parent
sys.path.insert(0, str(ALGO))
sys.path.insert(0, str(HERE))

from phase4_sizing_sim import (
    load_kaggle, build_extra_features,
    CORE_PARAMS, PST_WIDTH_BPS, PST_HARD_FLOOR, PST_MAX_WAIT, H_THRESH,
)
from upgrade_backtest import candidate_signals

OUT = ALGO / "reports" / "sprint_v15_revised_6b_shadow.json"


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
    equity_curve = []
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
        # NOTE: F3c gate INTENTIONALLY REMOVED (revised 6b).

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
    if nt == 0:
        return {"trades": 0, "compound_ret_pct": 0, "max_dd": 0,
                "win_rate": 0, "pf": 0, "knife_rate": 0,
                "trades_long": 0, "trades_short": 0}
    pnls_usd = [t["pnl_usd"] for t in trades]
    wins = sum(1 for p in pnls_usd if p > 0)
    eq_arr = np.asarray(equity_curve)
    pk = np.maximum.accumulate(eq_arr)
    mdd = float(((pk - eq_arr) / pk * 100).max())
    ret = (equity - 10000)/10000*100
    bps = [t["pnl_bps"] for t in trades]
    win_bps = sum(b for b in bps if b > 0)
    loss_bps = sum(b for b in bps if b <= 0)
    pf = abs(win_bps/loss_bps) if loss_bps != 0 else float("inf")
    knife = sum(1 for t in trades if t["pnl_bps"] <= 0 and t["peak_bps"] < 20)
    return {
        "trades": nt, "wins": wins, "win_rate": wins/nt,
        "compound_ret_pct": ret, "max_dd": mdd, "pf": pf,
        "knife_rate": knife/nt, "knife_catchers": knife,
        "trades_long": sum(1 for t in trades if t["direction"] == 1),
        "trades_short": sum(1 for t in trades if t["direction"] == -1),
    }


def main():
    import phase2_filter_ablation_v2 as P2
    # ensure H_THRESH default
    P2.H_THRESH = H_THRESH
    print("Loading Kaggle 60d OOS...", flush=True)
    kg_feats, kg_extra = load_kaggle(60)
    print(f"  {kg_feats['n']:,} bars")

    # Kaggle refit params (from phase2_kaggle_refit.json): sl=50, tp=150,
    # knife=100, ext=None
    kg_params = {"sl": 50, "tp": 150, "knife_bps": 100, "ext_cap_bps": None}
    print(f"  using Kaggle refit params: {kg_params}")

    print("\n[cell] lev=5, entropy_weighted=OFF, e3_overlay=ON, F3c=OFF (revised 6b)")
    r = run_cell_nof3c(kg_feats, kg_extra,
                        sl=kg_params["sl"], tp=kg_params["tp"],
                        knife_bps=kg_params["knife_bps"],
                        ext_cap_bps=kg_params["ext_cap_bps"],
                        leverage=5, e3_overlay=True)
    for k, v in r.items():
        if isinstance(v, float): print(f"  {k}: {v:.4f}")
        else:                    print(f"  {k}: {v}")

    # Phase 4 reference for context
    phase4 = json.load(open(ALGO / "reports" / "phase4_sizing_sim.json"))
    orig = next((c for c in phase4["cells"] if c["name"] == "lev5x_ewoff_e3on"),
                 None)
    orig_oos = orig["oos"] if orig else None

    out = {
        "cell": "lev5x_ewoff_e3on",
        "scenario": "revised 6b (F3c removed, timeout_trail+E3+5x only)",
        "data": "Kaggle ETH/USDT 60d OOS, refit params (sl=50,tp=150,knife=100,no ext)",
        "original_with_f3c_oos": orig_oos,
        "revised_no_f3c_oos": r,
        "delta_vs_original": (
            {
                "compound_ret_pct": r["compound_ret_pct"] - orig_oos["compound_ret_pct"],
                "max_dd": r["max_dd"] - orig_oos["max_dd"],
                "pf": r["pf"] - orig_oos["pf"],
                "knife_rate": r["knife_rate"] - orig_oos["knife_rate"],
                "trades": r["trades"] - orig_oos["trades"],
            } if orig_oos else None
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {OUT}")
    if orig_oos:
        print("\nContext -- original lev5x_ewoff_e3on WITH F3c (from phase4):")
        print(f"  ret={orig_oos['compound_ret_pct']:+.2f}%  "
              f"DD={orig_oos['max_dd']:.2f}%  "
              f"PF={orig_oos['pf']:.2f}  "
              f"knife={orig_oos['knife_rate']*100:.1f}%  "
              f"trades={orig_oos['trades']}")


if __name__ == "__main__":
    main()
