"""
Phase 4 sizing + leverage sim.

Test matrix (8 cells):
  leverage           in {5x, 10x}
  entropy-weighted   in {off, on}
  E3 overlay         in {off, on}

Fixed ON: F3c entry gate, (b) timeout_trail (Phase 3 promoted).
Fixed OFF: all other Phase-3 exit variants.

Entropy-weighted sizing (per Addendum #1):
    z = (H_threshold - H_at_entry) / H_threshold        # 0 at thresh, ->1 as H->0
    size_multiplier = clamp(0.5 + 1.5 * z, 0.5, 1.5)
    notional = margin * leverage * size_multiplier

Datasets:
  IS  = Pi Feb 18 - Apr 7 with live exit params (SL=50, TP=200)
  OOS = last 60d Kaggle ETH/USDT with refit params (SL=50, TP=150)

Same corrected promotion rule as Phase 3.
Output: reports/phase4_sizing_sim.json
"""
import json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ALGO = HERE.parent
sys.path.insert(0, str(ALGO))

from ob_entropy import (
    NUM_STATES_OB, rolling_entropy_ob, load_orderbook_range,
    compute_ob_features, classify_ob_states,
)
from kaggle_ob_trainer import (
    resample_pi_to_1min,
    load_kaggle_csv, parse_kaggle_to_ob_features,
)
from upgrade_backtest import make_features, candidate_signals

OUT = ALGO / "reports" / "phase4_sizing_sim.json"
REFIT_FILE = ALGO / "reports" / "phase2_kaggle_refit.json"
H_THRESH = 0.4352
CORE_PARAMS = {"imb_min": 0.05, "spread_max": 20, "ret_low": 20, "ret_high": 80}

# post_signal_trail (timeout_trail uses these)
PST_WIDTH_BPS = 20
PST_HARD_FLOOR = -60
PST_MAX_WAIT = 30


# ---------- helpers ----------
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

def build_extra_features(feats):
    mids = feats["mid"]; highs = feats["high"]; lows = feats["low"]
    atr_30 = wilder_atr(highs, lows, mids, 30)
    return {"atr_30": atr_30,
            "ema_fast": ema_series(mids, 30),
            "ema_slow": ema_series(mids, 150)}


def size_multiplier(H_at_entry):
    """Entropy-weighted scale factor. clamp(0.5 + 1.5 * (H_thresh - H)/H_thresh, 0.5, 1.5)."""
    if H_at_entry is None or np.isnan(H_at_entry) or H_THRESH == 0:
        return 1.0
    z = (H_THRESH - H_at_entry) / H_THRESH
    m = 0.5 + 1.5 * z
    return max(0.5, min(1.5, m))


def run_cell(feats, extra, sl, tp, knife_bps, ext_cap_bps,
              leverage, entropy_weighted, e3_overlay):
    """Single Phase-4 simulation. F3c always on, timeout_trail always on."""
    n = feats['n']
    mids = feats['mid']; highs = feats['high']; lows = feats['low']
    atr_bps_arr = feats['atr_bps']; imb = feats['imb5']; H = feats['H']
    dH_5 = feats['dH_5']; ret_60 = feats['ret_60']
    ret_150 = np.zeros(n); ret_150[150:] = (mids[150:] / mids[:-150] - 1) * 10000
    atr_30 = extra['atr_30']
    ema_fast = extra['ema_fast']; ema_slow = extra['ema_slow']

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
    H_at_entry = None
    size_mult = 1.0
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

            # E3 time-decayed SL
            if (exit_reason is None and e3_overlay and not e3_tightened
                and (i - entry_idx) >= 60 and peak_pnl < 50):
                sl_current = -25
                e3_tightened = True

            # Timeout branch: convert to ATR-trail if in profit, else
            # to post_signal_trail (b timeout_trail always ON)
            if exit_reason is None and (i - entry_idx) >= 240 \
               and not trailing_active and not pst_active:
                if curr_bps > 0:
                    trailing_active = True
                else:
                    pst_active = True
                    pst_entry_bar = i
                    pst_peak_bps = best_bps

            if exit_reason:
                fee = notional * 5.0 / 10000
                realized = (exit_pnl / 10000) * notional - fee
                equity += realized
                pos_pnl_bps = (realized / initial_notional * 10000
                               if initial_notional > 0 else exit_pnl)
                trades.append({
                    "direction": direction,
                    "pnl_bps": pos_pnl_bps,
                    "pnl_usd": realized,
                    "reason": exit_reason,
                    "peak_bps": peak_pnl,
                    "size_mult": size_mult,
                    "H_at_entry": H_at_entry,
                })
                in_trade = False
                tp_trailing = False; trailing_active = False
                pst_active = False; e3_tightened = False
            equity_curve.append(equity)
            continue

        # -- entry evaluation (F3c always on) --
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

        # F3c
        if atr_30[i] > 0:
            norm = (ema_fast[i] - ema_slow[i]) / atr_30[i]
        else:
            norm = 0.0
        if d == 1 and norm < 0.3: continue
        if d == -1 and norm > -0.3: continue

        # Sizing
        H_entry = float(H[i]) if (i < len(H) and not np.isnan(H[i])) else None
        size_mult = size_multiplier(H_entry) if entropy_weighted else 1.0
        margin = equity * 0.90
        notional = margin * leverage * size_mult
        initial_notional = notional
        equity -= notional * 5.0 / 10000
        in_trade = True; entry_idx = i; entry_price = mids[i]
        direction = d; peak_pnl = 0
        tp_trailing = False; trailing_active = False; pst_active = False
        e3_tightened = False
        H_at_entry = H_entry
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
                       "peak_bps": peak_pnl,
                       "size_mult": size_mult, "H_at_entry": H_at_entry})
        equity_curve.append(equity)

    # Summary stats
    nt = len(trades)
    if nt == 0:
        return {"trades": 0, "compound_ret_pct": 0, "max_dd": 0,
                "win_rate": 0, "pf": 0, "knife_rate": 0,
                "sharpe_daily": 0, "kelly_f": None,
                "avg_size_mult": 1.0}

    pnls_usd = [t["pnl_usd"] for t in trades]
    wins = sum(1 for p in pnls_usd if p > 0)
    eq_arr = np.asarray(equity_curve)
    if len(eq_arr) > 1:
        pk = np.maximum.accumulate(eq_arr)
        dd_curve = (pk - eq_arr) / pk * 100
        mdd = float(dd_curve.max())
    else:
        mdd = 0.0
    ret = (equity - 10000)/10000*100
    bps = [t["pnl_bps"] for t in trades]
    win_bps = sum(b for b in bps if b > 0)
    loss_bps = sum(b for b in bps if b <= 0)
    pf = abs(win_bps/loss_bps) if loss_bps != 0 else float("inf")

    # Daily Sharpe from daily equity returns
    bars_per_day = 1440
    if len(eq_arr) > bars_per_day:
        eod_idx = np.arange(bars_per_day, len(eq_arr), bars_per_day)
        eod_eq = eq_arr[eod_idx]
        daily_ret = np.diff(eod_eq) / eod_eq[:-1]
        sharpe = ((daily_ret.mean() / daily_ret.std()) * np.sqrt(365)
                  if daily_ret.std() > 0 else 0.0)
    else:
        sharpe = 0.0

    # Kelly fraction: f = (p*b - q) / b where b = avg_win / |avg_loss|
    wins_usd = [p for p in pnls_usd if p > 0]
    losses_usd = [p for p in pnls_usd if p <= 0]
    if wins_usd and losses_usd:
        avg_w = np.mean(wins_usd); avg_l = abs(np.mean(losses_usd))
        b = avg_w / avg_l if avg_l > 0 else 0
        p = wins / nt; q = 1 - p
        kelly = (p * b - q) / b if b > 0 else None
    else:
        kelly = None

    knife = sum(1 for t in trades if t["pnl_bps"] <= 0 and t["peak_bps"] < 20)

    return {
        "trades": nt, "wins": wins, "win_rate": wins/nt,
        "compound_ret_pct": ret, "max_dd": mdd,
        "pf": pf,
        "knife_rate": knife/nt, "knife_catchers": knife,
        "sharpe_daily": sharpe,
        "kelly_f": kelly,
        "avg_size_mult": float(np.mean([t["size_mult"] for t in trades])),
        "trades_long": sum(1 for t in trades if t["direction"] == 1),
        "trades_short": sum(1 for t in trades if t["direction"] == -1),
    }


def load_pi():
    print("Loading ETH Pi orderbook...", flush=True)
    df = load_orderbook_range("data/orderbook_pi", "ob_PF_ETHUSD")
    df_1m = resample_pi_to_1min(df)
    df_1m = compute_ob_features(df_1m)
    df_1m = classify_ob_states(df_1m, window=60)
    ent = rolling_entropy_ob(df_1m['state_ob'].values, NUM_STATES_OB, 30)
    feats = make_features(df_1m, ent)
    extra = build_extra_features(feats)
    return feats, extra


def load_kaggle(days=60):
    print(f"Loading Kaggle last {days}d...", flush=True)
    raw = load_kaggle_csv("data/kaggle/ETH_USDT.csv")
    df = parse_kaggle_to_ob_features(raw)
    df = compute_ob_features(df)
    df_1m = resample_pi_to_1min(df)
    df_1m = compute_ob_features(df_1m)
    bars = days * 1440
    if len(df_1m) > bars: df_1m = df_1m.iloc[-bars:].copy()
    df_1m = classify_ob_states(df_1m, window=60)
    ent = rolling_entropy_ob(df_1m['state_ob'].values, NUM_STATES_OB, 30)
    feats = make_features(df_1m, ent)
    extra = build_extra_features(feats)
    return feats, extra


def evaluate_against_baseline(r, baseline):
    """Same Phase-3 promotion rule."""
    fails = []
    is_ = r["in_sample"]; oos = r["oos"]
    bis = baseline["in_sample"]; boos = baseline["oos"]
    if is_["compound_ret_pct"] <= 0:
        fails.append(f"IS return {is_['compound_ret_pct']:+.2f}% <= 0")
    if oos["compound_ret_pct"] <= 0:
        fails.append(f"OOS return {oos['compound_ret_pct']:+.2f}% <= 0")
    pf_imp_oos = ((oos["pf"] - boos["pf"]) / abs(boos["pf"]) * 100
                  if boos["pf"] else None)
    if pf_imp_oos is None or pf_imp_oos < 15:
        fails.append(f"OOS PF improvement {pf_imp_oos}% < 15%")
    if oos["max_dd"] > boos["max_dd"]:
        fails.append(f"OOS max_dd {oos['max_dd']:.2f}% > baseline {boos['max_dd']:.2f}%")
    if is_["trades"] < 15:
        fails.append(f"IS trades {is_['trades']} < 15")
    if oos["trades"] < 30:
        fails.append(f"OOS trades {oos['trades']} < 30")
    for ds_name, ds, base_ds in [("IS", is_, bis), ("OOS", oos, boos)]:
        for metric, direction in [("pf", "up"), ("win_rate", "up"),
                                   ("knife_rate", "down")]:
            bv = base_ds.get(metric, 0); cv = ds.get(metric, 0)
            if direction == "up" and bv > 0:
                if (cv - bv) / abs(bv) * 100 < -15:
                    fails.append(f"{ds_name} {metric} {cv:.3f} vs {bv:.3f} worse >15%")
            elif direction == "down" and bv > 0:
                if (cv - bv) / abs(bv) * 100 > 15:
                    fails.append(f"{ds_name} knife_rate {cv:.3f} vs {bv:.3f} worse >15%")
    return {"passes": len(fails) == 0, "fails": fails,
            "oos_pf_imp": pf_imp_oos}


def main():
    refit = json.load(open(REFIT_FILE))
    kg_p = refit["selected_refit_baseline"]
    print(f"Kaggle refit params: sl={kg_p['sl']}, tp={kg_p['tp']}, "
          f"knife={kg_p['knife_bps']}, ext={kg_p['ext_cap_bps']}")

    pi_sl, pi_tp, pi_knife, pi_ext = 50, 200, 50, 100
    pi_feats, pi_extra = load_pi()
    kg_feats, kg_extra = load_kaggle(60)
    print(f"  Pi: {pi_feats['n']} bars   Kaggle: {kg_feats['n']} bars\n",
          flush=True)

    # 8-cell matrix: 2 leverage x 2 sizing x 2 E3
    cells = []
    for lev in (5, 10):
        for ew in (False, True):
            for e3 in (False, True):
                name = f"lev{lev}x_ew{'on' if ew else 'off'}_e3{'on' if e3 else 'off'}"
                r_is = run_cell(pi_feats, pi_extra, pi_sl, pi_tp, pi_knife, pi_ext,
                                leverage=lev, entropy_weighted=ew, e3_overlay=e3)
                r_oos = run_cell(kg_feats, kg_extra, kg_p["sl"], kg_p["tp"],
                                 kg_p["knife_bps"], kg_p["ext_cap_bps"],
                                 leverage=lev, entropy_weighted=ew, e3_overlay=e3)
                cells.append({
                    "name": name,
                    "leverage": lev,
                    "entropy_weighted": ew,
                    "e3_overlay": e3,
                    "in_sample": r_is,
                    "oos": r_oos,
                })
                print(f"  {name:30} IS ret={r_is['compound_ret_pct']:+7.1f}% "
                      f"DD={r_is['max_dd']:4.1f}% PF={r_is['pf']:.2f} "
                      f"Sh={r_is['sharpe_daily']:+.2f} K={r_is['kelly_f']}  |  "
                      f"OOS ret={r_oos['compound_ret_pct']:+7.1f}% "
                      f"DD={r_oos['max_dd']:4.1f}% PF={r_oos['pf']:.2f} "
                      f"Sh={r_oos['sharpe_daily']:+.2f}", flush=True)

    # Baseline = 10x, no sizing, no E3 (Phase 3 promoted config at status-quo leverage)
    baseline = next(c for c in cells
                    if c["leverage"] == 10 and not c["entropy_weighted"]
                    and not c["e3_overlay"])
    # Evaluate every non-baseline
    evals = []
    for c in cells:
        if c is baseline: continue
        ev = evaluate_against_baseline(c, baseline)
        evals.append({"name": c["name"], **ev})

    passing = [e for e in evals if e["passes"]]
    print("\n=== PHASE 4 PROMOTION EVAL ===")
    print(f"Baseline: {baseline['name']}")
    print(f"  IS ret={baseline['in_sample']['compound_ret_pct']:+.1f}% "
          f"DD={baseline['in_sample']['max_dd']:.1f}% "
          f"PF={baseline['in_sample']['pf']:.2f}  |  "
          f"OOS ret={baseline['oos']['compound_ret_pct']:+.1f}% "
          f"DD={baseline['oos']['max_dd']:.1f}% "
          f"PF={baseline['oos']['pf']:.2f}")
    if passing:
        print(f"\n{len(passing)} variants pass:")
        for e in passing:
            print(f"  {e['name']}: OOS PF+{e['oos_pf_imp']:.1f}%")
    else:
        evals.sort(key=lambda e: (len(e["fails"]),
                                   -(e["oos_pf_imp"] or -999)))
        print("\nNo variant passes. Closest:")
        for e in evals[:5]:
            print(f"  {e['name']}: {len(e['fails'])} fails, "
                  f"OOS PF+{e['oos_pf_imp']}%")
            for f in e["fails"][:3]:
                print(f"     - {f}")

    OUT.write_text(json.dumps({
        "config": {
            "entry_gate": "F3c always on",
            "timeout_trail": "always on (Phase 3 promoted)",
            "pi_params": {"sl": pi_sl, "tp": pi_tp,
                          "knife_bps": pi_knife, "ext_cap_bps": pi_ext},
            "kaggle_params": {"sl": kg_p["sl"], "tp": kg_p["tp"],
                              "knife_bps": kg_p["knife_bps"],
                              "ext_cap_bps": kg_p["ext_cap_bps"]},
            "entropy_weight_formula":
                "size_mult = clamp(0.5 + 1.5*(H_thresh - H_at_entry)/H_thresh, 0.5, 1.5)",
        },
        "baseline_cell": baseline["name"],
        "cells": cells,
        "evaluations": evals,
        "passing": [e["name"] for e in passing],
        "decision": "PROMOTE" if passing else "NO_PROMOTE",
    }, indent=2, default=str))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
