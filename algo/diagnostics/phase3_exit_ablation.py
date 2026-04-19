"""
Phase 3 exit ablation. F3c entry gate is now settled; every run has it.

Ablation lines:
  (a) Baseline (F3c + current exits)
  (b) Timeout-trail-only (Addendum #2 default)
  (c) E5-long-only  (min_hold=20, h_exit=0.4852, mfe_guard=20, width=20,
      direction == +1 only)
  (d) E1  (lower tp_trail: activate @+80 bps, 30 bps trail width)
  (e) E2  (partial TP: close 50% at +100 bps)
  (f) E3  (time-decayed SL: if bars_in_trade==60 AND peak<+50 -> SL=-25)
  (g) E4  (ATR-initial SL: sl = max(35, 1.5*ATR_30_at_entry_bps))
  (h) Best of (c)-(g) stacked with (b)
  (i) All stacked

Dual-dataset eval:
  IS  = Feb 18 - Apr 7 Pi ETH with live exit params (SL=50, TP=200)
  OOS = last 60d Kaggle ETH/USDT with refit params (SL=50, TP=150)

F3c ALWAYS ON on top of both datasets' baselines.

Corrected promotion rule (Phase 3 onward):
  Primary gate (all must pass):
    (a) Both IS and OOS return > 0
    (b) OOS PF improvement vs baseline >= 15%
    (c) OOS max_dd <= baseline max_dd
    (d) IS trade count >= 15
    (e) OOS trade count >= 30
  Secondary: no single metric worse by >15% on either dataset
  (PF, knife_rate, win_rate).

Output: reports/phase3_exit_ablation.json
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

OUT = ALGO / "reports" / "phase3_exit_ablation.json"
REFIT_FILE = ALGO / "reports" / "phase2_kaggle_refit.json"
H_THRESH = 0.4352
CORE_PARAMS = {"imb_min": 0.05, "spread_max": 20, "ret_low": 20, "ret_high": 80}

# post_signal_trail config (Addendum #2 defaults)
PST_WIDTH_BPS = 20
PST_HARD_FLOOR = -60
PST_MAX_WAIT = 30
# E5-long-only params (per the user prompt)
E5L_MIN_HOLD = 20
E5L_H_EXIT = 0.4852
E5L_MFE_GUARD = 20
E5L_WIDTH = 20


# ---------------- shared helpers (same as v2) ----------------

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
    atr_30_bps = np.where(mids > 0, atr_30 / mids * 10000, 0)
    return {"atr_30": atr_30, "atr_30_bps": atr_30_bps,
            "ema_fast": ema_series(mids, 30),
            "ema_slow": ema_series(mids, 150)}


# ---------------- core engine with all exit variants ----------------

def run_ablation(feats, extra, sl, tp, knife_bps, ext_cap_bps, variant):
    """Run with F3c entry gate always on + a specific exit variant.

    variant is a dict with flags:
      F3c_always = True (implicit)
      timeout_trail = False           (b)
      e5_long_only = False            (c)
      e1_lower_trail = False          (d)      -> trail_after=80, trail_bps=30
      e2_partial_tp = False           (e)
      e3_time_decayed_sl = False      (f)
      e4_atr_initial_sl = False       (g)
    """
    n = feats['n']
    mids = feats['mid']; highs = feats['high']; lows = feats['low']
    atr_bps_arr = feats['atr_bps']; imb = feats['imb5']; H = feats['H']
    dH_5 = feats['dH_5']; ret_60 = feats['ret_60']
    ret_150 = np.zeros(n); ret_150[150:] = (mids[150:] / mids[:-150] - 1) * 10000
    atr_30 = extra['atr_30']
    atr_30_bps = extra['atr_30_bps']
    ema_fast = extra['ema_fast']; ema_slow = extra['ema_slow']

    cands = set(candidate_signals(feats, H_THRESH, CORE_PARAMS))

    # Baseline tp_trail: (150, 50). E1 override: (80, 30).
    if variant.get("e1_lower_trail"):
        trail_after = 80; trail_bps = 30
    else:
        trail_after = 150; trail_bps = 50

    equity = 10000.0
    in_trade = False
    entry_idx = entry_price = direction = 0
    notional = peak_pnl = 0.0
    initial_notional = 0.0        # notional at entry (for pnl_bps aggregation)
    position_realized_usd = 0.0   # sum of partial fills before full close
    position_had_partial = False
    tp_trailing = trailing_active = pst_active = False
    pst_peak_bps = 0.0; pst_entry_bar = 0
    sl_current = -sl
    e3_tightened = False
    e2_partial_taken = False
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

            # SL is always force-exit (priority 1)
            if worst_bps <= sl_current:
                exit_reason = 'sl'; exit_pnl = sl_current

            # tp_trail (priority 2)
            if exit_reason is None and not tp_trailing and peak_pnl >= trail_after:
                tp_trailing = True
            if exit_reason is None and tp_trailing:
                floor = max(peak_pnl - trail_bps, sl_current)
                if worst_bps <= floor:
                    exit_reason = 'tp_trail'; exit_pnl = max(floor, curr_bps)

            # ATR-trail at timeout if in profit (priority 3)
            if exit_reason is None and trailing_active:
                tw = 2.0 * atr_bps_arr[i] if atr_bps_arr[i] > 0 else 50
                floor = max(peak_pnl - tw, sl_current)
                if worst_bps <= floor:
                    exit_reason = 'trail_stop'; exit_pnl = max(floor, curr_bps)
                elif best_bps >= tp:
                    exit_reason = 'tp'; exit_pnl = tp

            # post_signal_trail resolution (priority 4)
            if exit_reason is None and pst_active:
                bars_in_trail = i - pst_entry_bar
                # update pst_peak
                pst_peak_bps = max(pst_peak_bps, best_bps)
                floor_bps = max(pst_peak_bps - PST_WIDTH_BPS, sl_current)
                if worst_bps <= floor_bps:
                    exit_reason = 'pst_trail'; exit_pnl = max(floor_bps, curr_bps)
                elif worst_bps <= PST_HARD_FLOOR:
                    exit_reason = 'pst_floored'; exit_pnl = PST_HARD_FLOOR
                elif bars_in_trail >= PST_MAX_WAIT:
                    exit_reason = 'pst_timeout'; exit_pnl = curr_bps

            # Fixed TP only reachable before tp_trail activates
            if exit_reason is None and not tp_trailing and not pst_active \
               and not trailing_active and best_bps >= tp:
                exit_reason = 'tp'; exit_pnl = tp

            # E2: partial TP at +100 bps. Aggregated into the position --
            # does NOT emit its own trade row (see bug note 2026-04-19).
            if (exit_reason is None and variant.get("e2_partial_tp")
                and not e2_partial_taken and best_bps >= 100):
                half = notional * 0.5
                fee_half = half * 5.0 / 10000
                realized_half = (100 / 10000) * half - fee_half
                equity += realized_half
                position_realized_usd += realized_half
                position_had_partial = True
                notional = half
                e2_partial_taken = True

            # E3: time-decayed SL
            if (exit_reason is None and variant.get("e3_time_decayed_sl")
                and not e3_tightened and (i - entry_idx) >= 60
                and peak_pnl < 50):
                sl_current = -25  # tighten
                e3_tightened = True

            # Timeout branch (priority after TP): activate ATR-trail in profit
            # OR transition to post_signal_trail (b) if timeout-trail variant
            # active and not in profit. Otherwise exit at current pnl.
            if exit_reason is None and (i - entry_idx) >= 240 \
               and not trailing_active and not pst_active:
                if curr_bps > 0:
                    trailing_active = True
                elif variant.get("timeout_trail"):
                    # transition to post_signal_trail
                    pst_active = True
                    pst_entry_bar = i
                    pst_peak_bps = best_bps
                else:
                    exit_reason = 'timeout'; exit_pnl = curr_bps

            # E5 long-only (c): entropy-decay trail for LONGS only
            if (exit_reason is None and variant.get("e5_long_only")
                and direction == 1 and not pst_active and not tp_trailing
                and (i - entry_idx) >= E5L_MIN_HOLD and H[i] is not None
                and not np.isnan(H[i])
                and H[i] > E5L_H_EXIT and peak_pnl < E5L_MFE_GUARD):
                pst_active = True
                pst_entry_bar = i
                pst_peak_bps = best_bps

            if exit_reason:
                fee = notional * 5.0 / 10000
                realized = (exit_pnl / 10000) * notional - fee
                equity += realized
                total_realized = position_realized_usd + realized
                # pnl_bps on the whole position = total USD / initial_notional
                pos_pnl_bps = (total_realized / initial_notional * 10000
                               if initial_notional > 0 else exit_pnl)
                trades.append({
                    "direction": direction,
                    "pnl_bps": pos_pnl_bps,          # position-level effective bps
                    "close_bps": exit_pnl,           # per-fill exit level (for ref)
                    "pnl_usd": total_realized,
                    "reason": exit_reason,
                    "peak_bps": peak_pnl,
                    "had_partial": position_had_partial,
                })
                in_trade = False
                tp_trailing = False; trailing_active = False
                pst_active = False; e3_tightened = False
                e2_partial_taken = False
                position_realized_usd = 0.0
                position_had_partial = False
            continue

        # ---- not in trade: entry evaluation with F3c ALWAYS on ----
        if i not in cands or equity <= 0: continue
        if not np.isnan(dH_5[i]) and dH_5[i] >= 0: continue
        d = 1 if imb[i] > 0 else -1
        if knife_bps and d == 1 and ret_60[i] < -knife_bps: continue
        if knife_bps and d == -1 and ret_60[i] > knife_bps: continue
        if ext_cap_bps is not None:
            r150 = ret_150[i]
            if d == 1 and r150 > ext_cap_bps: continue
            if d == -1 and r150 < -ext_cap_bps: continue

        # F3c: require positive alignment
        if atr_30[i] > 0:
            norm = (ema_fast[i] - ema_slow[i]) / atr_30[i]
        else:
            norm = 0.0
        if d == 1 and norm < 0.3: continue
        if d == -1 and norm > -0.3: continue

        margin = equity * 0.90
        notional = margin * 10
        initial_notional = notional
        equity -= notional * 5.0 / 10000
        in_trade = True; entry_idx = i; entry_price = mids[i]
        direction = d; peak_pnl = 0
        tp_trailing = False; trailing_active = False; pst_active = False
        e3_tightened = False; e2_partial_taken = False
        position_realized_usd = 0.0
        position_had_partial = False

        # E4: ATR-initial SL
        if variant.get("e4_atr_initial_sl"):
            atr_bps_at_entry = atr_30_bps[i]
            sl_use = max(35, 1.5 * atr_bps_at_entry)
            sl_current = -sl_use
        else:
            sl_current = -sl

    if in_trade:
        curr = direction * (mids[-1]/entry_price - 1) * 10000
        fee = notional * 5.0 / 10000
        realized = (curr/10000) * notional - fee
        equity += realized
        total_realized = position_realized_usd + realized
        pos_pnl_bps = (total_realized / initial_notional * 10000
                       if initial_notional > 0 else curr)
        trades.append({"direction": direction, "pnl_bps": pos_pnl_bps,
                       "close_bps": curr, "pnl_usd": total_realized,
                       "reason": "end", "peak_bps": peak_pnl,
                       "had_partial": position_had_partial})

    # ---- summary ----
    nt = len(trades)
    # For partial TP, both slices are counted; that's fine for aggregate stats
    if nt == 0:
        return {"trades": 0, "compound_ret_pct": 0, "max_dd": 0,
                "win_rate": 0, "pf": 0, "knife_rate": 0, "knife_catchers": 0,
                "trades_long": 0, "trades_short": 0, "reasons": {}}
    pnls = [t["pnl_usd"] for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    pk = 10000; cum = 10000; mdd = 0
    for p in pnls:
        cum += p; pk = max(pk, cum); mdd = max(mdd, (pk-cum)/pk*100)
    ret = (equity - 10000)/10000 * 100
    bps = [t["pnl_bps"] for t in trades]
    win_bps = sum(b for b in bps if b > 0)
    loss_bps = sum(b for b in bps if b <= 0)
    pf = abs(win_bps/loss_bps) if loss_bps != 0 else float("inf")
    # knife catchers: losers with peak<20. Each trade row now corresponds
    # to a single logical position (partial fills rolled up), so just
    # iterate trades directly. Positions that had a partial by definition
    # had peak >= +100, so they are naturally excluded by peak_bps < 20.
    full_closes = trades
    knife = sum(1 for t in full_closes
                if t["pnl_bps"] <= 0 and t["peak_bps"] < 20)
    reasons = {}
    for t in trades:
        reasons[t["reason"]] = reasons.get(t["reason"], 0) + 1
    return {
        "trades": nt, "n_full_closes": len(full_closes), "wins": wins,
        "win_rate": wins/nt,
        "compound_ret_pct": ret, "max_dd": mdd, "pf": pf,
        "knife_catchers": knife,
        "knife_rate": knife/max(1, len(full_closes)),
        "trades_long": sum(1 for t in trades if t["direction"] == 1),
        "trades_short": sum(1 for t in trades if t["direction"] == -1),
        "reasons": reasons,
    }


def load_pi_dataset():
    print("Loading ETH Pi orderbook...", flush=True)
    df = load_orderbook_range("data/orderbook_pi", "ob_PF_ETHUSD")
    df_1m = resample_pi_to_1min(df)
    df_1m = compute_ob_features(df_1m)
    df_1m = classify_ob_states(df_1m, window=60)
    ent = rolling_entropy_ob(df_1m['state_ob'].values, NUM_STATES_OB, 30)
    feats = make_features(df_1m, ent)
    extra = build_extra_features(feats)
    print(f"  Pi: {feats['n']} bars", flush=True)
    return feats, extra


def load_kaggle_eval(days=60):
    print(f"Loading Kaggle ETH/USDT last {days}d...", flush=True)
    raw = load_kaggle_csv("data/kaggle/ETH_USDT.csv")
    df = parse_kaggle_to_ob_features(raw)
    df = compute_ob_features(df)
    df_1m = resample_pi_to_1min(df)
    df_1m = compute_ob_features(df_1m)
    bars_wanted = days * 1440
    if len(df_1m) > bars_wanted:
        df_1m = df_1m.iloc[-bars_wanted:].copy()
    df_1m = classify_ob_states(df_1m, window=60)
    ent = rolling_entropy_ob(df_1m['state_ob'].values, NUM_STATES_OB, 30)
    feats = make_features(df_1m, ent)
    extra = build_extra_features(feats)
    print(f"  Kaggle eval: {feats['n']} bars", flush=True)
    return feats, extra


def variants_list(best_single_name=None):
    """Ablation (a)-(g). (h) and (i) assembled by main()."""
    return [
        ("a_baseline_F3c",     {}),
        ("b_timeout_trail",    {"timeout_trail": True}),
        ("c_E5_long_only",     {"e5_long_only": True}),
        ("d_E1_lower_trail",   {"e1_lower_trail": True}),
        ("e_E2_partial_tp",    {"e2_partial_tp": True}),
        ("f_E3_time_decay_sl", {"e3_time_decayed_sl": True}),
        ("g_E4_atr_initial_sl",{"e4_atr_initial_sl": True}),
    ]


def evaluate(r, baseline):
    """Apply corrected Phase-3 promotion rule. Return (passes, fails list)."""
    fails = []
    is_ = r["in_sample"]; oos = r["oos"]
    bis = baseline["in_sample"]; boos = baseline["oos"]
    # Primary
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
    # Secondary: PF, knife_rate, win_rate -- no metric >15% worse on either ds
    for ds_name, ds, base_ds in [("IS", is_, bis), ("OOS", oos, boos)]:
        for metric, direction in [("pf", "up"),
                                   ("win_rate", "up"),
                                   ("knife_rate", "down")]:
            bv = base_ds.get(metric, 0); cv = ds.get(metric, 0)
            if direction == "up" and bv > 0:
                if (cv - bv) / abs(bv) * 100 < -15:
                    fails.append(f"{ds_name} {metric} {cv:.3f} "
                                 f"vs baseline {bv:.3f} (>15% worse)")
            elif direction == "down" and bv > 0:
                # higher knife rate is worse
                if (cv - bv) / abs(bv) * 100 > 15:
                    fails.append(f"{ds_name} knife_rate {cv:.3f} "
                                 f"vs baseline {bv:.3f} (>15% worse)")
    return {"passes": len(fails) == 0, "fails": fails,
            "oos_pf_improvement_pct": pf_imp_oos}


def main():
    refit = json.load(open(REFIT_FILE))
    kg_p = refit["selected_refit_baseline"]
    print(f"Kaggle refit params: sl={kg_p['sl']}, tp={kg_p['tp']}, "
          f"knife={kg_p['knife_bps']}, ext={kg_p['ext_cap_bps']}")

    pi_sl, pi_tp, pi_knife, pi_ext = 50, 200, 50, 100
    pi_feats, pi_extra = load_pi_dataset()
    kg_feats, kg_extra = load_kaggle_eval(60)

    variants = variants_list()

    # Run (a)-(g)
    runs = []
    print(f"\nRunning {len(variants)} variants x 2 datasets...", flush=True)
    for name, v in variants:
        r_is  = run_ablation(pi_feats, pi_extra, pi_sl, pi_tp, pi_knife, pi_ext, v)
        r_oos = run_ablation(kg_feats, kg_extra, kg_p["sl"], kg_p["tp"],
                             kg_p["knife_bps"], kg_p["ext_cap_bps"], v)
        runs.append({"name": name, "variant": v,
                     "in_sample": r_is, "oos": r_oos})
        print(f"  {name:28} IS tr={r_is['trades']:3} ret={r_is['compound_ret_pct']:+6.1f}% "
              f"DD={r_is['max_dd']:4.1f}% PF={r_is['pf']:.2f}  |  "
              f"OOS tr={r_oos['trades']:3} ret={r_oos['compound_ret_pct']:+6.1f}% "
              f"DD={r_oos['max_dd']:4.1f}% PF={r_oos['pf']:.2f}", flush=True)

    baseline = runs[0]
    # Evaluate (b)-(g)
    evals_a_g = []
    for r in runs[1:]:
        ev = evaluate(r, baseline)
        evals_a_g.append({"name": r["name"], **ev})

    # Pick best of (c)-(g) by OOS PF improvement (among those with IS/OOS trades floor met)
    cg = [r for r in runs if r["name"].startswith(("c_", "d_", "e_", "f_", "g_"))]
    # rank by OOS PF improvement (regardless of passing gate), used as seed for (h)
    def pf_imp(r):
        boos = baseline["oos"]; oos = r["oos"]
        return ((oos["pf"] - boos["pf"]) / abs(boos["pf"]) * 100
                if boos["pf"] else -999)
    cg_ranked = sorted(cg, key=lambda r: pf_imp(r), reverse=True)
    best_cg = cg_ranked[0]
    print(f"\nBest of (c)-(g) by OOS PF improvement: {best_cg['name']} "
          f"({pf_imp(best_cg):+.1f}% OOS PF)", flush=True)

    # (h) = best of (c)-(g) stacked with (b)
    h_variant = dict(best_cg["variant"])
    h_variant["timeout_trail"] = True
    r_is_h  = run_ablation(pi_feats, pi_extra, pi_sl, pi_tp, pi_knife, pi_ext, h_variant)
    r_oos_h = run_ablation(kg_feats, kg_extra, kg_p["sl"], kg_p["tp"],
                           kg_p["knife_bps"], kg_p["ext_cap_bps"], h_variant)
    h_run = {"name": f"h_best_cg_stack_b_{best_cg['name']}",
             "variant": h_variant,
             "in_sample": r_is_h, "oos": r_oos_h}
    runs.append(h_run)
    print(f"  {h_run['name']:28} IS tr={r_is_h['trades']:3} "
          f"ret={r_is_h['compound_ret_pct']:+6.1f}% DD={r_is_h['max_dd']:4.1f}% "
          f"PF={r_is_h['pf']:.2f}  |  OOS tr={r_oos_h['trades']:3} "
          f"ret={r_oos_h['compound_ret_pct']:+6.1f}% DD={r_oos_h['max_dd']:4.1f}% "
          f"PF={r_oos_h['pf']:.2f}", flush=True)

    # (i) All stacked = b+c+d+e+f+g
    all_variant = {"timeout_trail": True, "e5_long_only": True,
                   "e1_lower_trail": True, "e2_partial_tp": True,
                   "e3_time_decayed_sl": True, "e4_atr_initial_sl": True}
    r_is_i  = run_ablation(pi_feats, pi_extra, pi_sl, pi_tp, pi_knife, pi_ext, all_variant)
    r_oos_i = run_ablation(kg_feats, kg_extra, kg_p["sl"], kg_p["tp"],
                           kg_p["knife_bps"], kg_p["ext_cap_bps"], all_variant)
    i_run = {"name": "i_all_stacked", "variant": all_variant,
             "in_sample": r_is_i, "oos": r_oos_i}
    runs.append(i_run)
    print(f"  {i_run['name']:28} IS tr={r_is_i['trades']:3} "
          f"ret={r_is_i['compound_ret_pct']:+6.1f}% DD={r_is_i['max_dd']:4.1f}% "
          f"PF={r_is_i['pf']:.2f}  |  OOS tr={r_oos_i['trades']:3} "
          f"ret={r_oos_i['compound_ret_pct']:+6.1f}% DD={r_oos_i['max_dd']:4.1f}% "
          f"PF={r_oos_i['pf']:.2f}", flush=True)

    # Evaluate all non-baseline vs baseline
    all_evals = []
    for r in runs[1:]:
        ev = evaluate(r, baseline)
        all_evals.append({"name": r["name"], **ev})
    passing = [e for e in all_evals if e["passes"]]

    print("\n=== PROMOTION EVAL (Phase 3, corrected rule) ===")
    if passing:
        print(f"{len(passing)} variants pass:")
        for e in passing:
            print(f"  {e['name']}: OOS PF +{e['oos_pf_improvement_pct']:.1f}%")
    else:
        all_evals.sort(key=lambda e: (len(e["fails"]),
                                       -(e["oos_pf_improvement_pct"] or -999)))
        print("No variant passes. Closest:")
        for e in all_evals[:5]:
            print(f"  {e['name']}: fails={len(e['fails'])}, "
                  f"OOS PF+{e['oos_pf_improvement_pct']}%")
            for f in e["fails"][:3]:
                print(f"     - {f}")

    OUT.write_text(json.dumps({
        "config": {
            "entry_gate": "F3c always on",
            "pi_params": {"sl": pi_sl, "tp": pi_tp,
                          "knife_bps": pi_knife, "ext_cap_bps": pi_ext},
            "kaggle_params": {"sl": kg_p["sl"], "tp": kg_p["tp"],
                              "knife_bps": kg_p["knife_bps"],
                              "ext_cap_bps": kg_p["ext_cap_bps"]},
            "post_signal_trail": {"width_bps": PST_WIDTH_BPS,
                                  "hard_floor": PST_HARD_FLOOR,
                                  "max_wait_bars": PST_MAX_WAIT},
            "e5_long_only": {"min_hold": E5L_MIN_HOLD,
                              "h_exit": E5L_H_EXIT,
                              "mfe_guard": E5L_MFE_GUARD,
                              "width": E5L_WIDTH},
        },
        "promotion_rule": {
            "primary": [
                "IS return > 0", "OOS return > 0",
                "OOS PF improvement >= 15%",
                "OOS max_dd <= baseline max_dd",
                "IS trades >= 15", "OOS trades >= 30",
            ],
            "secondary": "no single metric (PF / knife_rate / win_rate) "
                         "worse by >15% on either dataset",
        },
        "runs": runs,
        "evaluations": all_evals,
        "passing": [e["name"] for e in passing],
        "decision": "PROMOTE" if passing else "NO_PROMOTE",
        "best_single_cg": best_cg["name"],
    }, indent=2, default=str))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
