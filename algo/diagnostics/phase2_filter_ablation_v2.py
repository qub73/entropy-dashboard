"""
Phase 2 filter ablation v2 -- with Kaggle-fair baseline.

Unlike v1 which applied Kraken-fit params to Kaggle OOS (baseline lost
42%), v2 uses:
  - Pi IS baseline  = current live config (sl=50, tp=200, knife=50, ext=100)
  - Kaggle OOS base = refit params from reports/phase2_kaggle_refit.json
    (currently sl=50, tp=150, knife=100, ext=None)

Filters F1/F2/F3a/F3b/F3c/F4 are overlayed on top of each dataset's
dataset-specific baseline. Measures PF improvement against the
relevant baseline for that dataset.

Writes reports/phase2_filter_ablation_v2.json.
Promotion rule unchanged (Addendum #1 + Pi-side).
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

OUT = ALGO / "reports" / "phase2_filter_ablation_v2.json"
REFIT_FILE = ALGO / "reports" / "phase2_kaggle_refit.json"
H_THRESH = 0.4352
CORE_PARAMS = {"imb_min": 0.05, "spread_max": 20, "ret_low": 20, "ret_high": 80}


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


def rolling_pct40(vol, lookback_bars=1440):
    out = np.full_like(vol, np.nan, dtype=float)
    v = np.asarray(vol, dtype=float); n = len(v)
    for i in range(lookback_bars, n):
        out[i] = np.percentile(v[i-lookback_bars:i], 40)
    return out


def build_extra_features(feats, volumes):
    mids = feats["mid"]; highs = feats["high"]; lows = feats["low"]
    n = len(mids)
    atr_5  = wilder_atr(highs, lows, mids, 5)
    atr_30 = wilder_atr(highs, lows, mids, 30)
    atr_5_bps  = np.where(mids > 0, atr_5  / mids * 10000, 0)
    atr_30_bps = np.where(mids > 0, atr_30 / mids * 10000, 0)
    ema_fast = ema_series(mids, 30)
    ema_slow = ema_series(mids, 150)
    if volumes is None:
        vol_60 = np.zeros(n); vol_pct40 = np.zeros(n)
    else:
        v = np.asarray(volumes, dtype=float)
        vol_60 = pd.Series(v).rolling(60, min_periods=10).sum().fillna(0).values
        vol_pct40 = rolling_pct40(vol_60, lookback_bars=min(1440, max(60, n//4)))
    return {"atr_5": atr_5, "atr_30": atr_30,
            "atr_5_bps": atr_5_bps, "atr_30_bps": atr_30_bps,
            "ema_fast": ema_fast, "ema_slow": ema_slow,
            "vol_60": vol_60, "vol_pct40": vol_pct40}


def run_ablation(feats, extra, params, filt, sl, tp, knife_bps, ext_cap_bps):
    """Same engine as v1, parameterized by baseline params per dataset."""
    n = feats['n']
    mids = feats['mid']; highs = feats['high']; lows = feats['low']
    atr_bps = feats['atr_bps']; imb = feats['imb5']
    dH_5 = feats['dH_5']; ret_60 = feats['ret_60']; ret_5 = feats['ret_5']
    ret_150 = np.zeros(n); ret_150[150:] = (mids[150:] / mids[:-150] - 1) * 10000
    atr_5_bps  = extra["atr_5_bps"]
    atr_30     = extra["atr_30"]
    atr_30_bps = extra["atr_30_bps"]
    ema_fast = extra["ema_fast"]
    ema_slow = extra["ema_slow"]
    vol_60   = extra["vol_60"]
    vol_pct40= extra["vol_pct40"]

    if filt.get("F4"):
        lp = dict(params); lp["ret_low"] = -999; lp["ret_high"] = 999
        cands_base = set(candidate_signals(feats, H_THRESH, lp))
        cands = {i for i in cands_base
                 if atr_30_bps[i] > 0
                 and 0.3 <= abs(ret_5[i]) / atr_30_bps[i] <= 1.2}
    else:
        cands = set(candidate_signals(feats, H_THRESH, params))

    trail_after = 150; trail_bps = 50
    equity = 10000.0
    in_trade = False
    entry_idx = entry_price = direction = 0
    notional = peak_pnl = 0.0
    tp_trailing = trailing_active = False
    trades = []
    pending_entry = None

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
                fee = notional * 5.0 / 10000
                realized = (exit_pnl/10000) * notional - fee
                equity += realized
                trades.append({"direction": direction, "pnl_bps": exit_pnl,
                               "pnl_usd": realized, "reason": exit_reason,
                               "peak_bps": peak_pnl})
                in_trade = False
                tp_trailing = False; trailing_active = False
            continue

        if pending_entry is not None:
            pe = pending_entry
            pending_entry = None
            prev_close = mids[pe["bar"]]; cur_close = mids[i]
            move_bps = pe["direction"] * (cur_close - prev_close) / prev_close * 10000
            thresh = 0.5 * atr_5_bps[pe["bar"]]
            if move_bps < thresh:
                pass
            else:
                margin = equity * 0.90; notional = margin * 10
                equity -= notional * 5.0 / 10000
                in_trade = True; entry_idx = i; entry_price = mids[i]
                direction = pe["direction"]; peak_pnl = 0
                tp_trailing = False; trailing_active = False
                continue

        if i not in cands or equity <= 0: continue
        if not np.isnan(dH_5[i]) and dH_5[i] >= 0: continue
        d = 1 if imb[i] > 0 else -1
        if knife_bps and d == 1 and ret_60[i] < -knife_bps: continue
        if knife_bps and d == -1 and ret_60[i] > knife_bps: continue
        if ext_cap_bps is not None:
            r150 = ret_150[i]
            if d == 1 and r150 > ext_cap_bps: continue
            if d == -1 and r150 < -ext_cap_bps: continue

        if filt.get("F3a") or filt.get("F3b") or filt.get("F3c"):
            norm = (ema_fast[i] - ema_slow[i]) / atr_30[i] if atr_30[i] > 0 else 0.0
            if filt.get("F3a"):
                block = (d == 1 and norm < -0.5) or (d == -1 and norm > 0.5)
            elif filt.get("F3b"):
                block = (d == 1 and norm < -0.3) or (d == -1 and norm > 0.3)
            elif filt.get("F3c"):
                block = (d == 1 and norm < 0.3) or (d == -1 and norm > -0.3)
            else:
                block = False
            if block: continue

        if filt.get("F2"):
            if vol_pct40[i] > 0 and vol_60[i] < vol_pct40[i]:
                continue

        if filt.get("F1"):
            pending_entry = {"bar": i, "direction": d}
            continue

        margin = equity * 0.90; notional = margin * 10
        equity -= notional * 5.0 / 10000
        in_trade = True; entry_idx = i; entry_price = mids[i]
        direction = d; peak_pnl = 0
        tp_trailing = False; trailing_active = False

    if in_trade:
        curr = direction * (mids[-1]/entry_price - 1) * 10000
        fee = notional * 5.0 / 10000
        realized = (curr/10000) * notional - fee
        equity += realized
        trades.append({"direction": direction, "pnl_bps": curr,
                       "pnl_usd": realized, "reason": "end", "peak_bps": peak_pnl})

    nt = len(trades)
    if nt == 0:
        return {"trades": 0, "compound_ret_pct": 0, "max_dd": 0,
                "win_rate": 0, "pf": 0, "knife_rate": 0, "knife_catchers": 0,
                "trades_long": 0, "trades_short": 0}
    pnls = [t["pnl_usd"] for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    pk = 10000; cum = 10000; mdd = 0
    for p in pnls:
        cum += p; pk = max(pk, cum); mdd = max(mdd, (pk-cum)/pk*100)
    ret = (equity - 10000)/10000*100
    bps = [t["pnl_bps"] for t in trades]
    win_bps = sum(b for b in bps if b > 0)
    loss_bps = sum(b for b in bps if b <= 0)
    pf = abs(win_bps/loss_bps) if loss_bps != 0 else float("inf")
    knife = sum(1 for t in trades if t["pnl_bps"] <= 0 and t["peak_bps"] < 20)
    return {
        "trades": nt, "wins": wins, "win_rate": wins/nt,
        "compound_ret_pct": ret, "max_dd": mdd, "pf": pf,
        "knife_catchers": knife, "knife_rate": knife/nt,
        "trades_long": sum(1 for t in trades if t["direction"] == 1),
        "trades_short": sum(1 for t in trades if t["direction"] == -1),
    }


def load_pi_dataset():
    print("Loading ETH Pi orderbook...", flush=True)
    df = load_orderbook_range("data/orderbook_pi", "ob_PF_ETHUSD")
    df_1m = resample_pi_to_1min(df)
    df_1m = compute_ob_features(df_1m)
    df_1m = classify_ob_states(df_1m, window=60)
    ent = rolling_entropy_ob(df_1m['state_ob'].values, NUM_STATES_OB, 30)
    feats = make_features(df_1m, ent)
    volumes = None
    if "bid_depth_5" in df_1m.columns and "ask_depth_5" in df_1m.columns:
        volumes = (df_1m["bid_depth_5"].values + df_1m["ask_depth_5"].values)
    extra = build_extra_features(feats, volumes)
    print(f"  Pi: {feats['n']} bars ({feats['n']/1440:.1f} days)", flush=True)
    return feats, extra


def load_kaggle_eval_dataset(days=60):
    print(f"Loading Kaggle ETH/USDT (last {days} days)...", flush=True)
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
    volumes = None
    if "bid_depth_5" in df_1m.columns and "ask_depth_5" in df_1m.columns:
        volumes = (df_1m["bid_depth_5"].values + df_1m["ask_depth_5"].values)
    extra = build_extra_features(feats, volumes)
    print(f"  Kaggle eval: {feats['n']} bars ({feats['n']/1440:.1f} days)",
          flush=True)
    return feats, extra


def filter_configs():
    return [
        ("baseline",         {}),
        ("F3a",              {"F3a": True}),
        ("F3b",              {"F3b": True}),
        ("F3c",              {"F3c": True}),
        ("F1",               {"F1": True}),
        ("F2",               {"F2": True}),
        ("F4",               {"F4": True}),
        ("F3a+F4",           {"F3a": True, "F4": True}),
        ("F3b+F4",           {"F3b": True, "F4": True}),
        ("F3c+F4",           {"F3c": True, "F4": True}),
        ("F1+F2",            {"F1": True, "F2": True}),
        ("F1+F4",            {"F1": True, "F4": True}),
        ("F3a+F1",           {"F3a": True, "F1": True}),
        ("F3b+F1",           {"F3b": True, "F1": True}),
        ("F3c+F1",           {"F3c": True, "F1": True}),
        ("F3a+F2",           {"F3a": True, "F2": True}),
        ("F3b+F2",           {"F3b": True, "F2": True}),
        ("F3c+F2",           {"F3c": True, "F2": True}),
        ("F3a+F1+F2+F4",     {"F3a": True, "F1": True, "F2": True, "F4": True}),
        ("F3b+F1+F2+F4",     {"F3b": True, "F1": True, "F2": True, "F4": True}),
        ("F3c+F1+F2+F4",     {"F3c": True, "F1": True, "F2": True, "F4": True}),
    ]


def pct_improvement(val, baseline):
    if baseline is None or baseline == 0: return None
    return (val - baseline) / abs(baseline) * 100


def main():
    refit_data = json.load(open(REFIT_FILE))
    kg_params = refit_data["selected_refit_baseline"]
    print(f"Refit baseline for Kaggle: sl={kg_params['sl']}, tp={kg_params['tp']}, "
          f"knife={kg_params['knife_bps']}, ext={kg_params['ext_cap_bps']}")

    # Pi: current live config (Kraken-fit)
    pi_sl, pi_tp, pi_knife, pi_ext = 50, 200, 50, 100

    pi_feats, pi_extra = load_pi_dataset()
    kg_feats, kg_extra = load_kaggle_eval_dataset(days=60)

    configs = filter_configs()
    runs = []
    print(f"\nRunning {len(configs)} configs x 2 datasets...", flush=True)
    for name, filt in configs:
        res_is = run_ablation(pi_feats, pi_extra, CORE_PARAMS, filt,
                              pi_sl, pi_tp, pi_knife, pi_ext)
        res_oos = run_ablation(kg_feats, kg_extra, CORE_PARAMS, filt,
                               kg_params["sl"], kg_params["tp"],
                               kg_params["knife_bps"], kg_params["ext_cap_bps"])
        runs.append({"name": name, "filt": filt,
                     "in_sample": res_is, "oos": res_oos})
        print(f"  {name:22}  IS tr={res_is['trades']:3} "
              f"ret={res_is['compound_ret_pct']:+6.1f}% "
              f"DD={res_is['max_dd']:4.1f}% PF={res_is['pf']:.2f} "
              f"knife={res_is['knife_rate']:.0%}  |  "
              f"OOS tr={res_oos['trades']:3} "
              f"ret={res_oos['compound_ret_pct']:+6.1f}% "
              f"DD={res_oos['max_dd']:4.1f}% PF={res_oos['pf']:.2f}",
              flush=True)

    baseline = next(r for r in runs if r["name"] == "baseline")

    def evaluate(r):
        fails = []
        is_ = r["in_sample"]; oos = r["oos"]
        bis = baseline["in_sample"]; boos = baseline["oos"]
        if is_["knife_rate"] >= 0.15:
            fails.append(f"IS knife_rate {is_['knife_rate']:.1%} >= 15%")
        pf_imp_is = pct_improvement(is_["pf"], bis["pf"])
        if pf_imp_is is None or pf_imp_is < 25:
            fails.append(f"IS PF improvement {pf_imp_is}% < 25%")
        if bis["trades"] > 0 and is_["trades"] < 0.7 * bis["trades"]:
            fails.append(f"IS trades {is_['trades']} < 0.7x baseline {bis['trades']}")
        pf_imp_oos = pct_improvement(oos["pf"], boos["pf"])
        if pf_imp_oos is None or pf_imp_oos < 15:
            fails.append(f"OOS PF improvement {pf_imp_oos}% < 15%")
        if oos["max_dd"] > boos["max_dd"]:
            fails.append(f"OOS max_dd {oos['max_dd']:.2f}% > baseline {boos['max_dd']:.2f}%")
        for ds_name, ds, base_ds in [("IS", is_, bis), ("OOS", oos, boos)]:
            for metric in ("compound_ret_pct", "pf", "win_rate"):
                bv = base_ds.get(metric, 0); cv = ds.get(metric, 0)
                if bv > 0 and (cv - bv) / abs(bv) * 100 < -10:
                    fails.append(f"{ds_name} {metric} {cv:.2f} worse than baseline {bv:.2f} by >10%")
        return {"passes": len(fails) == 0, "fails": fails,
                "is_pf_imp": pf_imp_is, "oos_pf_imp": pf_imp_oos}

    evals = [{"name": r["name"], **evaluate(r)} for r in runs if r["name"] != "baseline"]
    passing = [e for e in evals if e["passes"]]

    print("\n=== PROMOTION EVAL (v2) ===")
    if passing:
        print(f"{len(passing)} configurations pass:")
        for e in passing:
            print(f"  {e['name']}: IS PF +{e['is_pf_imp']:.1f}%, OOS PF +{e['oos_pf_imp']:.1f}%")
    else:
        evals.sort(key=lambda e: (len(e["fails"]), -(e["is_pf_imp"] or -999)))
        print("No configuration passes. Closest:")
        for e in evals[:5]:
            print(f"  {e['name']}: fails={len(e['fails'])}, "
                  f"IS PF+{e['is_pf_imp']:.1f}%, OOS PF+{e['oos_pf_imp']:.1f}%")
            for f in e["fails"][:3]:
                print(f"     - {f}")

    OUT.write_text(json.dumps({
        "pi_params": {"sl": pi_sl, "tp": pi_tp,
                      "knife_bps": pi_knife, "ext_cap_bps": pi_ext},
        "kaggle_params": {"sl": kg_params["sl"], "tp": kg_params["tp"],
                          "knife_bps": kg_params["knife_bps"],
                          "ext_cap_bps": kg_params["ext_cap_bps"]},
        "runs": runs,
        "evaluations": evals,
        "passing": [e["name"] for e in passing],
        "decision": "PROMOTE" if passing else "NO_PROMOTE",
    }, indent=2, default=str))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
