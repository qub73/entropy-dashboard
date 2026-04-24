"""
D2 -- Feb-Mar baseline long/short split.

Re-run the ETH baseline on Feb 18 - Apr 7 Pi orderbook data with
current live config. Decompose returns by trade direction. Compare
side PnL against the period's underlying ETH price path.

If ≥85% of the compound return came from one side and that side
aligns with the period's net price direction, flag the baseline as
regime-fit rather than edge-demonstrating.

Outputs:
  - reports/d2_baseline_lshort_split.json
"""
import json, os, sys
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

HERE = Path(__file__).resolve().parent
ALGO = HERE.parent
sys.path.insert(0, str(ALGO))

from ob_entropy import (
    NUM_STATES_OB, rolling_entropy_ob, load_orderbook_range,
    compute_ob_features, classify_ob_states,
)
from kaggle_ob_trainer import resample_pi_to_1min
from upgrade_backtest import make_features, candidate_signals

OUT = ALGO / "reports" / "d2_baseline_lshort_split.json"


def ema_series(arr, n):
    a = np.asarray(arr, dtype=float)
    alpha = 2.0 / (n + 1.0)
    out = np.empty_like(a)
    if len(a) == 0: return out
    out[0] = a[0]
    for i in range(1, len(a)):
        out[i] = alpha * a[i] + (1 - alpha) * out[i-1]
    return out


def run_baseline(feats, h_thresh, params, sl, tp, trail_after, trail_bps,
                 cooldown=0, knife_bps=None, extended_cap_bps=None):
    """Current production config. Returns list of detailed trades."""
    n = feats['n']
    mids = feats['mid']; highs = feats['high']; lows = feats['low']
    atr_arr = feats['atr_bps']; imb = feats['imb5']
    dH_5 = feats['dH_5']; ret_60 = feats['ret_60']
    ret_150 = np.zeros(n)
    ret_150[150:] = (mids[150:] / mids[:-150] - 1) * 10000

    cands = set(candidate_signals(feats, h_thresh, params))
    equity = 10000.0
    in_trade = False
    entry_idx = entry_price = direction = 0
    notional = peak_pnl = 0.0
    tp_trailing = trailing_active = False
    trades = []
    last_loss_bar = -999

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
                tw = 2.0 * atr_arr[i] if atr_arr[i] > 0 else 50
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
                    if curr > 0:
                        trailing_active = True
                    else:
                        exit_reason = 'timeout'; exit_pnl = curr
            if exit_reason:
                fee = notional * 5.0 / 10000
                realized = (exit_pnl / 10000) * notional - fee
                equity += realized
                trades.append({
                    "entry_bar": entry_idx, "exit_bar": i,
                    "entry_price": entry_price,
                    "pnl_bps": exit_pnl,
                    "pnl_usd": realized,
                    "direction": direction,
                    "reason": exit_reason,
                    "hold_min": i - entry_idx,
                })
                if exit_pnl < 0:
                    last_loss_bar = i
                in_trade = False
                tp_trailing = False; trailing_active = False
        else:
            if i not in cands or equity <= 0: continue
            if cooldown > 0 and (i - last_loss_bar) < cooldown: continue
            if not np.isnan(dH_5[i]) and dH_5[i] >= 0: continue
            d = 1 if imb[i] > 0 else -1
            if knife_bps and d == 1 and ret_60[i] < -knife_bps: continue
            if extended_cap_bps is not None:
                r150 = ret_150[i]
                if d == 1 and r150 > extended_cap_bps: continue
                if d == -1 and r150 < -extended_cap_bps: continue
            margin = equity * 0.90
            notional = margin * 10
            equity -= notional * 5.0 / 10000
            in_trade = True; entry_idx = i; entry_price = mids[i]
            direction = d; peak_pnl = 0
            tp_trailing = False; trailing_active = False

    if in_trade:
        curr = direction * (mids[-1]/entry_price - 1) * 10000
        fee = notional * 5.0 / 10000
        realized = (curr / 10000) * notional - fee
        equity += realized
        trades.append({
            "entry_bar": entry_idx, "exit_bar": n-1,
            "entry_price": entry_price, "pnl_bps": curr,
            "pnl_usd": realized, "direction": direction,
            "reason": "end", "hold_min": n - 1 - entry_idx,
        })
    return trades, equity


def side_stats(trades, side):
    ts = [t for t in trades if t["direction"] == side]
    if not ts:
        return {"n": 0}
    wins = [t for t in ts if t["pnl_bps"] > 0]
    losses = [t for t in ts if t["pnl_bps"] <= 0]
    bps = [t["pnl_bps"] for t in ts]
    win_bps = sum(t["pnl_bps"] for t in wins)
    loss_bps = sum(t["pnl_bps"] for t in losses)
    return {
        "n": len(ts),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(ts),
        "avg_winner_bps": float(np.mean([t["pnl_bps"] for t in wins])) if wins else None,
        "avg_loser_bps": float(np.mean([t["pnl_bps"] for t in losses])) if losses else None,
        "total_bps_gross": float(sum(bps)),
        "mean_bps": float(np.mean(bps)),
        "pf": float(abs(win_bps / loss_bps)) if loss_bps != 0 else None,
        "reasons": {r: sum(1 for t in ts if t["reason"] == r)
                    for r in set(t["reason"] for t in ts)},
        "usd_total": float(sum(t["pnl_usd"] for t in ts)),
    }


def compound_contribution(trades):
    """Simulate compounded equity, then split the realized compound by
    attributing each trade's multiplicative factor to its side."""
    eq = 10000.0
    side_equity = {1: 0.0, -1: 0.0}
    for t in trades:
        before = eq
        eq += t["pnl_usd"]
        side_equity[t["direction"]] += t["pnl_usd"]
    final = eq
    ret_pct = (final - 10000) / 10000 * 100
    return {
        "final_equity": final,
        "compound_ret_pct": ret_pct,
        "long_usd_contrib": side_equity[1],
        "short_usd_contrib": side_equity[-1],
        # contribution shares of the absolute USD gained/lost
        "long_share_of_gross": side_equity[1] / (abs(side_equity[1]) + abs(side_equity[-1]))
                                if (abs(side_equity[1]) + abs(side_equity[-1])) > 0 else None,
        "short_share_of_gross": side_equity[-1] / (abs(side_equity[1]) + abs(side_equity[-1]))
                                 if (abs(side_equity[1]) + abs(side_equity[-1])) > 0 else None,
    }


def main():
    print("Loading ETH Pi orderbook (Feb 18 - Apr 7)...", flush=True)
    df = load_orderbook_range("data/orderbook_pi", "ob_PF_ETHUSD")
    df_1m = resample_pi_to_1min(df)
    df_1m = compute_ob_features(df_1m)
    df_1m = classify_ob_states(df_1m, window=60)
    ent = rolling_entropy_ob(df_1m['state_ob'].values, NUM_STATES_OB, 30)
    feats = make_features(df_1m, ent)
    n = feats['n']
    print(f"  {n} bars, {n/1440:.1f} days\n", flush=True)

    params = {'imb_min': 0.05, 'spread_max': 20, 'ret_low': 20, 'ret_high': 80}

    # Primary baseline = current live config (with ext_filter=100)
    trades_live, equity_live = run_baseline(
        feats, 0.4352, params,
        sl=50, tp=200, trail_after=150, trail_bps=50,
        cooldown=0, knife_bps=50, extended_cap_bps=100)
    # 51-trade variant = same without ext_filter (for comparison)
    trades_51, equity_51 = run_baseline(
        feats, 0.4352, params,
        sl=50, tp=200, trail_after=150, trail_bps=50,
        cooldown=0, knife_bps=50, extended_cap_bps=None)

    # Price-path context
    mids = feats['mid']
    close_start = float(mids[0])
    close_end = float(mids[-1])
    start_to_end_bps = (close_end / close_start - 1) * 10000
    max_up_from_start = (float(mids.max()) / close_start - 1) * 10000
    max_dn_from_start = (float(mids.min()) / close_start - 1) * 10000
    # Trend fraction via EMA crossover (using reasonable ETH parameters on 1-min)
    ema_fast = ema_series(mids, 30)
    ema_slow = ema_series(mids, 150)
    frac_ema_up = float((ema_fast > ema_slow).mean())
    # Cumulative-return regime: slope of close over windows
    first_half = (mids[n//2] / mids[0] - 1) * 10000
    second_half = (mids[-1] / mids[n//2] - 1) * 10000

    out = {}
    for label, trades, equity in [
        ("live_config_with_ext_filter", trades_live, equity_live),
        ("no_ext_filter_51_trade_variant", trades_51, equity_51),
    ]:
        all_side_stats = {
            "long": side_stats(trades, 1),
            "short": side_stats(trades, -1),
        }
        comp = compound_contribution(trades)
        ret_pct = comp["compound_ret_pct"]
        long_share = comp.get("long_share_of_gross") or 0
        short_share = comp.get("short_share_of_gross") or 0

        # flag: ≥85% of compound from one side AND that side aligns with
        # net price direction (start_to_end_bps)
        price_up = start_to_end_bps > 0
        # Use USD contribution as proxy for compound share
        dominant_side = "long" if abs(comp["long_usd_contrib"]) >= abs(comp["short_usd_contrib"]) else "short"
        dominant_mag = max(abs(comp["long_usd_contrib"]), abs(comp["short_usd_contrib"]))
        total_mag = abs(comp["long_usd_contrib"]) + abs(comp["short_usd_contrib"])
        share_of_dominant = dominant_mag / total_mag if total_mag > 0 else None
        dominant_aligned = (dominant_side == "long" and price_up) \
                        or (dominant_side == "short" and not price_up)
        flag_regime_fit = (share_of_dominant is not None
                           and share_of_dominant >= 0.85
                           and dominant_aligned)

        out[label] = {
            "trades": len(trades),
            "final_equity": comp["final_equity"],
            "compound_ret_pct": ret_pct,
            "long": all_side_stats["long"],
            "short": all_side_stats["short"],
            "usd_contribution": {
                "long": comp["long_usd_contrib"],
                "short": comp["short_usd_contrib"],
                "long_share_of_gross": comp["long_share_of_gross"],
                "short_share_of_gross": comp["short_share_of_gross"],
            },
            "regime_fit_flag": {
                "dominant_side": dominant_side,
                "share_of_dominant": share_of_dominant,
                "dominant_aligned_with_price": dominant_aligned,
                "flagged": flag_regime_fit,
            },
        }

    out["price_path"] = {
        "start_mid": close_start,
        "end_mid": close_end,
        "start_to_end_bps": start_to_end_bps,
        "max_up_from_start_bps": max_up_from_start,
        "max_dn_from_start_bps": max_dn_from_start,
        "first_half_bps": first_half,
        "second_half_bps": second_half,
        "frac_minutes_ema_up": frac_ema_up,
    }

    OUT.write_text(json.dumps(out, indent=2, default=str))

    # Console summary
    print("=== D2 SUMMARY ===")
    for label in ["live_config_with_ext_filter", "no_ext_filter_51_trade_variant"]:
        r = out[label]
        print(f"\n--- {label} ---")
        print(f"  Trades: {r['trades']}, Compound return: {r['compound_ret_pct']:+.1f}%")
        for s in ("long", "short"):
            ss = r[s]
            if ss["n"] == 0:
                print(f"  {s.upper():6}: 0 trades")
                continue
            print(f"  {s.upper():6}: n={ss['n']}, WR={ss['win_rate']*100:.0f}%, "
                  f"mean={ss['mean_bps']:+.1f} bps, PF={ss['pf']}, "
                  f"total={ss['total_bps_gross']:+.0f} bps")
        rf = r['regime_fit_flag']
        print(f"  Contribution shares: long={r['usd_contribution']['long_share_of_gross']:.2f} "
              f"short={r['usd_contribution']['short_share_of_gross']:.2f}")
        print(f"  Dominant side: {rf['dominant_side']} ({rf['share_of_dominant']:.2%}), "
              f"aligned with price: {rf['dominant_aligned_with_price']}")
        print(f"  **REGIME-FIT FLAG**: {rf['flagged']}")
    pp = out["price_path"]
    print(f"\nPrice path: start=${pp['start_mid']:.2f}, end=${pp['end_mid']:.2f}, "
          f"net={pp['start_to_end_bps']:+.0f} bps")
    print(f"  max up/down from start: {pp['max_up_from_start_bps']:+.0f} / "
          f"{pp['max_dn_from_start_bps']:+.0f} bps")
    print(f"  first/second half: {pp['first_half_bps']:+.0f} / "
          f"{pp['second_half_bps']:+.0f} bps")
    print(f"  fraction of minutes ema_fast > ema_slow: {pp['frac_minutes_ema_up']:.2%}")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
