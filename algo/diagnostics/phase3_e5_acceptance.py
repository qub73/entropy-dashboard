"""
Phase 3 E5 acceptance check (with Addendum #2 trail mechanic).

Rule under test: once `minutes_in_trade >= min_hold_bars` AND
`H > h_exit_threshold` AND `peak_pnl_bps < mfe_guard_bps`, transition
the position into `post_signal_trail` (NOT force-exit) with state
(trail_peak, entry_bar, orig_reason='entropy_decay_trail').

`post_signal_trail` resolution per bar:
  - for a LONG: trail_peak = max(trail_peak, bar_high)
  - exit conditions (first hit wins):
     * bar_low <= trail_peak - width_bps                  -> tag = orig
     * bar_pnl_bps <= hard_floor_bps_from_entry           -> tag + _floored
     * bars_in_trail >= max_wait_bars                     -> tag + _timeout
Symmetric for SHORT.

Acceptance (Addendum #2 revised):
  - For every baseline winner with final PnL >= +80 bps:
       E5-trail must NOT exit below (peak_pnl_at_exit - 25 bps).
    Count violators.
  - For winners whose peak PnL never exceeded +80 bps, E5 firing is OK.

Grid sensitivity: h_exit_threshold x mfe_guard_bps x width_bps,
  3 x 3 x 2 = 18 cells.
  h_exit_threshold in {0.465, 0.4852, 0.505}
  mfe_guard_bps    in {20, 30, 50}
  width_bps        in {15, 25}

Default (width_bps=20) tested separately as the main acceptance run.

Outputs:
  - reports/phase3_e5_acceptance.json
"""
import json, os, sys
from pathlib import Path
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

OUT = ALGO / "reports" / "phase3_e5_acceptance.json"

# Post-trail config defaults (can be overridden by grid)
HARD_FLOOR_BPS = -60
MAX_WAIT_BARS = 30
MIN_HOLD_BARS = 10  # E5 ignition requires >= this many minutes held
# Acceptance slack
WINNER_GIVEBACK_MAX_BPS = 25  # trail width 20 + 5 slippage


def run_baseline_with_trajectories(feats, h_thresh, params, sl, tp,
                                    trail_after, trail_bps,
                                    cooldown=0, knife_bps=None,
                                    extended_cap_bps=None):
    """Current live exit logic. Also records for each trade the FULL
    per-bar trajectory (H, mid, high, low, peak_pnl, curr_pnl) so we
    can replay E5 logic without re-running the whole engine."""
    n = feats['n']
    mids = feats['mid']; highs = feats['high']; lows = feats['low']
    atr_arr = feats['atr_bps']; imb = feats['imb5']; H = feats['H']
    dH_5 = feats['dH_5']; ret_60 = feats['ret_60']
    ret_150 = np.zeros(n)
    ret_150[150:] = (mids[150:] / mids[:-150] - 1) * 10000

    cands = set(candidate_signals(feats, h_thresh, params))
    equity = 10000.0
    in_trade = False
    entry_idx = entry_price = direction = 0
    notional = peak_pnl_bps = 0.0
    tp_trailing = trailing_active = False
    trades = []
    last_loss_bar = -999
    cur_traj = None

    for i in range(n):
        if in_trade:
            d = direction
            if d == 1:
                worst = (lows[i]/entry_price - 1) * 10000
                best = (highs[i]/entry_price - 1) * 10000
            else:
                worst = -(highs[i]/entry_price - 1) * 10000
                best = -(lows[i]/entry_price - 1) * 10000
            curr = d * (mids[i]/entry_price - 1) * 10000
            peak_pnl_bps = max(peak_pnl_bps, best)
            cur_traj.append({
                "bar": i,
                "mid": float(mids[i]),
                "high": float(highs[i]),
                "low": float(lows[i]),
                "H": float(H[i]) if not np.isnan(H[i]) else None,
                "curr_bps": float(curr),
                "peak_bps": float(peak_pnl_bps),
            })

            exit_reason = exit_pnl = None
            if not tp_trailing and peak_pnl_bps >= trail_after:
                tp_trailing = True
            if tp_trailing:
                floor = max(peak_pnl_bps - trail_bps, -sl)
                if worst <= floor:
                    exit_reason = 'tp_trail'; exit_pnl = max(floor, curr)
            elif trailing_active:
                tw = 2.0 * atr_arr[i] if atr_arr[i] > 0 else 50
                floor = max(peak_pnl_bps - tw, -sl)
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
                    "entry_price": entry_price, "direction": direction,
                    "pnl_bps": float(exit_pnl),
                    "pnl_usd": float(realized),
                    "reason": exit_reason,
                    "hold_bars": i - entry_idx,
                    "peak_bps_final": float(peak_pnl_bps),
                    "trajectory": cur_traj,
                })
                if exit_pnl < 0: last_loss_bar = i
                in_trade = False
                tp_trailing = False; trailing_active = False
                cur_traj = None
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
            direction = d; peak_pnl_bps = 0
            tp_trailing = False; trailing_active = False
            cur_traj = []

    # trailing open position at end: ignore for this analysis
    return trades


def replay_e5_on_trade(trade, h_exit_threshold, mfe_guard_bps, width_bps,
                        hard_floor=HARD_FLOOR_BPS, max_wait=MAX_WAIT_BARS,
                        min_hold=MIN_HOLD_BARS):
    """Given a baseline trade's trajectory, determine whether E5 would
    have fired, and if so, what exit (bps) it would have produced under
    the Addendum-#2 trail mechanic.

    Returns dict:
      {
        "fired": bool,
        "fire_bar_offset": int or None,       # bars after entry
        "H_at_fire": float or None,
        "peak_at_fire": float or None,
        "e5_exit_bps": float or None,         # trail/floor/timeout exit PnL
        "e5_exit_reason": str or None,
        "e5_giveback_bps": float or None,     # peak_at_resolve - e5_exit_bps
      }
    """
    traj = trade["trajectory"]
    if not traj:
        return {"fired": False}
    d = trade["direction"]
    entry_price = trade["entry_price"]

    # Find E5 ignition bar in the baseline trajectory
    fire_idx = None
    for k, bar in enumerate(traj):
        if k < min_hold:
            continue
        if bar["H"] is None:
            continue
        # Use the peak tracked in trajectory up to this bar
        peak = bar["peak_bps"]
        if bar["H"] > h_exit_threshold and peak < mfe_guard_bps:
            fire_idx = k
            break
    if fire_idx is None:
        return {"fired": False}

    fire_bar = traj[fire_idx]
    # Start post_signal_trail. trail_peak starts at current peak or current
    # bar's high (whichever is better in trade direction); reset bars_in_trail
    if d == 1:
        trail_peak_price = fire_bar["high"]
    else:
        trail_peak_price = fire_bar["low"]

    resolve_exit_bps = None
    resolve_reason = None
    bars_in_trail = 0
    # Evaluate trailing over remaining bars in baseline trajectory.
    # If baseline exited before max_wait, we also stop there -- we can't
    # know what would have happened past the baseline exit without
    # re-running the engine for those bars. We conservatively resolve at
    # baseline-exit if trail hasn't closed first.
    for j in range(fire_idx, len(traj)):
        bar = traj[j]
        bars_in_trail = j - fire_idx
        # update trail_peak in price space
        if d == 1:
            if bar["high"] > trail_peak_price: trail_peak_price = bar["high"]
        else:
            if bar["low"] < trail_peak_price: trail_peak_price = bar["low"]
        # trail_peak in bps from entry
        trail_peak_bps = d * (trail_peak_price / entry_price - 1) * 10000
        # floor in bps = trail_peak - width
        floor_bps = trail_peak_bps - width_bps
        # this bar's worst PnL
        if d == 1:
            worst_bps = (bar["low"]  / entry_price - 1) * 10000
            best_bps  = (bar["high"] / entry_price - 1) * 10000
        else:
            worst_bps = -(bar["high"] / entry_price - 1) * 10000
            best_bps  = -(bar["low"]  / entry_price - 1) * 10000

        # Exit conditions in spec order
        if worst_bps <= floor_bps:
            resolve_exit_bps = floor_bps
            resolve_reason = "entropy_decay_trail"
            break
        if worst_bps <= hard_floor:
            resolve_exit_bps = hard_floor
            resolve_reason = "entropy_decay_trail_floored"
            break
        if bars_in_trail >= max_wait:
            resolve_exit_bps = float(bar["curr_bps"])
            resolve_reason = "entropy_decay_trail_timeout"
            break

    if resolve_exit_bps is None:
        # Baseline trajectory ended before trail resolved -- resolve at
        # baseline's own exit PnL as a conservative estimate.
        last = traj[-1]
        resolve_exit_bps = float(last["curr_bps"])
        resolve_reason = "conservative_basline_exit"

    return {
        "fired": True,
        "fire_bar_offset": fire_idx,
        "H_at_fire": fire_bar["H"],
        "peak_at_fire": fire_bar["peak_bps"],
        "e5_exit_bps": resolve_exit_bps,
        "e5_exit_reason": resolve_reason,
        "e5_giveback_bps": (d * (trail_peak_price / entry_price - 1) * 10000) - resolve_exit_bps
                            if resolve_exit_bps is not None else None,
    }


def apply_e5_everywhere(trades, h_exit_threshold, mfe_guard_bps, width_bps):
    """Apply E5 replay to every trade, return list of merged records."""
    out = []
    for t in trades:
        r = replay_e5_on_trade(t, h_exit_threshold, mfe_guard_bps, width_bps)
        rec = {k: t[k] for k in ("entry_bar", "exit_bar", "direction",
                                  "pnl_bps", "reason", "hold_bars",
                                  "peak_bps_final")}
        rec["e5"] = r
        out.append(rec)
    return out


def summarize_e5(records):
    """Produce acceptance-style summary from E5-enriched records."""
    winners = [r for r in records if r["pnl_bps"] > 0]
    losers = [r for r in records if r["pnl_bps"] <= 0]
    winners_80 = [r for r in winners if r["pnl_bps"] >= 80]

    # Winners that E5 fired on, with final PnL >= 80: check trail giveback
    violators = []
    acceptable_low_peak_fires = []
    for r in winners:
        e5 = r["e5"]
        if not e5.get("fired"): continue
        peak_at_fire = e5.get("peak_at_fire") or 0
        if r["pnl_bps"] >= 80:
            # Acceptance criterion: trail must not exit below (peak - 25)
            # "peak" here = the trade's ultimate peak in the baseline, which
            # the baseline trajectory knows. Use peak_bps_final.
            expected_floor = r["peak_bps_final"] - WINNER_GIVEBACK_MAX_BPS
            if (e5.get("e5_exit_bps") or -999) < expected_floor:
                violators.append({
                    "entry_bar": r["entry_bar"],
                    "baseline_pnl_bps": r["pnl_bps"],
                    "baseline_peak_bps": r["peak_bps_final"],
                    "e5_exit_bps": e5.get("e5_exit_bps"),
                    "giveback_vs_peak": r["peak_bps_final"] - (e5.get("e5_exit_bps") or 0),
                    "fire_bar_offset": e5.get("fire_bar_offset"),
                    "direction": r["direction"],
                })
        else:
            # Winner with peak < 80: E5 firing is acceptable regardless
            if e5.get("fired"):
                acceptable_low_peak_fires.append({
                    "baseline_pnl_bps": r["pnl_bps"],
                    "baseline_peak_bps": r["peak_bps_final"],
                    "e5_exit_bps": e5.get("e5_exit_bps"),
                })

    # Losers: how much E5 would have saved vs baseline exit
    loser_deltas = []
    for r in losers:
        e5 = r["e5"]
        if e5.get("fired"):
            delta = (e5.get("e5_exit_bps") or r["pnl_bps"]) - r["pnl_bps"]
            loser_deltas.append({
                "baseline_pnl_bps": r["pnl_bps"],
                "e5_exit_bps": e5.get("e5_exit_bps"),
                "delta_bps": delta,
                "direction": r["direction"],
            })

    # Per-direction stats
    by_dir = {}
    for d in (1, -1):
        subs = [r for r in records if r["direction"] == d]
        fired = [r for r in subs if r["e5"].get("fired")]
        by_dir["long" if d == 1 else "short"] = {
            "n_trades": len(subs),
            "n_e5_fired": len(fired),
            "fired_on_winners": sum(1 for r in fired if r["pnl_bps"] > 0),
            "fired_on_losers": sum(1 for r in fired if r["pnl_bps"] <= 0),
            "violator_count": sum(1 for v in violators
                                   if v["direction"] == d),
        }

    return {
        "n_trades": len(records),
        "n_winners_total": len(winners),
        "n_winners_ge80": len(winners_80),
        "n_winners_cut_below_peak_minus_25":
            len([v for v in violators]),
        "n_winners_low_peak_e5_fired": len(acceptable_low_peak_fires),
        "n_losers": len(losers),
        "n_losers_e5_fired": len(loser_deltas),
        "loser_avg_save_bps": (
            float(np.mean([l["delta_bps"] for l in loser_deltas]))
            if loser_deltas else 0.0),
        "loser_total_save_bps": (
            float(sum(l["delta_bps"] for l in loser_deltas))
            if loser_deltas else 0.0),
        "violators": violators,
        "low_peak_fires": acceptable_low_peak_fires[:10],  # cap
        "by_direction": by_dir,
    }


def main():
    print("Loading ETH Pi orderbook...", flush=True)
    df = load_orderbook_range("data/orderbook_pi", "ob_PF_ETHUSD")
    df_1m = resample_pi_to_1min(df)
    df_1m = compute_ob_features(df_1m)
    df_1m = classify_ob_states(df_1m, window=60)
    ent = rolling_entropy_ob(df_1m['state_ob'].values, NUM_STATES_OB, 30)
    feats = make_features(df_1m, ent)
    n = feats['n']
    print(f"  {n} bars, {n/1440:.1f} days", flush=True)

    params = {'imb_min': 0.05, 'spread_max': 20, 'ret_low': 20, 'ret_high': 80}
    print("Running baseline with full trajectories...", flush=True)
    trades = run_baseline_with_trajectories(
        feats, 0.4352, params,
        sl=50, tp=200, trail_after=150, trail_bps=50,
        cooldown=0, knife_bps=50, extended_cap_bps=100)
    print(f"  baseline trades: {len(trades)}", flush=True)

    # ---- Main acceptance run: defaults (0.4852 / 30 / 20) ----
    main_recs = apply_e5_everywhere(trades, 0.4852, 30, 20)
    main_summary = summarize_e5(main_recs)

    # ---- Grid sensitivity (3x3x2) ----
    h_grid = [0.465, 0.4852, 0.505]
    mfe_grid = [20, 30, 50]
    w_grid = [15, 25]
    grid = []
    for h in h_grid:
        for m in mfe_grid:
            for w in w_grid:
                recs = apply_e5_everywhere(trades, h, m, w)
                s = summarize_e5(recs)
                grid.append({
                    "h_exit_threshold": h,
                    "mfe_guard_bps": m,
                    "width_bps": w,
                    "n_winners_cut": s["n_winners_cut_below_peak_minus_25"],
                    "n_losers_fired": s["n_losers_e5_fired"],
                    "loser_total_save_bps": s["loser_total_save_bps"],
                    "loser_avg_save_bps": s["loser_avg_save_bps"],
                })

    out = {
        "config_defaults": {
            "h_exit_threshold": 0.4852,
            "mfe_guard_bps": 30,
            "width_bps": 20,
            "hard_floor_bps": HARD_FLOOR_BPS,
            "max_wait_bars": MAX_WAIT_BARS,
            "min_hold_bars": MIN_HOLD_BARS,
        },
        "baseline": {
            "n_trades": len(trades),
            "long_n": sum(1 for t in trades if t["direction"] == 1),
            "short_n": sum(1 for t in trades if t["direction"] == -1),
        },
        "main_acceptance": main_summary,
        "grid_sensitivity": grid,
    }
    OUT.write_text(json.dumps(out, indent=2, default=str))

    # Console summary
    print("\n=== E5 ACCEPTANCE (defaults 0.4852 / 30 / 20) ===")
    print(f"Trades: {main_summary['n_trades']} "
          f"(winners={main_summary['n_winners_total']}, "
          f"winners>=80={main_summary['n_winners_ge80']})")
    print(f"Winners cut below (peak - 25): "
          f"{main_summary['n_winners_cut_below_peak_minus_25']}")
    print(f"Winners with peak<80 where E5 fired: "
          f"{main_summary['n_winners_low_peak_e5_fired']}")
    print(f"Losers where E5 fired: {main_summary['n_losers_e5_fired']} / "
          f"{main_summary['n_losers']}, "
          f"avg save {main_summary['loser_avg_save_bps']:+.1f} bps, "
          f"total {main_summary['loser_total_save_bps']:+.1f} bps")
    for side in ("long", "short"):
        d = main_summary["by_direction"][side]
        print(f"  {side:5} n={d['n_trades']}, e5_fired={d['n_e5_fired']} "
              f"(w={d['fired_on_winners']}, l={d['fired_on_losers']}), "
              f"violators={d['violator_count']}")

    # Grid top-3 best (zero winner-cuts, max loser-save)
    eligible = [g for g in grid if g["n_winners_cut"] == 0]
    if eligible:
        eligible.sort(key=lambda g: -g["loser_total_save_bps"])
        print("\nTop 3 grid cells with zero winner-cuts:")
        for g in eligible[:3]:
            print(f"  h={g['h_exit_threshold']}, mfe={g['mfe_guard_bps']}, "
                  f"w={g['width_bps']}: loser_save={g['loser_total_save_bps']:+.0f} bps "
                  f"({g['n_losers_fired']} trades)")
    else:
        print("\nNo grid cell has zero winner-cuts.")
        grid.sort(key=lambda g: (g["n_winners_cut"], -g["loser_total_save_bps"]))
        print("Best (fewest winner-cuts first, then largest loser-save):")
        for g in grid[:5]:
            print(f"  h={g['h_exit_threshold']}, mfe={g['mfe_guard_bps']}, "
                  f"w={g['width_bps']}: cuts={g['n_winners_cut']}, "
                  f"loser_save={g['loser_total_save_bps']:+.0f} bps")

    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
