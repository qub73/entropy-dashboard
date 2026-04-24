"""
E5 expanded grid sweep with min_hold_bars as a new dimension, plus
H-trajectory diagnostic against the current baseline trades.

Acceptance (this run):
  (a) zero TOTAL winner-cuts below (peak - 25)
  (b) zero LONG-side winner-cuts
  (c) loser_total_save_bps >= 300
If multiple cells meet all three, pick the one with lowest
min_hold_bars (preserves more loser-save).

Fallback: no cell passes -> report closest, do not promote.

Inputs: Feb 18 - Apr 7 Pi baseline trades with H trajectories
(generated inline from the same engine as phase3_e5_acceptance.py).

Outputs:
  - reports/phase3_e5_min_hold_sweep.json (grid + acceptance decision)
  - reports/phase3_e5_h_trajectory.json   (diagnostic trajectory stats)
  - reports/phase3_e5_h_trajectory.png    (plot of H paths grouped)
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
from upgrade_backtest import make_features
# reuse the engine + E5 replay from the acceptance script
from phase3_e5_acceptance import (
    run_baseline_with_trajectories,
    replay_e5_on_trade,
    HARD_FLOOR_BPS, MAX_WAIT_BARS, WINNER_GIVEBACK_MAX_BPS,
)

OUT_GRID = ALGO / "reports" / "phase3_e5_min_hold_sweep.json"
OUT_TRAJ_JSON = ALGO / "reports" / "phase3_e5_h_trajectory.json"
OUT_TRAJ_PNG  = ALGO / "reports" / "phase3_e5_h_trajectory.png"


def summarize_e5_expanded(records):
    winners = [r for r in records if r["pnl_bps"] > 0]
    losers  = [r for r in records if r["pnl_bps"] <= 0]
    violators = []
    for r in winners:
        e5 = r["e5"]
        if not e5.get("fired"): continue
        if r["pnl_bps"] < 80: continue
        exp_floor = r["peak_bps_final"] - WINNER_GIVEBACK_MAX_BPS
        if (e5.get("e5_exit_bps") or -999) < exp_floor:
            violators.append({
                "direction": r["direction"],
                "baseline_pnl": r["pnl_bps"],
                "baseline_peak": r["peak_bps_final"],
                "e5_exit": e5.get("e5_exit_bps"),
            })
    loser_deltas = []
    for r in losers:
        e5 = r["e5"]
        if e5.get("fired"):
            delta = (e5.get("e5_exit_bps") or r["pnl_bps"]) - r["pnl_bps"]
            loser_deltas.append(delta)
    long_cuts  = sum(1 for v in violators if v["direction"] ==  1)
    short_cuts = sum(1 for v in violators if v["direction"] == -1)
    return {
        "n_winners_cut_total": len(violators),
        "long_winner_cuts": long_cuts,
        "short_winner_cuts": short_cuts,
        "n_losers_e5_fired": len(loser_deltas),
        "loser_total_save_bps": float(sum(loser_deltas)),
        "loser_avg_save_bps": float(np.mean(loser_deltas)) if loser_deltas else 0.0,
    }


def replay_with_min_hold(trade, h_thresh, mfe_guard, width, min_hold):
    """Same as replay_e5_on_trade but with configurable min_hold_bars."""
    from phase3_e5_acceptance import replay_e5_on_trade as base_replay
    # The base replay signature doesn't take min_hold; re-implement with
    # the same logic but injectable min_hold_bars.
    traj = trade["trajectory"]
    if not traj: return {"fired": False}
    d = trade["direction"]; entry_price = trade["entry_price"]
    fire_idx = None
    for k, bar in enumerate(traj):
        if k < min_hold: continue
        if bar["H"] is None: continue
        peak = bar["peak_bps"]
        if bar["H"] > h_thresh and peak < mfe_guard:
            fire_idx = k; break
    if fire_idx is None: return {"fired": False}
    fire_bar = traj[fire_idx]
    if d == 1: trail_peak_price = fire_bar["high"]
    else:      trail_peak_price = fire_bar["low"]
    resolve_exit_bps = resolve_reason = None
    for j in range(fire_idx, len(traj)):
        bar = traj[j]
        bars_in_trail = j - fire_idx
        if d == 1:
            if bar["high"] > trail_peak_price: trail_peak_price = bar["high"]
        else:
            if bar["low"]  < trail_peak_price: trail_peak_price = bar["low"]
        trail_peak_bps = d * (trail_peak_price / entry_price - 1) * 10000
        floor_bps = trail_peak_bps - width
        if d == 1:
            worst_bps = (bar["low"]  / entry_price - 1) * 10000
        else:
            worst_bps = -(bar["high"] / entry_price - 1) * 10000
        if worst_bps <= floor_bps:
            resolve_exit_bps = floor_bps; resolve_reason = "trail"; break
        if worst_bps <= HARD_FLOOR_BPS:
            resolve_exit_bps = HARD_FLOOR_BPS; resolve_reason = "floored"; break
        if bars_in_trail >= MAX_WAIT_BARS:
            resolve_exit_bps = float(bar["curr_bps"]); resolve_reason = "timeout"; break
    if resolve_exit_bps is None:
        resolve_exit_bps = float(traj[-1]["curr_bps"])
        resolve_reason = "conservative"
    return {
        "fired": True,
        "fire_bar_offset": fire_idx,
        "H_at_fire": fire_bar["H"],
        "peak_at_fire": fire_bar["peak_bps"],
        "e5_exit_bps": resolve_exit_bps,
        "e5_exit_reason": resolve_reason,
    }


def apply_e5(trades, h_thresh, mfe_guard, width, min_hold):
    out = []
    for t in trades:
        r = replay_with_min_hold(t, h_thresh, mfe_guard, width, min_hold)
        rec = {k: t[k] for k in ("direction", "pnl_bps", "peak_bps_final")}
        rec["e5"] = r
        out.append(rec)
    return out


def h_trajectory_stats(trades):
    """For the diagnostic plot: aggregate H vs bar-offset, grouped by
    (outcome, direction). Returns:
       { group_key: { 'n': int, 'mean_H': [...], 'median_H': [...],
                      'frac_above_thresh': [...] } }
    """
    groups = {}
    MAX_OFFSET = 60  # first 60 minutes of each trade
    for t in trades:
        traj = t["trajectory"]
        if not traj: continue
        # outcome = winner if pnl>=80, loser if <=0, mid otherwise
        if t["pnl_bps"] >= 80: outcome = "winner_ge80"
        elif t["pnl_bps"] <= 0: outcome = "loser"
        else: outcome = "mid"
        side = "long" if t["direction"] == 1 else "short"
        key = f"{side}_{outcome}"
        if key not in groups:
            groups[key] = {"n": 0, "h_by_offset": [[] for _ in range(MAX_OFFSET)]}
        g = groups[key]; g["n"] += 1
        for k, bar in enumerate(traj[:MAX_OFFSET]):
            if bar["H"] is not None:
                g["h_by_offset"][k].append(bar["H"])
    # reduce
    out = {}
    for key, g in groups.items():
        mean = [float(np.mean(x)) if x else None for x in g["h_by_offset"]]
        median = [float(np.median(x)) if x else None for x in g["h_by_offset"]]
        frac_above = [float(np.mean([1.0 if h > 0.4852 else 0.0 for h in x]))
                      if x else None for x in g["h_by_offset"]]
        out[key] = {"n": g["n"], "mean_H": mean,
                    "median_H": median, "frac_above_0.4852": frac_above}
    return out


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
    print("Running baseline with trajectories...", flush=True)
    trades = run_baseline_with_trajectories(
        feats, 0.4352, params,
        sl=50, tp=200, trail_after=150, trail_bps=50,
        cooldown=0, knife_bps=50, extended_cap_bps=100)
    print(f"  {len(trades)} baseline trades", flush=True)

    # ------------------- Diagnostic: H trajectories -------------------
    traj_stats = h_trajectory_stats(trades)
    OUT_TRAJ_JSON.write_text(json.dumps(traj_stats, indent=2, default=str))
    print("\nH trajectory groups:")
    for k, g in sorted(traj_stats.items()):
        # time (in minutes) at which median H first crosses 0.4852
        cross = None
        for off, v in enumerate(g["median_H"]):
            if v is not None and v > 0.4852:
                cross = off; break
        print(f"  {k:25} n={g['n']:3}  median_H_crosses_0.4852_at_bar = {cross}")

    # Plot if matplotlib is available
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.patch.set_facecolor("#0d1117")
        for ax in axes:
            ax.set_facecolor("#161b22")
            ax.tick_params(colors="#e6edf3")
            for s in ax.spines.values(): s.set_color("#30363d")
            ax.grid(True, color="#30363d", alpha=0.4)
            ax.xaxis.label.set_color("#e6edf3"); ax.yaxis.label.set_color("#e6edf3")
            ax.title.set_color("#e6edf3")
        colors = {
            "long_winner_ge80": "#3fb950", "long_loser": "#f85149", "long_mid": "#d29922",
            "short_winner_ge80": "#79c0ff", "short_loser": "#bc8cff", "short_mid": "#e6edf3",
        }
        for key in sorted(traj_stats.keys()):
            g = traj_stats[key]
            if g["n"] < 2: continue
            ax = axes[0] if key.startswith("long") else axes[1]
            x = list(range(len(g["median_H"])))
            y = [v if v is not None else np.nan for v in g["median_H"]]
            ax.plot(x, y, label=f"{key} (n={g['n']})",
                    color=colors.get(key, "#58a6ff"), linewidth=1.8)
        for ax, title in ((axes[0], "LONG trades  -  median H by bars held"),
                          (axes[1], "SHORT trades  -  median H by bars held")):
            ax.axhline(0.4352, color="#3fb950", linestyle="--", linewidth=1, alpha=0.7, label="H_threshold=0.4352")
            ax.axhline(0.4852, color="#f85149", linestyle="--", linewidth=1, alpha=0.7, label="E5 h_exit=0.4852")
            ax.axvline(10, color="#6e7681", linestyle=":", linewidth=1, alpha=0.6, label="current min_hold=10")
            ax.set_title(title); ax.set_xlabel("bars held"); ax.set_ylabel("H")
            ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=9)
        fig.suptitle("E5 mechanical-H-drift diagnostic  (Feb-Mar Pi baseline)",
                     color="#e6edf3", fontsize=13, y=0.99)
        fig.tight_layout()
        fig.savefig(OUT_TRAJ_PNG, dpi=110, facecolor=fig.get_facecolor())
        print(f"Plot saved: {OUT_TRAJ_PNG}")
    except Exception as e:
        print(f"(plot skipped: {e})")

    # ------------------- Expanded E5 grid -------------------
    h_grid = [0.465, 0.4852, 0.505]
    mfe_grid = [20, 30, 50]
    w_grid = [15, 20, 25]
    mh_grid = [10, 20, 30, 45, 60]
    grid = []
    total = len(h_grid) * len(mfe_grid) * len(w_grid) * len(mh_grid)
    print(f"\nRunning expanded grid ({total} cells)...", flush=True)
    n_done = 0
    for h in h_grid:
        for m in mfe_grid:
            for w in w_grid:
                for mh in mh_grid:
                    recs = apply_e5(trades, h, m, w, mh)
                    s = summarize_e5_expanded(recs)
                    grid.append({
                        "h_exit_threshold": h, "mfe_guard_bps": m,
                        "width_bps": w, "min_hold_bars": mh,
                        **s,
                    })
                    n_done += 1
        print(f"  done h={h}: {n_done}/{total}", flush=True)

    # ------------------- Acceptance -------------------
    passing = [g for g in grid
               if g["n_winners_cut_total"] == 0
               and g["long_winner_cuts"] == 0
               and g["loser_total_save_bps"] >= 300]
    if passing:
        passing.sort(key=lambda g: (g["min_hold_bars"], -g["loser_total_save_bps"]))
        winner = passing[0]
        decision = "ACCEPT"
        print("\n=== ACCEPTANCE: PASS ===")
        print(f"Cells meeting all three: {len(passing)}")
        print(f"Selected (lowest min_hold_bars): {winner}")
    else:
        # rank cells by how close they are: min total cuts, then min long cuts,
        # then max loser save
        grid_ranked = sorted(grid,
            key=lambda g: (g["n_winners_cut_total"], g["long_winner_cuts"],
                           -g["loser_total_save_bps"]))
        winner = grid_ranked[0]
        decision = "REJECT"
        print("\n=== ACCEPTANCE: FAIL ===")
        print("No cell meets all three criteria.")
        print("Closest (by cuts, then loser-save):")
        for g in grid_ranked[:5]:
            print(f"  h={g['h_exit_threshold']}, mfe={g['mfe_guard_bps']}, "
                  f"w={g['width_bps']}, min_hold={g['min_hold_bars']}: "
                  f"cuts total/long={g['n_winners_cut_total']}/"
                  f"{g['long_winner_cuts']}, losers_fired={g['n_losers_e5_fired']}, "
                  f"L_save={g['loser_total_save_bps']:+.0f}")

    out = {
        "acceptance_rule": {
            "a_zero_total_winner_cuts": True,
            "b_zero_long_winner_cuts": True,
            "c_loser_save_bps_min": 300,
            "tiebreaker": "lowest min_hold_bars, then highest loser_save",
        },
        "decision": decision,
        "selected": winner,
        "n_cells_passing": len(passing) if passing else 0,
        "n_cells_total": len(grid),
        "all_cells": grid,
    }
    OUT_GRID.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {OUT_GRID}")


if __name__ == "__main__":
    main()
