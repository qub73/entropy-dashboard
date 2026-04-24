"""Parametrized exit engine derived from the deployed mean-reversion
strategy (run_cell_nof3c). F3c stays off (per sprint v1.5 RED). Entry
logic unchanged from deployed; only exit-stack params vary.

Per-bar exit priority:
  1. E3 time-decayed SL (if enabled)
  2. SL (with dynamic floor if E3 has tightened)
  3. tp_trail (activates at peak >= trail_after; trails at peak - trail_bps)
  4. ATR trail (post-timeout-in-profit; trail at peak - atr_mult*atr_bps)
  5. post_signal_trail (post-timeout-in-loss; width + floor + max_wait)
  6. Fixed TP (if none of the above fired and peak never hit trail_after)
  7. Max-hold (timeout) — switch to ATR-trail or post_signal_trail

Extra "radical" modes:
  exit_mode in:
    "standard"     -- as above
    "atr_trail"    -- once peak>=trail_after, use peak - atr_mult*atr_bps
                      (no fixed trail_bps; ATR-scaled)
    "dual_stage"   -- peak<100 bps -> tight trail (trail_bps_tight);
                      peak>=100 bps -> loose trail (trail_bps_loose)
                      addresses mode (a) by letting winners run wider
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class ExitParams:
    # Entry-side (unchanged from deployed)
    sl_bps: float = 50.0
    tp_bps: float = 200.0
    knife_bps: float = 50.0
    ext_cap_bps: float = 100.0
    h_thresh: float = 0.4352
    # Exit stack (the sweep targets these)
    trail_after: float = 150.0        # peak threshold to activate tp_trail
    trail_bps: float = 50.0           # tp_trail width
    trail_bps_tight: float = 30.0     # dual_stage: peak<100
    trail_bps_loose: float = 75.0     # dual_stage: peak>=100
    dual_stage_pivot: float = 100.0   # peak bps at which trail widens
    exit_mode: str = "standard"       # standard | atr_trail | dual_stage
    atr_trail_mult: float = 2.0       # used in standard post-timeout AND atr_trail modes
    # post_signal_trail (timeout-in-loss)
    pst_width_bps: float = 20.0
    pst_hard_floor_bps: float = -60.0
    pst_max_wait_bars: int = 30
    # E3 time-decayed SL
    e3_enabled: bool = True
    e3_after_bars: int = 60
    e3_peak_threshold_bps: float = 50.0
    e3_tightened_sl_bps: float = 25.0
    # Timeout
    timeout_bars: int = 240
    # Sizing
    leverage: int = 5
    equity_fraction: float = 0.90
    fee_bps_per_side: float = 5.0


def run_exit_variant(feats: dict, extra: dict, p: ExitParams) -> Dict:
    """Mirror run_cell_nof3c logic with parametrized exit stack. Returns
    aggregate metrics + trade list."""
    n = feats["n"]
    mids = feats["mid"]; highs = feats["high"]; lows = feats["low"]
    atr_bps_arr = feats["atr_bps"]; imb = feats["imb5"]
    dH_5 = feats["dH_5"]; ret_60 = feats["ret_60"]
    ret_150 = np.zeros(n)
    ret_150[150:] = (mids[150:] / mids[:-150] - 1) * 10000

    # Entry candidates (reuse signal pipeline from upgrade_backtest)
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from upgrade_backtest import candidate_signals
    CORE_PARAMS = {"imb_min": 0.05, "spread_max": 20,
                    "ret_low": 20, "ret_high": 80}
    cands = set(candidate_signals(feats, p.h_thresh, CORE_PARAMS))

    equity = 10000.0; eq_curve = [equity]
    in_trade = False
    entry_idx = entry_price = direction = 0
    notional = peak_pnl = 0.0; initial_notional = 0.0
    tp_trailing = trailing_active = pst_active = False
    pst_peak_bps = 0.0; pst_entry_bar = 0
    sl_current = -p.sl_bps
    e3_tightened = False
    trades: List[dict] = []

    for i in range(n):
        if in_trade:
            d = direction
            if d == 1:
                worst_bps = (lows[i]/entry_price - 1)*10000
                best_bps  = (highs[i]/entry_price - 1)*10000
            else:
                worst_bps = -(highs[i]/entry_price - 1)*10000
                best_bps  = -(lows[i]/entry_price - 1)*10000
            curr_bps = d * (mids[i]/entry_price - 1) * 10000
            peak_pnl = max(peak_pnl, best_bps)
            bars_in = i - entry_idx
            exit_reason = None; exit_pnl = None

            # 1. E3 pre-check (tighten SL for subsequent checks)
            if (p.e3_enabled and not e3_tightened
                and bars_in >= p.e3_after_bars
                and peak_pnl < p.e3_peak_threshold_bps):
                sl_current = -p.e3_tightened_sl_bps
                e3_tightened = True

            # 2. SL
            if worst_bps <= sl_current and not pst_active:
                exit_reason = "sl"; exit_pnl = sl_current

            # 3. tp_trail activation (by exit_mode)
            if exit_reason is None and not tp_trailing:
                if peak_pnl >= p.trail_after:
                    tp_trailing = True

            # 4. Apply trail logic per mode
            if exit_reason is None and tp_trailing and not pst_active:
                if p.exit_mode == "standard":
                    floor = max(peak_pnl - p.trail_bps, sl_current)
                elif p.exit_mode == "atr_trail":
                    aw = p.atr_trail_mult * atr_bps_arr[i] \
                         if atr_bps_arr[i] > 0 else 50.0
                    floor = max(peak_pnl - aw, sl_current)
                elif p.exit_mode == "dual_stage":
                    tw = p.trail_bps_tight if peak_pnl < p.dual_stage_pivot \
                         else p.trail_bps_loose
                    floor = max(peak_pnl - tw, sl_current)
                else:
                    raise ValueError(f"unknown exit_mode: {p.exit_mode}")
                if worst_bps <= floor:
                    exit_reason = "tp_trail"; exit_pnl = max(floor, curr_bps)

            # 5. Post-timeout ATR trail
            if exit_reason is None and trailing_active and not pst_active:
                tw = p.atr_trail_mult * atr_bps_arr[i] \
                     if atr_bps_arr[i] > 0 else 50.0
                floor = max(peak_pnl - tw, sl_current)
                if worst_bps <= floor:
                    exit_reason = "trail_stop"; exit_pnl = max(floor, curr_bps)
                elif best_bps >= p.tp_bps:
                    exit_reason = "tp"; exit_pnl = p.tp_bps

            # 6. post_signal_trail (timeout-in-loss)
            if exit_reason is None and pst_active:
                pst_peak_bps = max(pst_peak_bps, best_bps)
                bars_pst = i - pst_entry_bar
                floor = max(pst_peak_bps - p.pst_width_bps, sl_current)
                if worst_bps <= floor:
                    exit_reason = "pst_trail"; exit_pnl = max(floor, curr_bps)
                elif worst_bps <= p.pst_hard_floor_bps:
                    exit_reason = "pst_floored"; exit_pnl = p.pst_hard_floor_bps
                elif bars_pst >= p.pst_max_wait_bars:
                    exit_reason = "pst_timeout"; exit_pnl = curr_bps

            # 7. Fixed TP (only if not in trail modes)
            if (exit_reason is None and not tp_trailing and not pst_active
                and not trailing_active and best_bps >= p.tp_bps):
                exit_reason = "tp"; exit_pnl = p.tp_bps

            # 8. Timeout transition
            if (exit_reason is None and bars_in >= p.timeout_bars
                and not trailing_active and not pst_active):
                if curr_bps > 0:
                    trailing_active = True
                else:
                    pst_active = True; pst_entry_bar = i
                    pst_peak_bps = best_bps

            if exit_reason is not None:
                fee = notional * p.fee_bps_per_side / 10000
                realized = (exit_pnl / 10000) * notional - fee
                equity += realized
                pos_pnl_bps = (realized / initial_notional * 10000
                                if initial_notional > 0 else exit_pnl)
                trades.append({
                    "entry_idx": entry_idx, "exit_idx": i,
                    "direction": direction, "bars_held": bars_in,
                    "pnl_bps": pos_pnl_bps, "peak_bps": peak_pnl,
                    "reason": exit_reason,
                })
                in_trade = False
                tp_trailing = False; trailing_active = False
                pst_active = False; e3_tightened = False
                sl_current = -p.sl_bps
            eq_curve.append(equity)
            continue

        eq_curve.append(equity)
        if i not in cands or equity <= 0: continue
        if not np.isnan(dH_5[i]) and dH_5[i] >= 0: continue
        d = 1 if imb[i] > 0 else -1
        if p.knife_bps and d == 1 and ret_60[i] < -p.knife_bps: continue
        if p.knife_bps and d == -1 and ret_60[i] > p.knife_bps: continue
        if p.ext_cap_bps is not None:
            r150 = ret_150[i]
            if d == 1 and r150 > p.ext_cap_bps: continue
            if d == -1 and r150 < -p.ext_cap_bps: continue

        margin = equity * p.equity_fraction
        notional = margin * p.leverage
        initial_notional = notional
        equity -= notional * p.fee_bps_per_side / 10000
        in_trade = True; entry_idx = i; entry_price = mids[i]
        direction = d; peak_pnl = 0
        tp_trailing = False; trailing_active = False; pst_active = False
        e3_tightened = False; sl_current = -p.sl_bps

    # End-of-sample close
    if in_trade:
        curr = direction * (mids[-1]/entry_price - 1) * 10000
        fee = notional * p.fee_bps_per_side / 10000
        realized = (curr/10000) * notional - fee
        equity += realized
        pos_pnl_bps = (realized / initial_notional * 10000
                        if initial_notional > 0 else curr)
        trades.append({
            "entry_idx": entry_idx, "exit_idx": n-1,
            "direction": direction, "bars_held": n-1-entry_idx,
            "pnl_bps": pos_pnl_bps, "peak_bps": peak_pnl,
            "reason": "end_of_sample",
        })
        eq_curve.append(equity)

    return _metrics(trades, np.asarray(eq_curve))


def _metrics(trades, eq):
    nt = len(trades)
    if nt == 0:
        return {"n_trades": 0, "compound_ret_pct": 0.0, "max_dd_pct": 0.0,
                "win_rate": 0.0, "pf": 0.0, "sharpe_daily": 0.0,
                "knife_rate": 0.0, "avg_winner_bps": 0.0,
                "avg_loser_bps": 0.0, "mean_pnl_bps": 0.0,
                "reason_distribution": {}}
    bps = [t["pnl_bps"] for t in trades]
    wins = [p for p in bps if p > 0]; losses = [p for p in bps if p <= 0]
    pk = np.maximum.accumulate(eq)
    dd = float(((pk - eq)/pk*100).max()) if len(eq) > 1 else 0.0
    ret = (eq[-1] - 10000)/100.0
    pf = (abs(sum(wins)/sum(losses)) if losses and sum(losses) != 0
          else (float("inf") if wins else 0.0))
    knife = sum(1 for t in trades if t["pnl_bps"] <= 0 and t["peak_bps"] < 20)
    reasons = {}
    for t in trades:
        reasons[t["reason"]] = reasons.get(t["reason"], 0) + 1
    bars_per_day = 1440
    if len(eq) > bars_per_day:
        eod = np.arange(bars_per_day, len(eq), bars_per_day)
        eod_eq = eq[eod]
        if len(eod_eq) > 1:
            d = np.diff(eod_eq) / eod_eq[:-1]
            sharpe = (d.mean() / d.std() * np.sqrt(365)
                      if d.std() > 0 else 0.0)
        else: sharpe = 0.0
    else: sharpe = 0.0
    return {
        "n_trades": nt, "compound_ret_pct": float(ret), "max_dd_pct": dd,
        "win_rate": len(wins)/nt, "pf": float(pf),
        "sharpe_daily": float(sharpe),
        "knife_rate": knife/nt, "knife_catchers": knife,
        "avg_winner_bps": float(np.mean(wins)) if wins else 0.0,
        "avg_loser_bps": float(np.mean(losses)) if losses else 0.0,
        "mean_pnl_bps": float(np.mean(bps)),
        "trades_long": sum(1 for t in trades if t["direction"] == 1),
        "trades_short": sum(1 for t in trades if t["direction"] == -1),
        "reason_distribution": reasons,
        "trades": trades,
    }
