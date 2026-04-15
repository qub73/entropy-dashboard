"""
Test variants of trailing stop in place of hard timeout exit.

Hypothesis: when a trade reaches the 4-hour timeout while in profit
and price is still trending in our favour, exiting at market gives up
upside. A trailing stop should lock in some gain while letting winners run.

Variants:
  BASELINE         — current live: hard exit at timeout
  T1_simple        — at timeout, if PnL>0 switch to trailing-stop (trail 50 bps below peak)
  T2_aggressive    — same trigger, trail tighter (30 bps)
  T3_loose         — trail 80 bps below peak
  T4_only_if_rising— trail only if last 30-bar return is positive in trade direction
  T5_atr_trail     — trail at 2x ATR below peak
  T6_no_timeout    — never timeout: replace timeout with permanent trailing stop (50 bps)
  T7_partial_exit  — exit half at TP, trail the other half (approximated via average)
  T8_trail_after_50— trail kicks in once trade is +50 bps in profit (before timeout)
  T9_trail_after_100— trail kicks in once +100 bps in profit
"""

import sys, os, json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ob_entropy import (
    OBSignalParams, NUM_STATES_OB,
    rolling_entropy_ob, load_orderbook_range,
    compute_ob_features, classify_ob_states,
)
from kaggle_ob_trainer import (
    load_kaggle_csv, parse_kaggle_to_ob_features, resample_pi_to_1min,
)
from upgrade_backtest import make_features, candidate_signals


def run_variant_trailing(feats, h_thresh, variant,
                          leverage=10.0, capital=10000.0,
                          taker_fee_bps=5.0, equity_frac=0.90):
    """Walk the bar series with peak/trough tracking + variant exit logic."""
    p = variant.get('params', {
        'imb_min': 0.05, 'spread_max': 20.0,
        'ret_low': 20.0, 'ret_high': 80.0,
    })
    candidates = candidate_signals(feats, h_thresh, p)
    cand_set = set(candidates)
    sl_bps = variant.get('sl_bps', 65)
    tp_bps = variant.get('tp_bps', 200)
    timeout_bars = variant.get('timeout_bars', 240)

    # Trailing config
    trail_at_timeout = variant.get('trail_at_timeout', False)
    trail_bps = variant.get('trail_bps', 50)
    trail_min_pnl = variant.get('trail_min_pnl', 0)
    trail_require_rising = variant.get('trail_require_rising', False)
    trail_atr = variant.get('trail_atr', None)  # None or multiplier
    no_timeout = variant.get('no_timeout', False)
    trail_after_pnl = variant.get('trail_after_pnl', None)  # bps threshold

    n = feats['n']
    mids = feats['mid']
    highs = feats['high']
    lows = feats['low']
    atr_bps = feats['atr_bps']

    in_trade = False
    entry_idx = entry_price = direction = 0
    peak_pnl = 0.0
    notional = 0.0
    equity = capital
    trades = []
    trailing_active = False
    trail_floor_bps = -sl_bps  # stop level in bps (most-loss-acceptable)

    for i in range(n):
        if in_trade:
            if direction == 1:
                worst_pnl = (lows[i] / entry_price - 1) * 10000
                best_pnl = (highs[i] / entry_price - 1) * 10000
            else:
                worst_pnl = -(highs[i] / entry_price - 1) * 10000
                best_pnl = -(lows[i] / entry_price - 1) * 10000
            curr_pnl = direction * (mids[i] / entry_price - 1) * 10000

            # Track peak
            if best_pnl > peak_pnl:
                peak_pnl = best_pnl

            # Activate trail-after-pnl mode early
            if trail_after_pnl is not None and not trailing_active:
                if peak_pnl >= trail_after_pnl:
                    trailing_active = True

            exit_reason = None
            exit_pnl = None

            # Hard SL always applies (or trailing floor if active)
            if trailing_active:
                # trailing stop: peak - trail_amount
                if trail_atr is not None:
                    trail_amount = trail_atr * atr_bps[i]
                else:
                    trail_amount = trail_bps
                effective_floor = peak_pnl - trail_amount
                # Don't allow it to be looser than fixed SL
                effective_floor = max(effective_floor, -sl_bps)
                if worst_pnl <= effective_floor:
                    exit_reason = 'trail_stop'
                    exit_pnl = effective_floor
            else:
                if worst_pnl <= -sl_bps:
                    exit_reason = 'sl'
                    exit_pnl = -sl_bps

            if exit_reason is None:
                if best_pnl >= tp_bps:
                    exit_reason = 'tp'
                    exit_pnl = tp_bps

            if exit_reason is None:
                elapsed = i - entry_idx
                if elapsed >= timeout_bars:
                    if trail_at_timeout and curr_pnl > trail_min_pnl:
                        # Optional: also require recent rising
                        rising_ok = True
                        if trail_require_rising:
                            ret_30 = direction * (mids[i] / mids[max(0, i-30)] - 1) * 10000
                            rising_ok = ret_30 > 0
                        if rising_ok:
                            # Activate trailing instead of exiting
                            trailing_active = True
                            # Reset peak to current
                            peak_pnl = max(peak_pnl, curr_pnl)
                        else:
                            exit_reason = 'timeout'
                            exit_pnl = curr_pnl
                    elif no_timeout:
                        # No timeout — activate trailing always
                        trailing_active = True
                        peak_pnl = max(peak_pnl, curr_pnl)
                    else:
                        exit_reason = 'timeout'
                        exit_pnl = curr_pnl

            if exit_reason:
                fee = notional * taker_fee_bps / 10000
                realized = (exit_pnl / 10000) * notional - fee
                equity += realized
                trades.append({
                    'entry_idx': entry_idx,
                    'exit_idx': i,
                    'direction': direction,
                    'entry_price': entry_price,
                    'pnl_bps': exit_pnl,
                    'pnl_bps_lev': exit_pnl * leverage,
                    'pnl_usd': realized,
                    'reason': exit_reason,
                    'hold': i - entry_idx,
                    'peak_pnl_seen': peak_pnl,
                })
                in_trade = False
                trailing_active = False
                peak_pnl = 0.0
        else:
            if i in cand_set and equity > 0:
                d = 1 if feats['imb5'][i] > 0 else -1
                margin = equity * equity_frac
                notional = margin * leverage
                entry_fee = notional * taker_fee_bps / 10000
                equity -= entry_fee
                in_trade = True
                entry_idx = i
                entry_price = mids[i]
                direction = d
                peak_pnl = 0.0
                trailing_active = False

    # Force close
    if in_trade:
        curr_pnl = direction * (mids[-1] / entry_price - 1) * 10000
        fee = notional * taker_fee_bps / 10000
        realized = (curr_pnl / 10000) * notional - fee
        equity += realized
        trades.append({
            'entry_idx': entry_idx, 'exit_idx': n-1,
            'direction': direction, 'entry_price': entry_price,
            'pnl_bps': curr_pnl, 'pnl_bps_lev': curr_pnl * leverage,
            'pnl_usd': realized, 'reason': 'end', 'hold': n-1-entry_idx,
            'peak_pnl_seen': peak_pnl,
        })

    n_t = len(trades)
    if n_t == 0:
        return {'name': variant['name'], 'trades': 0, 'wins': 0, 'win_rate': 0,
                'ret_pct': 0, 'pnl_usd': 0, 'max_dd': 0, 'calmar': 0,
                'avg_bps': 0, 'avg_hold': 0, 'reasons': {}}

    pnls = [t['pnl_usd'] for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    peak = capital; cum = capital; max_dd = 0
    for p in pnls:
        cum += p
        peak = max(peak, cum)
        max_dd = max(max_dd, (peak - cum) / peak * 100)
    ret = (equity - capital) / capital * 100
    calmar = ret / max_dd if max_dd > 0 else 0
    reasons = {}
    for t in trades:
        reasons[t['reason']] = reasons.get(t['reason'], 0) + 1

    # How often did peak_pnl > exit_pnl (we gave up upside)?
    gave_up = [t['peak_pnl_seen'] - t['pnl_bps'] for t in trades if t['peak_pnl_seen'] > 0]
    avg_gave_up = float(np.mean(gave_up)) if gave_up else 0

    return {
        'name': variant['name'],
        'trades': n_t, 'wins': wins, 'losses': n_t - wins,
        'win_rate': wins / n_t,
        'ret_pct': ret, 'pnl_usd': sum(pnls),
        'max_dd': max_dd, 'calmar': calmar,
        'avg_bps': float(np.mean([t['pnl_bps'] for t in trades])),
        'avg_hold': float(np.mean([t['hold'] for t in trades])),
        'avg_peak_minus_exit': avg_gave_up,
        'reasons': reasons,
    }


def variants():
    base = {
        'sl_bps': 65, 'tp_bps': 200, 'timeout_bars': 240,
    }
    return [
        {'name': 'BASELINE', **base},
        {'name': 'T1_trail_at_TO_50bps',  **base, 'trail_at_timeout': True, 'trail_bps': 50, 'trail_min_pnl': 0},
        {'name': 'T2_trail_at_TO_30bps',  **base, 'trail_at_timeout': True, 'trail_bps': 30, 'trail_min_pnl': 0},
        {'name': 'T3_trail_at_TO_80bps',  **base, 'trail_at_timeout': True, 'trail_bps': 80, 'trail_min_pnl': 0},
        {'name': 'T4_TO_50_if_rising',    **base, 'trail_at_timeout': True, 'trail_bps': 50, 'trail_min_pnl': 0, 'trail_require_rising': True},
        {'name': 'T5_TO_atr_2x',          **base, 'trail_at_timeout': True, 'trail_atr': 2.0, 'trail_min_pnl': 0},
        {'name': 'T5b_TO_atr_3x',         **base, 'trail_at_timeout': True, 'trail_atr': 3.0, 'trail_min_pnl': 0},
        {'name': 'T6_no_TO_trail50',      **base, 'no_timeout': True, 'trail_bps': 50},
        {'name': 'T6b_no_TO_trail30',     **base, 'no_timeout': True, 'trail_bps': 30},
        {'name': 'T8_trail_after_50bps',  **base, 'trail_after_pnl': 50, 'trail_bps': 50},
        {'name': 'T9_trail_after_100bps', **base, 'trail_after_pnl': 100, 'trail_bps': 50},
        {'name': 'T10_trail_after_50_30', **base, 'trail_after_pnl': 50, 'trail_bps': 30},
        {'name': 'T11_trail_after_TPish_150', **base, 'trail_after_pnl': 150, 'trail_bps': 30},
        # Combo with min PnL gate at timeout
        {'name': 'T12_TO_trail50_min30',  **base, 'trail_at_timeout': True, 'trail_bps': 50, 'trail_min_pnl': 30},
        {'name': 'T13_TO_trail50_min50',  **base, 'trail_at_timeout': True, 'trail_bps': 50, 'trail_min_pnl': 50},
    ]


def run(kaggle_csv, pi_data_dir, pi_pair, pair_label):
    print(f"\n{'='*100}\nTRAILING-STOP BACKTEST — {pair_label}\n{'='*100}")
    df_k_raw = load_kaggle_csv(kaggle_csv)
    df_k = parse_kaggle_to_ob_features(df_k_raw)
    df_k = compute_ob_features(df_k)
    df_k = classify_ob_states(df_k, window=60)

    df_pi = load_orderbook_range(pi_data_dir, pi_pair)
    df_pi_1m = resample_pi_to_1min(df_pi)
    df_pi_1m = compute_ob_features(df_pi_1m)
    df_pi_1m = classify_ob_states(df_pi_1m, window=60)

    k_ent = rolling_entropy_ob(df_k['state_ob'].values, NUM_STATES_OB, 30)
    pi_ent = rolling_entropy_ob(df_pi_1m['state_ob'].values, NUM_STATES_OB, 30)
    k_split = int(len(df_k) * 0.7)
    h_thresh = float(np.percentile(k_ent[:k_split][~np.isnan(k_ent[:k_split])], 3))
    print(f"H threshold: {h_thresh:.4f}")
    feats = make_features(df_pi_1m, pi_ent)

    rows = []
    print(f"\n{'Variant':<28} {'Tr':>4} {'WR':>4} {'Ret%':>7} {'DD%':>5} {'Calmar':>7} "
          f"{'AvgBps':>7} {'AvgHold':>7} {'GaveUp':>7} | Reasons")
    print('-'*120)

    for v in variants():
        r = run_variant_trailing(feats, h_thresh, v)
        rows.append(r)
        reasons = ", ".join(f"{k}:{v}" for k, v in r['reasons'].items())
        print(f"{r['name']:<28} {r['trades']:>4} {r['win_rate']*100:>3.0f}% "
              f"{r['ret_pct']:>+6.1f}% {r['max_dd']:>4.1f}% {r['calmar']:>+6.2f} "
              f"{r['avg_bps']:>+6.1f} {r['avg_hold']:>6.0f}m "
              f"{r['avg_peak_minus_exit']:>+6.1f} | {reasons}")

    print(f"\nRANKED BY CALMAR ({pair_label}):")
    valid = sorted([r for r in rows if r['trades'] >= 5], key=lambda x: x['calmar'], reverse=True)
    base_calmar = next((r['calmar'] for r in rows if r['name']=='BASELINE'), 0)
    base_ret = next((r['ret_pct'] for r in rows if r['name']=='BASELINE'), 0)
    for r in valid:
        marker = '+' if r['calmar'] > base_calmar else ' '
        print(f"  {marker} {r['name']:<28} Calmar={r['calmar']:+.2f} (d{r['calmar']-base_calmar:+.2f}) | "
              f"Ret={r['ret_pct']:+.1f}% (d{r['ret_pct']-base_ret:+.1f}) | DD={r['max_dd']:.1f}%")

    out_dir = Path(__file__).parent / 'reports'
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / f'trailing_{pair_label}.json', 'w') as f:
        json.dump(rows, f, indent=2, default=str)
    return rows


if __name__ == '__main__':
    rows_btc = run('data/kaggle/BTC_USDT.csv', 'data/orderbook_pi', 'ob_PF_XBTUSD', 'BTC')
    rows_eth = run('data/kaggle/ETH_USDT.csv', 'data/orderbook_pi', 'ob_PF_ETHUSD', 'ETH')
