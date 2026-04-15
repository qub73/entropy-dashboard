"""
Upgrade backtest — test multiple variants of the entropy strategy.

Train: Kaggle BTC/ETH (Oct 2023 - Oct 2024)
Test:  Pi Kraken Futures orderbook (Feb 18 - Mar 11 2026)

Each variant inherits from baseline and overrides one or more of:
  - Entry condition (signal generation)
  - Direction rule
  - Stop loss method
  - Position size

Same execution engine for all to keep comparison fair.
"""

import sys, os, json, time
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


# =====================================================================
# Trade engine — same for all variants; only filters/stops change
# =====================================================================

def compute_atr_bps(mids, highs, lows, window=14):
    """Rolling ATR in bps."""
    n = len(mids)
    tr = np.zeros(n)
    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - mids[i-1])
        lc = abs(lows[i] - mids[i-1])
        tr[i] = max(hl, hc, lc)
    atr = pd.Series(tr).rolling(window, min_periods=1).mean().values
    return np.where(mids > 0, atr / mids * 10000, 0)


def compute_vwap(mids, volumes, window):
    """Rolling VWAP. If volumes unavailable, returns simple MA."""
    if volumes is None or np.all(volumes == 0):
        return pd.Series(mids).rolling(window, min_periods=1).mean().values
    vp = pd.Series(mids * volumes).rolling(window, min_periods=1).sum().values
    v = pd.Series(volumes).rolling(window, min_periods=1).sum().values
    return np.where(v > 0, vp / v, mids)


def make_features(df, entropy):
    """Pre-compute every feature any variant might need."""
    n = len(df)
    mids = df['mid'].values
    spreads = df['spread_bps'].values
    imb5 = df['imbalance_5'].values
    imb10 = df['imbalance_10'].values

    # high/low approximation from spread for chandelier ATR
    if 'high' in df.columns and df['high'].max() > 0:
        highs = df['high'].values
        lows = df['low'].values
    else:
        highs = mids * (1 + spreads / 20000)
        lows = mids * (1 - spreads / 20000)

    # Volume proxy from depth (Pi data lacks executed volume)
    bd5 = df['bid_depth_5'].values if 'bid_depth_5' in df.columns else np.ones(n)
    ad5 = df['ask_depth_5'].values if 'ask_depth_5' in df.columns else np.ones(n)
    book_volume = bd5 + ad5

    # Returns at multiple horizons
    ret_1 = np.zeros(n); ret_1[1:] = (mids[1:] / mids[:-1] - 1) * 10000
    ret_5 = np.zeros(n); ret_5[5:] = (mids[5:] / mids[:-5] - 1) * 10000
    ret_15 = np.zeros(n); ret_15[15:] = (mids[15:] / mids[:-15] - 1) * 10000
    ret_30 = np.zeros(n); ret_30[30:] = (mids[30:] / mids[:-30] - 1) * 10000
    ret_60 = np.zeros(n); ret_60[60:] = (mids[60:] / mids[:-60] - 1) * 10000

    atr_bps = compute_atr_bps(mids, highs, lows, window=14)

    vwap_30 = compute_vwap(mids, book_volume, 30)
    vwap_60 = compute_vwap(mids, book_volume, 60)
    vwap_240 = compute_vwap(mids, book_volume, 240)

    # Entropy delta
    H = np.array(entropy, dtype=float)
    dH_5 = np.zeros(n)
    dH_5[5:] = H[5:] - H[:-5]
    dH_15 = np.zeros(n)
    dH_15[15:] = H[15:] - H[:-15]

    # Rolling moving averages (price)
    ma_30 = pd.Series(mids).rolling(30, min_periods=1).mean().values
    ma_60 = pd.Series(mids).rolling(60, min_periods=1).mean().values

    # Rolling vol of returns
    vol_30 = pd.Series(ret_1).rolling(30, min_periods=5).std().fillna(0).values

    return {
        'n': n,
        'mid': mids, 'high': highs, 'low': lows,
        'spread_bps': spreads,
        'imb5': imb5, 'imb10': imb10,
        'H': H, 'dH_5': dH_5, 'dH_15': dH_15,
        'ret_1': ret_1, 'ret_5': ret_5, 'ret_15': ret_15,
        'ret_30': ret_30, 'ret_60': ret_60,
        'atr_bps': atr_bps,
        'vwap_30': vwap_30, 'vwap_60': vwap_60, 'vwap_240': vwap_240,
        'ma_30': ma_30, 'ma_60': ma_60,
        'vol_30': vol_30,
    }


def candidate_signals(feats, h_thresh, params):
    """Default baseline candidate signals: H<thresh, |imb|>0.05, spread<20, |ret_5|∈[20,80]."""
    n = feats['n']
    H = feats['H']
    imb = feats['imb5']
    spread = feats['spread_bps']
    ret_5 = feats['ret_5']

    valid_H = ~np.isnan(H)
    cond = (
        valid_H
        & (H < h_thresh)
        & (np.abs(imb) > params['imb_min'])
        & (spread < params['spread_max'])
        & (np.abs(ret_5) >= params['ret_low'])
        & (np.abs(ret_5) <= params['ret_high'])
    )
    return np.where(cond)[0]


def run_variant(feats, h_thresh, variant, leverage=10.0,
                 capital=10000.0, taker_fee_bps=5.0, equity_frac=0.90):
    """
    Run one variant and return performance metrics.
    Variant is a dict with overrides:
      entry_extra:  callable(feats, idx) -> bool (additional entry filter)
      direction_fn: callable(feats, idx) -> 1 / -1 / 0  (overrides imbalance)
      sl_fn:        callable(feats, idx) -> float (SL in bps; default 65)
      tp_fn:        callable(feats, idx) -> float (TP in bps; default 200)
      timeout_bars: int (default 240)
      size_fn:      callable(equity, n_consecutive_losses) -> float (margin)
      params:       baseline signal params dict
    """
    p = variant.get('params', {
        'imb_min': 0.05, 'spread_max': 20.0, 'ret_low': 20.0, 'ret_high': 80.0,
    })
    candidates = candidate_signals(feats, h_thresh, p)

    direction_fn = variant.get('direction_fn',
                               lambda f, i: 1 if f['imb5'][i] > 0 else -1)
    entry_extra = variant.get('entry_extra', lambda f, i: True)
    sl_fn = variant.get('sl_fn', lambda f, i: 65.0)
    tp_fn = variant.get('tp_fn', lambda f, i: 200.0)
    timeout_bars = variant.get('timeout_bars', 240)
    size_fn = variant.get('size_fn', lambda eq, nl: eq * equity_frac)

    n = feats['n']
    mids = feats['mid']
    highs = feats['high']
    lows = feats['low']

    in_trade = False
    entry_idx = entry_price = direction = sl_bps = tp_bps = 0.0
    notional = 0.0
    equity = capital
    consecutive_losses = 0
    trades = []
    candidates_set = set(candidates)

    # Cooldown-after-loss
    cooldown_until = -1
    cooldown_bars = variant.get('cooldown_after_loss', 0)

    skipped_filter = 0
    skipped_cooldown = 0

    for i in range(n):
        if in_trade:
            curr_pnl = direction * (mids[i] / entry_price - 1) * 10000
            best_pnl = direction * ((highs[i] if direction==1 else -lows[i]) / entry_price - direction) * 10000
            worst_pnl = direction * ((lows[i] if direction==1 else -highs[i]) / entry_price - direction) * 10000
            # use simpler bar logic
            if direction == 1:
                worst_pnl = (lows[i] / entry_price - 1) * 10000
                best_pnl = (highs[i] / entry_price - 1) * 10000
            else:
                worst_pnl = -(highs[i] / entry_price - 1) * 10000
                best_pnl = -(lows[i] / entry_price - 1) * 10000

            exit_reason = None
            exit_pnl = None

            if worst_pnl <= -sl_bps:
                exit_reason = 'sl'
                exit_pnl = -sl_bps
            elif best_pnl >= tp_bps:
                exit_reason = 'tp'
                exit_pnl = tp_bps
            elif (i - entry_idx) >= timeout_bars:
                exit_reason = 'timeout'
                exit_pnl = curr_pnl

            if exit_reason:
                fee = notional * taker_fee_bps / 10000.0
                realized = (exit_pnl / 10000.0) * notional - fee
                equity += realized
                trades.append({
                    'entry_idx': entry_idx, 'exit_idx': i,
                    'direction': direction,
                    'entry_price': entry_price,
                    'pnl_bps': exit_pnl,
                    'pnl_bps_leveraged': exit_pnl * leverage,
                    'pnl_usd': realized,
                    'reason': exit_reason,
                    'hold': i - entry_idx,
                    'sl_used': sl_bps, 'tp_used': tp_bps,
                })
                if realized < 0:
                    consecutive_losses += 1
                    cooldown_until = i + cooldown_bars
                else:
                    consecutive_losses = 0
                in_trade = False
        else:
            if i in candidates_set and equity > 0:
                if i < cooldown_until:
                    skipped_cooldown += 1
                    continue
                if not entry_extra(feats, i):
                    skipped_filter += 1
                    continue
                d = direction_fn(feats, i)
                if d == 0:
                    skipped_filter += 1
                    continue
                margin = size_fn(equity, consecutive_losses)
                notional = margin * leverage
                entry_fee = notional * taker_fee_bps / 10000.0
                equity -= entry_fee
                in_trade = True
                entry_idx = i
                entry_price = mids[i]
                direction = d
                sl_bps = sl_fn(feats, i)
                tp_bps = tp_fn(feats, i)

    # Force close
    if in_trade:
        curr_pnl = direction * (mids[-1] / entry_price - 1) * 10000
        fee = notional * taker_fee_bps / 10000.0
        realized = (curr_pnl / 10000.0) * notional - fee
        equity += realized
        trades.append({
            'entry_idx': entry_idx, 'exit_idx': n-1,
            'direction': direction, 'entry_price': entry_price,
            'pnl_bps': curr_pnl, 'pnl_bps_leveraged': curr_pnl * leverage,
            'pnl_usd': realized, 'reason': 'end', 'hold': n-1-entry_idx,
            'sl_used': sl_bps, 'tp_used': tp_bps,
        })

    # Metrics
    n_t = len(trades)
    wins = sum(1 for t in trades if t['pnl_usd'] > 0)
    if n_t == 0:
        return {'name': variant['name'], 'trades': 0, 'candidates': len(candidates),
                'skipped_filter': skipped_filter, 'skipped_cooldown': skipped_cooldown,
                'win_rate': 0, 'ret_pct': 0, 'pnl_usd': 0,
                'max_dd': 0, 'calmar': 0, 'sharpe': 0, 'reasons': {}}

    pnls = [t['pnl_usd'] for t in trades]
    peak = capital; cum = capital; max_dd = 0
    for p in pnls:
        cum += p
        peak = max(peak, cum)
        max_dd = max(max_dd, (peak - cum) / peak * 100)
    ret = (equity - capital) / capital * 100
    calmar = ret / max_dd if max_dd > 0 else 0
    # Daily Sharpe via per-bar p&l
    daily_pnl = np.array(pnls)
    sharpe = (daily_pnl.mean() / daily_pnl.std() * np.sqrt(len(daily_pnl)) ) if daily_pnl.std() > 0 else 0
    reasons = {}
    for t in trades:
        reasons[t['reason']] = reasons.get(t['reason'], 0) + 1
    longs = [t for t in trades if t['direction'] == 1]
    shorts = [t for t in trades if t['direction'] == -1]
    long_wr = sum(1 for t in longs if t['pnl_usd']>0)/len(longs) if longs else 0
    short_wr = sum(1 for t in shorts if t['pnl_usd']>0)/len(shorts) if shorts else 0

    return {
        'name': variant['name'],
        'trades': n_t,
        'candidates': len(candidates),
        'skipped_filter': skipped_filter,
        'skipped_cooldown': skipped_cooldown,
        'wins': wins, 'losses': n_t - wins,
        'win_rate': wins / n_t,
        'ret_pct': ret, 'pnl_usd': sum(pnls),
        'max_dd': max_dd, 'calmar': calmar,
        'sharpe': float(sharpe),
        'avg_bps': float(np.mean([t['pnl_bps'] for t in trades])),
        'avg_hold': float(np.mean([t['hold'] for t in trades])),
        'reasons': reasons,
        'longs': len(longs), 'long_wr': long_wr,
        'shorts': len(shorts), 'short_wr': short_wr,
    }


# =====================================================================
# Variants
# =====================================================================

def baseline_variant():
    return {'name': 'BASELINE_live'}


def variant_dh_negative():
    """Entry only when entropy is FALLING (dH_5 < 0)."""
    return {
        'name': 'V1_dH<0',
        'entry_extra': lambda f, i: f['dH_5'][i] < 0,
    }


def variant_dh_negative_strong(threshold=-0.02):
    return {
        'name': f'V1b_dH<{threshold}',
        'entry_extra': lambda f, i: f['dH_5'][i] < threshold,
    }


def variant_imb_strong(thresh=0.30):
    """Higher imbalance threshold (stronger directional signal)."""
    return {
        'name': f'V2_imb>{thresh}',
        'params': {'imb_min': thresh, 'spread_max': 20, 'ret_low': 20, 'ret_high': 80},
    }


def variant_atr_sl(mult=2.5):
    """SL scaled by 1-hour ATR, floor=30bps, cap=120bps."""
    def sl_fn(f, i):
        atr = f['atr_bps'][i]
        return float(np.clip(mult * atr, 30, 120))
    return {'name': f'V3_atrSL_{mult}x', 'sl_fn': sl_fn}


def variant_atr_tp_sl(sl_mult=2.0, tp_mult=8.0):
    """Both SL and TP scaled by ATR (4:1 reward/risk)."""
    def sl_fn(f, i):
        return float(np.clip(sl_mult * f['atr_bps'][i], 25, 120))
    def tp_fn(f, i):
        return float(np.clip(tp_mult * f['atr_bps'][i], 100, 400))
    return {'name': f'V3b_atrSL{sl_mult}x_TP{tp_mult}x', 'sl_fn': sl_fn, 'tp_fn': tp_fn}


def variant_vwap_filter(window=240):
    """Block longs below VWAP, block shorts above VWAP."""
    def entry(f, i):
        m = f['mid'][i]; v = f[f'vwap_{window}'][i]
        if v <= 0:
            return True
        # Allow only with-trend entries
        # Don't actually know direction yet here — use imbalance sign as proxy
        imb = f['imb5'][i]
        d = 1 if imb > 0 else -1
        return (d == 1 and m >= v) or (d == -1 and m <= v)
    return {'name': f'V4_vwap{window}', 'entry_extra': entry}


def variant_trend_ma(window=60):
    """Block longs when MA falling, shorts when MA rising. Uses 60-bar MA slope."""
    def entry(f, i):
        if i < window + 5:
            return True
        ma_now = f[f'ma_{window}'][i]
        ma_prev = f[f'ma_{window}'][i - 5]
        slope = ma_now - ma_prev
        d = 1 if f['imb5'][i] > 0 else -1
        return (d == 1 and slope >= 0) or (d == -1 and slope <= 0)
    return {'name': f'V5_trendMA{window}', 'entry_extra': entry}


def variant_cooldown(bars=60):
    """Cooldown N bars after a losing trade."""
    return {'name': f'V6_cooldown{bars}', 'cooldown_after_loss': bars}


def variant_persistence(min_bars=3):
    """Signal must hold across N bars (entropy stays low for `min_bars`)."""
    def entry(f, i):
        if i < min_bars:
            return False
        H = f['H']
        # last `min_bars` all below threshold (use rolling check via imb5 sign too)
        recent = H[i - min_bars + 1:i + 1]
        return np.all(~np.isnan(recent))  # H_thresh check is in candidates already
        # But check imbalance sign hasn't flipped:
    def entry2(f, i):
        if i < min_bars:
            return False
        imb_recent = f['imb5'][i - min_bars + 1:i + 1]
        # all same sign
        return np.all(imb_recent > 0) or np.all(imb_recent < 0)
    return {'name': f'V7_persistence{min_bars}', 'entry_extra': entry2}


def variant_microprice_dir(min_dev_bps=0.5):
    """Direction from microprice deviation, not raw imbalance."""
    def dir_fn(f, i):
        # microprice already encoded in imbalance to some extent
        # use imbalance10 and imbalance5 agreement as proxy
        i5 = f['imb5'][i]; i10 = f['imb10'][i]
        # require both same sign
        if i5 > 0 and i10 > 0:
            return 1
        if i5 < 0 and i10 < 0:
            return -1
        return 0
    return {'name': 'V8_imb5+imb10_agree', 'direction_fn': dir_fn}


def variant_combo_dh_vwap_atr():
    """Combination: ΔH<0 entry + VWAP filter + ATR-scaled SL."""
    def entry(f, i):
        if f['dH_5'][i] >= 0:
            return False
        m = f['mid'][i]; v = f['vwap_240'][i]
        if v <= 0: return True
        d = 1 if f['imb5'][i] > 0 else -1
        return (d == 1 and m >= v) or (d == -1 and m <= v)
    def sl_fn(f, i):
        return float(np.clip(2.5 * f['atr_bps'][i], 30, 120))
    return {'name': 'COMBO_dH+vwap+atrSL', 'entry_extra': entry, 'sl_fn': sl_fn}


def variant_combo_strong():
    """All filters: ΔH<-0.005 + imb>0.15 + VWAP + trend60 + ATR SL."""
    def entry(f, i):
        if f['dH_5'][i] >= -0.005:
            return False
        m = f['mid'][i]; v = f['vwap_240'][i]
        d = 1 if f['imb5'][i] > 0 else -1
        if v > 0:
            if d == 1 and m < v: return False
            if d == -1 and m > v: return False
        if i >= 65:
            ma_now = f['ma_60'][i]; ma_prev = f['ma_60'][i-5]
            slope = ma_now - ma_prev
            if d == 1 and slope < 0: return False
            if d == -1 and slope > 0: return False
        return True
    def sl_fn(f, i):
        return float(np.clip(2.5 * f['atr_bps'][i], 30, 120))
    return {
        'name': 'COMBO_all_filters',
        'entry_extra': entry,
        'sl_fn': sl_fn,
        'params': {'imb_min': 0.15, 'spread_max': 20, 'ret_low': 20, 'ret_high': 80},
    }


def variant_anti_martingale():
    """Halve size after each consecutive loss (max 4 reductions)."""
    def size_fn(eq, nl):
        return eq * 0.90 * (0.5 ** min(nl, 3))
    return {'name': 'V9_antiMartingale', 'size_fn': size_fn}


def variant_long_only_above_vwap():
    """Block longs unless price > VWAP_240. Allow shorts anywhere."""
    def entry(f, i):
        d = 1 if f['imb5'][i] > 0 else -1
        if d != 1:
            return True
        m = f['mid'][i]; v = f['vwap_240'][i]
        return v > 0 and m >= v
    return {'name': 'V4b_long_above_vwap', 'entry_extra': entry}


def variant_no_recent_loss_price(width_bps=30):
    """Don't enter near the price level of a recent SL."""
    # This needs trade history awareness — implemented inline in run_variant_with_state
    return {'name': f'V10_no_replay_{width_bps}bps', '__needs_state__': True,
            'replay_width_bps': width_bps}


# =====================================================================
# Main
# =====================================================================

def run(kaggle_csv='data/kaggle/BTC_USDT.csv',
        pi_data_dir='data/orderbook_pi',
        pi_pair='ob_PF_XBTUSD',
        pair_label='BTC'):

    print(f"\n{'='*100}")
    print(f"UPGRADE BACKTEST — {pair_label}")
    print(f"Train: Kaggle | Test: Pi Kraken Futures")
    print(f"{'='*100}")

    print("Loading Kaggle...")
    df_k_raw = load_kaggle_csv(kaggle_csv)
    df_k = parse_kaggle_to_ob_features(df_k_raw)
    df_k = compute_ob_features(df_k)
    df_k = classify_ob_states(df_k, window=60)

    print("Loading Pi...")
    df_pi = load_orderbook_range(pi_data_dir, pi_pair)
    df_pi_1m = resample_pi_to_1min(df_pi)
    df_pi_1m = compute_ob_features(df_pi_1m)
    df_pi_1m = classify_ob_states(df_pi_1m, window=60)

    print("Computing entropy...")
    k_ent = rolling_entropy_ob(df_k['state_ob'].values, NUM_STATES_OB, 30)
    pi_ent = rolling_entropy_ob(df_pi_1m['state_ob'].values, NUM_STATES_OB, 30)

    k_split = int(len(df_k) * 0.7)
    h_thresh = float(np.percentile(k_ent[:k_split][~np.isnan(k_ent[:k_split])], 3))
    print(f"H threshold (Kaggle 3rd pctile): {h_thresh:.4f}")

    print("Building features...")
    feats = make_features(df_pi_1m, pi_ent)
    print(f"Pi bars: {feats['n']}, candidate signals (baseline filter): "
          f"{len(candidate_signals(feats, h_thresh, {'imb_min':0.05,'spread_max':20,'ret_low':20,'ret_high':80}))}")

    # Define all variants
    variants = [
        baseline_variant(),
        variant_dh_negative(),
        variant_dh_negative_strong(-0.005),
        variant_dh_negative_strong(-0.02),
        variant_imb_strong(0.15),
        variant_imb_strong(0.30),
        variant_atr_sl(2.0),
        variant_atr_sl(2.5),
        variant_atr_sl(3.0),
        variant_atr_tp_sl(2.0, 8.0),
        variant_atr_tp_sl(2.5, 6.0),
        variant_vwap_filter(60),
        variant_vwap_filter(240),
        variant_trend_ma(60),
        variant_trend_ma(30),
        variant_cooldown(30),
        variant_cooldown(60),
        variant_cooldown(120),
        variant_persistence(2),
        variant_persistence(3),
        variant_microprice_dir(),
        variant_anti_martingale(),
        variant_long_only_above_vwap(),
        variant_combo_dh_vwap_atr(),
        variant_combo_strong(),
    ]

    rows = []
    print(f"\n{'Variant':<32} {'Tr':>4} {'Cand':>5} {'WR':>4} {'Ret%':>7} "
          f"{'PnL$':>9} {'MaxDD':>6} {'Calmar':>6} {'Sharpe':>6} {'AvgBps':>6} "
          f"{'L/S':>6} {'Lwr/Swr':>9}")
    print('-' * 120)

    for v in variants:
        try:
            r = run_variant(feats, h_thresh, v)
        except Exception as e:
            print(f"  {v['name']}: ERROR {e}")
            continue
        rows.append(r)
        ls_str = f"{r.get('longs',0)}/{r.get('shorts',0)}"
        wr_str = f"{r.get('long_wr',0)*100:.0f}/{r.get('short_wr',0)*100:.0f}"
        print(f"{r['name']:<32} {r['trades']:>4} {r.get('candidates',0):>5} "
              f"{r['win_rate']*100:>3.0f}% {r['ret_pct']:>+6.1f}% "
              f"{r['pnl_usd']:>+9.0f} {r['max_dd']:>5.1f}% {r['calmar']:>+5.2f} "
              f"{r['sharpe']:>+5.2f} {r['avg_bps']:>+5.0f} "
              f"{ls_str:>6} {wr_str:>9}")

    # Save and rank
    out_dir = Path(__file__).parent / 'reports'
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / f'upgrades_{pair_label}.json', 'w') as f:
        json.dump(rows, f, indent=2, default=str)

    print(f"\n{'='*100}")
    print(f"RANKED BY CALMAR — {pair_label}")
    print(f"{'='*100}")
    sorted_rows = sorted([r for r in rows if r['trades'] >= 5],
                         key=lambda x: x['calmar'], reverse=True)
    for i, r in enumerate(sorted_rows[:15]):
        print(f"  {i+1:>2}. {r['name']:<32} Ret={r['ret_pct']:>+6.1f}% "
              f"DD={r['max_dd']:.1f}% Calmar={r['calmar']:.2f} "
              f"WR={r['win_rate']*100:.0f}% Tr={r['trades']}")

    return rows


if __name__ == '__main__':
    rows_btc = run(kaggle_csv='data/kaggle/BTC_USDT.csv',
                   pi_data_dir='data/orderbook_pi',
                   pi_pair='ob_PF_XBTUSD',
                   pair_label='BTC')
    rows_eth = run(kaggle_csv='data/kaggle/ETH_USDT.csv',
                   pi_data_dir='data/orderbook_pi',
                   pi_pair='ob_PF_ETHUSD',
                   pair_label='ETH')
