"""
Phase 0 (adapted) -- engine sanity check.

We cannot reproduce the 17 live ETH trades (Apr 13-19) because the
orderbook feed for 13 of those dates is not in the repo. Instead, we:

 1) Confirm the backtest engine is deterministic and reproduces an
    existing stored result (param_sweep_validation.json, ETH SL50/TP150
    with ext-filter OFF) within rounding tolerance.
 2) Run the engine with the CURRENT LIVE CONFIG (SL50, TP200, trail at
    150/50, ext-move cap=100, knife=50, dH<0, per-pair cooldown) over
    the Feb 18 - Mar 11 Pi window (24 days, ~35k bars).
 3) Write a report comparing (trades, WR, ret%, DD, Calmar, PF) to the
    stored reference and the already-seen 'ext_filter prod' baseline.

Output: reports/phase0_engine_validation.json
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

OUT = ALGO / "reports" / "phase0_engine_validation.json"
REF = ALGO / "reports" / "param_sweep_validation.json"


def run_engine(feats, h_thresh, params, sl, tp, trail_after, trail_bps,
               cooldown=0, knife_bps=None, extended_cap_bps=None):
    """Same engine as production/backtest. Returns summary dict."""
    n = feats['n']
    mids = feats['mid']; highs = feats['high']; lows = feats['low']
    atr = feats['atr_bps']; imb = feats['imb5']
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
                worst = (lows[i]/entry_price - 1) * 10000
                best  = (highs[i]/entry_price - 1) * 10000
            else:
                worst = -(highs[i]/entry_price - 1) * 10000
                best  = -(lows[i]/entry_price - 1) * 10000
            curr = d * (mids[i]/entry_price - 1) * 10000
            peak_pnl = max(peak_pnl, best)
            exit_reason = exit_pnl = None
            # race-fix: activate trail before SL check
            if not tp_trailing and peak_pnl >= trail_after:
                tp_trailing = True
            if tp_trailing:
                floor = max(peak_pnl - trail_bps, -sl)
                if worst <= floor:
                    exit_reason = 'tp_trail'; exit_pnl = max(floor, curr)
            elif trailing_active:
                tw = 2.0 * atr[i] if atr[i] > 0 else 50
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
                trades.append({'pnl_bps': exit_pnl, 'pnl_usd': realized,
                               'reason': exit_reason})
                if exit_pnl < 0:
                    last_loss_bar = i
                in_trade = False
                tp_trailing = False; trailing_active = False
        else:
            if i not in cands or equity <= 0:
                continue
            if cooldown > 0 and (i - last_loss_bar) < cooldown:
                continue
            if not np.isnan(dH_5[i]) and dH_5[i] >= 0:
                continue
            d = 1 if imb[i] > 0 else -1
            if knife_bps and d == 1 and ret_60[i] < -knife_bps:
                continue
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
        trades.append({'pnl_bps': curr, 'pnl_usd': realized, 'reason': 'end'})

    nt = len(trades)
    if nt == 0:
        return {'trades': 0}
    pnls = [t['pnl_usd'] for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    pk = 10000; cum = 10000; mdd = 0
    for p in pnls:
        cum += p; pk = max(pk, cum)
        mdd = max(mdd, (pk-cum)/pk*100)
    ret = (equity - 10000) / 10000 * 100
    bps_list = [t['pnl_bps'] for t in trades]
    win_bps = sum(b for b in bps_list if b > 0)
    loss_bps = sum(b for b in bps_list if b <= 0)
    pf = abs(win_bps / loss_bps) if loss_bps != 0 else float('inf')
    reasons = {}
    longs = sum(1 for t in trades if t.get('pnl_bps', 0) is not None)  # placeholder
    for t in trades:
        reasons[t['reason']] = reasons.get(t['reason'], 0) + 1
    return {
        'trades': nt, 'wins': wins,
        'win_rate': wins/nt,
        'ret_pct': ret, 'max_dd': mdd,
        'calmar': ret/mdd if mdd > 0 else 0,
        'pf': pf,
        'reasons': reasons,
    }


def main():
    params = {'imb_min': 0.05, 'spread_max': 20, 'ret_low': 20, 'ret_high': 80}
    print("Loading ETH Pi orderbook (Feb 18 - Apr 7)...", flush=True)
    df = load_orderbook_range("data/orderbook_pi", "ob_PF_ETHUSD")
    df_1m = resample_pi_to_1min(df)
    df_1m = compute_ob_features(df_1m)
    df_1m = classify_ob_states(df_1m, window=60)
    ent = rolling_entropy_ob(df_1m['state_ob'].values, NUM_STATES_OB, 30)
    feats = make_features(df_1m, ent)
    n_days = feats['n'] / 1440
    print(f"  {feats['n']} bars, {n_days:.1f} days\n", flush=True)

    # (A) Reproduce the stored reference row: ETH SL50/TP150 (no ext, no knife, no cooldown, no dH... wait, the baseline uses cooldown+knife)
    # Looking at param_sweep_validation.py: it ran with cooldown=0, knife=50 for ETH
    print("A) Reproducing stored reference (ETH SL50/TP150/trail@150/50, knife=50, ext=none)")
    reproduced = run_engine(feats, 0.4352, params,
                            sl=50, tp=200, trail_after=150, trail_bps=50,
                            cooldown=0, knife_bps=50, extended_cap_bps=None)
    print(f"  -> {reproduced}\n", flush=True)

    # Stored reference (from reports/param_sweep_validation.json, ETH sl50/tp150):
    # Wait -- the reference was tp=150, but current live is tp=200. Let's re-run with tp=150 too
    print("A') Same but TP=150 (to match stored reference)")
    reproduced_tp150 = run_engine(feats, 0.4352, params,
                                   sl=50, tp=150, trail_after=150, trail_bps=50,
                                   cooldown=0, knife_bps=50, extended_cap_bps=None)
    print(f"  -> {reproduced_tp150}\n", flush=True)

    # Load stored reference
    ref_rows = json.load(open(REF))
    ref = next((r for r in ref_rows
                if r.get('pair') == 'ETH' and r.get('sl') == 50
                and r.get('tp') == 150 and r.get('timeout') == 240), None)

    # (B) Current live config: SL=50, TP=200, trail@150/50, knife=50, ext_cap=100
    print("B) Current production config (SL50/TP200/trail@150/50, knife=50, ext=100)")
    live_cfg = run_engine(feats, 0.4352, params,
                          sl=50, tp=200, trail_after=150, trail_bps=50,
                          cooldown=0, knife_bps=50, extended_cap_bps=100)
    print(f"  -> {live_cfg}\n", flush=True)

    out = {
        "n_bars": int(feats['n']),
        "n_days": round(n_days, 2),
        "reference_from_param_sweep": {
            "config": {"sl": 50, "tp": 150, "trail_after": 150,
                       "trail_bps": 50, "knife_bps": 50, "ext_cap": None},
            "stored": {"trades": ref['trades'], "wins": ref['wins'],
                       "win_rate": ref['win_rate'], "ret_pct": ref['ret_pct'],
                       "max_dd": ref['max_dd'], "calmar": ref['calmar'],
                       "pf": ref['pf']} if ref else None,
            "re_run": reproduced_tp150,
            "reproduces_within_tolerance": (ref is not None and
                reproduced_tp150.get('trades') == ref['trades'] and
                abs(reproduced_tp150.get('ret_pct', 0) - ref['ret_pct']) < 1.0),
        },
        "current_live_config": {
            "config": {"sl": 50, "tp": 200, "trail_after": 150,
                       "trail_bps": 50, "knife_bps": 50, "ext_cap": 100},
            "result": live_cfg,
        },
        "tp200_no_ext": {
            "config": {"sl": 50, "tp": 200, "trail_after": 150,
                       "trail_bps": 50, "knife_bps": 50, "ext_cap": None},
            "result": reproduced,
        },
    }
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {OUT}")
    print(f"\n=== SUMMARY ===")
    print(f"Reference row (stored):  trades={ref['trades']}, ret={ref['ret_pct']:.1f}%, "
          f"Cal={ref['calmar']:.2f}")
    print(f"Reference row (re-run):  trades={reproduced_tp150['trades']}, "
          f"ret={reproduced_tp150['ret_pct']:.1f}%, Cal={reproduced_tp150['calmar']:.2f}")
    print(f"Current live config:     trades={live_cfg['trades']}, "
          f"ret={live_cfg['ret_pct']:.1f}%, Cal={live_cfg['calmar']:.2f}")


if __name__ == "__main__":
    main()
