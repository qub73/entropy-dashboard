"""One-off bootstrap helper for 6b go-live brief. Read-only. Not committed."""
import sys
import numpy as np

sys.path.insert(0, 'algo')
sys.path.insert(0, 'algo/diagnostics')

from status_2026_04_21_replay import run_cell_nof3c
from phase4_sizing_sim import (
    load_pi, load_kaggle, CORE_PARAMS,
    PST_WIDTH_BPS, PST_HARD_FLOOR, PST_MAX_WAIT, H_THRESH,
)
from upgrade_backtest import candidate_signals


def replay_with_trades(feats, extra, sl, tp, knife_bps, ext_cap_bps, leverage, e3_overlay):
    n = feats['n']
    mids = feats['mid']; highs = feats['high']; lows = feats['low']
    atr_bps_arr = feats['atr_bps']; imb = feats['imb5']
    dH_5 = feats['dH_5']; ret_60 = feats['ret_60']
    ret_150 = np.zeros(n); ret_150[150:] = (mids[150:] / mids[:-150] - 1) * 10000
    cands = set(candidate_signals(feats, H_THRESH, CORE_PARAMS))
    trail_after, trail_bps = 150, 50
    equity = 10000.0; in_trade=False
    entry_idx=entry_price=direction=0; notional=peak_pnl=0.0; initial_notional=0.0
    tp_trailing=trailing_active=pst_active=False
    pst_peak_bps=0.0; pst_entry_bar=0; sl_current=-sl; e3_tightened=False
    trades=[]
    for i in range(n):
        if in_trade:
            d=direction
            if d==1:
                worst=(lows[i]/entry_price-1)*10000; best=(highs[i]/entry_price-1)*10000
            else:
                worst=-(highs[i]/entry_price-1)*10000; best=-(lows[i]/entry_price-1)*10000
            curr=d*(mids[i]/entry_price-1)*10000
            peak_pnl=max(peak_pnl,best); exit_reason=exit_pnl=None
            if worst<=sl_current: exit_reason='sl'; exit_pnl=sl_current
            if exit_reason is None and not tp_trailing and peak_pnl>=trail_after: tp_trailing=True
            if exit_reason is None and tp_trailing:
                floor=max(peak_pnl-trail_bps,sl_current)
                if worst<=floor: exit_reason='tp_trail'; exit_pnl=max(floor,curr)
            if exit_reason is None and trailing_active:
                tw=2.0*atr_bps_arr[i] if atr_bps_arr[i]>0 else 50
                floor=max(peak_pnl-tw,sl_current)
                if worst<=floor: exit_reason='trail_stop'; exit_pnl=max(floor,curr)
                elif best>=tp: exit_reason='tp'; exit_pnl=tp
            if exit_reason is None and pst_active:
                b=i-pst_entry_bar; pst_peak_bps=max(pst_peak_bps,best)
                floor=max(pst_peak_bps-PST_WIDTH_BPS,sl_current)
                if worst<=floor: exit_reason='pst_trail'; exit_pnl=max(floor,curr)
                elif worst<=PST_HARD_FLOOR: exit_reason='pst_floored'; exit_pnl=PST_HARD_FLOOR
                elif b>=PST_MAX_WAIT: exit_reason='pst_timeout'; exit_pnl=curr
            if exit_reason is None and not tp_trailing and not pst_active and not trailing_active and best>=tp:
                exit_reason='tp'; exit_pnl=tp
            if exit_reason is None and e3_overlay and not e3_tightened and (i-entry_idx)>=60 and peak_pnl<50:
                sl_current=-25; e3_tightened=True
            if exit_reason is None and (i-entry_idx)>=240 and not trailing_active and not pst_active:
                if curr>0: trailing_active=True
                else: pst_active=True; pst_entry_bar=i; pst_peak_bps=best
            if exit_reason:
                fee=notional*5.0/10000; realized=(exit_pnl/10000)*notional-fee; equity+=realized
                pos_pnl_bps=realized/initial_notional*10000 if initial_notional>0 else exit_pnl
                trades.append({'pnl_bps': pos_pnl_bps, 'direction': direction, 'peak_bps': peak_pnl})
                in_trade=False; tp_trailing=False; trailing_active=False; pst_active=False; e3_tightened=False
            continue
        if i not in cands or equity<=0: continue
        if not np.isnan(dH_5[i]) and dH_5[i]>=0: continue
        d=1 if imb[i]>0 else -1
        if knife_bps and d==1 and ret_60[i]<-knife_bps: continue
        if knife_bps and d==-1 and ret_60[i]>knife_bps: continue
        if ext_cap_bps is not None:
            r150=ret_150[i]
            if d==1 and r150>ext_cap_bps: continue
            if d==-1 and r150<-ext_cap_bps: continue
        margin=equity*0.90; notional=margin*leverage; initial_notional=notional
        equity-=notional*5.0/10000
        in_trade=True; entry_idx=i; entry_price=mids[i]; direction=d; peak_pnl=0
        tp_trailing=False; trailing_active=False; pst_active=False; e3_tightened=False; sl_current=-sl
    return trades


np.random.seed(42)
print("Loading Pi + Kaggle...")
pi_feats, pi_extra = load_pi()
kg_feats, kg_extra = load_kaggle(60)
pi_trades = replay_with_trades(pi_feats, pi_extra, 50, 200, 50, 100, 5, True)
kg_trades = replay_with_trades(kg_feats, kg_extra, 50, 150, 100, None, 5, True)
pi_pnls = np.array([t['pnl_bps'] for t in pi_trades])
kg_pnls = np.array([t['pnl_bps'] for t in kg_trades])
print(f"Pi n={len(pi_pnls)} mean={pi_pnls.mean():+.1f} std={pi_pnls.std():.1f}")
print(f"Kg n={len(kg_pnls)} mean={kg_pnls.mean():+.1f} std={kg_pnls.std():.1f}")

n_pi_30d = int(round(44/21.3*30))
n_kg_30d = int(round(131/60*30))
print(f"Expected 30d trades: Pi={n_pi_30d}, Kaggle={n_kg_30d}")

N=10000
# pos_pnl_bps is on initial_notional (= margin * leverage = 0.9 * leverage * equity).
# Equity multiplier per trade = 1 + (0.9 * leverage / 10000) * (pos_pnl_bps - 5)
# where the -5 accounts for the asymmetric fee (entry fee charged separately).
# For 5x leverage + 90% margin: multiplier = 1 + 4.5e-4 * (pos_pnl_bps - 5)
EQ_SCALE_5X = 0.9 * 5 / 10000  # = 4.5e-4
FEE_OFFSET = 5.0

pi_r=[]; kg_r=[]; pi_d=[]; kg_d=[]
for _ in range(N):
    s=np.random.choice(pi_pnls,n_pi_30d,replace=True)
    eq=10000; el=[eq]
    for p in s:
        eq *= (1 + EQ_SCALE_5X * (p - FEE_OFFSET))
        el.append(max(eq, 0))  # floor at 0 to avoid negative equity
    a=np.array(el); pk=np.maximum.accumulate(a)
    pi_r.append((eq-10000)/100); pi_d.append(((pk-a)/pk*100).max())
    s=np.random.choice(kg_pnls,n_kg_30d,replace=True)
    eq=10000; el=[eq]
    for p in s:
        eq *= (1 + EQ_SCALE_5X * (p - FEE_OFFSET))
        el.append(max(eq, 0))
    a=np.array(el); pk=np.maximum.accumulate(a)
    kg_r.append((eq-10000)/100); kg_d.append(((pk-a)/pk*100).max())

def pct(a, p): return np.percentile(a, p)
print("\n30d Pi @5x:")
print(f"  ret p10/p25/p50/p75/p90: {pct(pi_r,10):+.1f}% / {pct(pi_r,25):+.1f}% / {pct(pi_r,50):+.1f}% / {pct(pi_r,75):+.1f}% / {pct(pi_r,90):+.1f}%")
print(f"  DD  p10/p25/p50/p75/p90: {pct(pi_d,10):.1f}% / {pct(pi_d,25):.1f}% / {pct(pi_d,50):.1f}% / {pct(pi_d,75):.1f}% / {pct(pi_d,90):.1f}%")
print(f"  P(ret<0): {np.mean(np.array(pi_r)<0)*100:.1f}%  P(DD>20%): {np.mean(np.array(pi_d)>20)*100:.1f}%")

print("\n30d Kaggle @5x:")
print(f"  ret p10/p25/p50/p75/p90: {pct(kg_r,10):+.1f}% / {pct(kg_r,25):+.1f}% / {pct(kg_r,50):+.1f}% / {pct(kg_r,75):+.1f}% / {pct(kg_r,90):+.1f}%")
print(f"  DD  p10/p25/p50/p75/p90: {pct(kg_d,10):.1f}% / {pct(kg_d,25):.1f}% / {pct(kg_d,50):.1f}% / {pct(kg_d,75):.1f}% / {pct(kg_d,90):.1f}%")
print(f"  P(ret<0): {np.mean(np.array(kg_r)<0)*100:.1f}%  P(DD>20%): {np.mean(np.array(kg_d)>20)*100:.1f}%")

w_r=0.7*np.array(pi_r)+0.3*np.array(kg_r)
w_d=0.7*np.array(pi_d)+0.3*np.array(kg_d)
print("\n70/30 Pi/Kaggle (weighted point estimate):")
print(f"  ret p10/p50/p90: {pct(w_r,10):+.1f}% / {pct(w_r,50):+.1f}% / {pct(w_r,90):+.1f}%")
print(f"  DD  p10/p50/p90: {pct(w_d,10):.1f}% / {pct(w_d,50):.1f}% / {pct(w_d,90):.1f}%")
