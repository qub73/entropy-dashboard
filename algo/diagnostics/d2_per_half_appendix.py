"""
D2 per-half appendix.

Split the Feb 18 - Apr 7 Pi window in half (by bar count). For each
half, compute per-direction: n trades, win rate, profit factor,
compound contribution.

Output: reports/d2_per_half_appendix.json
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
# reuse the baseline runner
from d2_baseline_lshort_split import run_baseline, side_stats, compound_contribution

OUT = ALGO / "reports" / "d2_per_half_appendix.json"


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
    trades, _ = run_baseline(feats, 0.4352, params,
                             sl=50, tp=200, trail_after=150, trail_bps=50,
                             cooldown=0, knife_bps=50, extended_cap_bps=100)
    mid = n // 2
    first = [t for t in trades if t["entry_bar"] < mid]
    second = [t for t in trades if t["entry_bar"] >= mid]

    mids = feats['mid']
    out = {
        "split_bar": mid,
        "split_time_note": "mid of bar index; approximate halfway through window",
        "full_window_price_bps": float((mids[-1]/mids[0]-1)*10000),
        "first_half_price_bps":  float((mids[mid]/mids[0]-1)*10000),
        "second_half_price_bps": float((mids[-1]/mids[mid]-1)*10000),
    }
    for label, half in (("first_half", first), ("second_half", second)):
        out[label] = {
            "n_trades": len(half),
            "long": side_stats(half, 1),
            "short": side_stats(half, -1),
            "compound": compound_contribution(half),
        }
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nFull window:     net {out['full_window_price_bps']:+.0f} bps "
          f"({len(trades)} trades)")
    print(f"First half:      net {out['first_half_price_bps']:+.0f} bps "
          f"({len(first)} trades)")
    print(f"  LONG: n={out['first_half']['long']['n']}, "
          f"WR={out['first_half']['long'].get('win_rate', 0)*100:.0f}%, "
          f"PF={out['first_half']['long'].get('pf')}, "
          f"USD={out['first_half']['compound']['long_usd_contrib']:+.0f}")
    print(f"  SHORT: n={out['first_half']['short']['n']}, "
          f"WR={out['first_half']['short'].get('win_rate', 0)*100:.0f}%, "
          f"PF={out['first_half']['short'].get('pf')}, "
          f"USD={out['first_half']['compound']['short_usd_contrib']:+.0f}")
    print(f"Second half:     net {out['second_half_price_bps']:+.0f} bps "
          f"({len(second)} trades)")
    print(f"  LONG: n={out['second_half']['long']['n']}, "
          f"WR={out['second_half']['long'].get('win_rate', 0)*100:.0f}%, "
          f"PF={out['second_half']['long'].get('pf')}, "
          f"USD={out['second_half']['compound']['long_usd_contrib']:+.0f}")
    print(f"  SHORT: n={out['second_half']['short']['n']}, "
          f"WR={out['second_half']['short'].get('win_rate', 0)*100:.0f}%, "
          f"PF={out['second_half']['short'].get('pf')}, "
          f"USD={out['second_half']['compound']['short_usd_contrib']:+.0f}")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
