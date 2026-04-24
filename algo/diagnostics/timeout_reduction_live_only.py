"""Live-retrospective piece only. Uses Binance 1m OHLC (deep history).
Pi and Kaggle substrates already stored in timeout_reduction_ablation.json.
Merges into the existing file."""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from timeout_reduction_ablation import run_live_ablation, OUT

def main():
    existing = json.loads(OUT.read_text()) if OUT.exists() else {}
    print("Running live 17-trade ablation (Binance 1m OHLC)...")
    live = run_live_ablation()
    existing["live_retrospective"] = live
    OUT.write_text(json.dumps(existing, indent=2, default=str))
    print(f"merged into {OUT}")

    # short summary
    print("\n== LIVE results ==")
    print(f"reconstructed: {live.get('n_reconstructed')}/{live.get('n_pre_sprint_live_trades')}")
    for T in [90, 120, 150, 180, 240]:
        m = live['metrics_by_variant'].get(f'T_{T}', {})
        if not m: continue
        rd = m.get('reason_distribution', {})
        print(f"  T_{T}: n={m['n_trades']} ret={m['compound_ret_pct']:+.2f}% "
              f"DD={m['max_dd_pct']:.2f}% WR={m['win_rate']*100:.1f}% "
              f"knife={m['knife_rate']*100:.1f}% "
              f"avg_w={m['avg_winner_bps']:+.1f} avg_l={m['avg_loser_bps']:+.1f}")
        print(f"    reasons: {rd}")

if __name__ == "__main__":
    main()
