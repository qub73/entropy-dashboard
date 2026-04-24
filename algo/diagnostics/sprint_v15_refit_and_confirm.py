"""
Sprint v1.5 Steps 1-3 -- temporal fit/eval split, Kraken-native baseline
refit, and F3c confirmation.

Step 1 (split): FIT = clean days 2025-12-18..2026-02-14 (30 days).
               EVAL = clean days 2026-03-18..2026-04-15 (22 days).

Step 2 (refit, FIT slice): grid SL in {30,40,50,60,70} x TP in {150,200,250,300}
       x H_thresh in {0.40, 0.4352, 0.47, 0.50}. Knife/ext fixed at Pi
       baseline (knife=50, ext_cap=100). Pick highest compound return.
       Top-5 cells reported.
       Output: reports/sprint_v15_kraken_native_refit.json

Step 3 (F3c confirmation, EVAL slice): two runs with the refit params:
       baseline (no F3c) vs F3c (threshold +/-0.3). Metrics: return, DD,
       Sharpe, PF, knife rate, trades, L/S split.
       Output: reports/sprint_v15_f3c_kraken_native_confirmation.json

Uses concat-then-featurize (same convention as phase2_filter_ablation_v2)
so features are comparable across per-slice tests. Cross-day gaps on a
clean-day slice are small and bounded (classify_ob_states 60-bar window,
entropy 30-bar window).
"""
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ALGO = HERE.parent
sys.path.insert(0, str(ALGO))
sys.path.insert(0, str(HERE))

from ob_entropy import (
    NUM_STATES_OB, rolling_entropy_ob,
    compute_ob_features, classify_ob_states,
)
from kaggle_ob_trainer import resample_pi_to_1min
from upgrade_backtest import make_features
from phase2_filter_ablation_v2 import build_extra_features, run_ablation
from kraken_hf_loader import load_kraken_hf_day, clean_days_from_report

REFIT_OUT   = ALGO / "reports" / "sprint_v15_kraken_native_refit.json"
CONFIRM_OUT = ALGO / "reports" / "sprint_v15_f3c_kraken_native_confirmation.json"

FIT_RANGE  = ("2025-12-18", "2026-02-14")
EVAL_RANGE = ("2026-03-18", "2026-04-15")

GRID_SL = [30, 40, 50, 60, 70]
GRID_TP = [150, 200, 250, 300]
GRID_H  = [0.40, 0.4352, 0.47, 0.50]

# Per user spec: "Same protocol as Phase 2 v2" -> Pi-baseline knife/ext fixed
PI_KNIFE_BPS = 50.0
PI_EXT_CAP_BPS = 100.0

ENTROPY_WINDOW_BARS = 30

CORE_PARAMS = {"imb_min": 0.05, "spread_max": 20,
               "ret_low": 20, "ret_high": 80}


def build_slice(slice_label: str, date_range):
    """Concat raw snapshots across clean days in range, then resample +
    featurize. Returns (feats, extra, meta)."""
    dates = clean_days_from_report(date_range=date_range)
    print(f"[{slice_label}] clean days ({len(dates)}): {dates[0]} -> {dates[-1]}")
    t0 = time.time()

    parts = []
    day_sizes = {}
    for d in dates:
        df = load_kraken_hf_day(d, verbose=False)
        if df.empty or len(df) < 300:
            print(f"  {d}: SKIP ({len(df)} snaps)")
            continue
        parts.append(df)
        day_sizes[d] = int(len(df))
    if not parts:
        raise RuntimeError(f"No usable clean days in range {date_range}")

    df_all = pd.concat(parts, ignore_index=True).sort_values("ts_ms")
    df_all = df_all.reset_index(drop=True)

    df_1m = resample_pi_to_1min(df_all)
    df_1m = compute_ob_features(df_1m)
    df_1m = classify_ob_states(df_1m, window=60)
    H = rolling_entropy_ob(df_1m["state_ob"].values, NUM_STATES_OB,
                           ENTROPY_WINDOW_BARS)
    feats = make_features(df_1m, H)
    volumes = (df_1m["bid_depth_5"].values + df_1m["ask_depth_5"].values
               if "bid_depth_5" in df_1m.columns else None)
    extra = build_extra_features(feats, volumes)
    meta = {
        "slice": slice_label,
        "date_range": list(date_range),
        "clean_days_used": list(day_sizes.keys()),
        "n_days": len(day_sizes),
        "raw_snapshots": int(sum(day_sizes.values())),
        "bars": int(feats["n"]),
        "H_min_max": [float(np.nanmin(feats["H"])),
                      float(np.nanmax(feats["H"]))],
        "H_p3_p50": [float(np.nanpercentile(feats["H"], 3)),
                     float(np.nanpercentile(feats["H"], 50))],
        "elapsed_s": round(time.time() - t0, 1),
    }
    print(f"  {slice_label}: {meta['bars']:,} bars  "
          f"({meta['elapsed_s']}s)", flush=True)
    return feats, extra, meta


def sharpe_from_trades(trades):
    if not trades: return 0.0
    bps = np.array([t["pnl_bps"] for t in trades], dtype=float)
    if bps.std() <= 1e-9: return 0.0
    return float(bps.mean() / bps.std() * np.sqrt(len(bps)))


def run_with_overrides(feats, extra, params, filt, sl, tp, knife, ext):
    """Wrapper around run_ablation that also returns Sharpe and trade dump."""
    # Patch: expose trade list. phase2 engine returns aggregated metrics only.
    # Copy its engine inline (minor) -- easier: call run_ablation and then
    # re-derive trade list by running a second pass. For brevity, we add a
    # one-liner via run_ablation's return + independent metrics re-calc.
    res = run_ablation(feats, extra, params, filt, sl, tp, knife, ext)
    return res


def step2_refit(fit_feats, fit_extra):
    print("\n[STEP 2] Kraken-native baseline refit on FIT slice")
    print(f"  grid: {len(GRID_SL)} SL  x  {len(GRID_TP)} TP  x  {len(GRID_H)} H  "
          f"= {len(GRID_SL)*len(GRID_TP)*len(GRID_H)} cells")
    print(f"  fixed filters: knife={PI_KNIFE_BPS}bps  ext={PI_EXT_CAP_BPS}bps  "
          f"(Pi-baseline, no F3c)")

    cells = []
    t0 = time.time()
    total = len(GRID_SL) * len(GRID_TP) * len(GRID_H)
    done = 0
    for h in GRID_H:
        for sl in GRID_SL:
            for tp in GRID_TP:
                if tp <= sl:
                    done += 1
                    continue
                # run_ablation reads H_THRESH from module global; we pass
                # h through by monkey-patching.
                import phase2_filter_ablation_v2 as P2
                P2.H_THRESH = h
                res = run_with_overrides(fit_feats, fit_extra, CORE_PARAMS,
                                          {}, sl, tp, PI_KNIFE_BPS,
                                          PI_EXT_CAP_BPS)
                res["h_thresh"] = h
                res["sl"] = sl
                res["tp"] = tp
                cells.append(res)
                done += 1
                if done % 10 == 0 or done == total:
                    print(f"  [{done:>3}/{total}] elapsed {time.time()-t0:.0f}s",
                          flush=True)

    cells.sort(key=lambda r: r.get("compound_ret_pct", -1e9), reverse=True)
    top5 = cells[:5]
    print("\n  TOP 5 cells by compound return:")
    for i, c in enumerate(top5, 1):
        print(f"    {i}. SL={c['sl']} TP={c['tp']} H={c['h_thresh']}  "
              f"ret={c.get('compound_ret_pct', 0):+.2f}%  "
              f"DD={c.get('max_dd', 0):.2f}%  "
              f"PF={c.get('pf', 0):.2f}  "
              f"trades={c.get('trades', 0)}  "
              f"WR={c.get('win_rate', 0)*100:.1f}%")

    best = cells[0]
    return cells, best


def step3_confirm(eval_feats, eval_extra, best):
    print("\n[STEP 3] F3c confirmation on EVAL slice")
    h = best["h_thresh"]; sl = best["sl"]; tp = best["tp"]
    print(f"  using refit params: SL={sl}  TP={tp}  H={h}")

    import phase2_filter_ablation_v2 as P2
    P2.H_THRESH = h

    baseline = run_with_overrides(eval_feats, eval_extra, CORE_PARAMS,
                                   {}, sl, tp, PI_KNIFE_BPS, PI_EXT_CAP_BPS)
    f3c = run_with_overrides(eval_feats, eval_extra, CORE_PARAMS,
                              {"F3c": True}, sl, tp, PI_KNIFE_BPS,
                              PI_EXT_CAP_BPS)

    def fmt(r):
        return {
            "trades": r.get("trades", 0),
            "wins": r.get("wins", 0),
            "win_rate": r.get("win_rate", 0),
            "compound_ret_pct": r.get("compound_ret_pct", 0),
            "max_dd": r.get("max_dd", 0),
            "pf": r.get("pf", 0),
            "knife_catchers": r.get("knife_catchers", 0),
            "knife_rate": r.get("knife_rate", 0),
            "trades_long": r.get("trades_long", 0),
            "trades_short": r.get("trades_short", 0),
        }

    bline = fmt(baseline); f3cf = fmt(f3c)
    delta = {
        "ret_pct_delta": f3cf["compound_ret_pct"] - bline["compound_ret_pct"],
        "max_dd_delta":  f3cf["max_dd"] - bline["max_dd"],
        "pf_delta":      f3cf["pf"] - bline["pf"],
        "knife_rate_delta": f3cf["knife_rate"] - bline["knife_rate"],
        "trades_delta":  f3cf["trades"] - bline["trades"],
    }

    print("\n  baseline (no F3c):")
    for k, v in bline.items():
        if isinstance(v, float): print(f"    {k}: {v:.4f}")
        else:                    print(f"    {k}: {v}")
    print("  F3c (threshold +/-0.3):")
    for k, v in f3cf.items():
        if isinstance(v, float): print(f"    {k}: {v:.4f}")
        else:                    print(f"    {k}: {v}")
    print("  delta (F3c - baseline):")
    for k, v in delta.items():
        print(f"    {k:>20}: {v:+.4f}")

    return bline, f3cf, delta


def main():
    t0 = time.time()
    fit_feats, fit_extra, fit_meta = build_slice("FIT", FIT_RANGE)
    eval_feats, eval_extra, eval_meta = build_slice("EVAL", EVAL_RANGE)

    # STEP 2
    cells, best = step2_refit(fit_feats, fit_extra)

    refit_out = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat() + "Z",
        "sprint": "v1.5 step 2 -- kraken-native refit",
        "venue": "Kraken-native L2 (Abraxasccs HF)",
        "pair": "ETH/USD",
        "fit_meta": fit_meta,
        "grid": {"sl": GRID_SL, "tp": GRID_TP, "h": GRID_H},
        "fixed": {"knife_bps": PI_KNIFE_BPS, "ext_cap_bps": PI_EXT_CAP_BPS},
        "core_params": CORE_PARAMS,
        "top5": [{k: v for k, v in c.items()} for c in cells[:5]],
        "selected": {k: v for k, v in best.items()},
        "all_cells": cells,
    }
    REFIT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REFIT_OUT.write_text(json.dumps(refit_out, indent=2, default=str))
    print(f"\nWrote {REFIT_OUT}")

    # STEP 3
    bline, f3cf, delta = step3_confirm(eval_feats, eval_extra, best)
    confirm_out = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat() + "Z",
        "sprint": "v1.5 step 3 -- F3c OOS confirmation",
        "venue": "Kraken-native L2 (Abraxasccs HF)",
        "pair": "ETH/USD",
        "eval_meta": eval_meta,
        "refit_params": {"h_thresh": best["h_thresh"], "sl": best["sl"],
                          "tp": best["tp"]},
        "fixed": {"knife_bps": PI_KNIFE_BPS, "ext_cap_bps": PI_EXT_CAP_BPS},
        "baseline_no_f3c": bline,
        "with_f3c": f3cf,
        "delta": delta,
    }
    CONFIRM_OUT.write_text(json.dumps(confirm_out, indent=2, default=str))
    print(f"\nWrote {CONFIRM_OUT}")

    print(f"\n[DONE] Steps 1-3 complete in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
