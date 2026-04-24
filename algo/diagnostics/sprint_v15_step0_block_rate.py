"""
Sprint v1.5 Step 0 -- F3c block-rate check on the clean-day subset of
Kraken-native L2 (Abraxasccs/kraken-market-data).

Pipeline per clean day:
  1. Load all ETH/USD snapshots, apply ask-cleanliness workaround.
  2. Resample to 1-min bars (same as Pi pipeline + Kaggle refit engine).
  3. Compute OB features -> 60-bar classify -> 30-bar rolling entropy
     (matches phase2_filter_ablation_v2).
  4. Emit per-day H series and candidate pre-filter conditions.

Two-pass design:
  Pass 1: compute H distribution across all clean days, derive H_THRESH
          at percentile 3 (live config) to get matched candidate density.
  Pass 2: for every candidate bar (base sig passes AND dH<0 AND knife AND
          ext cap, i.e. "all filters except F3c"), compute
          ema_diff_norm = (ema_fast - ema_slow) / atr_30 and tally block count
          under F3c threshold +/-0.3.

Report both the calibrated-H block rate and (for transparency) the count at
the original Pi-calibrated H=0.4352.

MUST PASS: overall block rate >=15% (expected 20-45%). Pi baseline F3c
blocked 48/124 = 38.7% of trades.

Output: algo/reports/sprint_v15_f3c_block_rate.json
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
from upgrade_backtest import make_features, candidate_signals
from phase2_filter_ablation_v2 import build_extra_features
from kraken_hf_loader import load_kraken_hf_day, clean_days_from_report

OUT = ALGO / "reports" / "sprint_v15_f3c_block_rate.json"

# Base signal pipeline (matches entropy_live_multi.py SHARED_CONFIG +
# phase2_filter_ablation_v2 CORE_PARAMS). Live uses entropy_percentile=3.
CORE_PARAMS = {"imb_min": 0.05, "spread_max": 20,
               "ret_low": 20, "ret_high": 80}
ENTROPY_WINDOW_BARS = 30                # matches phase2_filter_ablation_v2
ENTROPY_PERCENTILE_LIVE = 3.0           # live SHARED_CONFIG.entropy_percentile
H_THRESH_PI = 0.4352                    # live hard-coded threshold (for ref)
F3C_THRESHOLD = 0.3
# Pi baseline filter stack for F3c reference (phase2_filter_ablation_v2):
PI_BASELINE_KNIFE_BPS = 50.0
PI_BASELINE_EXT_CAP_BPS = 100.0
MIN_BARS_PER_DAY = 200


def load_and_featurize(date: str):
    """Load one clean day, resample, compute features and H array."""
    df_raw = load_kraken_hf_day(date, verbose=False)
    if df_raw.empty or len(df_raw) < 300:
        return None, f"too_few_snapshots({len(df_raw)})"

    df_1m = resample_pi_to_1min(df_raw)
    if len(df_1m) < MIN_BARS_PER_DAY:
        return None, f"too_few_bars({len(df_1m)})"

    df_1m = compute_ob_features(df_1m)
    df_1m = classify_ob_states(df_1m, window=60)
    H = rolling_entropy_ob(df_1m["state_ob"].values, NUM_STATES_OB,
                           ENTROPY_WINDOW_BARS)

    feats = make_features(df_1m, H)
    extra = build_extra_features(feats, volumes=None)
    return {"date": date, "bars": int(len(df_1m)),
            "raw": int(len(df_raw)), "feats": feats, "extra": extra}, None


def count_block(d, date, h_thresh, label):
    """Apply filter stack + F3c decision; return counts."""
    feats = d["feats"]; extra = d["extra"]
    n = feats["n"]
    mids = feats["mid"]; imb = feats["imb5"]
    dH_5 = feats["dH_5"]; ret_60 = feats["ret_60"]
    ema_fast = extra["ema_fast"]; ema_slow = extra["ema_slow"]
    atr_30 = extra["atr_30"]
    ret_150 = np.zeros(n)
    ret_150[150:] = (mids[150:] / mids[:-150] - 1) * 10000

    cands = candidate_signals(feats, h_thresh, CORE_PARAMS)
    base_cands = int(len(cands))

    n_dH = n_knife = n_ext = n_ema = 0
    longs = shorts = 0
    blocked = 0; block_long = block_short = 0
    norm_values = []

    for i in cands:
        if not np.isnan(dH_5[i]) and dH_5[i] >= 0:
            continue
        n_dH += 1
        d_dir = 1 if imb[i] > 0 else -1
        if d_dir == 1 and ret_60[i] < -PI_BASELINE_KNIFE_BPS:
            continue
        if d_dir == -1 and ret_60[i] > PI_BASELINE_KNIFE_BPS:
            continue
        n_knife += 1
        if d_dir == 1 and ret_150[i] > PI_BASELINE_EXT_CAP_BPS:
            continue
        if d_dir == -1 and ret_150[i] < -PI_BASELINE_EXT_CAP_BPS:
            continue
        n_ext += 1
        if atr_30[i] <= 0:
            continue
        n_ema += 1
        if d_dir == 1: longs += 1
        else: shorts += 1
        norm = (ema_fast[i] - ema_slow[i]) / atr_30[i]
        norm_values.append(float(norm))
        if d_dir == 1 and norm < F3C_THRESHOLD:
            blocked += 1; block_long += 1
        elif d_dir == -1 and norm > -F3C_THRESHOLD:
            blocked += 1; block_short += 1

    block_rate = (blocked / n_ema) if n_ema > 0 else None
    return {
        "label": label, "h_thresh": h_thresh,
        "date": date,
        "base_cands": base_cands,
        "after_dH": n_dH,
        "after_knife": n_knife,
        "after_ext": n_ext,
        "final_cands_with_ema": n_ema,
        "longs": longs, "shorts": shorts,
        "blocked": blocked,
        "block_long": block_long, "block_short": block_short,
        "block_rate": block_rate,
        "norm_mean": float(np.mean(norm_values)) if norm_values else None,
        "norm_p10": float(np.percentile(norm_values, 10)) if norm_values else None,
        "norm_p50": float(np.percentile(norm_values, 50)) if norm_values else None,
        "norm_p90": float(np.percentile(norm_values, 90)) if norm_values else None,
    }


def main():
    t0 = time.time()
    clean_days = clean_days_from_report()
    print(f"Sprint v1.5 Step 0 -- F3c block-rate check")
    print(f"  clean days to process: {len(clean_days)}")
    print(f"  first -> last: {clean_days[0]} -> {clean_days[-1]}")
    print(f"  entropy window (bars): {ENTROPY_WINDOW_BARS}")
    print(f"  live entropy percentile: {ENTROPY_PERCENTILE_LIVE}")
    print(f"  Pi-calibrated H_thresh: {H_THRESH_PI}")
    print(f"  Pi baseline filters: knife<={PI_BASELINE_KNIFE_BPS}bps  "
          f"ext<={PI_BASELINE_EXT_CAP_BPS}bps")
    print(f"  F3c threshold: +/-{F3C_THRESHOLD}")
    print(f"  MUST PASS: block rate >=15% (expected 20-45%)")
    print(f"  Pi reference: 48/124 = 38.7% of trades blocked by F3c")
    print()

    # PASS 1 -- load everything and collect H arrays
    print(f"[PASS 1] Loading + featurizing {len(clean_days)} days...", flush=True)
    loaded = []
    skipped = []
    all_H = []
    for i, d in enumerate(clean_days, 1):
        rec, err = load_and_featurize(d)
        if rec is None:
            skipped.append({"date": d, "reason": err})
            print(f"  [{i:>2}/{len(clean_days)}] {d}: SKIP {err}", flush=True)
            continue
        loaded.append(rec)
        H = rec["feats"]["H"]
        all_H.extend(H[~np.isnan(H)].tolist())
        dt = time.time() - t0
        print(f"  [{i:>2}/{len(clean_days)}] {d}: "
              f"raw={rec['raw']:>5} bars={rec['bars']:>4}  "
              f"H_valid={int((~np.isnan(H)).sum()):>4}  "
              f"elapsed={dt:.0f}s", flush=True)

    print(f"\n[PASS 1] Summary: loaded {len(loaded)}  skipped {len(skipped)}")
    print(f"  aggregate H bars: {len(all_H):,}")
    if not all_H:
        print("  NO H DATA. Abort."); return

    H_arr = np.asarray(all_H, dtype=float)
    h_pct3 = float(np.percentile(H_arr, 3.0))
    h_pct5 = float(np.percentile(H_arr, 5.0))
    h_pct10 = float(np.percentile(H_arr, 10.0))
    print(f"  H min/p3/p5/p10/p50/p90/max: "
          f"{H_arr.min():.4f} / {h_pct3:.4f} / {h_pct5:.4f} / {h_pct10:.4f} / "
          f"{np.percentile(H_arr,50):.4f} / {np.percentile(H_arr,90):.4f} / "
          f"{H_arr.max():.4f}")
    print(f"  Pi-calibrated H={H_THRESH_PI}: "
          f"{int((H_arr < H_THRESH_PI).sum())} of {len(H_arr)} bars pass")

    # PASS 2 -- block-rate at multiple H thresholds
    print(f"\n[PASS 2] Counting F3c blocks per-day at 4 H thresholds...")
    thresholds = [("calibrated_p3", h_pct3),
                  ("calibrated_p5", h_pct5),
                  ("calibrated_p10", h_pct10),
                  ("pi_0.4352", H_THRESH_PI)]

    per_day_all = {lbl: [] for lbl, _ in thresholds}
    for rec in loaded:
        for lbl, h in thresholds:
            per_day_all[lbl].append(count_block(rec, rec["date"], h, lbl))

    summary_by_threshold = {}
    for lbl, h in thresholds:
        per_day = per_day_all[lbl]
        agg_cands = sum(r["final_cands_with_ema"] for r in per_day)
        agg_blocked = sum(r["blocked"] for r in per_day)
        agg_longs = sum(r["longs"] for r in per_day)
        agg_shorts = sum(r["shorts"] for r in per_day)
        agg_block_long = sum(r["block_long"] for r in per_day)
        agg_block_short = sum(r["block_short"] for r in per_day)
        overall = (agg_blocked / agg_cands) if agg_cands > 0 else None
        long_rate = (agg_block_long / agg_longs) if agg_longs > 0 else None
        short_rate = (agg_block_short / agg_shorts) if agg_shorts > 0 else None
        summary_by_threshold[lbl] = {
            "h_thresh": h,
            "base_cands_total": sum(r["base_cands"] for r in per_day),
            "after_dH_total": sum(r["after_dH"] for r in per_day),
            "after_knife_total": sum(r["after_knife"] for r in per_day),
            "after_ext_total": sum(r["after_ext"] for r in per_day),
            "final_cands_with_ema": agg_cands,
            "longs": agg_longs, "shorts": agg_shorts,
            "blocked_total": agg_blocked,
            "blocked_long": agg_block_long,
            "blocked_short": agg_block_short,
            "block_rate": overall,
            "block_rate_long": long_rate,
            "block_rate_short": short_rate,
        }
        if overall is not None:
            print(f"  {lbl:>16}  H={h:.4f}  cands={agg_cands:>4}  "
                  f"blocked={agg_blocked:>4}  rate={overall*100:>5.2f}%")
        else:
            print(f"  {lbl:>16}  H={h:.4f}  ZERO candidates")

    # Verdict -- use the live-equivalent (p3) threshold as primary
    primary = summary_by_threshold["calibrated_p3"]
    br = primary["block_rate"]
    if br is None:
        verdict = "FAIL"
    elif br >= 0.15:
        verdict = "PASS"
    else:
        verdict = "FAIL"

    out = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat() + "Z",
        "sprint": "v1.5 step 0",
        "venue": "Kraken-native L2 (Abraxasccs HF)",
        "pair": "ETH/USD",
        "clean_days_input": len(clean_days),
        "clean_days_loaded": len(loaded),
        "clean_days_skipped": len(skipped),
        "skipped": skipped,
        "base_signal_params": CORE_PARAMS,
        "pi_baseline_filters": {
            "knife_bps": PI_BASELINE_KNIFE_BPS,
            "ext_cap_bps": PI_BASELINE_EXT_CAP_BPS,
            "dH_5_negative_required": True,
        },
        "entropy_window_bars": ENTROPY_WINDOW_BARS,
        "entropy_percentile_live": ENTROPY_PERCENTILE_LIVE,
        "f3c_threshold": F3C_THRESHOLD,
        "h_distribution": {
            "bars_total": len(H_arr),
            "min": float(H_arr.min()),
            "p3": h_pct3, "p5": h_pct5, "p10": h_pct10,
            "p50": float(np.percentile(H_arr, 50)),
            "p90": float(np.percentile(H_arr, 90)),
            "max": float(H_arr.max()),
            "bars_below_pi_0.4352": int((H_arr < H_THRESH_PI).sum()),
        },
        "summary_by_threshold": summary_by_threshold,
        "primary_threshold_label": "calibrated_p3",
        "primary_block_rate": br,
        "verdict": verdict,
        "verdict_threshold": 0.15,
        "per_day": {lbl: per_day_all[lbl] for lbl, _ in thresholds},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))

    print()
    print("=" * 70)
    print("=== F3c block-rate summary (primary: calibrated_p3) ===")
    print(f"  days loaded/input:    {len(loaded)} / {len(clean_days)}")
    if br is None:
        print(f"  block rate:           (undefined, zero candidates)")
    else:
        p = primary
        print(f"  base cands (p3):      {p['base_cands_total']:,}")
        print(f"  after dH<0:           {p['after_dH_total']:,}")
        print(f"  after knife:          {p['after_knife_total']:,}")
        print(f"  after ext cap:        {p['after_ext_total']:,}")
        print(f"  final (valid EMA):    {p['final_cands_with_ema']:,}  "
              f"L={p['longs']} S={p['shorts']}")
        print(f"  F3c blocked:          {p['blocked_total']:,}  "
              f"(L:{p['blocked_long']}, S:{p['blocked_short']})")
        print(f"  BLOCK RATE:           {br*100:.2f}%   "
              f"(L:{(p['block_rate_long'] or 0)*100:.1f}%  "
              f"S:{(p['block_rate_short'] or 0)*100:.1f}%)")
    print(f"  verdict:              {verdict} (threshold >=15%)")
    print(f"  Pi reference:         38.7%")
    print(f"  elapsed:              {time.time()-t0:.0f}s")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
