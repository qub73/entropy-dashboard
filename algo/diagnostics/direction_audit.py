"""
Phase 1 -- Direction signal audit.

Hypothesis: 17/17 LONG is not chance. Either `sign(imbalance)` is
structurally long-biased on Kraken Futures ETH, or the entropy threshold
only fires in long-leaning states.

This script:
 1) Loads all Pi ETH orderbook data (Feb 18 - Apr 7, 21 consecutive days
    + 3 scattered), resamples to 1-min bars, builds the full feature set.
 2) Characterizes the imbalance_5 distribution at every bar.
 3) Characterizes the imbalance distribution at bars where EVERY entry
    filter EXCEPT the direction-selection step has passed.
 4) Splits H into imb>0 vs imb<0 conditional distributions.
 5) Prototypes trade-flow imbalance using the last ~24h of Kraken public
    trade feed (paginated REST) since we have no historical trade archive.

Outputs:
 - reports/direction_audit.json
 - reports/direction_audit.png  (via matplotlib if available)
"""
import json, os, sys, time, urllib.request
from pathlib import Path
from datetime import datetime, timezone
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

OUT_JSON = ALGO / "reports" / "direction_audit.json"
OUT_PNG = ALGO / "reports" / "direction_audit.png"


def pct(x, p): return float(np.percentile(x, p)) if len(x) else None


def describe(arr, name):
    a = np.asarray(arr, dtype=float)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return {"n": 0}
    return {
        "n": int(len(a)),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "std": float(a.std()),
        "p5": pct(a, 5),
        "p25": pct(a, 25),
        "p75": pct(a, 75),
        "p95": pct(a, 95),
        "min": float(a.min()),
        "max": float(a.max()),
        "frac_positive": float((a > 0).mean()),
        "frac_negative": float((a < 0).mean()),
        "frac_abs_above_0_05": float((np.abs(a) > 0.05).mean()),
    }


def fetch_kraken_trades(pair="XETHZUSD", hours_back=24):
    """Paginate Kraken /Trades back from now until hours_back reached.
    Returns list of [price, volume, ts, side] tuples where side is 'b' or 's'."""
    end_s = time.time()
    start_s = end_s - hours_back * 3600
    since_ns = int(start_s * 1e9)
    all_trades = []
    last_ns = since_ns
    max_pages = 60  # safety cap (~60000 trades)
    for page in range(max_pages):
        url = f"https://api.kraken.com/0/public/Trades?pair={pair}&since={last_ns}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            resp = urllib.request.urlopen(req, timeout=20)
            data = json.loads(resp.read())
        except Exception as e:
            print(f"    trades fetch error page {page}: {e}", flush=True)
            time.sleep(2); continue
        if data.get("error"):
            err = data["error"]
            if any("Too many" in str(e) for e in err):
                time.sleep(3); continue
            print(f"    kraken trades err: {err}", flush=True); break
        result = data.get("result", {})
        new_last = result.get("last")
        rows = [v for k, v in result.items() if k != "last"]
        rows = rows[0] if rows else []
        if not rows:
            break
        all_trades.extend(rows)
        if new_last is None or new_last == last_ns:
            break
        last_ns = int(new_last)
        latest_ts = float(rows[-1][2])
        if latest_ts >= end_s:
            break
        time.sleep(1.1)
    # Normalize: [price, vol, ts, side]
    out = []
    for r in all_trades:
        try:
            out.append([float(r[0]), float(r[1]), float(r[2]), str(r[3])])
        except Exception:
            continue
    return out


def main():
    print("Loading ETH Pi orderbook (Feb 18 - Apr 7)...", flush=True)
    df = load_orderbook_range("data/orderbook_pi", "ob_PF_ETHUSD")
    df_1m = resample_pi_to_1min(df)
    df_1m = compute_ob_features(df_1m)
    df_1m = classify_ob_states(df_1m, window=60)
    ent = rolling_entropy_ob(df_1m['state_ob'].values, NUM_STATES_OB, 30)
    feats = make_features(df_1m, ent)
    print(f"  {feats['n']} bars, {feats['n']/1440:.1f} days\n", flush=True)

    imb = feats['imb5']
    H = feats['H']
    spread = feats['spread_bps']
    ret_5 = feats['ret_5']
    ret_60 = feats['ret_60']
    dH_5 = feats['dH_5']
    n = feats['n']

    # --- All-bars distribution ---
    all_dist = describe(imb, "imb_all")

    # --- Filter pass (except direction) mask ---
    # Same filters as entry logic, minus the imbalance-sign selection.
    params = {'imb_min': 0.05, 'spread_max': 20, 'ret_low': 20, 'ret_high': 80}
    cands = set(candidate_signals(feats, 0.4352, params))

    # Apply the same dH and knife filters on top. Extended-move too.
    ret_150 = np.zeros(n)
    ret_150[150:] = (feats['mid'][150:] / feats['mid'][:-150] - 1) * 10000

    cand_idx = []
    blocked_dH = blocked_knife_long = blocked_knife_short = 0
    blocked_ext_long = blocked_ext_short = 0
    for i in sorted(cands):
        if not np.isnan(dH_5[i]) and dH_5[i] >= 0:
            blocked_dH += 1; continue
        # Note: knife filter is direction-dependent, but for the audit we
        # record the PRE-direction candidate pool; knife only drops once
        # direction is chosen by sign(imb). We apply it symmetrically here
        # just to mirror the live filter set.
        d = 1 if imb[i] > 0 else -1
        if d == 1 and ret_60[i] < -50:
            blocked_knife_long += 1; continue
        if d == -1 and ret_60[i] > 50:
            blocked_knife_short += 1; continue
        if d == 1 and ret_150[i] > 100:
            blocked_ext_long += 1; continue
        if d == -1 and ret_150[i] < -100:
            blocked_ext_short += 1; continue
        cand_idx.append(i)

    cand_idx = np.array(cand_idx, dtype=int)
    imb_at_cands = imb[cand_idx] if len(cand_idx) else np.array([])
    H_at_cands = H[cand_idx] if len(cand_idx) else np.array([])

    cand_dist = describe(imb_at_cands, "imb_at_candidates")

    # --- H conditional on imbalance sign (over all bars) ---
    mask_pos = imb > 0
    mask_neg = imb < 0
    H_pos = H[mask_pos & ~np.isnan(H)]
    H_neg = H[mask_neg & ~np.isnan(H)]

    # --- Also H at candidate bars split by imb sign ---
    cand_imb_pos_mask = imb_at_cands > 0 if len(cand_idx) else np.array([], dtype=bool)
    cand_imb_neg_mask = imb_at_cands < 0 if len(cand_idx) else np.array([], dtype=bool)

    # --- Asymmetric over narrow price-bucketed subsets ---
    # Has the entropy threshold been more attainable in one regime or the other?
    h_thresh = 0.4352
    below_thresh = H < h_thresh
    below_pos = (below_thresh & mask_pos).sum()
    below_neg = (below_thresh & mask_neg).sum()

    # Combined tier: H<thresh AND |imb|>0.05
    strong_imb = np.abs(imb) > 0.05
    combined_long = int((below_thresh & strong_imb & (imb > 0)).sum())
    combined_short = int((below_thresh & strong_imb & (imb < 0)).sum())

    # ----- Trade-flow prototype (last ~24h) -----
    print("Fetching last 24h of Kraken ETH/USD trades for trade-flow prototype...", flush=True)
    trades = fetch_kraken_trades("XETHZUSD", hours_back=24)
    print(f"  got {len(trades)} trades\n", flush=True)
    trade_flow_summary = None
    if trades:
        # Bucket by 5-min window and compute imbalance
        trades.sort(key=lambda r: r[2])
        t0 = trades[0][2]; t1 = trades[-1][2]
        buckets = {}  # bucket_start -> [buy_vol, sell_vol]
        for p, v, ts, side in trades:
            bkt = int(ts // 300) * 300
            if bkt not in buckets:
                buckets[bkt] = [0.0, 0.0]
            if side == "b":
                buckets[bkt][0] += v
            else:
                buckets[bkt][1] += v
        tfi_series = []
        for bkt in sorted(buckets):
            b, s = buckets[bkt]
            total = b + s
            if total > 0:
                tfi_series.append((b - s) / total)
        tfi_dist = describe(np.asarray(tfi_series), "trade_flow_imb_5min")
        trade_flow_summary = {
            "window_hours": 24,
            "n_trades": len(trades),
            "n_5min_buckets": len(tfi_series),
            "first_ts": t0, "last_ts": t1,
            "dist": tfi_dist,
        }

    # ------------------------- Summary JSON -------------------------
    result = {
        "data": {
            "source": "Pi Kraken Futures L2 orderbook",
            "date_range_utc": [
                str(df_1m.index[0]) if hasattr(df_1m.index, "__getitem__") else None,
                str(df_1m.index[-1]) if hasattr(df_1m.index, "__getitem__") else None,
            ],
            "n_bars_1min": int(n),
            "n_days": round(n/1440, 2),
        },
        "all_bars": {
            "imbalance_dist": all_dist,
            "H_dist_when_imb_pos": describe(H_pos, "H_imb_pos"),
            "H_dist_when_imb_neg": describe(H_neg, "H_imb_neg"),
        },
        "filter_pass_except_direction": {
            "n_candidates": int(len(cand_idx)),
            "blocked_by_dH": int(blocked_dH),
            "blocked_by_knife_long": int(blocked_knife_long),
            "blocked_by_knife_short": int(blocked_knife_short),
            "blocked_by_ext_long": int(blocked_ext_long),
            "blocked_by_ext_short": int(blocked_ext_short),
            "imbalance_dist": cand_dist,
            "signal_candidates_long":  int((imb_at_cands > 0).sum()) if len(cand_idx) else 0,
            "signal_candidates_short": int((imb_at_cands < 0).sum()) if len(cand_idx) else 0,
            "long_to_short_ratio":
                (int((imb_at_cands > 0).sum()) / max(1, int((imb_at_cands < 0).sum())))
                if len(cand_idx) else None,
        },
        "entropy_crossings": {
            "n_below_thresh_total": int(below_thresh.sum()),
            "n_below_thresh_when_imb_pos": int(below_pos),
            "n_below_thresh_when_imb_neg": int(below_neg),
            "long_to_short_ratio_below_thresh":
                (int(below_pos) / max(1, int(below_neg))),
        },
        "combined_h_and_imb_signal": {
            "n_long": combined_long,
            "n_short": combined_short,
            "long_to_short_ratio": combined_long / max(1, combined_short),
        },
        "trade_flow_prototype": trade_flow_summary,
        "headline_metrics": {
            "imbalance_mean":   all_dist["mean"],
            "imbalance_median": all_dist["median"],
            "frac_positive":    all_dist["frac_positive"],
            "signal_candidates_long":  int((imb_at_cands > 0).sum()) if len(cand_idx) else 0,
            "signal_candidates_short": int((imb_at_cands < 0).sum()) if len(cand_idx) else 0,
            "H_p50_when_imb_pos": pct(H_pos, 50),
            "H_p50_when_imb_neg": pct(H_neg, 50),
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, default=str))

    # ------------------------- Plot -------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.patch.set_facecolor("#0d1117")
        for ax in axes.flat:
            ax.set_facecolor("#161b22")
            ax.tick_params(colors="#e6edf3")
            for s in ax.spines.values(): s.set_color("#30363d")
            ax.grid(True, color="#30363d", alpha=0.4)
            ax.xaxis.label.set_color("#e6edf3"); ax.yaxis.label.set_color("#e6edf3")
            ax.title.set_color("#e6edf3")

        axes[0, 0].hist(imb[~np.isnan(imb)], bins=60, color="#58a6ff", edgecolor="none", alpha=0.85)
        axes[0, 0].axvline(0, color="#d29922", linestyle="--", linewidth=1)
        axes[0, 0].set_title(f"All-bar imbalance (n={(~np.isnan(imb)).sum()}, mean={all_dist['mean']:.3f})")
        axes[0, 0].set_xlabel("imbalance_5"); axes[0, 0].set_ylabel("count")

        if len(imb_at_cands):
            axes[0, 1].hist(imb_at_cands, bins=40, color="#3fb950", edgecolor="none", alpha=0.85)
            axes[0, 1].axvline(0, color="#d29922", linestyle="--", linewidth=1)
            axes[0, 1].axvline(+0.05, color="#f85149", linestyle=":", linewidth=1)
            axes[0, 1].axvline(-0.05, color="#f85149", linestyle=":", linewidth=1)
            axes[0, 1].set_title(f"Imbalance at signal candidates (n={len(imb_at_cands)}, "
                                 f"long/short={int((imb_at_cands>0).sum())}/{int((imb_at_cands<0).sum())})")
            axes[0, 1].set_xlabel("imbalance_5"); axes[0, 1].set_ylabel("count")

        if len(H_pos) and len(H_neg):
            bins = np.linspace(0, 1, 50)
            axes[1, 0].hist(H_pos, bins=bins, color="#3fb950", alpha=0.6, label=f"imb>0 (n={len(H_pos)})")
            axes[1, 0].hist(H_neg, bins=bins, color="#f85149", alpha=0.6, label=f"imb<0 (n={len(H_neg)})")
            axes[1, 0].axvline(0.4352, color="#d29922", linestyle="--", linewidth=1, label="threshold")
            axes[1, 0].set_title("H distribution conditional on imbalance sign")
            axes[1, 0].set_xlabel("H"); axes[1, 0].set_ylabel("count")
            axes[1, 0].legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")

        if trade_flow_summary and trade_flow_summary.get("n_5min_buckets", 0) > 10:
            tfi_vals = []
            for bkt in sorted(buckets):
                b, s = buckets[bkt]
                total = b + s
                if total > 0:
                    tfi_vals.append((b - s) / total)
            tfi_vals = np.asarray(tfi_vals)
            axes[1, 1].hist(tfi_vals, bins=30, color="#bc8cff", edgecolor="none", alpha=0.85)
            axes[1, 1].axvline(0, color="#d29922", linestyle="--", linewidth=1)
            axes[1, 1].set_title(f"Trade-flow imbalance 5min (24h, n={len(tfi_vals)}, "
                                 f"mean={tfi_vals.mean():.3f})")
            axes[1, 1].set_xlabel("(buy - sell) / total"); axes[1, 1].set_ylabel("count")
        else:
            axes[1, 1].text(0.5, 0.5, "trade-flow data unavailable\n(would need paginated Kraken trades)",
                            ha="center", va="center", color="#e6edf3", transform=axes[1, 1].transAxes)
            axes[1, 1].set_title("Trade-flow imbalance prototype")

        fig.suptitle("Direction-signal audit  --  Pi ETH 21d", color="#e6edf3", fontsize=14, y=0.995)
        fig.tight_layout()
        fig.savefig(OUT_PNG, dpi=110, facecolor=fig.get_facecolor())
        print(f"Saved plot: {OUT_PNG}")
    except Exception as e:
        print(f"(plot error, skipping: {e})")

    # Console summary
    print("\n=== HEADLINE ===")
    print(f"Imbalance all bars:  mean={all_dist['mean']:.3f}, median={all_dist['median']:.3f}, "
          f"frac_positive={all_dist['frac_positive']:.3f}")
    print(f"Imbalance at signal candidates: n={len(imb_at_cands)}, long/short = "
          f"{int((imb_at_cands>0).sum())}/{int((imb_at_cands<0).sum())}")
    print(f"H median when imb>0: {pct(H_pos,50):.4f}")
    print(f"H median when imb<0: {pct(H_neg,50):.4f}")
    print(f"Combined (H<thresh AND |imb|>0.05): long={combined_long}, short={combined_short}")
    if trade_flow_summary:
        td = trade_flow_summary["dist"]
        print(f"Trade-flow 5m (24h): mean={td['mean']:.3f}, frac_positive={td['frac_positive']:.3f}, "
              f"n_buckets={trade_flow_summary['n_5min_buckets']}")
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
