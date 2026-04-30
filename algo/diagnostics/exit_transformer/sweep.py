"""Sweep exit thresholds × horizons to find the best policy on Pi trades."""
from __future__ import annotations
import json
import sys
import datetime as dt
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data import build_features, SEQ_LEN, TARGETS, MAX_HORIZON
from model import ExitTransformer

ROOT = HERE.parent.parent.parent
TRADE_HISTORY = ROOT / "algo" / "state" / "pi_pull_2026_04_30" / "trade_history.jsonl"
MODEL_PATH = ROOT / "algo" / "state" / "exit_transformer" / "exit_model.pt"
OHLC_CACHE = ROOT / "algo" / "state" / "pi_pull_2026_04_30" / "eth_1m_60d.json"
DEPLOY_TS = dt.datetime(2026, 4, 24, 13, 29, 39, tzinfo=dt.timezone.utc).timestamp()


def main():
    bars = np.array(json.loads(OHLC_CACHE.read_text()), dtype=np.float64)
    feats = build_features(bars)
    trades = [json.loads(l) for l in open(TRADE_HISTORY)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    norm = ckpt["norm"]
    model = ExitTransformer(
        f_seq=cfg["f_seq"], f_scalar=cfg["f_scalar"], n_targets=len(cfg["targets"]),
        d_model=cfg["d_model"], nhead=cfg["nhead"], num_layers=cfg["num_layers"],
        dim_ff=cfg["dim_ff"], seq_len=cfg["seq_len"], dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Pre-compute model predictions at every minute inside every trade
    seq_mean = np.array(norm["seq_mean"], dtype=np.float32)
    seq_std = np.array(norm["seq_std"], dtype=np.float32)
    sca_mean = np.array(norm["sca_mean"], dtype=np.float32)
    sca_std = np.array(norm["sca_std"], dtype=np.float32)
    y_mean = np.array(norm["y_mean"], dtype=np.float32)
    y_std = np.array(norm["y_std"], dtype=np.float32)
    ts_arr = bars[:, 0]
    closes = bars[:, 4]

    # For each trade collect per-minute predictions
    trade_preds = []
    for t in trades:
        exit_unix_ms = int(dt.datetime.fromisoformat(t["time"]).timestamp() * 1000)
        entry_unix_ms = exit_unix_ms - int(t["hold_min"] * 60 * 1000)
        in_trade_idx = np.where((ts_arr >= entry_unix_ms - 60_000) &
                                (ts_arr <= exit_unix_ms + 60_000))[0]
        rows = []
        for i in in_trade_idx:
            if i < MAX_HORIZON:
                continue
            seq = feats["seq_feats"][i - SEQ_LEN + 1: i + 1]
            if seq.shape[0] != SEQ_LEN:
                continue
            scalar = feats["scalar_feats"][i]
            seq_n = (seq - seq_mean) / seq_std
            sca_n = (scalar - sca_mean) / sca_std
            with torch.no_grad():
                xs = torch.from_numpy(seq_n).unsqueeze(0).to(device)
                xc = torch.from_numpy(sca_n).unsqueeze(0).to(device)
                pred_n = model(xs, xc).cpu().numpy()[0]
            pred_bps = pred_n * y_std + y_mean
            rows.append((i, pred_bps.copy(), float(closes[i])))
        trade_preds.append((t, in_trade_idx, rows))

    # Sweep thresholds × horizons
    hs = [5, 15, 60, 240]
    thrs = [0, -5, -10, -15, -25]
    print(f"{'horizon':<8}{'thr_bps':<10}"
          f"{'PRE_sum':<12}{'PRE_wr':<10}"
          f"{'POST_sum':<12}{'POST_wr':<10}"
          f"{'ALL_sum':<12}{'ALL_wr':<10}{'mean_n_exits':<14}")
    print("-" * 120)
    summaries = []
    for h in hs:
        h_idx = TARGETS.index(h)
        for thr in thrs:
            pre_pnls, post_pnls, all_pnls, n_exits = [], [], [], []
            for t, in_trade_idx, rows in trade_preds:
                d = t["direction"]
                exit_minute = None
                exit_close = None
                for k, (i, pred_bps, c) in enumerate(rows):
                    adj = d * pred_bps[h_idx]
                    if adj < thr:
                        exit_minute = k
                        exit_close = c
                        break
                if exit_minute is not None:
                    pnl = d * (exit_close / t["entry_price"] - 1.0) * 1e4
                    n_exits.append(1)
                else:
                    pnl = float(t["pnl_bps"])
                    n_exits.append(0)
                all_pnls.append(pnl)
                if dt.datetime.fromisoformat(t["time"]).timestamp() > DEPLOY_TS:
                    post_pnls.append(pnl)
                else:
                    pre_pnls.append(pnl)

            pa = np.array(pre_pnls)
            po = np.array(post_pnls)
            al = np.array(all_pnls)
            row = dict(horizon=h, thr=thr,
                       pre_sum=pa.sum(), pre_wr=(pa > 0).mean() * 100,
                       post_sum=po.sum(), post_wr=(po > 0).mean() * 100,
                       all_sum=al.sum(), all_wr=(al > 0).mean() * 100,
                       n_exits=int(np.sum(n_exits)))
            summaries.append(row)
            print(f"{h:<8}{thr:<10}"
                  f"{pa.sum():+11.0f} {(pa>0).mean()*100:8.0f}%"
                  f"{po.sum():+11.0f} {(po>0).mean()*100:8.0f}%"
                  f"{al.sum():+11.0f} {(al>0).mean()*100:8.0f}%"
                  f"{int(np.sum(n_exits)):>14}/30")

    # Find best by ALL_sum
    best = max(summaries, key=lambda r: r["all_sum"])
    print(f"\nBest policy (by ALL sum): horizon={best['horizon']}m  thr={best['thr']}bps")
    print(f"  PRE  sum={best['pre_sum']:+.0f}  wr={best['pre_wr']:.0f}%")
    print(f"  POST sum={best['post_sum']:+.0f}  wr={best['post_wr']:.0f}%")
    print(f"  ALL  sum={best['all_sum']:+.0f}  wr={best['all_wr']:.0f}%  exits={best['n_exits']}/30")

    out = ROOT / "algo" / "state" / "exit_transformer" / "sweep_results.json"
    out.write_text(json.dumps(summaries, indent=2, default=str))
    print(f"Saved sweep to {out}")


if __name__ == "__main__":
    main()
