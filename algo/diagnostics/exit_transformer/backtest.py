"""Replay each historical Pi trade with the exit transformer and compare
hypothetical exit pnl vs the actual realized pnl.

Inputs:
  - algo/state/pi_pull_2026_04_30/trade_history.jsonl  (30 trades)
  - PF_ETHUSD 1-min OHLCV (Kraken Futures public, fetched on demand)
  - exit_model.pt (trained model + norm stats)

For each trade, we walk through every minute inside the trade window. At each
minute, we compute features over the trailing 60-min sequence + multi-horizon
context, then ask the model to predict forward returns at [5,15,60,240]
minute horizons. The model recommends EXIT if the position-direction-adjusted
forward return is below threshold.

We then compare:
  - Actual exit:   what the bot actually did (recorded in trade_history)
  - Model exit:    hypothetical first-time the model says exit
  - Best-possible exit:  oracle that exits at the maximum of best_seen
                          (an upper bound for reference)

Output: backtest report per trade + aggregate summary.
"""
from __future__ import annotations
import json
import sys
import time
import urllib.request
import datetime as dt
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data import build_features, SEQ_LEN, HORIZONS, TARGETS, MAX_HORIZON
from model import ExitTransformer

ROOT = HERE.parent.parent.parent
TRADE_HISTORY = ROOT / "algo" / "state" / "pi_pull_2026_04_30" / "trade_history.jsonl"
MODEL_PATH = ROOT / "algo" / "state" / "exit_transformer" / "exit_model.pt"
OHLC_CACHE = ROOT / "algo" / "state" / "pi_pull_2026_04_30" / "eth_1m_60d.json"
OUT = ROOT / "algo" / "state" / "exit_transformer" / "backtest_results.json"

DEPLOY_TS = dt.datetime(2026, 4, 24, 13, 29, 39, tzinfo=dt.timezone.utc).timestamp()

# Exit policy: trade direction times predicted forward return is adverse if < -EXIT_THRESH bps
EXIT_THRESH_BPS = 10  # require strong adverse signal
HORIZON_USED = 60  # which horizon to use for the exit decision (must be in TARGETS)

# Risk floor (model never overrides hard SL)
HARD_SL_BPS = 50


def load_60d_bars() -> np.ndarray:
    bars = json.loads(OHLC_CACHE.read_text())
    return np.array(bars, dtype=np.float64)


def load_model():
    ckpt = torch.load(MODEL_PATH, map_location="cuda" if torch.cuda.is_available() else "cpu",
                      weights_only=False)
    cfg = ckpt["config"]
    model = ExitTransformer(
        f_seq=cfg["f_seq"], f_scalar=cfg["f_scalar"], n_targets=len(cfg["targets"]),
        d_model=cfg["d_model"], nhead=cfg["nhead"], num_layers=cfg["num_layers"],
        dim_ff=cfg["dim_ff"], seq_len=cfg["seq_len"], dropout=0.0,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, ckpt["norm"], cfg, device


def replay_trade(trade: dict, bars_arr: np.ndarray, feats: dict,
                 model, norm, cfg, device) -> dict:
    """Walk through the trade minute-by-minute and find the first minute the
    model recommends exit. Compute hypothetical pnl at that minute."""
    entry_px = trade["entry_price"]
    direction = trade["direction"]
    exit_iso = trade["time"]
    exit_unix_ms = int(dt.datetime.fromisoformat(exit_iso).timestamp() * 1000)
    entry_unix_ms = exit_unix_ms - int(trade["hold_min"] * 60 * 1000)

    # Find indices in bars covering the trade
    ts_arr = bars_arr[:, 0]
    in_trade_idx = np.where((ts_arr >= entry_unix_ms - 60_000) &
                            (ts_arr <= exit_unix_ms + 60_000))[0]
    if len(in_trade_idx) < 5:
        return {"order_id": trade["order_id"], "error": "no_bars"}

    seq_mean = np.array(norm["seq_mean"], dtype=np.float32)
    seq_std = np.array(norm["seq_std"], dtype=np.float32)
    sca_mean = np.array(norm["sca_mean"], dtype=np.float32)
    sca_std = np.array(norm["sca_std"], dtype=np.float32)
    y_mean = np.array(norm["y_mean"], dtype=np.float32)
    y_std = np.array(norm["y_std"], dtype=np.float32)

    h_idx = TARGETS.index(HORIZON_USED)

    decisions = []  # (minute_offset, predicted ret_h, model_exit?)
    model_exit_minute = None
    model_exit_close = None
    closes = bars_arr[:, 4]

    for k, i in enumerate(in_trade_idx):
        # need MAX_HORIZON of past data and SEQ_LEN of seq context
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
        pred_h = float(pred_bps[h_idx])

        # Position-direction-adjusted forward return: positive = trade is winning,
        # negative = trade is losing in next H minutes. For long, equals pred_h.
        adj = direction * pred_h
        decisions.append((k, pred_h, adj))

        if model_exit_minute is None and adj < -EXIT_THRESH_BPS:
            model_exit_minute = k
            model_exit_close = float(closes[i])

    # Compute pnl at model exit
    if model_exit_minute is not None and model_exit_close is not None:
        pnl_bps_model = direction * (model_exit_close / entry_px - 1.0) * 1e4
    else:
        # Model never said exit -- assume held to actual exit
        pnl_bps_model = float(trade["pnl_bps"])

    # Best-possible: oracle that exits at the best price achieved
    best_close = entry_px
    for i in in_trade_idx:
        c = bars_arr[i, 4]
        if direction == 1:
            if c > best_close:
                best_close = c
        else:
            if c < best_close:
                best_close = c
    pnl_bps_oracle = direction * (best_close / entry_px - 1.0) * 1e4

    return {
        "order_id": trade["order_id"],
        "entry_iso": dt.datetime.fromtimestamp(entry_unix_ms / 1000, tz=dt.timezone.utc).isoformat(),
        "exit_iso": exit_iso,
        "direction": direction,
        "hold_min": trade["hold_min"],
        "actual_pnl_bps": float(trade["pnl_bps"]),
        "actual_reason": trade["reason"],
        "model_exit_minute": model_exit_minute,
        "model_pnl_bps": float(pnl_bps_model),
        "oracle_pnl_bps": float(pnl_bps_oracle),
        "n_decisions": len(decisions),
        "min_adj_pred": min((d[2] for d in decisions), default=None),
        "max_adj_pred": max((d[2] for d in decisions), default=None),
    }


def main():
    trades = [json.loads(l) for l in open(TRADE_HISTORY)]
    print(f"Loaded {len(trades)} trades")

    # Use the cached 60d bars to cover all trades
    bars_arr = load_60d_bars()
    print(f"60d bars: {bars_arr.shape}, span "
          f"{dt.datetime.fromtimestamp(bars_arr[0,0]/1000, tz=dt.timezone.utc)} to "
          f"{dt.datetime.fromtimestamp(bars_arr[-1,0]/1000, tz=dt.timezone.utc)}")

    feats = build_features(bars_arr)

    model, norm, cfg, device = load_model()
    print(f"model loaded on {device}")

    results = []
    for t in trades:
        r = replay_trade(t, bars_arr, feats, model, norm, cfg, device)
        results.append(r)
        if "error" in r:
            print(f"{r['order_id'][:8]}  SKIP: {r['error']}")
            continue
        is_post = dt.datetime.fromisoformat(t["time"]).timestamp() > DEPLOY_TS
        tag = "POST" if is_post else "PRE "
        ema = r["model_exit_minute"] if r["model_exit_minute"] is not None else "—"
        print(f"{tag} {r['order_id'][:8]} dir={r['direction']:+d} "
              f"hold={r['hold_min']:6.1f}m  "
              f"actual={r['actual_pnl_bps']:+7.1f}bps ({r['actual_reason']:>11})  "
              f"model_exit_at={ema:>4}m  model={r['model_pnl_bps']:+7.1f}bps  "
              f"oracle={r['oracle_pnl_bps']:+7.1f}bps")

    # Aggregate
    pre = [r for r, t in zip(results, trades)
           if "error" not in r
           and dt.datetime.fromisoformat(t["time"]).timestamp() <= DEPLOY_TS]
    post = [r for r, t in zip(results, trades)
            if "error" not in r
            and dt.datetime.fromisoformat(t["time"]).timestamp() > DEPLOY_TS]

    def summarize(rs, label):
        if not rs:
            return
        actual = np.array([r["actual_pnl_bps"] for r in rs])
        model = np.array([r["model_pnl_bps"] for r in rs])
        oracle = np.array([r["oracle_pnl_bps"] for r in rs])
        print(f"\n=== {label} (n={len(rs)}) ===")
        print(f"  actual:   sum={actual.sum():+.0f}  mean={actual.mean():+.1f}  "
              f"win_rate={(actual>0).mean()*100:.0f}%")
        print(f"  model:    sum={model.sum():+.0f}  mean={model.mean():+.1f}  "
              f"win_rate={(model>0).mean()*100:.0f}%")
        print(f"  oracle:   sum={oracle.sum():+.0f}  mean={oracle.mean():+.1f}")
        delta = (model - actual).sum()
        print(f"  model - actual sum: {delta:+.0f} bps  (positive = model better)")

    summarize(pre, "PRE-DEPLOY (10x leverage)")
    summarize(post, "POST-DEPLOY (5x leverage)")
    summarize([r for r in results if "error" not in r], "ALL")

    OUT.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
