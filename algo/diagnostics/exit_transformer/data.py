"""Feature pipeline for the exit-decision transformer.

Inputs are derived solely from price/volume series — all values are
relative (log-returns, bps, log-vol-z) so the model sees no absolute
prices and cannot memorise specific price levels.

Per-minute features (sequence, length T=60):
  - log_return_1m
  - hl_range_bps          (high-low)/close * 1e4
  - log_vol_z             log(v) - rolling-60-mean of log(v)

Scalar context features (10):
  - pct_chg_60m, _240m, _360m, _720m, _1440m   (in bps)
  - vol_60m, vol_240m, vol_360m, vol_720m, vol_1440m
       (annualized stdev of 1-min log returns, %)

Targets (multi-horizon forward log-return in bps):
  - ret_5m, ret_15m, ret_60m, ret_240m
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np

SEQ_LEN = 60
HORIZONS = [60, 240, 360, 720, 1440]
TARGETS = [5, 15, 60, 240]
MAX_HORIZON = max(HORIZONS)
MAX_TARGET = max(TARGETS)


def load_bars(path: Path) -> np.ndarray:
    bars = json.loads(path.read_text())
    arr = np.array(bars, dtype=np.float64)  # [n, 6] = ts, o, h, l, c, v
    return arr


def build_features(bars: np.ndarray) -> dict:
    """Return dict of arrays aligned to bar index, plus a `valid` mask.

    Keys:
      seq_feats:  (n, 3) per-minute features  (log_ret, hl_range_bps, log_vol_z)
      scalar_feats: (n, 10) scalar features at end-of-sequence
      targets: (n, 4) forward log-returns in bps at each TARGETS step
      valid:   bool mask, True where all features+targets are computable
    """
    ts = bars[:, 0]
    closes = bars[:, 4]
    highs = bars[:, 2]
    lows = bars[:, 3]
    vols = bars[:, 5]
    n = len(closes)

    # Per-minute features
    log_ret = np.zeros(n)
    log_ret[1:] = np.log(closes[1:] / closes[:-1])
    hl_range_bps = (highs - lows) / closes * 1e4
    log_vol = np.log(np.maximum(vols, 1e-9))
    # rolling-60 mean of log_vol (causal)
    rmean = np.zeros(n)
    cs = np.cumsum(log_vol)
    for i in range(n):
        lo = max(0, i - 59)
        rmean[i] = (cs[i] - (cs[lo - 1] if lo > 0 else 0)) / (i - lo + 1)
    log_vol_z = log_vol - rmean

    seq_feats = np.stack([log_ret, hl_range_bps, log_vol_z], axis=1).astype(np.float32)

    # Scalar features: pct_chg at each horizon (bps), and vol (stdev * sqrt(525600/h)*100 -- annualized)
    scalar = np.zeros((n, 2 * len(HORIZONS)), dtype=np.float32)
    for k, h in enumerate(HORIZONS):
        # pct change in bps over horizon h
        pct = np.zeros(n)
        pct[h:] = (closes[h:] / closes[:-h] - 1.0) * 1e4
        scalar[:, k] = pct.astype(np.float32)
        # rolling vol over h: stdev of log_ret[i-h+1:i+1]
        vol = np.zeros(n)
        # use cumulative sum approach for speed: stdev = sqrt(E[r^2] - E[r]^2)
        r2 = log_ret * log_ret
        cs_r = np.cumsum(log_ret)
        cs_r2 = np.cumsum(r2)
        for i in range(h, n):
            sum_r = cs_r[i] - cs_r[i - h]
            sum_r2 = cs_r2[i] - cs_r2[i - h]
            mean = sum_r / h
            var = sum_r2 / h - mean * mean
            if var > 0:
                vol[i] = math.sqrt(var) * math.sqrt(1440 * 365) * 100  # annualized %
        scalar[:, len(HORIZONS) + k] = vol.astype(np.float32)

    # Targets
    targets = np.zeros((n, len(TARGETS)), dtype=np.float32)
    for k, t in enumerate(TARGETS):
        ret = np.zeros(n)
        if n > t:
            ret[:n - t] = (closes[t:] / closes[:n - t] - 1.0) * 1e4
        targets[:, k] = ret.astype(np.float32)

    valid = np.zeros(n, dtype=bool)
    valid[MAX_HORIZON: n - MAX_TARGET] = True

    return dict(seq_feats=seq_feats, scalar_feats=scalar,
                targets=targets, valid=valid, ts=ts)


def make_dataset(feats: dict, sample_stride: int = 5) -> tuple:
    """Build (X_seq, X_scalar, Y) tensors from feature dict.

    Strides through valid indices every `sample_stride` minutes.
    """
    valid_idx = np.where(feats["valid"])[0]
    sampled = valid_idx[::sample_stride]
    X_seq = np.zeros((len(sampled), SEQ_LEN, 3), dtype=np.float32)
    X_scalar = feats["scalar_feats"][sampled]
    Y = feats["targets"][sampled]
    seq_arr = feats["seq_feats"]
    for j, i in enumerate(sampled):
        X_seq[j] = seq_arr[i - SEQ_LEN + 1: i + 1]
    return X_seq, X_scalar, Y, sampled


def normalize_train_test(X_seq_train, X_scalar_train,
                         X_seq_val, X_scalar_val):
    """Compute z-score stats from train, apply to both."""
    seq_mean = X_seq_train.reshape(-1, 3).mean(axis=0)
    seq_std = X_seq_train.reshape(-1, 3).std(axis=0) + 1e-6
    sca_mean = X_scalar_train.mean(axis=0)
    sca_std = X_scalar_train.std(axis=0) + 1e-6
    X_seq_train = (X_seq_train - seq_mean) / seq_std
    X_seq_val = (X_seq_val - seq_mean) / seq_std
    X_scalar_train = (X_scalar_train - sca_mean) / sca_std
    X_scalar_val = (X_scalar_val - sca_mean) / sca_std
    return (X_seq_train, X_scalar_train, X_seq_val, X_scalar_val,
            dict(seq_mean=seq_mean.tolist(), seq_std=seq_std.tolist(),
                 sca_mean=sca_mean.tolist(), sca_std=sca_std.tolist()))


if __name__ == "__main__":
    p = Path("D:/ai/timesFM/algo/state/pi_pull_2026_04_30/eth_1m_60d.json")
    bars = load_bars(p)
    print(f"bars: {bars.shape}")
    feats = build_features(bars)
    print(f"valid: {feats['valid'].sum()} / {len(feats['valid'])}")
    X_seq, X_scalar, Y, idx = make_dataset(feats, sample_stride=5)
    print(f"samples: X_seq={X_seq.shape}  X_scalar={X_scalar.shape}  Y={Y.shape}")
    print(f"sample scalar (first 5):\n{X_scalar[0]}")
    print(f"sample targets (first 5):\n{Y[:5]}")
