"""
Load clean-day Kraken L2 parquets from the Abraxasccs/kraken-market-data HF
dataset and produce an OB-features DataFrame matching the shape produced by
parse_kaggle_to_ob_features (in kaggle_ob_trainer.py).

Known bias: the Kraken WebSocket book-reconstruction bug (missed
volume-zero-means-delete messages during aggressive moves) corrupts asks on
many snapshots. We filter asks < best_bid out, require >=5 clean levels,
and drop unrecoverable rows. Clean days in this script are those where
>=80% of ETH/USD snapshots survive the filter (pre-computed in
reports/kraken_hf_cleanliness.json).

Usage:
    from kraken_hf_loader import load_kraken_hf_day, load_kraken_hf_days
    df = load_kraken_hf_day("2026-04-13")   # all ETH/USD snapshots, 10 levels
    dfs = load_kraken_hf_days(["2026-04-13", "2026-04-14"])

Cache location: data/kraken_hf_sample/book/{date}_{name}.parquet
"""
import json, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLE_DIR = ROOT / "data" / "kraken_hf_sample" / "book"
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

TREE = "https://huggingface.co/api/datasets/Abraxasccs/kraken-market-data/tree/main"
RAW  = "https://huggingface.co/datasets/Abraxasccs/kraken-market-data/resolve/main"

TARGET_PAIR = "ETH/USD"
MIN_CLEAN_LEVELS = 5
MAX_LEVELS = 10


def _hf_json(path, retries=4):
    for i in range(retries):
        try:
            req = urllib.request.Request(f"{TREE}/{path}",
                                         headers={"User-Agent": "kraken-hf-loader"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except Exception:
            time.sleep(1.0 + 0.5 * i)
    raise RuntimeError(f"failed to list {path}")


def _pull_file(date, name, retries=3):
    out = SAMPLE_DIR / f"{date}_{name}"
    if out.exists() and out.stat().st_size > 0:
        return out
    for i in range(retries):
        try:
            urllib.request.urlretrieve(
                f"{RAW}/data/crypto/book/{date}/{name}", out)
            return out
        except Exception:
            time.sleep(0.8 + 0.3 * i)
    raise RuntimeError(f"download failed: {date}/{name}")


def list_parquets_for_day(date: str) -> List[str]:
    entries = _hf_json(f"data/crypto/book/{date}")
    names = sorted(
        (e["path"].split("/")[-1] for e in entries
         if e.get("type") == "file" and e["path"].endswith(".parquet")),
        key=lambda nm: int(nm.split(".")[0]),
    )
    return names


def _row_to_features(bids_json: str, asks_json: str):
    """
    Parse one snapshot. Apply ask-cleanliness workaround: filter out ask
    levels whose price < best_bid (Kraken WS reconstruction artifact), then
    take top MAX_LEVELS on each side. Require >=MIN_CLEAN_LEVELS on each side.

    Returns dict with fields matching parse_kaggle_to_ob_features output
    (excluding ts columns, which are set by the caller). None if unusable.
    """
    try:
        bids = json.loads(bids_json)
        asks = json.loads(asks_json)
    except Exception:
        return None
    if not bids or not asks:
        return None
    try:
        best_bid = float(bids[0][0])
    except (IndexError, ValueError, TypeError):
        return None
    clean_asks = [a for a in asks if float(a[0]) >= best_bid]
    if len(clean_asks) < MIN_CLEAN_LEVELS or len(bids) < MIN_CLEAN_LEVELS:
        return None
    bids = bids[:MAX_LEVELS]
    clean_asks = clean_asks[:MAX_LEVELS]

    try:
        bp = np.array([float(b[0]) for b in bids], dtype=np.float64)
        bq = np.array([float(b[1]) for b in bids], dtype=np.float64)
        ap = np.array([float(a[0]) for a in clean_asks], dtype=np.float64)
        aq = np.array([float(a[1]) for a in clean_asks], dtype=np.float64)
    except (ValueError, TypeError, IndexError):
        return None

    best_bid = bp[0]; best_ask = ap[0]
    if best_ask <= 0 or best_bid <= 0:
        return None
    mid = (best_bid + best_ask) / 2.0
    spread = best_ask - best_bid
    spread_bps = spread / mid * 10000 if mid > 0 else 0.0

    use_5 = min(5, len(bp), len(ap))
    use_10 = min(10, len(bp), len(ap))

    bid_depth_5 = float(bq[:use_5].sum())
    ask_depth_5 = float(aq[:use_5].sum())
    bid_depth_10 = float(bq[:use_10].sum())
    ask_depth_10 = float(aq[:use_10].sum())

    total_5 = bid_depth_5 + ask_depth_5
    imbalance_5 = (bid_depth_5 - ask_depth_5) / total_5 if total_5 > 0 else 0.0
    total_10 = bid_depth_10 + ask_depth_10
    imbalance_10 = (bid_depth_10 - ask_depth_10) / total_10 if total_10 > 0 else 0.0

    l1_total = bq[0] + aq[0]
    microprice = ((bp[0] * aq[0] + ap[0] * bq[0]) / l1_total
                  if l1_total > 0 else mid)

    return {
        "mid": mid,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread": spread,
        "spread_bps": spread_bps,
        "imbalance_5": imbalance_5,
        "imbalance_10": imbalance_10,
        "bid_depth_5": bid_depth_5,
        "ask_depth_5": ask_depth_5,
        "bid_depth_10": bid_depth_10,
        "ask_depth_10": ask_depth_10,
        "microprice": microprice,
    }


def load_kraken_hf_day(date: str, verbose: bool = False) -> pd.DataFrame:
    """
    Load every ETH/USD snapshot for `date`, apply ask-cleanliness workaround,
    return feature DataFrame matching parse_kaggle_to_ob_features output.

    Columns: ts_s, ts_ms, mid, best_bid, best_ask, spread, spread_bps,
             imbalance_5, imbalance_10, bid_depth_5, ask_depth_5,
             bid_depth_10, ask_depth_10, microprice.
    Sorted by ts_ms ascending.
    """
    t0 = time.time()
    names = list_parquets_for_day(date)
    if not names:
        return pd.DataFrame()

    # Parallel download (cache-aware; _pull_file short-circuits if cached)
    uncached = [nm for nm in names
                if not (SAMPLE_DIR / f"{date}_{nm}").exists()]
    if uncached:
        with ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(lambda nm: _pull_file(date, nm), uncached))

    rows = []
    for nm in names:
        fp = SAMPLE_DIR / f"{date}_{nm}"
        try:
            df = pd.read_parquet(fp)
        except Exception:
            continue
        eth = df[df["pair"] == TARGET_PAIR]
        if len(eth) == 0:
            continue
        for _, r in eth.iterrows():
            feats = _row_to_features(r["bids_json"], r["asks_json"])
            if feats is None:
                continue
            ts_ms = int(r["ts"])
            feats["ts_ms"] = ts_ms
            feats["ts_s"] = ts_ms / 1000.0
            rows.append(feats)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).sort_values("ts_ms").reset_index(drop=True)
    # Re-order to match parse_kaggle_to_ob_features schema
    cols = ["ts_s", "ts_ms", "mid", "best_bid", "best_ask", "spread",
            "spread_bps", "imbalance_5", "imbalance_10",
            "bid_depth_5", "ask_depth_5", "bid_depth_10", "ask_depth_10",
            "microprice"]
    out = out[cols]
    if verbose:
        dt = time.time() - t0
        print(f"  {date}: {len(out):,} rows from {len(names)} parquets "
              f"({dt:.1f}s)", flush=True)
    return out


def load_kraken_hf_days(dates: List[str], verbose: bool = True) -> pd.DataFrame:
    """Concatenate multiple clean days into a single DataFrame, sorted by ts."""
    parts = []
    for d in dates:
        df = load_kraken_hf_day(d, verbose=verbose)
        if not df.empty:
            parts.append(df)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True).sort_values("ts_ms")
    return out.reset_index(drop=True)


def clean_days_from_report(report_path: Optional[Path] = None,
                           date_range: Optional[tuple] = None) -> List[str]:
    """Read clean-day list from the cleanliness audit report."""
    if report_path is None:
        report_path = (Path(__file__).resolve().parent.parent
                       / "reports" / "kraken_hf_cleanliness.json")
    r = json.loads(Path(report_path).read_text())
    days = list(r["clean_days"])
    if date_range:
        lo, hi = date_range
        days = [d for d in days if lo <= d <= hi]
    return days


if __name__ == "__main__":
    # Smoke test on a single clean day
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else "2026-04-15"
    df = load_kraken_hf_day(d, verbose=True)
    print(f"\n{d}: {len(df):,} clean snapshots")
    if len(df):
        print(f"  time range: {pd.Timestamp(df.ts_ms.iloc[0], unit='ms', tz='UTC')} "
              f"-> {pd.Timestamp(df.ts_ms.iloc[-1], unit='ms', tz='UTC')}")
        print(f"  mid range:  ${df.mid.min():.2f} -> ${df.mid.max():.2f}")
        print(f"  mean spread_bps: {df.spread_bps.mean():.2f}")
        print(f"  mean imb_5:      {df.imbalance_5.mean():.4f}")
