"""
Step 2: snapshot density + schema consistency check.

Pull 3 sample date dirs (spread across the 120-day range), inspect:
  - files per day
  - first/last timestamps
  - ETH/USD snapshot inter-arrival time
  - depth consistency (always 100 levels?)
  - ask-ordering quirk (are first N asks below best bid?)
"""
import json, time, urllib.request, sys, statistics
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLE_DIR = ROOT / "data" / "kraken_hf_sample" / "book"
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

TREE = "https://huggingface.co/api/datasets/Abraxasccs/kraken-market-data/tree/main"
RAW  = "https://huggingface.co/datasets/Abraxasccs/kraken-market-data/resolve/main"

SAMPLE_DATES = ["2025-12-17", "2026-02-15", "2026-04-15"]  # spread across range
TARGET_PAIR = "ETH/USD"


def list_files(date):
    url = f"{TREE}/data/crypto/book/{date}"
    req = urllib.request.Request(url, headers={"User-Agent": "audit"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def pull_file(date, name):
    url = f"{RAW}/data/crypto/book/{date}/{name}"
    out = SAMPLE_DIR / f"{date}_{name}"
    if out.exists() and out.stat().st_size > 0:
        return out
    urllib.request.urlretrieve(url, out)
    return out


def main():
    report = {}
    for date in SAMPLE_DATES:
        print(f"\n=== {date} ===", flush=True)
        entries = list_files(date)
        parquets = sorted(
            [e for e in entries if e.get("type") == "file" and e["path"].endswith(".parquet")],
            key=lambda e: e["path"]
        )
        first_name = parquets[0]["path"].split("/")[-1]
        last_name  = parquets[-1]["path"].split("/")[-1]
        print(f"  files:  {len(parquets)}")
        print(f"  first:  {first_name}  ({parquets[0].get('size',0):,}b)")
        print(f"  last:   {last_name}   ({parquets[-1].get('size',0):,}b)")

        # Pull first and last parquet only (no need for all of them at this stage)
        first_path = pull_file(date, first_name)
        last_path  = pull_file(date, last_name) if last_name != first_name else first_path
        df_first = pd.read_parquet(first_path)
        df_last  = pd.read_parquet(last_path)
        print(f"  first parquet rows: {len(df_first)}, cols: {list(df_first.columns)}")
        print(f"  pairs in first file: {sorted(df_first['pair'].unique().tolist())}")

        t_first = int(df_first["ts"].min())
        t_last  = int(df_last["ts"].max())
        print(f"  first snapshot ts: {t_first} "
              f"({pd.Timestamp(t_first, unit='ms', tz='UTC')})")
        print(f"  last  snapshot ts: {t_last} "
              f"({pd.Timestamp(t_last, unit='ms', tz='UTC')})")

        # ETH/USD snapshot inter-arrival: need to scan every file for this date.
        # Sample 8 files evenly to estimate.
        n = len(parquets)
        sample_idxs = sorted(set([0, n//7, 2*n//7, 3*n//7, 4*n//7, 5*n//7, 6*n//7, n-1]))
        sample_idxs = [i for i in sample_idxs if 0 <= i < n]
        eth_ts = []
        pair_counts = {}
        depths_bid = []; depths_ask = []
        ask_below_bid_count = 0
        total_rows = 0
        for i in sample_idxs:
            nm = parquets[i]["path"].split("/")[-1]
            fp = pull_file(date, nm)
            df = pd.read_parquet(fp)
            total_rows += len(df)
            for p in df["pair"].unique():
                pair_counts[p] = pair_counts.get(p, 0) + int((df["pair"] == p).sum())
            eth_sub = df[df["pair"] == TARGET_PAIR]
            for _, row in eth_sub.iterrows():
                eth_ts.append(int(row["ts"]))
                bids = json.loads(row["bids_json"])
                asks = json.loads(row["asks_json"])
                depths_bid.append(len(bids)); depths_ask.append(len(asks))
                if bids and asks:
                    best_bid = float(bids[0][0])
                    # count asks below best bid
                    below = sum(1 for a in asks[:10] if float(a[0]) < best_bid)
                    if below > 0:
                        ask_below_bid_count += 1
            time.sleep(0.1)
        eth_ts.sort()
        gaps = [eth_ts[i+1]-eth_ts[i] for i in range(len(eth_ts)-1)]
        gap_stats = None
        if gaps:
            gap_stats = {
                "min_ms": min(gaps),
                "p50_ms": int(statistics.median(gaps)),
                "p90_ms": int(sorted(gaps)[int(len(gaps)*0.9)]),
                "max_ms": max(gaps),
                "mean_ms": int(sum(gaps)/len(gaps)),
            }
        print(f"  sampled {len(sample_idxs)} parquets out of {n}; total rows: {total_rows}")
        print(f"  pair counts across samples: {pair_counts}")
        print(f"  ETH/USD snapshots in sample: {len(eth_ts)}")
        if gap_stats:
            print(f"  ETH/USD gap (ms): "
                  f"min={gap_stats['min_ms']}  p50={gap_stats['p50_ms']}  "
                  f"p90={gap_stats['p90_ms']}  max={gap_stats['max_ms']}  "
                  f"mean={gap_stats['mean_ms']}")
        if depths_bid:
            print(f"  bid depths seen: min={min(depths_bid)} max={max(depths_bid)} "
                  f"unique={len(set(depths_bid))}")
            print(f"  ask depths seen: min={min(depths_ask)} max={max(depths_ask)} "
                  f"unique={len(set(depths_ask))}")
        print(f"  rows with at least one ask price below best-bid: "
              f"{ask_below_bid_count} / {len(depths_bid)}")

        # Persist per-date
        report[date] = {
            "n_files_day": n,
            "first_parquet": first_name,
            "last_parquet":  last_name,
            "first_ts_ms": t_first,
            "last_ts_ms": t_last,
            "day_span_minutes": (t_last - t_first) / 60000.0,
            "sampled_parquets": len(sample_idxs),
            "total_sampled_rows": total_rows,
            "pair_counts_in_sample": pair_counts,
            "eth_usd_snapshots_in_sample": len(eth_ts),
            "eth_usd_gap_stats_ms": gap_stats,
            "bid_depth_range": [min(depths_bid), max(depths_bid)] if depths_bid else None,
            "ask_depth_range": [min(depths_ask), max(depths_ask)] if depths_ask else None,
            "bid_depth_unique_values": sorted(set(depths_bid)) if depths_bid else [],
            "ask_depth_unique_values": sorted(set(depths_ask)) if depths_ask else [],
            "rows_with_asks_below_bid": ask_below_bid_count,
            "rows_inspected_for_ordering": len(depths_bid),
        }

    # Show a sample row verbatim for human-eyeball
    print(f"\n=== sample ETH/USD snapshot (from 2026-02-15 first parquet) ===")
    df = pd.read_parquet(list(SAMPLE_DIR.glob("2026-02-15_*.parquet"))[0])
    eth = df[df["pair"] == TARGET_PAIR].head(1)
    if len(eth):
        row = eth.iloc[0]
        bids = json.loads(row["bids_json"])
        asks = json.loads(row["asks_json"])
        print(f"  ts: {row['ts']} ({pd.Timestamp(int(row['ts']), unit='ms', tz='UTC')})")
        print(f"  first 5 bids: {bids[:5]}")
        print(f"  first 5 asks: {asks[:5]}")
        if bids and asks:
            bb = float(bids[0][0]); ba = float(asks[0][0])
            print(f"  best bid={bb}, best ask={ba}, spread={(ba-bb):.2f} "
                  f"({(ba-bb)/bb*10000:.1f} bps)")

    out = Path(__file__).resolve().parent.parent / "reports" / "kraken_hf_schema_inspect.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
