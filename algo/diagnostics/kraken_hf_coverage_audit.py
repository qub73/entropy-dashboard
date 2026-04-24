"""
Sprint v1.5 re-open -- coverage audit of Abraxasccs/kraken-market-data
data/crypto/book/ subdirs.

My prior pass reported 151 "empty" date subdirs but HF's tree API
returns size=0 for directory entries regardless of contents. This
pass recurses into each date dir and counts parquet files per day.

Output: reports/kraken_hf_coverage.json
"""
import json, time, urllib.request
from pathlib import Path
from datetime import datetime
import sys

OUT = Path(__file__).resolve().parent.parent / "reports" / "kraken_hf_coverage.json"
BASE_TREE = "https://huggingface.co/api/datasets/Abraxasccs/kraken-market-data/tree/main"
BASE_BOOK = "data/crypto/book"


def hf_tree(path):
    """Fetch a single level of tree listing. Returns list of dict entries."""
    url = f"{BASE_TREE}/{path}"
    for _ in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "audit/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except Exception as e:
            time.sleep(1.5)
    raise RuntimeError(f"failed to fetch {url}")


def main():
    print(f"Listing {BASE_BOOK}/ ...", flush=True)
    top = hf_tree(BASE_BOOK)
    date_dirs = [e["path"].split("/")[-1] for e in top if e.get("type") == "directory"]
    date_dirs.sort()
    print(f"  {len(date_dirs)} date dirs: {date_dirs[0]} -> {date_dirs[-1]}", flush=True)

    per_day = {}
    n_done = 0
    print("\nRecursing into each date dir...", flush=True)
    for d in date_dirs:
        try:
            entries = hf_tree(f"{BASE_BOOK}/{d}")
            files = [e for e in entries if e.get("type") == "file"]
            parquets = [e for e in files if e["path"].endswith(".parquet")]
            per_day[d] = {
                "n_files": len(files),
                "n_parquets": len(parquets),
                "parquet_names": [p["path"].split("/")[-1] for p in parquets][:20],
                "total_parquet_bytes": sum(p.get("size", 0) for p in parquets),
            }
        except Exception as e:
            per_day[d] = {"error": str(e)}
        n_done += 1
        if n_done % 30 == 0:
            print(f"  {n_done}/{len(date_dirs)} dirs audited "
                  f"({sum(1 for v in per_day.values() if v.get('n_parquets', 0) > 0)} with data so far)",
                  flush=True)
        time.sleep(0.1)  # gentle to HF API

    # Summary
    days_with_data = [d for d, v in per_day.items()
                      if v.get("n_parquets", 0) > 0]
    days_empty = [d for d in date_dirs if d not in days_with_data]
    total_parquets = sum(v.get("n_parquets", 0) for v in per_day.values())
    total_bytes = sum(v.get("total_parquet_bytes", 0) for v in per_day.values())

    # Sample 3 dates across the range for deeper inspection (pick from with-data)
    sample_dates = []
    if len(days_with_data) >= 3:
        idx = [0, len(days_with_data)//2, len(days_with_data)-1]
        sample_dates = [days_with_data[i] for i in idx]
    elif days_with_data:
        sample_dates = list(days_with_data)

    out = {
        "audit_generated_at": datetime.utcnow().isoformat() + "Z",
        "date_range": [date_dirs[0], date_dirs[-1]] if date_dirs else [None, None],
        "n_days_total": len(date_dirs),
        "n_days_with_data": len(days_with_data),
        "n_days_empty": len(days_empty),
        "total_parquets": total_parquets,
        "total_parquet_bytes": total_bytes,
        "sample_dates_for_deeper_inspection": sample_dates,
        "per_day_counts": {d: v.get("n_parquets", 0) for d, v in per_day.items()},
        "per_day_sizes": {d: v.get("total_parquet_bytes", 0) for d, v in per_day.items()},
        "days_empty": days_empty,
        "first_5_data_days": days_with_data[:5],
        "last_5_data_days": days_with_data[-5:],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))

    print(f"\n=== SUMMARY ===")
    print(f"  date range:        {out['date_range'][0]} .. {out['date_range'][1]}")
    print(f"  days total:        {out['n_days_total']}")
    print(f"  days WITH data:    {out['n_days_with_data']}")
    print(f"  days empty:        {out['n_days_empty']}")
    print(f"  total parquets:    {out['total_parquets']:,}")
    print(f"  total size:        {out['total_parquet_bytes']/1e9:.2f} GB")
    print(f"  sample dates:      {sample_dates}")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
