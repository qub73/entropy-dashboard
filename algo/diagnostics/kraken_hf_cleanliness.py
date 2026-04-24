"""
Per-snapshot ask-cleanliness sweep for the Abraxasccs/kraken-market-data
L2 book archive. 10 files per day across 120 days.

Labels per ETH/USD snapshot:
  clean                 -- score == 0 (no asks below best_bid)
  light                 -- 0 < score < 0.1 (small subset below bid)
  heavy                 -- score >= 0.1 (large fraction below bid)

Additional "recoverable" flag: after filtering asks<=best_bid out,
at least 5 levels remain. Both clean and recoverable light snapshots
count as "usable" for entropy state reconstruction.

Per-day rollup: clean_day = >=80% of ETH/USD snapshots are usable.

Output: reports/kraken_hf_cleanliness.json
"""
import json, time, urllib.request
from pathlib import Path
from datetime import datetime
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "reports" / "kraken_hf_cleanliness.json"
SAMPLE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "kraken_hf_sample" / "book"
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

TREE = "https://huggingface.co/api/datasets/Abraxasccs/kraken-market-data/tree/main"
RAW  = "https://huggingface.co/datasets/Abraxasccs/kraken-market-data/resolve/main"

N_FILES_PER_DAY = 10
RECOVERABLE_MIN_LEVELS = 5
CLEAN_DAY_USABLE_FRAC = 0.80
HEAVY_SCORE_THRESHOLD = 0.1


def hf_json(path):
    for _ in range(4):
        try:
            req = urllib.request.Request(f"{TREE}/{path}", headers={"User-Agent": "audit"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except Exception:
            time.sleep(1.5)
    raise RuntimeError(f"failed: {path}")


def pull_file(date, name):
    out = SAMPLE_DIR / f"{date}_{name}"
    if out.exists() and out.stat().st_size > 0:
        return out
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(f"{RAW}/data/crypto/book/{date}/{name}", out)
            return out
        except Exception:
            time.sleep(1.0)
    raise RuntimeError(f"download failed: {date}/{name}")


def score_snapshot(bids_json, asks_json):
    """Return (ask_corruption_score, best_bid, clean_ask_levels_after_filter)."""
    bids = json.loads(bids_json); asks = json.loads(asks_json)
    if not bids or not asks:
        return None, None, 0
    best_bid = float(bids[0][0])
    n_below = sum(1 for a in asks if float(a[0]) < best_bid)
    score = n_below / len(asks)
    clean_levels = len(asks) - n_below
    return score, best_bid, clean_levels


def main():
    t0 = time.time()
    print("Listing top-level book dirs...", flush=True)
    top = hf_json("data/crypto/book")
    dates = sorted(e["path"].split("/")[-1] for e in top if e.get("type") == "directory")
    print(f"  {len(dates)} dates: {dates[0]} -> {dates[-1]}", flush=True)

    per_day = {}
    n_done = 0
    for d in dates:
        try:
            entries = hf_json(f"data/crypto/book/{d}")
            parquets = sorted(
                (e["path"].split("/")[-1] for e in entries
                 if e.get("type") == "file" and e["path"].endswith(".parquet")),
                key=lambda nm: int(nm.split(".")[0])
            )
            n = len(parquets)
            if n == 0:
                per_day[d] = {"error": "no parquets"}
                continue
            # Pick 10 spread across the day
            if n <= N_FILES_PER_DAY:
                picks = list(parquets)
            else:
                step = n / N_FILES_PER_DAY
                picks = [parquets[int(i * step)] for i in range(N_FILES_PER_DAY)]

            clean = light = heavy = total = recoverable = 0
            for pname in picks:
                try:
                    fp = pull_file(d, pname)
                    df = pd.read_parquet(fp)
                except Exception as e:
                    continue
                eth = df[df["pair"] == "ETH/USD"]
                for _, r in eth.iterrows():
                    score, _, clean_levels = score_snapshot(
                        r["bids_json"], r["asks_json"])
                    if score is None: continue
                    total += 1
                    if score == 0:
                        clean += 1
                        recoverable += 1
                    elif score < HEAVY_SCORE_THRESHOLD:
                        light += 1
                        if clean_levels >= RECOVERABLE_MIN_LEVELS:
                            recoverable += 1
                    else:
                        heavy += 1
                        if clean_levels >= RECOVERABLE_MIN_LEVELS:
                            recoverable += 1
                time.sleep(0.05)

            usable = clean + (recoverable - clean)  # equals recoverable (clean subset of recoverable)
            usable_frac = (recoverable / total) if total > 0 else 0
            per_day[d] = {
                "parquets_in_day": n,
                "picks": len(picks),
                "eth_snapshots_scored": total,
                "clean":       clean,
                "light":       light,
                "heavy":       heavy,
                "recoverable": recoverable,
                "usable_frac": usable_frac,
                "clean_day":   usable_frac >= CLEAN_DAY_USABLE_FRAC,
            }
            n_done += 1
            if n_done % 20 == 0 or n_done == len(dates):
                clean_days = sum(1 for v in per_day.values()
                                 if v.get("clean_day"))
                elapsed = time.time() - t0
                print(f"  {n_done}/{len(dates)} days  |  "
                      f"clean_days_so_far={clean_days}  |  "
                      f"elapsed={elapsed:.0f}s", flush=True)
        except Exception as e:
            per_day[d] = {"error": str(e)}

    # Summary
    clean_days = [d for d, v in per_day.items() if v.get("clean_day")]
    corrupt_days = [d for d, v in per_day.items()
                    if v.get("clean_day") is False]
    error_days = [d for d, v in per_day.items() if "error" in v]

    total_snaps = sum(v.get("eth_snapshots_scored", 0) for v in per_day.values())
    total_clean = sum(v.get("clean", 0) for v in per_day.values())
    total_light = sum(v.get("light", 0) for v in per_day.values())
    total_heavy = sum(v.get("heavy", 0) for v in per_day.values())
    total_recov = sum(v.get("recoverable", 0) for v in per_day.values())

    out = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "n_files_per_day_sampled": N_FILES_PER_DAY,
        "clean_day_threshold": CLEAN_DAY_USABLE_FRAC,
        "recoverable_min_levels": RECOVERABLE_MIN_LEVELS,
        "heavy_score_threshold": HEAVY_SCORE_THRESHOLD,
        "date_range": [dates[0], dates[-1]],
        "n_days_total": len(dates),
        "n_clean_days": len(clean_days),
        "n_corrupt_days": len(corrupt_days),
        "n_error_days": len(error_days),
        "clean_days": clean_days,
        "corrupt_days": corrupt_days,
        "error_days": error_days,
        "aggregate": {
            "total_eth_snapshots": total_snaps,
            "clean": total_clean,
            "light": total_light,
            "heavy": total_heavy,
            "recoverable": total_recov,
            "clean_frac_all_snaps": total_clean / max(1, total_snaps),
            "recoverable_frac_all_snaps": total_recov / max(1, total_snaps),
        },
        "per_day": per_day,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))

    print("\n=== CLEANLINESS SUMMARY ===")
    print(f"  days total:        {out['n_days_total']}")
    print(f"  clean days:        {out['n_clean_days']}")
    print(f"  corrupt days:      {out['n_corrupt_days']}")
    print(f"  error days:        {out['n_error_days']}")
    print(f"  total ETH snaps:   {total_snaps:,}")
    print(f"  clean snaps:       {total_clean:,} ({total_clean/max(1,total_snaps)*100:.1f}%)")
    print(f"  light snaps:       {total_light:,}")
    print(f"  heavy snaps:       {total_heavy:,}")
    print(f"  recoverable snaps: {total_recov:,} ({total_recov/max(1,total_snaps)*100:.1f}%)")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
