"""Fetch ~60 days of PF_ETHUSD 1-min OHLCV from Kraken Futures public API.

Output: D:/ai/timesFM/algo/state/pi_pull_2026_04_30/eth_1m_60d.json
        as a JSON array of [ts_ms, o, h, l, c, v].
"""
import json
import urllib.request
import time
import datetime as dt
from pathlib import Path

OUT = Path("D:/ai/timesFM/algo/state/pi_pull_2026_04_30/eth_1m_60d.json")
DAYS_BACK = 60


def fetch_paginated(start_unix: int, end_unix: int) -> list:
    bars = []
    cur_from = start_unix
    iters = 0
    while cur_from < end_unix and iters < 200:
        iters += 1
        url = (f"https://futures.kraken.com/api/charts/v1/trade/PF_ETHUSD/1m"
               f"?from={cur_from}&to={end_unix}")
        req = urllib.request.Request(url, headers={"User-Agent": "fetch_eth_1m"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())
        candles = data.get("candles", [])
        if not candles:
            break
        chunk = []
        for c in candles:
            chunk.append([
                int(c["time"]),
                float(c["open"]),
                float(c["high"]),
                float(c["low"]),
                float(c["close"]),
                float(c.get("volume", 0)),
            ])
        chunk.sort(key=lambda b: b[0])
        bars.extend(chunk)
        last_ts = chunk[-1][0] // 1000
        more = data.get("more_candles", False)
        if not more or last_ts >= end_unix - 60:
            break
        cur_from = last_ts + 60
        if iters % 5 == 0:
            print(f"  iter {iters}: have {len(bars)} bars, "
                  f"last_ts={dt.datetime.fromtimestamp(last_ts, tz=dt.timezone.utc)}")
        time.sleep(0.25)
    seen = {}
    for b in bars:
        seen[b[0]] = b
    return sorted(seen.values(), key=lambda b: b[0])


def main():
    end_unix = int(time.time())
    start_unix = end_unix - DAYS_BACK * 86400
    print(f"Fetching {DAYS_BACK}d of PF_ETHUSD 1m: "
          f"{dt.datetime.fromtimestamp(start_unix, tz=dt.timezone.utc)} to "
          f"{dt.datetime.fromtimestamp(end_unix, tz=dt.timezone.utc)}")

    bars = fetch_paginated(start_unix, end_unix)
    print(f"Total bars: {len(bars)}")
    if bars:
        print(f"  span: {dt.datetime.fromtimestamp(bars[0][0]/1000, tz=dt.timezone.utc)} "
              f"to {dt.datetime.fromtimestamp(bars[-1][0]/1000, tz=dt.timezone.utc)}")
        # Check for gaps
        ts_diffs = [bars[i+1][0] - bars[i][0] for i in range(len(bars) - 1)]
        big_gaps = [(i, d) for i, d in enumerate(ts_diffs) if d > 120_000]
        print(f"  gaps > 2min: {len(big_gaps)}")
        if big_gaps[:3]:
            for i, d in big_gaps[:3]:
                t = dt.datetime.fromtimestamp(bars[i][0]/1000, tz=dt.timezone.utc)
                print(f"    gap of {d/60000:.1f}m at {t}")

    OUT.write_text(json.dumps(bars))
    print(f"Saved to {OUT}  ({OUT.stat().st_size/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
