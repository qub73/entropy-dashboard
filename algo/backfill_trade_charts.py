"""
One-shot: fetch ±1h price bars around each historical trade from Kraken
public OHLC and save them to state/trade_charts.json so the dashboard
can render the zoom chart for old trades.

Intervals chosen automatically by trade age (Kraken returns up to 720
bars per request):
    < 12 h:  1-min
    < 60 h:  5-min
    < 180 h: 15-min
    else:    60-min

Output format:
    { "<order_id>": {
          "entry_ts": int_ms,
          "exit_ts":  int_ms,
          "interval_min": int,
          "source":   "kraken",
          "bars": [[ts_ms, open, high, low, close, volume], ...]
      }, ... }
"""
import json, time, urllib.request
from pathlib import Path
from datetime import datetime

STATE = Path(__file__).parent / "state"
HIST = STATE / "trade_history.jsonl"
OUT = STATE / "trade_charts.json"


def fetch_kraken(interval_min, since_sec, max_attempts=5):
    url = (f"https://api.kraken.com/0/public/OHLC?"
           f"pair=ETHUSD&interval={interval_min}&since={since_sec}")
    delay = 2.0
    last_err = None
    for attempt in range(max_attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            resp = urllib.request.urlopen(req, timeout=30)
            data = json.loads(resp.read())
            err = data.get("error")
            if err and any("Too many" in str(e) for e in err):
                time.sleep(delay); delay = min(delay * 2, 30); last_err = err
                continue
            if err:
                raise RuntimeError(f"kraken: {err}")
            for k, v in data["result"].items():
                if k != "last":
                    return v
            return []
        except Exception as e:
            last_err = str(e)
            time.sleep(delay); delay = min(delay * 2, 30)
    raise RuntimeError(f"kraken failed after {max_attempts}: {last_err}")


def pick_interval(age_h):
    if age_h < 12:   return 1
    if age_h < 60:   return 5
    if age_h < 180:  return 15
    return 60


def load_trades():
    if not HIST.exists():
        return []
    out = []
    with open(HIST) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out


def main():
    trades = load_trades()
    if not trades:
        print("No trades to backfill.")
        return
    existing = {}
    if OUT.exists():
        try:
            existing = json.load(open(OUT))
        except Exception:
            existing = {}

    now_s = time.time()
    n_new = 0
    for i, t in enumerate(trades, 1):
        order_id = t.get("order_id") or f"idx_{i}"
        if order_id in existing and existing[order_id].get("bars"):
            continue
        exit_ts = datetime.fromisoformat(t["time"]).timestamp()
        entry_ts = exit_ts - (t.get("hold_min", 0) * 60)
        # window ±1h around the trade
        w_start = entry_ts - 3600
        w_end = exit_ts + 3600
        age_h = (now_s - exit_ts) / 3600
        interval = pick_interval(age_h)
        # request enough history: since = window start, capped to not overflow 720 bars
        max_span = 720 * interval * 60
        since_s = int(max(w_start, now_s - max_span))
        print(f"[{i}/{len(trades)}] {t.get('pair','ETH')} "
              f"{datetime.fromisoformat(t['time']).strftime('%Y-%m-%d %H:%M')} "
              f"age={age_h:.1f}h -> {interval}m", flush=True)
        try:
            raw = fetch_kraken(interval, since_s)
        except Exception as e:
            print(f"    ERROR: {e}")
            continue
        # Keep only bars in the ±1h window, dropping anything outside
        bars = []
        for row in raw:
            ts = int(row[0]) * 1000
            if w_start * 1000 <= ts <= w_end * 1000:
                # [ts_ms, o, h, l, c, v] (Kraken index 6 is volume)
                bars.append([ts, float(row[1]), float(row[2]),
                             float(row[3]), float(row[4]), float(row[6])])
        if not bars:
            print(f"    WARN: no bars in window; got {len(raw)} raw bars")
        existing[order_id] = {
            "entry_ts": int(entry_ts * 1000),
            "exit_ts":  int(exit_ts * 1000),
            "interval_min": interval,
            "source": "kraken",
            "bars": bars,
        }
        n_new += 1
        time.sleep(1.5)  # politeness: avoid Kraken rate limit (1s limit per IP)

    STATE.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(existing, f, separators=(",", ":"))
    total_bars = sum(len(v.get("bars", [])) for v in existing.values())
    print(f"\nSaved {len(existing)} trades ({n_new} new) to {OUT}")
    print(f"Total bars: {total_bars}, file size: "
          f"{OUT.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
