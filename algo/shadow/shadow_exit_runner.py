"""Shadow exit runner — sidecar daemon for the live entropy trader.

Polls the position state every 60s. When a position is open, fetches the most
recent ~1500 1-min PF_ETHUSD bars from Kraken Futures public API, computes
features, runs the exit transformer (NumPy-only), and appends the prediction
to `state/exit_shadow.jsonl`. The main bot is untouched.

Each log entry:
{
  "ts": "2026-04-30T12:34:00Z",
  "minute": 1745423640,
  "position": {"pair": "ETH", "direction": 1, "entry_price": 2235.98, "entry_time": ...},
  "elapsed_min": 12,
  "current_pnl_bps": 4.3,
  "pred_bps": {"5m": 1.2, "15m": 3.1, "60m": -8.4, "240m": -22.1},
  "adj_pred_bps": {"5m": 1.2, "15m": 3.1, "60m": -8.4, "240m": -22.1},   # direction-adjusted
  "would_exit": true,
  "policy": {"horizon": 60, "threshold": 0, "min_hold_min": 5}
}

When a position closes, we emit a `kind: trade_close` entry that summarizes
the trade and the first model-exit signal seen during it.
"""
from __future__ import annotations
import json
import math
import sys
import time
import urllib.request
import urllib.error
import datetime as dt
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from exit_model_numpy import ExitModelNumpy

# Default Pi paths; override via env vars for local testing
import os
ENTROPY_HOME = Path(os.environ.get("ENTROPY_HOME", "/home/user/entropy_trader"))
STATE_DIR = ENTROPY_HOME / "state"
MODEL_NPZ = ENTROPY_HOME / "state" / "exit_transformer" / "exit_model.npz"
SHADOW_LOG = STATE_DIR / "exit_shadow.jsonl"
SHADOW_SUMMARY = STATE_DIR / "exit_shadow_summary.json"

# Policy
HORIZON_BPS = 60        # which forward horizon to use for would_exit decision
THRESHOLD_BPS = 0       # exit if direction*pred[horizon] < threshold
MIN_HOLD_MIN = 5        # don't fire model exit before 5 min into trade
POLL_SECONDS = 60

# Feature config (must match training)
SEQ_LEN = 60
HORIZONS = [60, 240, 360, 720, 1440]
TARGETS = [5, 15, 60, 240]
MAX_HORIZON = max(HORIZONS)
N_BARS_NEEDED = MAX_HORIZON + SEQ_LEN + 5  # safety margin


def fetch_recent_bars(end_unix: int, n: int = 1600) -> np.ndarray:
    """Returns (n, 6) array of [ts_ms, o, h, l, c, v]."""
    start_unix = end_unix - n * 60
    bars = []
    cur_from = start_unix
    iters = 0
    while cur_from < end_unix and iters < 5:
        iters += 1
        url = (f"https://futures.kraken.com/api/charts/v1/trade/PF_ETHUSD/1m"
               f"?from={cur_from}&to={end_unix}")
        req = urllib.request.Request(url, headers={"User-Agent": "shadow-exit"})
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"[shadow] fetch failed: {e}", flush=True)
            return None
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
        more = data.get("more_candles", False)
        last_ts = chunk[-1][0] // 1000
        if not more or last_ts >= end_unix - 60:
            break
        cur_from = last_ts + 60
        time.sleep(0.2)
    if not bars:
        return None
    seen = {b[0]: b for b in bars}
    arr = np.array([seen[k] for k in sorted(seen)], dtype=np.float64)
    return arr


def compute_eval_features(bars: np.ndarray) -> tuple:
    """Compute per-minute sequence + scalar context for the LATEST bar only.

    Mirrors `algo/diagnostics/exit_transformer/data.py`:build_features for a
    single end-of-window evaluation point.

    Returns (seq[SEQ_LEN, 3], scalar[10], current_close).
    """
    closes = bars[:, 4]
    highs = bars[:, 2]
    lows = bars[:, 3]
    vols = bars[:, 5]
    n = len(closes)
    if n < MAX_HORIZON + SEQ_LEN:
        raise ValueError(f"need >= {MAX_HORIZON + SEQ_LEN} bars, have {n}")

    # Per-minute features (computed only for the last SEQ_LEN bars)
    log_ret = np.zeros(n)
    log_ret[1:] = np.log(closes[1:] / closes[:-1])
    hl_range_bps = (highs - lows) / closes * 1e4
    log_vol = np.log(np.maximum(vols, 1e-9))
    # rolling-60 mean of log_vol (causal) -- fast: cumulative sum
    rmean = np.zeros(n)
    cs = np.cumsum(log_vol)
    for i in range(n):
        lo = max(0, i - 59)
        rmean[i] = (cs[i] - (cs[lo - 1] if lo > 0 else 0)) / (i - lo + 1)
    log_vol_z = log_vol - rmean

    seq = np.stack([log_ret[-SEQ_LEN:],
                    hl_range_bps[-SEQ_LEN:],
                    log_vol_z[-SEQ_LEN:]], axis=1).astype(np.float32)

    # Scalar context features at the LATEST index
    i = n - 1
    scalar = np.zeros(2 * len(HORIZONS), dtype=np.float32)
    for k, h in enumerate(HORIZONS):
        scalar[k] = (closes[i] / closes[i - h] - 1.0) * 1e4
        # vol over h
        win = log_ret[i - h + 1: i + 1]
        var = win.var()
        if var > 0:
            scalar[len(HORIZONS) + k] = math.sqrt(var) * math.sqrt(1440 * 365) * 100

    return seq, scalar, float(closes[-1])


def load_state() -> dict:
    p = STATE_DIR / "multi_trader_state.json"
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def append_log(entry: dict) -> None:
    SHADOW_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(SHADOW_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


def update_summary() -> None:
    """Compute aggregate stats over all shadow predictions to date and write
    them to `exit_shadow_summary.json` for the dashboard."""
    decisions = []
    closes = []
    if not SHADOW_LOG.exists():
        # Emit empty heartbeat summary so the dashboard sees the runner is alive
        SHADOW_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
        SHADOW_SUMMARY.write_text(json.dumps({
            "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "runner_state": "awaiting first trade",
            "n_decisions_total": 0,
            "n_trade_closes_total": 0,
            "n_distinct_trades": 0,
            "policy": {
                "horizon_min": HORIZON_BPS, "threshold_bps": THRESHOLD_BPS,
                "min_hold_min": MIN_HOLD_MIN,
            },
            "last_decision": None,
            "model_vs_actual": {
                "model_better_count": 0, "model_worse_count": 0,
                "delta_sum_bps": 0.0, "delta_mean_bps": 0.0,
            },
            "paired_trades": [],
        }, indent=2))
        return
    with open(SHADOW_LOG) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            kind = d.get("kind", "decision")
            if kind == "decision":
                decisions.append(d)
            elif kind == "trade_close":
                closes.append(d)
    # By trade
    by_trade = {}
    for d in decisions:
        oid = d["position"]["order_id"]
        by_trade.setdefault(oid, []).append(d)

    # Pair each closed trade with its model decisions
    paired = []
    model_better_count = 0
    model_worse_count = 0
    delta_sum_bps = 0.0
    for c in closes:
        actual = c.get("actual_trade") or {}
        # Pair by the ENTRY order_id (closed_position.order_id) which matches
        # what was recorded in `by_trade` from decision events. The actual
        # trade row uses a DIFFERENT (exit) order_id from Kraken.
        oid = (c.get("closed_position") or {}).get("order_id")
        if not oid:
            continue
        ds = by_trade.get(oid, [])
        ds.sort(key=lambda r: r["minute"])
        first_exit = next((r for r in ds if r.get("would_exit")), None)
        actual_pnl = actual.get("pnl_bps")
        model_pnl = first_exit["current_pnl_bps"] if first_exit else actual_pnl
        delta = (model_pnl - actual_pnl) if (model_pnl is not None and actual_pnl is not None) else None
        if delta is not None:
            delta_sum_bps += delta
            if delta > 0:
                model_better_count += 1
            elif delta < 0:
                model_worse_count += 1
        paired.append({
            "order_id": oid,
            "direction": (c.get("closed_position") or {}).get("direction"),
            "actual_reason": actual.get("reason"),
            "actual_pnl_bps": actual_pnl,
            "actual_hold_min": actual.get("hold_min"),
            "model_exit_min": first_exit["elapsed_min"] if first_exit else None,
            "model_pnl_bps": model_pnl if first_exit else None,
            "delta_bps": delta,
            "n_decisions": len(ds),
        })

    last_decision = decisions[-1] if decisions else None

    # Determine runner state from last shadow log entry (decision OR trade_close)
    last_log_entry = None
    if SHADOW_LOG.exists():
        with open(SHADOW_LOG) as f:
            for line in f:
                try:
                    last_log_entry = json.loads(line.strip())
                except Exception:
                    pass
    if last_log_entry is None:
        state_msg = "awaiting first trade"
    elif last_log_entry.get("kind") == "decision":
        state_msg = "open position"
    else:
        state_msg = "between trades"

    summary = {
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "runner_state": state_msg,
        "n_decisions_total": len(decisions),
        "n_trade_closes_total": len(closes),
        "n_distinct_trades": len(by_trade),
        "policy": {
            "horizon_min": HORIZON_BPS, "threshold_bps": THRESHOLD_BPS,
            "min_hold_min": MIN_HOLD_MIN,
        },
        "last_decision": last_decision,
        "model_vs_actual": {
            "model_better_count": model_better_count,
            "model_worse_count": model_worse_count,
            "delta_sum_bps": round(delta_sum_bps, 1),
            "delta_mean_bps": round(delta_sum_bps / max(1, len(paired)), 1),
        },
        "paired_trades": paired[-15:],
    }
    SHADOW_SUMMARY.write_text(json.dumps(summary, indent=2))


def run() -> None:
    print(f"[shadow] starting; model={MODEL_NPZ}", flush=True)
    if not MODEL_NPZ.exists():
        print(f"[shadow] FATAL: model file not found at {MODEL_NPZ}", flush=True)
        sys.exit(2)
    model = ExitModelNumpy(MODEL_NPZ)
    print(f"[shadow] loaded model: layers={model.num_layers}  heads={model.n_heads}  "
          f"targets={model.targets}", flush=True)

    last_logged_minute = 0
    last_pos_order_id = None
    last_pos_summary = None  # for trade_close event
    h_idx = TARGETS.index(HORIZON_BPS)

    # Seed a summary at startup so the dashboard sees the runner is alive,
    # even before the first trade.
    try:
        update_summary()
    except Exception:
        pass

    while True:
        try:
            state = load_state()
            pos = state.get("position")

            now = int(time.time())
            current_min = now // 60

            if pos is not None:
                # detect new trade
                if pos.get("order_id") != last_pos_order_id:
                    last_pos_order_id = pos.get("order_id")
                    last_logged_minute = 0
                    last_pos_summary = {
                        "order_id": pos.get("order_id"),
                        "pair": pos.get("pair"),
                        "direction": pos.get("direction"),
                        "entry_price": pos.get("entry_price"),
                        "entry_time": pos.get("entry_time"),
                    }
                    append_log({
                        "kind": "trade_open", "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
                        "position": last_pos_summary,
                    })
                    print(f"[shadow] new trade {pos.get('order_id', '')[:8]} "
                          f"dir={pos.get('direction'):+d} "
                          f"entry={pos.get('entry_price')}", flush=True)

                if current_min > last_logged_minute:
                    # Run prediction
                    bars = fetch_recent_bars(now, n=N_BARS_NEEDED)
                    if bars is not None and len(bars) >= MAX_HORIZON + SEQ_LEN:
                        try:
                            seq, scalar, current_close = compute_eval_features(bars)
                            pred = model.predict(seq, scalar)[0]  # (4,)
                            entry = pos["entry_price"]
                            d = pos["direction"]
                            curr_pnl = d * (current_close / entry - 1.0) * 1e4
                            elapsed_min = (time.time() - pos["entry_time"]) / 60.0
                            adj = float(d * pred[h_idx])
                            would_exit = bool(adj < THRESHOLD_BPS and
                                              elapsed_min >= MIN_HOLD_MIN)

                            entry_log = {
                                "kind": "decision",
                                "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
                                "minute": int(current_min),
                                "position": last_pos_summary,
                                "elapsed_min": round(elapsed_min, 1),
                                "current_close": current_close,
                                "current_pnl_bps": round(float(curr_pnl), 1),
                                "pred_bps": {f"{TARGETS[i]}m": round(float(pred[i]), 1)
                                             for i in range(len(TARGETS))},
                                "adj_pred_60m_bps": round(adj, 1),
                                "would_exit": would_exit,
                                "policy": {"horizon": HORIZON_BPS, "threshold": THRESHOLD_BPS,
                                           "min_hold_min": MIN_HOLD_MIN},
                            }
                            append_log(entry_log)
                            update_summary()
                            print(f"[shadow] {pos.get('order_id', '')[:8]} "
                                  f"min={int(elapsed_min):3d} pnl={curr_pnl:+.1f}bps "
                                  f"pred60={pred[h_idx]:+.1f}bps adj={adj:+.1f} "
                                  f"would_exit={would_exit}", flush=True)
                            last_logged_minute = current_min
                        except Exception as e:
                            print(f"[shadow] inference error: {e}", flush=True)
            else:
                # No position. If we just had one, log a trade_close.
                if last_pos_order_id is not None:
                    # The LAST trade in trade_history.jsonl is almost certainly
                    # the trade we just observed close (its order_id is the
                    # EXIT order id, distinct from our position.order_id which
                    # is the ENTRY id). Pair by entry_price + entry_time match
                    # to be safe in case of unusual timing.
                    hist = STATE_DIR / "trade_history.jsonl"
                    last_trade = None
                    if hist.exists():
                        with open(hist) as f:
                            for line in f:
                                try:
                                    last_trade = json.loads(line)
                                except Exception:
                                    pass
                    matched = None
                    if last_trade and last_pos_summary:
                        ep_match = (abs(last_trade.get("entry_price", 0) -
                                        last_pos_summary.get("entry_price", -1)) < 0.01
                                    and last_trade.get("direction") ==
                                        last_pos_summary.get("direction"))
                        if ep_match:
                            matched = last_trade
                    closing = {
                        "kind": "trade_close",
                        "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
                        "closed_position": last_pos_summary,
                        "actual_trade": matched,
                    }
                    append_log(closing)
                    update_summary()
                    print(f"[shadow] trade closed: {last_pos_order_id[:8]}", flush=True)
                    last_pos_order_id = None
                    last_pos_summary = None
                    last_logged_minute = 0
        except Exception as e:
            print(f"[shadow] loop error: {e}", flush=True)

        # Heartbeat: refresh summary timestamp so dashboard sees we're alive
        try:
            update_summary()
        except Exception:
            pass

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    run()
