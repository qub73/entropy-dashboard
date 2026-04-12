"""
Multi-exchange entropy engine.
Reuses the 27-state Markov chain from ob_entropy but runs one engine per
(exchange, symbol) pair in parallel. Designed for real-time use.
"""

import time
import threading
from collections import deque
from typing import Dict, Optional
import numpy as np

NUM_STATES = 27


def _state_index(sign: int, imb_tercile: int, spread_regime: int) -> int:
    sign_idx = {-1: 0, 0: 1, 1: 2}[sign]
    return sign_idx * 9 + (imb_tercile - 1) * 3 + (spread_regime - 1)


class EntropyEngine:
    """Per-pair rolling 27-state Markov entropy. Thread-safe via internal lock."""

    def __init__(self, window: int = 30):
        self.window = window
        self.mids = deque(maxlen=window + 200)
        self.imbalances = deque(maxlen=window + 200)
        self.spreads_bps = deque(maxlen=window + 200)
        self.states = deque(maxlen=window + 200)
        self.counts = np.zeros((NUM_STATES, NUM_STATES), dtype=np.float64)
        self.bar_count = 0
        self.last_H: Optional[float] = None
        self.last_bar_ts = 0.0
        self._lock = threading.Lock()

    def on_bar(self, mid: float, imbalance: float, spread_bps: float,
               ts: Optional[float] = None):
        """Process one 1-minute bar aggregate."""
        with self._lock:
            self.mids.append(mid)
            self.imbalances.append(imbalance)
            self.spreads_bps.append(spread_bps)
            self.bar_count += 1
            self.last_bar_ts = ts or time.time()

            if len(self.mids) < 2:
                return None

            price_change = self.mids[-1] - self.mids[-2]
            sign = 1 if price_change > 0 else (-1 if price_change < 0 else 0)

            if len(self.imbalances) >= 30:
                arr = np.array(list(self.imbalances)[-min(self.window, len(self.imbalances)):])
                p33, p67 = np.percentile(arr, [33.3, 66.7])
                imb_t = 1 if imbalance <= p33 else (3 if imbalance >= p67 else 2)
            else:
                imb_t = 2

            if len(self.spreads_bps) >= 30:
                arr = np.array(list(self.spreads_bps)[-min(self.window, len(self.spreads_bps)):])
                sp33, sp67 = np.percentile(arr, [33.3, 66.7])
                spr_r = 1 if spread_bps <= sp33 else (3 if spread_bps >= sp67 else 2)
            else:
                spr_r = 2

            s = _state_index(sign, imb_t, spr_r)
            self.states.append(s)

            if len(self.states) < 2:
                return None

            self.counts[self.states[-2], self.states[-1]] += 1.0
            if len(self.states) > self.window:
                old = list(self.states)
                idx = len(old) - self.window - 1
                if idx >= 0 and idx + 1 < len(old):
                    self.counts[old[idx], old[idx + 1]] -= 1.0
                    self.counts[old[idx], old[idx + 1]] = max(
                        0, self.counts[old[idx], old[idx + 1]])

            if self.bar_count < self.window:
                return None

            P = self.counts + 0.01
            row_sums = P.sum(axis=1)
            for r in range(NUM_STATES):
                if row_sums[r] <= 0.01 * NUM_STATES + 1e-10:
                    P[r, :] = 1.0 / NUM_STATES
                else:
                    P[r, :] /= row_sums[r]

            pi = np.ones(NUM_STATES) / NUM_STATES
            for _ in range(50):
                pi_new = pi @ P
                if np.max(np.abs(pi_new - pi)) < 1e-10:
                    break
                pi = pi_new
            pi = np.abs(pi)
            s_sum = pi.sum()
            pi = pi / s_sum if s_sum > 1e-12 else np.ones(NUM_STATES) / NUM_STATES

            log_norm = np.log(NUM_STATES)
            P_safe = np.where(P > 1e-15, P, 1e-15)
            inner = np.sum(P * np.log(P_safe) * (P > 1e-15), axis=1)
            H = float(-np.dot(pi, inner) / log_norm)
            self.last_H = H
            return H

    def get_trailing_return_bps(self, lookback: int = 5):
        with self._lock:
            if len(self.mids) < lookback + 1:
                return None
            current = self.mids[-1]
            past = list(self.mids)[-lookback - 1]
            if past <= 0:
                return None
            return (current / past - 1.0) * 10000


class MultiExchangeEntropy:
    """
    Holds one EntropyEngine per (exchange, symbol) and per-bar aggregation
    buffers filled from live tick sampling.
    """

    def __init__(self, window: int = 30):
        self.window = window
        self.engines: Dict[tuple, EntropyEngine] = {}
        self.tick_buffers: Dict[tuple, dict] = {}
        self.last_bar_minute: Dict[tuple, int] = {}

    def add_pair(self, exchange: str, symbol: str):
        key = (exchange, symbol)
        self.engines[key] = EntropyEngine(window=self.window)
        self.tick_buffers[key] = {"mids": [], "imbs": [], "spreads": []}
        self.last_bar_minute[key] = 0

    def on_tick(self, exchange: str, symbol: str,
                mid: float, imbalance: float, spread_bps: float):
        """Called for each new orderbook tick; returns H if a new bar completed."""
        key = (exchange, symbol)
        if key not in self.engines:
            return None

        buf = self.tick_buffers[key]
        buf["mids"].append(mid)
        buf["imbs"].append(imbalance)
        buf["spreads"].append(spread_bps)

        current_minute = int(time.time()) // 60
        last_min = self.last_bar_minute[key]

        if current_minute > last_min and last_min > 0:
            # Flush bar
            if buf["mids"]:
                bar_mid = buf["mids"][-1]
                bar_imb = float(np.mean(buf["imbs"]))
                bar_spread = float(np.mean(buf["spreads"]))
                H = self.engines[key].on_bar(bar_mid, bar_imb, bar_spread)
                buf["mids"].clear()
                buf["imbs"].clear()
                buf["spreads"].clear()
                self.last_bar_minute[key] = current_minute
                return H

        self.last_bar_minute[key] = current_minute
        return None

    def get_H(self, exchange: str, symbol: str) -> Optional[float]:
        engine = self.engines.get((exchange, symbol))
        return engine.last_H if engine else None

    def get_snapshot(self) -> dict:
        """Returns {exchange_symbol: {H, bars, mid_last_bar, ...}} for dashboard."""
        out = {}
        for (ex, sym), engine in self.engines.items():
            out[f"{ex}:{sym}"] = {
                "H": engine.last_H,
                "bars": engine.bar_count,
                "last_bar_ts": engine.last_bar_ts,
            }
        return out
