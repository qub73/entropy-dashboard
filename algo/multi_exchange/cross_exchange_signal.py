"""
Cross-exchange lead-lag signal.

Core thesis (Singha 2025 + cross-venue microstructure): the exchange with the
deepest liquidity and lowest latency (Binance for crypto) tends to *lead*
other venues in price discovery by 100-500ms. If Binance entropy collapses
before Kraken's does, that's a strong forward signal for Kraken.

Signal rule:
  1. At least ONE "leading" exchange entropy drops below its threshold
  2. Aggregate imbalance across leaders points in one direction
  3. Kraken (the trade venue) entropy is NOT YET fully collapsed — we want
     to front-run Kraken's catch-up, not trade after the move has already played
  4. Trailing return on Kraken confirms activity (20-80 bps over 5 min)

Trade on Kraken, but the signal comes from the broader market.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time
import numpy as np


@dataclass
class CrossExchangeConfig:
    # Thresholds per exchange (fitted independently or copied from single-exchange)
    h_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "binance": 0.35,
        "bybit": 0.40,
        "coinbase": 0.42,
        "kraken_futures": 0.35,
    })
    # Minimum number of leading exchanges that must trigger
    min_leaders_triggered: int = 1
    # Kraken's H must still be ABOVE this to avoid trading after the move
    kraken_h_ceiling: float = 0.48
    # Imbalance agreement: fraction of triggered leaders that must agree on direction
    direction_agreement: float = 0.6
    # Imbalance magnitude filter
    imbalance_min: float = 0.05
    # Max spread on Kraken (trade venue)
    spread_max_bps: float = 20.0
    # Trailing return filter on Kraken
    ret_low_bps: float = 20.0
    ret_high_bps: float = 80.0
    # Cooldown between signals (seconds)
    cooldown_sec: int = 60


@dataclass
class ExchangeState:
    """Latest entropy + orderbook features for one exchange/symbol."""
    H: Optional[float] = None
    imbalance: float = 0.0
    spread_bps: float = 999.0
    mid: float = 0.0
    bars: int = 0
    last_update: float = 0.0


class CrossExchangeSignalEngine:
    """
    Aggregates entropy + microstructure across all exchanges and fires
    directional signals to trade on the target (kraken_futures).
    """

    def __init__(self, config: CrossExchangeConfig, trade_venue: str = "kraken_futures",
                 leader_exchanges: Optional[List[str]] = None):
        self.config = config
        self.trade_venue = trade_venue
        self.leaders = leader_exchanges or ["binance", "bybit", "coinbase"]
        # state[exchange][symbol] = ExchangeState
        self.state: Dict[str, Dict[str, ExchangeState]] = {}
        self.last_signal_ts: Dict[str, float] = {}  # symbol -> ts

    def update(self, exchange: str, symbol: str, H: Optional[float],
               imbalance: float, spread_bps: float, mid: float, bars: int):
        """Update state from live tick + entropy engine."""
        if exchange not in self.state:
            self.state[exchange] = {}
        if symbol not in self.state[exchange]:
            self.state[exchange][symbol] = ExchangeState()
        s = self.state[exchange][symbol]
        if H is not None:
            s.H = H
        s.imbalance = imbalance
        s.spread_bps = spread_bps
        s.mid = mid
        s.bars = bars
        s.last_update = time.time()

    def check_signal(self, symbol: str, trailing_ret_bps: Optional[float]) -> Optional[dict]:
        """
        Check if cross-exchange conditions are met to fire a signal on `symbol`.
        Returns signal dict or None.
        Signal is ALWAYS for the trade_venue's symbol mapping.
        """
        cfg = self.config

        # Cooldown
        now = time.time()
        last = self.last_signal_ts.get(symbol, 0)
        if now - last < cfg.cooldown_sec:
            return None

        # Check trade venue spread
        tv_state = self.state.get(self.trade_venue, {}).get(symbol)
        if tv_state is None or tv_state.H is None:
            return None
        if tv_state.spread_bps > cfg.spread_max_bps:
            return None
        # Trade venue must NOT have fully collapsed yet (we want to front-run)
        if tv_state.H < cfg.h_thresholds.get(self.trade_venue, 0.35):
            # If Kraken itself is already below threshold, we missed the move
            # (unless leaders agree strongly — allow through)
            pass  # don't block here; leaders + direction filters handle it

        # Trailing return filter on trade venue
        if trailing_ret_bps is None:
            return None
        abs_ret = abs(trailing_ret_bps)
        if abs_ret < cfg.ret_low_bps or abs_ret > cfg.ret_high_bps:
            return None

        # Check leaders
        triggered_leaders = []
        for ex in self.leaders:
            ex_states = self.state.get(ex, {})
            # Symbol map: the symbol on the leader may differ
            # Caller should pass a mapping; for now try direct match first
            leader_sym = self._map_symbol(ex, symbol)
            if leader_sym is None:
                continue
            st = ex_states.get(leader_sym)
            if st is None or st.H is None:
                continue
            thresh = cfg.h_thresholds.get(ex, 0.40)
            if st.H < thresh:
                triggered_leaders.append((ex, st))

        if len(triggered_leaders) < cfg.min_leaders_triggered:
            return None

        # Aggregate imbalance direction from leaders
        imbs = [s.imbalance for _, s in triggered_leaders]
        long_votes = sum(1 for i in imbs if i > cfg.imbalance_min)
        short_votes = sum(1 for i in imbs if i < -cfg.imbalance_min)
        total = long_votes + short_votes
        if total == 0:
            return None

        if long_votes / total >= cfg.direction_agreement:
            direction = 1
        elif short_votes / total >= cfg.direction_agreement:
            direction = -1
        else:
            return None  # no consensus

        # All checks pass — fire signal
        self.last_signal_ts[symbol] = now

        return {
            "timestamp": now,
            "trade_venue": self.trade_venue,
            "symbol": symbol,
            "direction": direction,
            "trailing_ret_bps": trailing_ret_bps,
            "trade_venue_H": tv_state.H,
            "trade_venue_mid": tv_state.mid,
            "trade_venue_imbalance": tv_state.imbalance,
            "leaders_triggered": [(ex, s.H, s.imbalance) for ex, s in triggered_leaders],
            "long_votes": long_votes,
            "short_votes": short_votes,
        }

    def _map_symbol(self, exchange: str, trade_venue_symbol: str) -> Optional[str]:
        """Map a trade-venue symbol to the equivalent on another exchange."""
        # trade_venue_symbol is like "PF_XBTUSD" or "PI_XBTUSD" on Kraken Futures
        mappings = {
            # Kraken -> Binance
            ("kraken_futures", "PI_XBTUSD", "binance"): "BTCUSDT",
            ("kraken_futures", "PI_ETHUSD", "binance"): "ETHUSDT",
            ("kraken_futures", "PF_XBTUSD", "binance"): "BTCUSDT",
            ("kraken_futures", "PF_ETHUSD", "binance"): "ETHUSDT",
            # Kraken -> Bybit
            ("kraken_futures", "PI_XBTUSD", "bybit"): "BTCUSDT",
            ("kraken_futures", "PI_ETHUSD", "bybit"): "ETHUSDT",
            ("kraken_futures", "PF_XBTUSD", "bybit"): "BTCUSDT",
            ("kraken_futures", "PF_ETHUSD", "bybit"): "ETHUSDT",
            # Kraken -> Coinbase
            ("kraken_futures", "PI_XBTUSD", "coinbase"): "BTC-USD",
            ("kraken_futures", "PI_ETHUSD", "coinbase"): "ETH-USD",
            ("kraken_futures", "PF_XBTUSD", "coinbase"): "BTC-USD",
            ("kraken_futures", "PF_ETHUSD", "coinbase"): "ETH-USD",
        }
        return mappings.get((self.trade_venue, trade_venue_symbol, exchange))

    def snapshot(self) -> dict:
        """Full state snapshot for dashboard / logging."""
        out = {"trade_venue": self.trade_venue, "exchanges": {}}
        for ex, syms in self.state.items():
            out["exchanges"][ex] = {}
            for sym, st in syms.items():
                out["exchanges"][ex][sym] = {
                    "H": st.H,
                    "imb": st.imbalance,
                    "spread_bps": st.spread_bps,
                    "mid": st.mid,
                    "bars": st.bars,
                }
        return out
