"""
Multi-exchange L2 orderbook collectors.

Each collector maintains a live L2 book in memory via WebSocket diff feeds.
Provides a unified `get_book_snapshot()` interface returning the same format
as the Pi JSONL logs: {ts, b: [[price,size]...], a: [[price,size]...]}.

Exchanges supported:
- Binance (spot + perpetual, largest global liquidity)
- Bybit (perpetuals, strong Asian flow)
- Coinbase (spot, US flow)
- Kraken (already have this via existing bot — included for consistency)

All collectors run in their own thread with automatic reconnect.
"""

import json
import time
import threading
import logging
import gzip
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Callable

try:
    import websocket  # websocket-client
except ImportError:
    websocket = None

try:
    import requests
except ImportError:
    requests = None

logger = logging.getLogger("mx_collectors")


# ---- Base class ----

class L2Book:
    """In-memory L2 book as sorted price -> size dict."""
    __slots__ = ("bids", "asks", "last_update_id", "last_update_ts")

    def __init__(self):
        self.bids = {}   # price -> size (bids)
        self.asks = {}   # price -> size (asks)
        self.last_update_id = 0
        self.last_update_ts = 0.0

    def apply_snapshot(self, bids, asks, update_id=0):
        self.bids = {float(p): float(s) for p, s in bids if float(s) > 0}
        self.asks = {float(p): float(s) for p, s in asks if float(s) > 0}
        self.last_update_id = update_id
        self.last_update_ts = time.time()

    def apply_delta(self, bids, asks):
        for p, s in bids:
            p, s = float(p), float(s)
            if s <= 0:
                self.bids.pop(p, None)
            else:
                self.bids[p] = s
        for p, s in asks:
            p, s = float(p), float(s)
            if s <= 0:
                self.asks.pop(p, None)
            else:
                self.asks[p] = s
        self.last_update_ts = time.time()

    def top_n(self, n=10):
        bids = sorted(self.bids.items(), key=lambda x: -x[0])[:n]
        asks = sorted(self.asks.items(), key=lambda x: x[0])[:n]
        return bids, asks

    def snapshot_dict(self, n=10):
        bids, asks = self.top_n(n)
        if not bids or not asks:
            return None
        return {
            "ts": int(time.time() * 1000),
            "b": [[float(p), float(s)] for p, s in bids],
            "a": [[float(p), float(s)] for p, s in asks],
        }

    @property
    def mid(self):
        bids, asks = self.top_n(1)
        if not bids or not asks:
            return 0.0
        return (bids[0][0] + asks[0][0]) / 2.0

    @property
    def spread_bps(self):
        bids, asks = self.top_n(1)
        if not bids or not asks:
            return 999.0
        m = (bids[0][0] + asks[0][0]) / 2.0
        return (asks[0][0] - bids[0][0]) / m * 10000 if m > 0 else 999.0


class BaseCollector:
    """Base class for per-exchange collectors. Subclasses implement WS protocol."""

    def __init__(self, name: str, symbols: list, on_snapshot: Optional[Callable] = None,
                 log_dir: Optional[Path] = None):
        self.name = name
        self.symbols = symbols
        self.on_snapshot = on_snapshot
        self.log_dir = log_dir
        self.books: Dict[str, L2Book] = {s: L2Book() for s in symbols}
        self._running = False
        self._thread = None
        self._reconnect_delay = 1
        self._snapshot_thread = None

    def get_book(self, symbol: str) -> Optional[L2Book]:
        return self.books.get(symbol)

    def start(self, snapshot_interval_sec: int = 10):
        if websocket is None:
            raise ImportError("websocket-client required: pip install websocket-client")
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        if snapshot_interval_sec > 0:
            self._snapshot_thread = threading.Thread(
                target=self._snapshot_loop,
                args=(snapshot_interval_sec,),
                daemon=True)
            self._snapshot_thread.start()

    def stop(self):
        self._running = False

    def _run_loop(self):
        raise NotImplementedError

    def _snapshot_loop(self, interval):
        while self._running:
            time.sleep(interval)
            for sym in self.symbols:
                book = self.books.get(sym)
                if book is None:
                    continue
                snap = book.snapshot_dict(n=10)
                if snap is None:
                    continue
                if self.on_snapshot:
                    try:
                        self.on_snapshot(self.name, sym, snap)
                    except Exception as e:
                        logger.error(f"{self.name} on_snapshot error: {e}")
                if self.log_dir:
                    self._write_jsonl(sym, snap)

    def _write_jsonl(self, sym: str, snap: dict):
        today = time.strftime("%Y%m%d", time.gmtime())
        path = self.log_dir / f"ob_{self.name}_{sym}_{today}.jsonl"
        try:
            with open(path, "a") as f:
                f.write(json.dumps(snap) + "\n")
        except Exception as e:
            logger.error(f"Write error {path}: {e}")


# ---- Binance ----

class BinanceCollector(BaseCollector):
    """
    Binance spot L2 collector via diff depth stream.
    Protocol:
      1. Connect to <symbol>@depth@100ms
      2. Buffer events
      3. REST snapshot from /api/v3/depth?symbol=X&limit=1000
      4. Drop buffered events with u <= lastUpdateId
      5. Apply remaining buffered + subsequent events (check U == last_u+1)
    """

    WS_URL = "wss://stream.binance.com:9443/stream"
    REST_URL = "https://api.binance.com/api/v3/depth"

    def __init__(self, symbols, on_snapshot=None, log_dir=None):
        # Binance uses lowercase symbols on WS, uppercase on REST
        super().__init__("binance", [s.upper() for s in symbols],
                         on_snapshot, log_dir)
        self._buffers = defaultdict(list)
        self._snapshot_done = {s: False for s in self.symbols}

    def _run_loop(self):
        streams = "/".join(f"{s.lower()}@depth@100ms" for s in self.symbols)
        url = f"{self.WS_URL}?streams={streams}"
        while self._running:
            try:
                ws = websocket.WebSocketApp(
                    url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=lambda ws, e: logger.error(f"Binance WS error: {e}"),
                    on_close=lambda ws, c, m: logger.info(f"Binance WS closed {c}"),
                )
                ws.run_forever(ping_interval=30)
            except Exception as e:
                logger.error(f"Binance connection error: {e}")
            if self._running:
                time.sleep(min(self._reconnect_delay, 60))
                self._reconnect_delay = min(self._reconnect_delay * 2, 60)

    def _on_open(self, ws):
        logger.info(f"Binance WS connected to {len(self.symbols)} streams")
        self._reconnect_delay = 1
        # Fetch initial snapshots
        for sym in self.symbols:
            threading.Thread(target=self._fetch_snapshot, args=(sym,), daemon=True).start()

    def _fetch_snapshot(self, sym):
        try:
            r = requests.get(self.REST_URL, params={"symbol": sym, "limit": 1000}, timeout=10)
            data = r.json()
            bids = [(p, s) for p, s in data.get("bids", [])]
            asks = [(p, s) for p, s in data.get("asks", [])]
            last_update_id = data.get("lastUpdateId", 0)
            self.books[sym].apply_snapshot(bids, asks, last_update_id)
            # Apply buffered events with u > lastUpdateId
            buf = self._buffers[sym]
            for event in buf:
                if event["u"] <= last_update_id:
                    continue
                self.books[sym].apply_delta(event["b"], event["a"])
            self._buffers[sym] = []
            self._snapshot_done[sym] = True
            logger.info(f"Binance {sym} snapshot loaded (lastUpdateId={last_update_id})")
        except Exception as e:
            logger.error(f"Binance snapshot fetch error for {sym}: {e}")

    def _on_message(self, ws, message):
        try:
            msg = json.loads(message)
            data = msg.get("data", msg)
            stream = msg.get("stream", "")
            sym = stream.split("@")[0].upper() if "@" in stream else ""
            if not sym:
                # Try to infer from event
                sym = data.get("s", "")
            if sym not in self.books:
                return
            event = {
                "U": data.get("U", 0),  # first update in event
                "u": data.get("u", 0),  # last update in event
                "b": data.get("b", []),
                "a": data.get("a", []),
            }
            if not self._snapshot_done[sym]:
                self._buffers[sym].append(event)
                return
            self.books[sym].apply_delta(event["b"], event["a"])
        except Exception as e:
            logger.error(f"Binance message error: {e}")


# ---- Bybit ----

class BybitCollector(BaseCollector):
    """
    Bybit perpetual L2 collector.
    Uses orderbook.50.<symbol> topic (50 levels, delta updates).
    """

    WS_URL = "wss://stream.bybit.com/v5/public/linear"

    def __init__(self, symbols, on_snapshot=None, log_dir=None):
        super().__init__("bybit", [s.upper() for s in symbols],
                         on_snapshot, log_dir)

    def _run_loop(self):
        while self._running:
            try:
                ws = websocket.WebSocketApp(
                    self.WS_URL,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=lambda ws, e: logger.error(f"Bybit WS error: {e}"),
                    on_close=lambda ws, c, m: logger.info(f"Bybit WS closed {c}"),
                )
                ws.run_forever(ping_interval=20)
            except Exception as e:
                logger.error(f"Bybit connection error: {e}")
            if self._running:
                time.sleep(min(self._reconnect_delay, 60))
                self._reconnect_delay = min(self._reconnect_delay * 2, 60)

    def _on_open(self, ws):
        logger.info(f"Bybit WS connected")
        self._reconnect_delay = 1
        args = [f"orderbook.50.{s}" for s in self.symbols]
        ws.send(json.dumps({"op": "subscribe", "args": args}))

    def _on_message(self, ws, message):
        try:
            msg = json.loads(message)
            if msg.get("topic", "").startswith("orderbook."):
                data = msg.get("data", {})
                sym = data.get("s", "")
                if sym not in self.books:
                    return
                msg_type = msg.get("type", "")  # 'snapshot' or 'delta'
                bids = data.get("b", [])
                asks = data.get("a", [])
                if msg_type == "snapshot":
                    self.books[sym].apply_snapshot(bids, asks)
                else:
                    self.books[sym].apply_delta(bids, asks)
        except Exception as e:
            logger.error(f"Bybit message error: {e}")


# ---- Coinbase ----

class CoinbaseCollector(BaseCollector):
    """
    Coinbase spot L2 collector.
    Uses level2 channel which provides full snapshot + incremental updates.
    """

    WS_URL = "wss://ws-feed.exchange.coinbase.com"

    def __init__(self, symbols, on_snapshot=None, log_dir=None):
        # Coinbase uses symbols like BTC-USD, ETH-USD
        super().__init__("coinbase", symbols, on_snapshot, log_dir)

    def _run_loop(self):
        while self._running:
            try:
                ws = websocket.WebSocketApp(
                    self.WS_URL,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=lambda ws, e: logger.error(f"Coinbase WS error: {e}"),
                    on_close=lambda ws, c, m: logger.info(f"Coinbase WS closed {c}"),
                )
                ws.run_forever(ping_interval=20)
            except Exception as e:
                logger.error(f"Coinbase connection error: {e}")
            if self._running:
                time.sleep(min(self._reconnect_delay, 60))
                self._reconnect_delay = min(self._reconnect_delay * 2, 60)

    def _on_open(self, ws):
        logger.info(f"Coinbase WS connected")
        self._reconnect_delay = 1
        sub = {
            "type": "subscribe",
            "product_ids": self.symbols,
            "channels": ["level2_batch"],
        }
        ws.send(json.dumps(sub))

    def _on_message(self, ws, message):
        try:
            msg = json.loads(message)
            mtype = msg.get("type", "")
            product = msg.get("product_id", "")
            if product not in self.books:
                return
            if mtype == "snapshot":
                bids = msg.get("bids", [])
                asks = msg.get("asks", [])
                self.books[product].apply_snapshot(bids, asks)
            elif mtype in ("l2update", "l2update_batch"):
                bids_delta = []
                asks_delta = []
                for change in msg.get("changes", []):
                    side, price, size = change[0], change[1], change[2]
                    if side == "buy":
                        bids_delta.append([price, size])
                    else:
                        asks_delta.append([price, size])
                self.books[product].apply_delta(bids_delta, asks_delta)
        except Exception as e:
            logger.error(f"Coinbase message error: {e}")


# ---- Kraken Futures (for unified interface; delegates to existing book logic) ----

class KrakenFuturesCollector(BaseCollector):
    """Kraken Futures via their public WS."""

    WS_URL = "wss://futures.kraken.com/ws/v1"

    def __init__(self, symbols, on_snapshot=None, log_dir=None):
        # symbols like PI_XBTUSD, PI_ETHUSD
        super().__init__("kraken_futures", symbols, on_snapshot, log_dir)

    def _run_loop(self):
        while self._running:
            try:
                ws = websocket.WebSocketApp(
                    self.WS_URL,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=lambda ws, e: logger.error(f"Kraken WS error: {e}"),
                    on_close=lambda ws, c, m: logger.info(f"Kraken WS closed {c}"),
                )
                ws.run_forever(ping_interval=30)
            except Exception as e:
                logger.error(f"Kraken connection error: {e}")
            if self._running:
                time.sleep(min(self._reconnect_delay, 60))
                self._reconnect_delay = min(self._reconnect_delay * 2, 60)

    def _on_open(self, ws):
        logger.info("Kraken Futures WS connected")
        self._reconnect_delay = 1
        ws.send(json.dumps({
            "event": "subscribe",
            "feed": "book",
            "product_ids": self.symbols,
        }))

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            feed = data.get("feed", "")
            product = data.get("product_id", "")
            if product not in self.books:
                return
            if feed == "book_snapshot":
                bids = [(l["price"], l["qty"]) for l in data.get("bids", [])]
                asks = [(l["price"], l["qty"]) for l in data.get("asks", [])]
                self.books[product].apply_snapshot(bids, asks)
            elif feed == "book":
                side = data.get("side", "")
                price = data.get("price", 0)
                qty = data.get("qty", 0)
                if side == "buy":
                    self.books[product].apply_delta([[price, qty]], [])
                else:
                    self.books[product].apply_delta([], [[price, qty]])
        except Exception as e:
            logger.error(f"Kraken message error: {e}")


# ---- Convenience: multi-exchange manager ----

class MultiExchangeCollector:
    """Runs all exchanges simultaneously."""

    def __init__(self, on_snapshot: Optional[Callable] = None,
                 log_dir: Optional[Path] = None):
        self.on_snapshot = on_snapshot
        self.log_dir = log_dir
        self.collectors = {}

    def add(self, exchange: str, symbols: list):
        if exchange == "binance":
            self.collectors["binance"] = BinanceCollector(
                symbols, self.on_snapshot, self.log_dir)
        elif exchange == "bybit":
            self.collectors["bybit"] = BybitCollector(
                symbols, self.on_snapshot, self.log_dir)
        elif exchange == "coinbase":
            self.collectors["coinbase"] = CoinbaseCollector(
                symbols, self.on_snapshot, self.log_dir)
        elif exchange == "kraken_futures":
            self.collectors["kraken_futures"] = KrakenFuturesCollector(
                symbols, self.on_snapshot, self.log_dir)
        else:
            raise ValueError(f"Unknown exchange: {exchange}")

    def start_all(self, snapshot_interval_sec: int = 10):
        for name, c in self.collectors.items():
            c.start(snapshot_interval_sec=snapshot_interval_sec)
            logger.info(f"Started {name} collector")

    def stop_all(self):
        for c in self.collectors.values():
            c.stop()

    def get_book(self, exchange: str, symbol: str) -> Optional[L2Book]:
        col = self.collectors.get(exchange)
        if col is None:
            return None
        return col.get_book(symbol)
