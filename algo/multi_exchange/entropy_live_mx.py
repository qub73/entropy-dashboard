#!/usr/bin/env python3
"""
Multi-Exchange Entropy Live Trader — Kraken Futures
====================================================
Collects L2 orderbook from Binance, Bybit, Coinbase, and Kraken Futures.
Computes entropy per exchange/symbol. Fires cross-exchange lead-lag signals.
Trades on Kraken Futures when leaders confirm direction.

Compatible with existing state/log format. Drop-in replacement for
entropy_live_multi.py with added cross-exchange intelligence.

Run locally:
    python algo/multi_exchange/entropy_live_mx.py
Or detached:
    pythonw.exe algo/multi_exchange/entropy_live_mx.py
"""

import os
import sys
import json
import time
import signal
import logging
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from collectors import MultiExchangeCollector, L2Book
from mx_entropy import MultiExchangeEntropy
from cross_exchange_signal import CrossExchangeConfig, CrossExchangeSignalEngine


# ---- Config ----

PAIRS = {
    # Trade venue symbol -> dict with exchange-specific symbols
    "BTC": {
        "kraken_futures": "PI_XBTUSD",  # Kraken WS symbol
        "kraken_trade_symbol": "PF_XBTUSD",  # Kraken order symbol
        "binance": "BTCUSDT",
        "bybit": "BTCUSDT",
        "coinbase": "BTC-USD",
        "stop_loss_bps": 65,
        "take_profit_bps": 200,
        "timeout_minutes": 240,
        "precision": 4,
    },
    "ETH": {
        "kraken_futures": "PI_ETHUSD",
        "kraken_trade_symbol": "PF_ETHUSD",
        "binance": "ETHUSDT",
        "bybit": "ETHUSDT",
        "coinbase": "ETH-USD",
        "stop_loss_bps": 50,
        "take_profit_bps": 200,
        "timeout_minutes": 240,
        "precision": 3,
    },
}

# Entropy thresholds per exchange — these would be refit on each exchange's
# own historical data. Using conservative starting values based on Kaggle training.
H_THRESHOLDS = {
    "binance": 0.35,
    "bybit": 0.40,
    "coinbase": 0.42,
    "kraken_futures": 0.35,  # BTC specific; ETH uses 0.4352
}

SHARED = {
    "leverage": 10,
    "equity_fraction": 0.75,
    "entropy_window": 30,
    "return_lookback": 5,
}

BASE = Path(__file__).parent
LOG_DIR = BASE / "logs"
STATE_DIR = BASE / "state"
DATA_DIR = BASE / "data"
for d in (LOG_DIR, STATE_DIR, DATA_DIR):
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "mx_live.log"),
    ],
)
logger = logging.getLogger("mx_live")

RUNNING = True


def _sigh(sig, frame):
    global RUNNING
    RUNNING = False
    logger.info("Shutdown requested")


signal.signal(signal.SIGINT, _sigh)
signal.signal(signal.SIGTERM, _sigh)


# ---- Kraken trading (reused from entropy_live_multi) ----

class KrakenFuturesAPI:
    def __init__(self, api_key, api_secret):
        from kraken.futures import User, Trade
        self._user = User(key=api_key, secret=api_secret)
        self._trade = Trade(key=api_key, secret=api_secret)

    def get_accounts(self):
        return self._user.get_wallets()

    def get_equity(self):
        accts = self.get_accounts()
        if accts.get("result") != "success":
            return None
        flex = accts.get("accounts", {}).get("flex", {})
        if flex:
            pv = flex.get("portfolioValue", 0)
            if pv > 0:
                return pv
        return None

    def send_order(self, symbol, side, size):
        return self._trade.create_order(
            orderType="mkt", symbol=symbol, side=side, size=size)


class PositionManager:
    def __init__(self, api, state_file):
        self.api = api
        self.state_file = Path(state_file)
        self.position = None
        self.trades = []
        self.daily_pnl = 0.0
        self._load()

    def _load(self):
        if self.state_file.exists():
            with open(self.state_file) as f:
                s = json.load(f)
            self.position = s.get("position")
            self.trades = s.get("trades", [])
            self.daily_pnl = s.get("daily_pnl", 0.0)
            logger.info(f"Loaded state: pos={self.position is not None}, trades={len(self.trades)}")

    def _save(self):
        state = {
            "position": self.position,
            "trades": self.trades[-100:],
            "daily_pnl": self.daily_pnl,
            "last_update": time.time(),
        }
        tmp = str(self.state_file) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, str(self.state_file))

    def has_position(self):
        return self.position is not None

    def _verify_fill(self, result, label):
        logger.info(f"  {label} raw: {json.dumps(result)[:400]}")
        if result.get("result") != "success":
            return None
        send = result.get("sendStatus", {})
        if send.get("status") != "placed":
            logger.error(f"  {label} status={send.get('status')}")
            return None
        executions = [e for e in send.get("orderEvents", []) if e.get("type") == "EXECUTION"]
        if not executions:
            logger.error(f"  {label} no executions")
            return None
        total_f = sum(e.get("amount", 0) for e in executions)
        total_v = sum(e.get("amount", 0) * e.get("price", 0) for e in executions)
        avg = total_v / total_f if total_f > 0 else 0
        return send.get("order_id", ""), avg, total_f

    def open_position(self, pair_name, direction, mid_price, trade_symbol, precision):
        if self.has_position():
            return False

        accts = self.api.get_accounts()
        flex = accts.get("accounts", {}).get("flex", {})
        available = flex.get("availableMargin", 0)
        if available <= 0:
            logger.error(f"No margin: ${available}")
            return False

        margin = available * SHARED["equity_fraction"]
        notional = margin * SHARED["leverage"]
        size = round(notional / mid_price, precision)
        if size < 0.0001:
            return False

        side = "buy" if direction == 1 else "sell"
        logger.info(f"SUBMITTING [{pair_name}]: {side.upper()} {size} {trade_symbol} @ ~{mid_price:.1f}")

        try:
            result = self.api.send_order(trade_symbol, side, size)
        except Exception as e:
            logger.error(f"ORDER EXCEPTION [{pair_name}]: {e}")
            return False

        fill = self._verify_fill(result, f"ENTRY [{pair_name}]")
        if fill is None:
            logger.error(f"ENTRY [{pair_name}] NOT FILLED")
            return False

        order_id, price, filled = fill
        self.position = {
            "pair": pair_name,
            "direction": direction,
            "entry_price": price,
            "entry_time": time.time(),
            "size": filled,
            "symbol": trade_symbol,
            "order_id": order_id,
        }
        self._save()
        logger.info(f"ENTRY [{pair_name}] CONFIRMED @ ${price:.2f} size={filled}")
        return True

    def close_position(self, current_price, reason):
        if not self.has_position():
            return None
        pos = self.position
        side = "sell" if pos["direction"] == 1 else "buy"
        try:
            result = self.api.send_order(pos["symbol"], side, pos["size"])
        except Exception as e:
            logger.error(f"CLOSE EXCEPTION [{pos['pair']}]: {e}")
            return None

        fill = self._verify_fill(result, f"EXIT [{pos['pair']}]")
        if fill is None:
            logger.error(f"EXIT [{pos['pair']}] NOT FILLED — still open!")
            return None

        order_id, fill_price, _ = fill
        pnl_bps = pos["direction"] * (fill_price / pos["entry_price"] - 1.0) * 10000
        pnl_lev = pnl_bps * SHARED["leverage"]
        hold_min = (time.time() - pos["entry_time"]) / 60

        trade = {
            "pair": pos["pair"],
            "direction": pos["direction"],
            "entry_price": pos["entry_price"],
            "exit_price": fill_price,
            "pnl_bps": pnl_bps,
            "pnl_bps_leveraged": pnl_lev,
            "hold_min": hold_min,
            "reason": reason,
            "time": datetime.now(timezone.utc).isoformat(),
            "order_id": order_id,
        }
        self.trades.append(trade)
        self.daily_pnl += pnl_lev
        logger.info(f"EXIT [{pos['pair']}] {reason}: {pnl_bps:.1f}bps ({pnl_lev:.1f}bps @10x), held {hold_min:.1f}min")
        self.position = None
        self._save()
        return trade

    def check_exit(self, current_price, pair_cfg):
        if not self.has_position():
            return None
        pos = self.position
        pnl_bps = pos["direction"] * (current_price / pos["entry_price"] - 1.0) * 10000
        elapsed = (time.time() - pos["entry_time"]) / 60
        if pnl_bps <= -pair_cfg["stop_loss_bps"]:
            return "sl"
        if pnl_bps >= pair_cfg["take_profit_bps"]:
            return "tp"
        if elapsed >= pair_cfg["timeout_minutes"]:
            return "timeout"
        return None


# ---- Main ----

def main():
    global RUNNING

    from dotenv import load_dotenv
    env_path = BASE.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()

    api_key = os.getenv("KRAKEN_API_FUTURES_KEY", "")
    api_secret = os.getenv("KRAKEN_API_FUTURES_SECRET", "")
    if not api_key:
        logger.error("KRAKEN_API_FUTURES_KEY missing")
        sys.exit(1)

    api = KrakenFuturesAPI(api_key, api_secret)
    try:
        eq = api.get_equity()
        logger.info(f"API OK, equity ${eq:.2f}")
    except Exception as e:
        logger.error(f"API fail: {e}")
        sys.exit(1)

    mgr = PositionManager(api, STATE_DIR / "mx_trader_state.json")

    # Multi-exchange entropy engine — one per (exchange, symbol)
    mx_entropy = MultiExchangeEntropy(window=SHARED["entropy_window"])
    for pair_name, cfg in PAIRS.items():
        for ex in ("binance", "bybit", "coinbase", "kraken_futures"):
            sym = cfg[ex]
            mx_entropy.add_pair(ex, sym)

    # Cross-exchange signal engine
    ce_cfg = CrossExchangeConfig(
        h_thresholds=dict(H_THRESHOLDS),
        min_leaders_triggered=1,
        direction_agreement=0.6,
    )
    ce_signal = CrossExchangeSignalEngine(
        ce_cfg, trade_venue="kraken_futures",
        leader_exchanges=["binance", "bybit", "coinbase"])

    signal_counts = {"total": 0}

    def on_snapshot(exchange, symbol, snap):
        """Called every 10s from any collector with a fresh book snapshot."""
        bids = snap["b"]
        asks = snap["a"]
        if not bids or not asks:
            return
        mid = (bids[0][0] + asks[0][0]) / 2.0
        spread = asks[0][0] - bids[0][0]
        spread_bps = spread / mid * 10000 if mid > 0 else 999
        # Imbalance across top 5
        bd = sum(s for _, s in bids[:5])
        ad = sum(s for _, s in asks[:5])
        total = bd + ad
        imb = (bd - ad) / total if total > 0 else 0

        # Feed entropy engine
        H = mx_entropy.on_tick(exchange, symbol, mid, imb, spread_bps)
        engine = mx_entropy.engines.get((exchange, symbol))
        bars = engine.bar_count if engine else 0

        # Update cross-exchange signal state
        ce_signal.update(exchange, symbol, H, imb, spread_bps, mid, bars)

        # If this was kraken_futures, check for a signal
        if exchange != "kraken_futures":
            return
        # Find pair config
        pair_name = None
        pair_cfg = None
        for pn, pc in PAIRS.items():
            if pc["kraken_futures"] == symbol:
                pair_name = pn
                pair_cfg = pc
                break
        if pair_name is None:
            return

        # Check exit first if we hold this pair
        if mgr.has_position() and mgr.position["pair"] == pair_name:
            reason = mgr.check_exit(mid, pair_cfg)
            if reason:
                mgr.close_position(mid, reason)
            return

        if mgr.has_position():
            return  # already positioned elsewhere

        # Check cross-exchange signal
        trailing_ret = engine.get_trailing_return_bps(SHARED["return_lookback"]) if engine else None
        sig = ce_signal.check_signal(symbol, trailing_ret)
        if sig is None:
            return

        signal_counts["total"] += 1
        leaders_str = ", ".join(
            f"{ex}:H={h:.3f},imb={imb:.2f}" for ex, h, imb in sig["leaders_triggered"])
        logger.info(
            f"CROSS-EXCHANGE SIGNAL #{signal_counts['total']} [{pair_name}]: "
            f"{'LONG' if sig['direction']==1 else 'SHORT'} | "
            f"Kraken H={sig['trade_venue_H']:.4f}, ret={trailing_ret:.1f}bps | "
            f"Leaders({len(sig['leaders_triggered'])}): {leaders_str}")

        mgr.open_position(
            pair_name, sig["direction"], mid,
            pair_cfg["kraken_trade_symbol"], pair_cfg["precision"])

    # Start collectors
    collector = MultiExchangeCollector(on_snapshot=on_snapshot, log_dir=DATA_DIR)
    # Binance
    binance_syms = list(set(cfg["binance"] for cfg in PAIRS.values()))
    collector.add("binance", binance_syms)
    # Bybit
    bybit_syms = list(set(cfg["bybit"] for cfg in PAIRS.values()))
    collector.add("bybit", bybit_syms)
    # Coinbase
    cb_syms = list(set(cfg["coinbase"] for cfg in PAIRS.values()))
    collector.add("coinbase", cb_syms)
    # Kraken Futures
    kf_syms = list(set(cfg["kraken_futures"] for cfg in PAIRS.values()))
    collector.add("kraken_futures", kf_syms)

    logger.info(f"Config: {json.dumps({p: {k:v for k,v in c.items() if not isinstance(v,(int,float)) or k in ('stop_loss_bps','take_profit_bps','timeout_minutes','precision')} for p,c in PAIRS.items()}, indent=2)}")
    logger.info(f"H thresholds: {H_THRESHOLDS}")
    logger.info(f"Leaders: binance, bybit, coinbase | Trade venue: kraken_futures")
    logger.info("Starting collectors...")
    collector.start_all(snapshot_interval_sec=10)

    # Status thread
    def status_loop():
        while RUNNING:
            time.sleep(300)
            snap = ce_signal.snapshot()
            status = {
                "position": mgr.position["pair"] if mgr.has_position() else None,
                "trades": len(mgr.trades),
                "daily_pnl": mgr.daily_pnl,
                "signals": signal_counts["total"],
            }
            # Add entropy per exchange/symbol
            for ex, syms in snap["exchanges"].items():
                for sym, st in syms.items():
                    key = f"{ex}:{sym}"
                    status[key] = {
                        "H": round(st["H"], 4) if st["H"] is not None else None,
                        "bars": st["bars"],
                        "mid": st["mid"],
                    }
            logger.info(f"STATUS: {json.dumps(status)}")

    threading.Thread(target=status_loop, daemon=True).start()

    try:
        while RUNNING:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    collector.stop_all()
    logger.info("Shutting down.")


if __name__ == "__main__":
    main()
