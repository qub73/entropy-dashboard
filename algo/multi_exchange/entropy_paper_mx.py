#!/usr/bin/env python3
"""
Paper-trading version of the multi-exchange entropy trader.

Identical signal logic to `entropy_live_mx.py` but:
- NO real orders sent to Kraken
- Fills simulated at trade-venue mid price
- Costs: 5 bps taker per side (Kraken Futures taker)
- Trades logged to state file + stdout
- Stats printed every 5 minutes (and on each trade event)

Run:
    python algo/multi_exchange/entropy_paper_mx.py
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

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from collectors import MultiExchangeCollector
from mx_entropy import MultiExchangeEntropy
from cross_exchange_signal import CrossExchangeConfig, CrossExchangeSignalEngine


# ---- Config (same as live) ----

PAIRS = {
    "BTC": {
        "kraken_futures": "PI_XBTUSD",
        "binance": "BTCUSDT",
        "bybit": "BTCUSDT",
        "coinbase": "BTC-USD",
        "stop_loss_bps": 65,
        "take_profit_bps": 200,
        "timeout_minutes": 240,
    },
    "ETH": {
        "kraken_futures": "PI_ETHUSD",
        "binance": "ETHUSDT",
        "bybit": "ETHUSDT",
        "coinbase": "ETH-USD",
        "stop_loss_bps": 50,
        "take_profit_bps": 200,
        "timeout_minutes": 240,
    },
}

H_THRESHOLDS = {
    "binance": 0.35,
    "bybit": 0.40,
    "coinbase": 0.42,
    "kraken_futures": 0.35,
}

SHARED = {
    "leverage": 10,
    "equity_fraction": 0.75,
    "entropy_window": 30,
    "return_lookback": 5,
    "taker_fee_bps": 5.0,  # Kraken Futures taker
    "slippage_bps": 0.3,
    "starting_equity": 10000.0,  # paper capital
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
        logging.FileHandler(LOG_DIR / "mx_paper.log"),
    ],
)
logger = logging.getLogger("mx_paper")

RUNNING = True


def _sigh(sig, frame):
    global RUNNING
    RUNNING = False
    logger.info("Shutdown requested")


signal.signal(signal.SIGINT, _sigh)
signal.signal(signal.SIGTERM, _sigh)


# ---- Paper trade manager ----

class PaperTradeManager:
    """
    Simulates one-position-at-a-time trading with realistic costs.
    No external API — all fills at current mid + slippage.
    """

    def __init__(self, state_file, starting_equity=10000.0):
        self.state_file = Path(state_file)
        self.starting_equity = starting_equity
        self.equity = starting_equity
        self.position = None  # {pair, direction, entry_price, entry_time, size, notional}
        self.trades = []
        self.daily_pnl = 0.0
        self.current_day = -1
        self._load()

    def _load(self):
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    s = json.load(f)
                self.equity = s.get("equity", self.starting_equity)
                self.position = s.get("position")
                self.trades = s.get("trades", [])
                self.daily_pnl = s.get("daily_pnl", 0.0)
                logger.info(f"Loaded paper state: equity=${self.equity:.2f}, "
                            f"pos={self.position is not None}, trades={len(self.trades)}")
            except Exception as e:
                logger.warning(f"State load error: {e}")

    def _save(self):
        state = {
            "equity": self.equity,
            "starting_equity": self.starting_equity,
            "position": self.position,
            "trades": self.trades[-500:],
            "daily_pnl": self.daily_pnl,
            "last_update": time.time(),
        }
        tmp = str(self.state_file) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, str(self.state_file))

    def has_position(self):
        return self.position is not None

    def open_position(self, pair_name, direction, mid_price):
        if self.has_position():
            return False
        if self.equity <= 0:
            logger.error("Equity depleted — no more paper trades")
            return False

        margin = self.equity * SHARED["equity_fraction"]
        notional = margin * SHARED["leverage"]
        # Apply slippage: buy = pay higher, sell = receive lower
        slip = SHARED["slippage_bps"] / 10000.0
        fill_price = mid_price * (1 + direction * slip)
        size = notional / fill_price
        entry_fee = notional * SHARED["taker_fee_bps"] / 10000.0
        self.equity -= entry_fee

        self.position = {
            "pair": pair_name,
            "direction": direction,
            "entry_price": fill_price,
            "entry_time": time.time(),
            "size": size,
            "notional": notional,
            "entry_fee": entry_fee,
        }
        self._save()
        logger.info(f"[PAPER ENTRY] {pair_name} {'LONG' if direction==1 else 'SHORT'} "
                    f"{size:.5f} @ ${fill_price:.2f} (notional=${notional:.0f}, "
                    f"fee=${entry_fee:.2f}, equity=${self.equity:.2f})")
        return True

    def close_position(self, mid_price, reason):
        if not self.has_position():
            return None
        pos = self.position

        slip = SHARED["slippage_bps"] / 10000.0
        # exit side slippage = opposite direction
        fill_price = mid_price * (1 - pos["direction"] * slip)

        pnl_bps = pos["direction"] * (fill_price / pos["entry_price"] - 1.0) * 10000
        pnl_lev_bps = pnl_bps * SHARED["leverage"]
        # Absolute P&L in USD
        pnl_usd = (pnl_bps / 10000.0) * pos["notional"]
        exit_fee = pos["notional"] * SHARED["taker_fee_bps"] / 10000.0
        net_pnl = pnl_usd - exit_fee
        self.equity += net_pnl

        hold_min = (time.time() - pos["entry_time"]) / 60
        trade = {
            "pair": pos["pair"],
            "direction": pos["direction"],
            "entry_price": pos["entry_price"],
            "exit_price": fill_price,
            "pnl_bps": pnl_bps,
            "pnl_bps_leveraged": pnl_lev_bps,
            "pnl_usd": net_pnl,
            "entry_fee": pos["entry_fee"],
            "exit_fee": exit_fee,
            "hold_min": hold_min,
            "reason": reason,
            "time": datetime.now(timezone.utc).isoformat(),
            "equity_after": self.equity,
        }
        self.trades.append(trade)
        self.daily_pnl += pnl_lev_bps

        wr_emoji = "+" if net_pnl > 0 else ""
        logger.info(f"[PAPER EXIT] {pos['pair']} ({reason}): "
                    f"{wr_emoji}{pnl_bps:.1f}bps ({pnl_lev_bps:+.1f}bps @10x), "
                    f"{wr_emoji}${net_pnl:.2f}, held {hold_min:.1f}min, "
                    f"equity=${self.equity:.2f}")
        self.position = None
        self._save()
        return trade

    def check_exit(self, current_price, pair_cfg):
        if not self.has_position():
            return None
        pos = self.position
        pnl_bps = pos["direction"] * (current_price / pos["entry_price"] - 1.0) * 10000
        elapsed_min = (time.time() - pos["entry_time"]) / 60
        if pnl_bps <= -pair_cfg["stop_loss_bps"]:
            return "sl"
        if pnl_bps >= pair_cfg["take_profit_bps"]:
            return "tp"
        if elapsed_min >= pair_cfg["timeout_minutes"]:
            return "timeout"
        return None

    def stats(self):
        n = len(self.trades)
        if n == 0:
            return {
                "trades": 0, "wins": 0, "losses": 0, "win_rate": 0,
                "total_return_pct": 0, "equity": self.equity,
                "mean_pnl_usd": 0, "best": 0, "worst": 0,
                "tp_count": 0, "sl_count": 0, "timeout_count": 0,
            }
        pnls = [t["pnl_usd"] for t in self.trades]
        wins = sum(1 for p in pnls if p > 0)
        losses = sum(1 for p in pnls if p < 0)
        total_ret = (self.equity - self.starting_equity) / self.starting_equity * 100
        reasons = {}
        for t in self.trades:
            r = t["reason"]
            reasons[r] = reasons.get(r, 0) + 1
        return {
            "trades": n, "wins": wins, "losses": losses,
            "win_rate": wins / n if n > 0 else 0,
            "total_return_pct": total_ret,
            "total_pnl_usd": self.equity - self.starting_equity,
            "equity": self.equity,
            "mean_pnl_usd": np.mean(pnls),
            "best": max(pnls),
            "worst": min(pnls),
            "tp_count": reasons.get("tp", 0),
            "sl_count": reasons.get("sl", 0),
            "timeout_count": reasons.get("timeout", 0),
        }


# ---- Main ----

def print_stats_header():
    logger.info("=" * 90)


def main():
    global RUNNING

    mgr = PaperTradeManager(STATE_DIR / "mx_paper_state.json",
                            starting_equity=SHARED["starting_equity"])

    mx_entropy = MultiExchangeEntropy(window=SHARED["entropy_window"])
    for pair_name, cfg in PAIRS.items():
        for ex in ("binance", "bybit", "coinbase", "kraken_futures"):
            mx_entropy.add_pair(ex, cfg[ex])

    ce_cfg = CrossExchangeConfig(
        h_thresholds=dict(H_THRESHOLDS),
        min_leaders_triggered=1,
        direction_agreement=0.6,
    )
    ce_signal = CrossExchangeSignalEngine(
        ce_cfg, trade_venue="kraken_futures",
        leader_exchanges=["binance", "bybit", "coinbase"])

    signal_counts = {"total": 0, "rejected": 0}

    def on_snapshot(exchange, symbol, snap):
        bids = snap["b"]
        asks = snap["a"]
        if not bids or not asks:
            return
        mid = (bids[0][0] + asks[0][0]) / 2.0
        spread = asks[0][0] - bids[0][0]
        spread_bps = spread / mid * 10000 if mid > 0 else 999
        bd = sum(s for _, s in bids[:5])
        ad = sum(s for _, s in asks[:5])
        total = bd + ad
        imb = (bd - ad) / total if total > 0 else 0

        H = mx_entropy.on_tick(exchange, symbol, mid, imb, spread_bps)
        engine = mx_entropy.engines.get((exchange, symbol))
        bars = engine.bar_count if engine else 0
        ce_signal.update(exchange, symbol, H, imb, spread_bps, mid, bars)

        if exchange != "kraken_futures":
            return

        pair_name = None
        pair_cfg = None
        for pn, pc in PAIRS.items():
            if pc["kraken_futures"] == symbol:
                pair_name = pn
                pair_cfg = pc
                break
        if pair_name is None:
            return

        # Exits
        if mgr.has_position() and mgr.position["pair"] == pair_name:
            reason = mgr.check_exit(mid, pair_cfg)
            if reason:
                mgr.close_position(mid, reason)
                _print_running_stats(mgr, signal_counts)
            return

        if mgr.has_position():
            return  # locked in another pair

        trailing_ret = engine.get_trailing_return_bps(SHARED["return_lookback"]) if engine else None
        sig = ce_signal.check_signal(symbol, trailing_ret)
        if sig is None:
            return

        signal_counts["total"] += 1
        leaders_str = ", ".join(
            f"{ex}(H={h:.3f},imb={imb:+.2f})" for ex, h, imb in sig["leaders_triggered"])
        logger.info(
            f"[SIGNAL #{signal_counts['total']}] {pair_name} "
            f"{'LONG' if sig['direction']==1 else 'SHORT'} | "
            f"Kraken H={sig['trade_venue_H']:.4f}, ret={trailing_ret:+.1f}bps | "
            f"Leaders: {leaders_str}")

        if mgr.open_position(pair_name, sig["direction"], mid):
            _print_running_stats(mgr, signal_counts)

    collector = MultiExchangeCollector(on_snapshot=on_snapshot, log_dir=DATA_DIR)
    collector.add("binance", list(set(c["binance"] for c in PAIRS.values())))
    collector.add("bybit", list(set(c["bybit"] for c in PAIRS.values())))
    collector.add("coinbase", list(set(c["coinbase"] for c in PAIRS.values())))
    collector.add("kraken_futures", list(set(c["kraken_futures"] for c in PAIRS.values())))

    logger.info("=" * 90)
    logger.info("PAPER TRADING MODE — Multi-Exchange Entropy Signal")
    logger.info("=" * 90)
    logger.info(f"Starting equity: ${SHARED['starting_equity']:.2f}")
    logger.info(f"Leverage: {SHARED['leverage']}x | Equity fraction: {SHARED['equity_fraction']*100:.0f}%")
    logger.info(f"Costs: {SHARED['taker_fee_bps']} bps taker fee + {SHARED['slippage_bps']} bps slippage per side")
    logger.info(f"Pairs: {list(PAIRS.keys())}")
    logger.info(f"H thresholds: {H_THRESHOLDS}")
    logger.info(f"Leaders: binance, bybit, coinbase | Trade venue: kraken_futures")
    logger.info("Starting collectors (this is where warmup begins — ~30 bars / ~30 min per exchange)")
    collector.start_all(snapshot_interval_sec=10)

    # Periodic stats thread
    def stats_loop():
        while RUNNING:
            time.sleep(300)  # 5 min
            _print_running_stats(mgr, signal_counts, show_entropy=True,
                                  mx_entropy=mx_entropy)

    threading.Thread(target=stats_loop, daemon=True).start()

    try:
        while RUNNING:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    collector.stop_all()
    logger.info("=" * 90)
    logger.info("FINAL STATS")
    _print_running_stats(mgr, signal_counts, final=True)


def _print_running_stats(mgr, signal_counts, show_entropy=False,
                          mx_entropy=None, final=False):
    s = mgr.stats()
    tag = "[FINAL]" if final else "[STATS]"
    logger.info("-" * 90)
    logger.info(f"{tag} equity=${s['equity']:.2f} "
                f"({s['total_return_pct']:+.2f}%) | "
                f"trades={s['trades']} (W:{s['wins']}/L:{s['losses']}) "
                f"WR={s['win_rate']*100:.0f}% | "
                f"signals fired={signal_counts['total']}")
    if s['trades'] > 0:
        logger.info(f"         TP:{s['tp_count']} SL:{s['sl_count']} TO:{s['timeout_count']} | "
                    f"mean ${s['mean_pnl_usd']:+.2f} | "
                    f"best ${s['best']:+.2f} | worst ${s['worst']:+.2f}")
    if show_entropy and mx_entropy:
        lines = []
        for (ex, sym), engine in mx_entropy.engines.items():
            h = engine.last_H
            thresh = H_THRESHOLDS.get(ex, 0.40)
            status = "BELOW" if h is not None and h < thresh else "above"
            h_str = f"{h:.4f}" if h is not None else "warming"
            lines.append(f"  {ex}:{sym} H={h_str} [{status} {thresh}] bars={engine.bar_count}")
        logger.info("Entropy state:")
        for l in lines:
            logger.info(l)
    logger.info("-" * 90)


if __name__ == "__main__":
    main()
