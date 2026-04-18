#!/usr/bin/env python3
"""
Multi-Pair Order-Flow Entropy Live Trader — Kraken Futures
==========================================================
Priority mode: BTC + ETH, lowest entropy gets the capital.
One position at a time, 90% equity, 10x leverage.
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

try:
    import requests
except ImportError:
    requests = None

try:
    import websocket
except ImportError:
    websocket = None

# ---- Configuration ----

PAIRS = {
    "BTC": {
        "futures_symbol": "PF_XBTUSD",
        "ws_symbol": "PI_XBTUSD",
        "stop_loss_bps": 65,
        "take_profit_bps": 200,
        "timeout_minutes": 240,
        "h_thresh": 0.3500,
        # Upgrade 1: cooldown 30 bars after a losing trade (BTC backtest +2.19 Calmar)
        "cooldown_bars_after_loss": 30,
        # No-falling-knife not used on BTC (backtest showed slight loss)
        "knife_threshold_bps": None,
        # T5: at timeout, if in profit, switch to 2x ATR trailing stop.
        "trailing_atr_mult": 2.0,
        # TP-trail: once profit reaches +150 bps, replace fixed TP with 50 bps trail.
        # Cross-pair winner: BTC Cal 13.54->15.58, ETH Cal 9.03->12.69. Combined +618% vs +500%.
        "trail_tp_after": 150,
        "trail_tp_bps": 50,
        # Skip entry if 2.5h (150-bar) move in trade direction already exceeds this.
        # Rationale: strategy is mean-reverting; entries after extended moves in the
        # same direction are chasing exhaustion. Backtest (ETH) Cal 31.30 -> 40.16.
        "extended_move_lookback": 150,
        "extended_move_cap_bps": 100,
    },
    "ETH": {
        "futures_symbol": "PF_ETHUSD",
        "ws_symbol": "PI_ETHUSD",
        "stop_loss_bps": 50,
        "take_profit_bps": 200,
        "timeout_minutes": 240,
        "h_thresh": 0.4352,
        "cooldown_bars_after_loss": 0,
        "knife_threshold_bps": 50,
        "trailing_atr_mult": 2.0,
        "trail_tp_after": 150,
        "trail_tp_bps": 50,
        "extended_move_lookback": 150,
        "extended_move_cap_bps": 100,
    },
}

SHARED_CONFIG = {
    "entropy_percentile": 3,
    "imbalance_threshold": 0.05,
    "spread_max_bps": 20,
    "return_lookback": 5,
    "return_low_bps": 20,
    "return_high_bps": 80,
    "entropy_window": 30,
    "leverage": 10,
    "equity_fraction": 0.90,
}

LOG_DIR = Path(__file__).parent / "logs"
STATE_DIR = Path(__file__).parent / "state"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "entropy_multi.log"),
    ],
)
logger = logging.getLogger("entropy_multi")

RUNNING = True

def _signal_handler(sig, frame):
    global RUNNING
    RUNNING = False
    logger.info("Shutdown requested")

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ---- Kraken API (via SDK) ----

class KrakenFuturesAPI:
    def __init__(self, api_key, api_secret):
        from kraken.futures import User, Trade, Market
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

    def cancel_all(self, symbol=None):
        if symbol:
            return self._trade.cancel_all_orders(symbol=symbol)
        return self._trade.cancel_all_orders()


# ---- Orderbook ----

class L2Book:
    __slots__ = ("bids", "asks", "last_update")
    def __init__(self):
        self.bids = {}
        self.asks = {}
        self.last_update = 0.0

    def apply_snapshot(self, bids, asks):
        self.bids = {float(p): float(s) for p, s in bids}
        self.asks = {float(p): float(s) for p, s in asks}
        self.bids = {p: s for p, s in self.bids.items() if s > 0}
        self.asks = {p: s for p, s in self.asks.items() if s > 0}
        self.last_update = time.time()

    def apply_delta(self, side, price, qty):
        book = self.bids if side == "buy" else self.asks
        p, s = float(price), float(qty)
        if s <= 0:
            book.pop(p, None)
        else:
            book[p] = s
        self.last_update = time.time()

    def top_n(self, n=10):
        bids = sorted(self.bids.items(), key=lambda x: -x[0])[:n]
        asks = sorted(self.asks.items(), key=lambda x: x[0])[:n]
        return bids, asks

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

    def imbalance_5(self):
        bids, asks = self.top_n(5)
        bd = sum(s for _, s in bids)
        ad = sum(s for _, s in asks)
        total = bd + ad
        return (bd - ad) / total if total > 0 else 0.0


# ---- Entropy Engine (per pair) ----

NUM_STATES = 27

def state_index(sign, imb_tercile, spread_regime):
    sign_idx = {-1: 0, 0: 1, 1: 2}[sign]
    return sign_idx * 9 + (imb_tercile - 1) * 3 + (spread_regime - 1)


class EntropyEngine:
    def __init__(self, window=30, atr_window=14):
        self.window = window
        self.atr_window = atr_window
        # Buffer sized to support up to 150-bar (2.5h) trailing returns + margin.
        self.mids = deque(maxlen=window + 300)
        self.imbalances = deque(maxlen=window + 300)
        self.spreads_bps = deque(maxlen=window + 300)
        self.states = deque(maxlen=window + 300)
        self.counts = np.zeros((NUM_STATES, NUM_STATES), dtype=np.float64)
        self.bar_count = 0
        self.last_H = None
        # Bar high/low for ATR (aggregated from tick data)
        self.bar_highs = deque(maxlen=atr_window + 50)
        self.bar_lows = deque(maxlen=atr_window + 50)
        self.true_ranges = deque(maxlen=atr_window + 50)
        self.last_atr_bps = None
        # H history for dH computation
        self.h_history = deque(maxlen=50)

    def on_bar(self, mid, imbalance, spread_bps, bar_high=None, bar_low=None):
        self.mids.append(mid)
        self.imbalances.append(imbalance)
        self.spreads_bps.append(spread_bps)
        self.bar_count += 1

        # Track high/low for ATR. Fallback: approximate from spread.
        if bar_high is None or bar_low is None:
            half_spread = mid * spread_bps / 2 / 10000
            bar_high = mid + half_spread
            bar_low = mid - half_spread
        self.bar_highs.append(bar_high)
        self.bar_lows.append(bar_low)

        # True range
        if len(self.mids) >= 2:
            prev_close = self.mids[-2]
            tr = max(bar_high - bar_low,
                     abs(bar_high - prev_close),
                     abs(bar_low - prev_close))
            self.true_ranges.append(tr)
            if len(self.true_ranges) >= 3:
                atr = float(np.mean(list(self.true_ranges)[-self.atr_window:]))
                self.last_atr_bps = (atr / mid * 10000) if mid > 0 else 0

        if len(self.mids) < 2:
            return None

        price_change = self.mids[-1] - self.mids[-2]
        sign = 1 if price_change > 0 else (-1 if price_change < 0 else 0)

        if len(self.imbalances) >= 30:
            imb_arr = np.array(list(self.imbalances)[-min(self.window, len(self.imbalances)):])
            p33, p67 = np.percentile(imb_arr, [33.3, 66.7])
            imb_t = 1 if imbalance <= p33 else (3 if imbalance >= p67 else 2)
        else:
            imb_t = 2

        if len(self.spreads_bps) >= 30:
            spr_arr = np.array(list(self.spreads_bps)[-min(self.window, len(self.spreads_bps)):])
            sp33, sp67 = np.percentile(spr_arr, [33.3, 66.7])
            spr_r = 1 if spread_bps <= sp33 else (3 if spread_bps >= sp67 else 2)
        else:
            spr_r = 2

        s = state_index(sign, imb_t, spr_r)
        self.states.append(s)

        if len(self.states) < 2:
            return None

        self.counts[self.states[-2], self.states[-1]] += 1.0
        if len(self.states) > self.window:
            old = list(self.states)
            idx = len(old) - self.window - 1
            if idx >= 0 and idx + 1 < len(old):
                self.counts[old[idx], old[idx+1]] -= 1.0
                self.counts[old[idx], old[idx+1]] = max(0, self.counts[old[idx], old[idx+1]])

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
        self.h_history.append(H)
        return H

    def get_trailing_return_bps(self, lookback=5):
        if len(self.mids) < lookback + 1:
            return None
        current = self.mids[-1]
        past = list(self.mids)[-lookback - 1]
        if past <= 0:
            return None
        return (current / past - 1.0) * 10000

    def get_dH(self, lookback=5):
        """Return H_now - H_{lookback bars ago}. Negative = entropy falling."""
        if len(self.h_history) < lookback + 1:
            return None
        h_list = list(self.h_history)
        return h_list[-1] - h_list[-lookback - 1]

    def snapshot(self):
        """Return engine state as a JSON-serializable dict for persistence."""
        return {
            "window": self.window,
            "atr_window": self.atr_window,
            "mids": list(self.mids),
            "imbalances": list(self.imbalances),
            "spreads_bps": list(self.spreads_bps),
            "states": list(self.states),
            "counts": self.counts.tolist(),
            "bar_count": self.bar_count,
            "last_H": self.last_H,
            "bar_highs": list(self.bar_highs),
            "bar_lows": list(self.bar_lows),
            "true_ranges": list(self.true_ranges),
            "last_atr_bps": self.last_atr_bps,
            "h_history": list(self.h_history),
        }

    def restore(self, snap):
        """Restore engine state from a snapshot dict. Returns True on success."""
        try:
            self.mids.clear(); self.mids.extend(snap.get("mids", []))
            self.imbalances.clear(); self.imbalances.extend(snap.get("imbalances", []))
            self.spreads_bps.clear(); self.spreads_bps.extend(snap.get("spreads_bps", []))
            self.states.clear(); self.states.extend(snap.get("states", []))
            counts = snap.get("counts")
            if counts:
                self.counts = np.asarray(counts, dtype=np.float64)
            self.bar_count = int(snap.get("bar_count", 0))
            self.last_H = snap.get("last_H")
            self.bar_highs.clear(); self.bar_highs.extend(snap.get("bar_highs", []))
            self.bar_lows.clear(); self.bar_lows.extend(snap.get("bar_lows", []))
            self.true_ranges.clear(); self.true_ranges.extend(snap.get("true_ranges", []))
            self.last_atr_bps = snap.get("last_atr_bps")
            self.h_history.clear(); self.h_history.extend(snap.get("h_history", []))
            return True
        except Exception:
            return False


# ---- Position & State ----

class MultiPairManager:
    def __init__(self, api, state_file):
        self.api = api
        self.state_file = Path(state_file)
        self.position = None  # {pair, direction, entry_price, entry_time, size, symbol}
        self.trades = []
        self.daily_pnl = 0.0
        self.current_day = -1
        self._load_state()

    def _load_state(self):
        # Full trade history lives in a separate append-only log
        history_file = self.state_file.parent / "trade_history.jsonl"
        self._history_file = history_file
        if history_file.exists():
            full_history = []
            with open(history_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        full_history.append(json.loads(line))
                    except Exception:
                        pass
            self.full_history = full_history
            logger.info(f"Loaded {len(full_history)} historical trades from {history_file.name}")
        else:
            self.full_history = []

        if self.state_file.exists():
            with open(self.state_file) as f:
                state = json.load(f)
            self.position = state.get("position")
            self.trades = state.get("trades", [])
            self.daily_pnl = state.get("daily_pnl", 0.0)
            logger.info(f"Loaded state: pos={self.position is not None}, "
                        f"recent trades={len(self.trades)}, total history={len(self.full_history)}")

    def _append_history(self, trade):
        """Append single trade to permanent history log (never truncated)."""
        try:
            with open(self._history_file, "a") as f:
                f.write(json.dumps(trade) + "\n")
            self.full_history.append(trade)
        except Exception as e:
            logger.error(f"History append error: {e}")

    def _save_state(self):
        state = {
            "position": self.position,
            "trades": self.trades[-100:],  # rolling recent for quick inspection
            "total_trades": len(self.full_history),
            "daily_pnl": self.daily_pnl,
            "last_update": time.time(),
            "cumulative": self._cumulative_stats(),
        }
        tmp = str(self.state_file) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, str(self.state_file))

    def _cumulative_stats(self):
        """Compute all-time stats from full_history.

        Note: total_pnl_bps_lev is a compound return (not arithmetic sum).
        pnl_bps_leveraged per trade is the leveraged price move, which
        equals the ~equity return for that trade. Compounding these is the
        correct way to get lifetime return; summing would overstate gains
        and understate losses because each trade is applied to current equity.
        """
        h = self.full_history
        if not h:
            return {"trades": 0, "wins": 0, "losses": 0, "win_rate": 0,
                    "total_pnl_bps_lev": 0, "best_bps_lev": 0, "worst_bps_lev": 0,
                    "tp": 0, "sl": 0, "timeout": 0, "other": 0,
                    "long_trades": 0, "long_wr": 0, "short_trades": 0, "short_wr": 0}
        pnls = [t.get("pnl_bps_leveraged", 0) for t in h]
        wins = sum(1 for p in pnls if p > 0)
        losses = sum(1 for p in pnls if p < 0)
        reasons = {"tp": 0, "sl": 0, "timeout": 0, "other": 0}
        for t in h:
            r = t.get("reason", "other")
            if r in reasons:
                reasons[r] += 1
            else:
                reasons["other"] += 1
        longs = [t for t in h if t.get("direction") == 1]
        shorts = [t for t in h if t.get("direction") == -1]
        long_wins = sum(1 for t in longs if t.get("pnl_bps_leveraged", 0) > 0)
        short_wins = sum(1 for t in shorts if t.get("pnl_bps_leveraged", 0) > 0)

        equity = 1.0
        for p in pnls:
            equity *= (1.0 + p / 10000.0)
        compound_bps = (equity - 1.0) * 10000

        return {
            "trades": len(h),
            "wins": wins,
            "losses": losses,
            "win_rate": wins / len(h) if h else 0,
            "total_pnl_bps_lev": compound_bps,
            "sum_pnl_bps_lev": sum(pnls),
            "best_bps_lev": max(pnls) if pnls else 0,
            "worst_bps_lev": min(pnls) if pnls else 0,
            "mean_bps_lev": sum(pnls) / len(pnls) if pnls else 0,
            "tp": reasons["tp"],
            "sl": reasons["sl"],
            "timeout": reasons["timeout"],
            "other": reasons["other"],
            "long_trades": len(longs),
            "long_wr": long_wins / len(longs) if longs else 0,
            "short_trades": len(shorts),
            "short_wr": short_wins / len(shorts) if shorts else 0,
        }

    def has_position(self):
        return self.position is not None

    def _verify_fill(self, result, label):
        """
        Verify order was actually filled. Returns (order_id, fill_price, fill_size) or None.
        Checks:
          1. result["result"] == "success"
          2. sendStatus["status"] == "placed"
          3. At least one EXECUTION event exists
          4. Logs full response for debugging
        """
        logger.info(f"  {label} raw response: {json.dumps(result)[:500]}")

        if result.get("result") != "success":
            logger.error(f"  {label} API result != success: {result.get('result')}")
            return None

        send = result.get("sendStatus", {})
        status = send.get("status", "")
        order_id = send.get("order_id", "unknown")

        if status != "placed":
            logger.error(f"  {label} order status={status} (not placed), id={order_id}")
            return None

        # Check for actual execution
        events = send.get("orderEvents", [])
        executions = [e for e in events if e.get("type") == "EXECUTION"]

        if not executions:
            logger.error(f"  {label} order placed but NO EXECUTION events, id={order_id}")
            return None

        # Sum up filled amount and get avg price
        total_filled = sum(e.get("amount", 0) for e in executions)
        total_value = sum(e.get("amount", 0) * e.get("price", 0) for e in executions)
        avg_price = total_value / total_filled if total_filled > 0 else 0

        logger.info(f"  {label} FILLED: {total_filled} @ avg ${avg_price:.2f}, "
                    f"{len(executions)} execution(s), id={order_id}")

        return order_id, avg_price, total_filled

    def open_position(self, pair, direction, mid_price, symbol):
        if self.has_position():
            return False

        equity = self.api.get_equity()
        if equity is None or equity <= 0:
            logger.error("Cannot size — equity unavailable")
            return False

        # Use availableMargin instead of portfolioValue to account for existing positions
        accts = self.api.get_accounts()
        flex = accts.get("accounts", {}).get("flex", {})
        available = flex.get("availableMargin", 0)
        if available <= 0:
            logger.error(f"No available margin (available=${available:.2f})")
            return False

        # Use 75% of available margin to leave buffer for Kraken's margin requirements
        margin = available * 0.75
        notional = margin * SHARED_CONFIG["leverage"]

        # Round to exchange precision: PF_XBTUSD=4dp, PF_ETHUSD=3dp
        precision = 4 if "XBT" in symbol else 3
        size = round(notional / mid_price, precision)

        if size < 0.0001:
            logger.warning(f"Size too small: {size}")
            return False

        side = "buy" if direction == 1 else "sell"
        logger.info(f"SUBMITTING [{pair}]: {side.upper()} {size} {symbol} "
                    f"@ ~{mid_price:.1f} (available=${available:.0f}, "
                    f"margin=${margin:.0f}, notional=${notional:.0f})")

        try:
            result = self.api.send_order(symbol, side, size)
        except Exception as e:
            logger.error(f"ORDER EXCEPTION [{pair}]: {e}")
            return False

        fill = self._verify_fill(result, f"ENTRY [{pair}]")
        if fill is None:
            logger.error(f"ENTRY [{pair}] NOT FILLED — no position opened")
            return False

        order_id, fill_price, fill_size = fill

        self.position = {
            "pair": pair,
            "direction": direction,
            "entry_price": fill_price,  # use actual fill price, not mid
            "entry_time": time.time(),
            "size": fill_size,  # use actual filled size
            "symbol": symbol,
            "order_id": order_id,
            # ATR trailing state:
            "peak_mid": fill_price,       # highest mid seen (longs) / lowest (shorts)
            "trailing_active": False,     # switched on when timeout hits with profit
        }
        self._save_state()
        logger.info(f"ENTRY [{pair}]: CONFIRMED {side.upper()} {fill_size} {symbol} "
                    f"@ ${fill_price:.2f}")
        return True

    def close_position(self, mid_price, reason):
        if not self.has_position():
            return None

        pos = self.position
        side = "sell" if pos["direction"] == 1 else "buy"

        # Round size to exchange precision (PF_XBTUSD=4dp, PF_ETHUSD=3dp)
        precision = 4 if "XBT" in pos["symbol"] else 3
        close_size = round(float(pos["size"]), precision)

        logger.info(f"CLOSING [{pos['pair']}] ({reason}): {side.upper()} "
                    f"{close_size} {pos['symbol']}")

        try:
            result = self.api.send_order(pos["symbol"], side, close_size)
        except Exception as e:
            logger.error(f"CLOSE EXCEPTION [{pos['pair']}]: {e}")
            return None

        fill = self._verify_fill(result, f"EXIT [{pos['pair']}]")
        if fill is None:
            logger.error(f"EXIT [{pos['pair']}] NOT FILLED — position still open!")
            return None

        order_id, fill_price, fill_size = fill

        # Use actual fill prices for PnL, not mid
        pnl_bps = pos["direction"] * (fill_price / pos["entry_price"] - 1.0) * 10000
        pnl_lev = pnl_bps * SHARED_CONFIG["leverage"]
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
        self._append_history(trade)  # permanent log — survives restarts
        self.daily_pnl += pnl_lev

        cum = self._cumulative_stats()
        logger.info(f"EXIT [{pos['pair']}] ({reason}): PnL={pnl_bps:.1f}bps "
                    f"({pnl_lev:.1f}bps @{SHARED_CONFIG['leverage']}x), "
                    f"held {hold_min:.1f}min, daily={self.daily_pnl:.1f}bps | "
                    f"LIFETIME: {cum['trades']}tr "
                    f"WR={cum['win_rate']*100:.0f}% "
                    f"total={cum['total_pnl_bps_lev']:.0f}bps@10x")

        self.position = None
        self._save_state()
        return trade

    def check_exit(self, mid_price, atr_bps=None, bar_high=None, bar_low=None):
        """Evaluate exit conditions for the current bar.

        bar_high/bar_low are intra-bar extremes; when provided, peak tracking
        uses bar_high/bar_low (matching backtest semantics) instead of mid-only.
        This avoids under-counting favorable excursions that occurred intra-bar.
        """
        if not self.has_position():
            return None
        pos = self.position
        pair_cfg = PAIRS[pos["pair"]]
        entry = pos["entry_price"]
        d = pos["direction"]
        pnl_bps = d * (mid_price / entry - 1.0) * 10000
        elapsed = (time.time() - pos["entry_time"]) / 60

        # Best/worst intrabar PnL (in bps) in the direction of the trade.
        if bar_high is not None and bar_low is not None:
            if d == 1:
                best_bps  = (bar_high / entry - 1.0) * 10000
                worst_bps = (bar_low  / entry - 1.0) * 10000
            else:
                best_bps  = -(bar_low  / entry - 1.0) * 10000
                worst_bps = -(bar_high / entry - 1.0) * 10000
        else:
            best_bps = worst_bps = pnl_bps

        # Update peak_mid using intrabar favourable price when available.
        favourable_px = bar_high if (d == 1 and bar_high is not None) else \
                        (bar_low if (d == -1 and bar_low is not None) else mid_price)
        prior_peak = pos.get("peak_mid", entry)
        if d == 1:
            if favourable_px > prior_peak: pos["peak_mid"] = favourable_px
        else:
            if favourable_px < prior_peak: pos["peak_mid"] = favourable_px

        peak_pnl = d * (pos["peak_mid"] / entry - 1.0) * 10000

        trailing_active = pos.get("trailing_active", False)
        trail_mult = pair_cfg.get("trailing_atr_mult")
        trail_tp_after = pair_cfg.get("trail_tp_after")
        trail_tp_bps = pair_cfg.get("trail_tp_bps", 50)
        sl_bps = pair_cfg["stop_loss_bps"]
        tp_bps = pair_cfg["take_profit_bps"]

        # --- Fix: activate tp_trailing BEFORE SL check when peak has
        # already crossed the trigger on this bar. Prevents the race where
        # a volatile bar hits both +trigger and -SL and SL wins.
        if (trail_tp_after is not None
                and not pos.get("tp_trailing", False)
                and peak_pnl >= trail_tp_after):
            pos["tp_trailing"] = True
            logger.info(
                f"[{pos['pair']}] Peak reached +{peak_pnl:.1f} bps (>={trail_tp_after}) "
                f"-- switching to {trail_tp_bps} bps trailing TP")
            self._save_state()

        tp_trailing = pos.get("tp_trailing", False)

        # --- ATR-trail after timeout (T5) ---
        if trailing_active and trail_mult and atr_bps and atr_bps > 0:
            trail_width_bps = trail_mult * atr_bps
            floor_bps = max(peak_pnl - trail_width_bps, -sl_bps)
            if worst_bps <= floor_bps:
                pos["_trail_floor_bps"] = floor_bps
                pos["_trail_peak_bps"] = peak_pnl
                return "trail_stop"
            if best_bps >= tp_bps:
                return "tp"
            return None

        # --- TP-trail mode (active after peak crossed trigger) ---
        if tp_trailing:
            floor = max(peak_pnl - trail_tp_bps, -sl_bps)
            if worst_bps <= floor:
                pos["_trail_floor_bps"] = floor
                pos["_trail_peak_bps"] = peak_pnl
                return "tp_trail"
            return None

        # --- Standard logic (no trail phase active) ---
        if worst_bps <= -sl_bps:
            return "sl"

        # Fixed TP only when no TP-trail configured for this pair
        if trail_tp_after is None and best_bps >= tp_bps:
            return "tp"

        # Timeout: if in profit AND ATR trailing configured, switch to ATR trail
        if elapsed >= pair_cfg["timeout_minutes"]:
            if trail_mult and pnl_bps > 0 and atr_bps and atr_bps > 0:
                pos["trailing_active"] = True
                logger.info(
                    f"[{pos['pair']}] TIMEOUT reached in profit ({pnl_bps:.1f} bps) "
                    f"-- switching to {trail_mult}x ATR trailing stop "
                    f"(ATR={atr_bps:.1f} bps, trail_width={trail_mult*atr_bps:.1f} bps)")
                self._save_state()
                return None
            return "timeout"
        return None


# ---- Main Loop ----

def main():
    global RUNNING

    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()

    api_key = os.getenv("KRAKEN_API_FUTURES_KEY", "")
    api_secret = os.getenv("KRAKEN_API_FUTURES_SECRET", "")
    if not api_key:
        logger.error("KRAKEN_API_FUTURES_KEY not set")
        sys.exit(1)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    api = KrakenFuturesAPI(api_key, api_secret)
    mgr = MultiPairManager(api, STATE_DIR / "multi_trader_state.json")

    # Verify API
    try:
        eq = api.get_equity()
        logger.info(f"API connected. Equity: ${eq:.2f}" if eq else "API connected. Equity: $0")
    except Exception as e:
        logger.error(f"API failed: {e}")
        sys.exit(1)

    # Per-pair state
    books = {p: L2Book() for p in PAIRS}
    engines = {p: EntropyEngine(window=SHARED_CONFIG["entropy_window"]) for p in PAIRS}
    last_bar_minute = {p: 0 for p in PAIRS}
    tick_data = {p: {"mids": [], "imbs": [], "spreads": []} for p in PAIRS}
    signal_counts = {p: 0 for p in PAIRS}
    # Per-pair cooldown: bar count when cooldown expires (0 = no cooldown)
    cooldown_until_bar = {p: 0 for p in PAIRS}
    # Filter rejection counts (for visibility)
    rejected_cooldown = {p: 0 for p in PAIRS}
    rejected_knife = {p: 0 for p in PAIRS}
    rejected_extended = {p: 0 for p in PAIRS}

    # ---- Engine state persistence (avoids cold-start warmup on restart) ----
    engine_state_file = STATE_DIR / "engine_state.json"
    ENGINE_STATE_MAX_AGE_SEC = 30 * 60  # 30 min: older than this, restart cold

    def _save_engine_state():
        try:
            data = {
                "saved_at": time.time(),
                "pairs": {p: engines[p].snapshot() for p in PAIRS},
                "cooldown_until_bar": cooldown_until_bar,
            }
            tmp = str(engine_state_file) + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f)
            os.replace(tmp, str(engine_state_file))
        except Exception as e:
            logger.warning(f"engine state save failed: {e}")

    if engine_state_file.exists():
        try:
            with open(engine_state_file) as f:
                saved = json.load(f)
            age = time.time() - saved.get("saved_at", 0)
            if age <= ENGINE_STATE_MAX_AGE_SEC:
                for p in PAIRS:
                    snap = saved.get("pairs", {}).get(p)
                    if snap and engines[p].restore(snap):
                        logger.info(
                            f"[{p}] engine state restored: {engines[p].bar_count} bars, "
                            f"H={engines[p].last_H}, age={age:.0f}s")
                cd = saved.get("cooldown_until_bar", {})
                for p, v in cd.items():
                    if p in cooldown_until_bar:
                        cooldown_until_bar[p] = int(v)
            else:
                logger.info(f"engine state file too old ({age:.0f}s > "
                            f"{ENGINE_STATE_MAX_AGE_SEC}s) -- cold start")
        except Exception as e:
            logger.warning(f"engine state restore failed: {e}")

    logger.info(f"Config: {json.dumps(SHARED_CONFIG, indent=2)}")
    for p, cfg in PAIRS.items():
        logger.info(f"  {p}: {json.dumps(cfg)}")
    logger.info("Starting multi-pair entropy trader (PRIORITY mode)...")

    def process_bar(pair):
        td = tick_data[pair]
        if not td["mids"]:
            return None

        bar_mid = td["mids"][-1]
        bar_imb = np.mean(td["imbs"])
        bar_spread = np.mean(td["spreads"])
        bar_high = max(td["mids"])
        bar_low = min(td["mids"])

        td["mids"].clear()
        td["imbs"].clear()
        td["spreads"].clear()

        H = engines[pair].on_bar(bar_mid, bar_imb, bar_spread, bar_high, bar_low)
        if H is None:
            return None

        cfg = PAIRS[pair]

        # Check exit if we hold this pair
        if mgr.has_position() and mgr.position["pair"] == pair:
            atr_bps = engines[pair].last_atr_bps
            reason = mgr.check_exit(bar_mid, atr_bps=atr_bps,
                                     bar_high=bar_high, bar_low=bar_low)
            if reason:
                trade = mgr.close_position(bar_mid, reason)
                # Set cooldown if this was a loss and pair has cooldown configured
                if trade and trade.get("pnl_bps", 0) < 0:
                    cooldown_bars = cfg.get("cooldown_bars_after_loss", 0)
                    if cooldown_bars > 0:
                        cooldown_until_bar[pair] = engines[pair].bar_count + cooldown_bars
                        logger.info(
                            f"[{pair}] Loss triggered cooldown — blocked until bar "
                            f"{cooldown_until_bar[pair]} ({cooldown_bars} bars from now)")
            return None  # don't enter same bar as exit

        # Check entry conditions
        if mgr.has_position():
            return None  # already in a trade (other pair)

        if H >= cfg["h_thresh"]:
            return None

        # --- I1: Entropy must be FALLING (dH_5 < 0) ---
        # Cross-pair Calmar +3.30 vs baseline. ETH 19.60 -> 27.23, DD 17% -> 13.5%.
        dH = engines[pair].get_dH(5)
        if dH is not None and dH >= 0:
            return None

        if abs(bar_imb) < SHARED_CONFIG["imbalance_threshold"]:
            return None

        if bar_spread > SHARED_CONFIG["spread_max_bps"]:
            return None

        ret = engines[pair].get_trailing_return_bps(SHARED_CONFIG["return_lookback"])
        if ret is None:
            return None
        if abs(ret) < SHARED_CONFIG["return_low_bps"] or abs(ret) > SHARED_CONFIG["return_high_bps"]:
            return None

        direction = 1 if bar_imb > 0 else -1

        # --- Upgrade 1: per-pair cooldown after losing trade ---
        cooldown_bars = cfg.get("cooldown_bars_after_loss", 0)
        if cooldown_bars > 0:
            current_bar = engines[pair].bar_count
            if current_bar < cooldown_until_bar[pair]:
                rejected_cooldown[pair] += 1
                logger.debug(
                    f"[{pair}] signal blocked by cooldown "
                    f"({cooldown_until_bar[pair] - current_bar} bars left)")
                return None

        # --- Upgrade 2: per-pair no-falling-knife filter (60-bar return) ---
        knife_bps = cfg.get("knife_threshold_bps")
        if knife_bps is not None:
            ret_60 = engines[pair].get_trailing_return_bps(60)
            if ret_60 is not None:
                if direction == 1 and ret_60 < -knife_bps:
                    rejected_knife[pair] += 1
                    logger.info(
                        f"[{pair}] LONG signal blocked: ret_60={ret_60:.1f} bps "
                        f"< -{knife_bps} (falling knife)")
                    return None
                if direction == -1 and ret_60 > knife_bps:
                    rejected_knife[pair] += 1
                    logger.info(
                        f"[{pair}] SHORT signal blocked: ret_60={ret_60:.1f} bps "
                        f"> +{knife_bps} (rising knife)")
                    return None

        # --- Upgrade 3: extended-move filter (skip chasing exhausted trends) ---
        # Don't enter in the direction the market has already moved strongly.
        ext_cap = cfg.get("extended_move_cap_bps")
        ext_lb = cfg.get("extended_move_lookback", 150)
        if ext_cap is not None:
            ret_ext = engines[pair].get_trailing_return_bps(ext_lb)
            if ret_ext is not None:
                if direction == 1 and ret_ext > ext_cap:
                    rejected_extended[pair] += 1
                    logger.info(
                        f"[{pair}] LONG signal blocked: ret_{ext_lb}={ret_ext:.1f} bps "
                        f"> +{ext_cap} (extended up-move; don't chase)")
                    return None
                if direction == -1 and ret_ext < -ext_cap:
                    rejected_extended[pair] += 1
                    logger.info(
                        f"[{pair}] SHORT signal blocked: ret_{ext_lb}={ret_ext:.1f} bps "
                        f"< -{ext_cap} (extended down-move; don't chase)")
                    return None

        return {
            "pair": pair,
            "direction": direction,
            "entropy": H,
            "imbalance": bar_imb,
            "mid": bar_mid,
            "ret": ret,
            "spread": bar_spread,
            "symbol": cfg["futures_symbol"],
        }

    def on_message(ws, message):
        try:
            data = json.loads(message)
            feed = data.get("feed")

            # Determine which pair this message is for
            product = data.get("product_id", "")
            pair = None
            for p, cfg in PAIRS.items():
                if cfg["ws_symbol"] == product:
                    pair = p
                    break

            if pair is None:
                if feed == "book_snapshot":
                    product = data.get("product_id", "")
                    for p, cfg in PAIRS.items():
                        if cfg["ws_symbol"] == product:
                            pair = p
                            break
                if pair is None:
                    return

            book = books[pair]

            if feed == "book_snapshot":
                bids = [(l.get("price", 0), l.get("qty", 0)) for l in data.get("bids", [])]
                asks = [(l.get("price", 0), l.get("qty", 0)) for l in data.get("asks", [])]
                book.apply_snapshot(bids, asks)

            elif feed == "book":
                price = data.get("price", 0)
                qty = data.get("qty", 0)
                side = data.get("side", "")
                if side and price:
                    book.apply_delta(side, price, qty)

            # Collect tick data
            if book.mid > 0:
                current_minute = int(time.time()) // 60
                td = tick_data[pair]
                td["mids"].append(book.mid)
                td["imbs"].append(book.imbalance_5())
                td["spreads"].append(book.spread_bps)

                if current_minute > last_bar_minute[pair] and last_bar_minute[pair] > 0:
                    # Process bars and collect signals
                    pending_signals = []
                    for p in PAIRS:
                        if current_minute > last_bar_minute[p] and last_bar_minute[p] > 0:
                            sig = process_bar(p)
                            if sig is not None:
                                pending_signals.append(sig)
                            last_bar_minute[p] = current_minute
                    # Persist engine state so a restart doesn't need a fresh warmup
                    _save_engine_state()

                    # PRIORITY: if multiple signals, pick lowest entropy
                    if pending_signals and not mgr.has_position():
                        pending_signals.sort(key=lambda s: s["entropy"])
                        best = pending_signals[0]
                        signal_counts[best["pair"]] += 1
                        logger.info(
                            f"SIGNAL #{sum(signal_counts.values())} [{best['pair']}]: "
                            f"H={best['entropy']:.4f} (<{PAIRS[best['pair']]['h_thresh']:.4f}), "
                            f"imb={best['imbalance']:.3f}, spread={best['spread']:.1f}bps, "
                            f"ret={best['ret']:.1f}bps -> "
                            f"{'LONG' if best['direction']==1 else 'SHORT'}"
                            + (f" [PRIORITY over {pending_signals[1]['pair']} "
                               f"H={pending_signals[1]['entropy']:.4f}]"
                               if len(pending_signals) > 1 else ""))
                        mgr.open_position(
                            best["pair"], best["direction"],
                            best["mid"], best["symbol"])

                last_bar_minute[pair] = current_minute

        except Exception as e:
            logger.error(f"Message error: {e}")

    def on_open(ws):
        logger.info("WebSocket connected")
        ws_symbols = [cfg["ws_symbol"] for cfg in PAIRS.values()]
        sub = {
            "event": "subscribe",
            "feed": "book",
            "product_ids": ws_symbols,
        }
        ws.send(json.dumps(sub))
        logger.info(f"Subscribed to: {ws_symbols}")

    def on_error(ws, error):
        logger.error(f"WebSocket error: {error}")

    def on_close(ws, code, msg):
        logger.info(f"WebSocket closed: {code} {msg}")

    # Status thread
    def status_loop():
        while RUNNING:
            time.sleep(300)
            status = {
                "position": mgr.position["pair"] if mgr.has_position() else None,
                "trades": len(mgr.trades),
                "daily_pnl": mgr.daily_pnl,
                "signals": dict(signal_counts),
                "rejected_cooldown": dict(rejected_cooldown),
                "rejected_knife": dict(rejected_knife),
                "rejected_extended": dict(rejected_extended),
            }
            for p in PAIRS:
                e = engines[p]
                status[p] = {
                    "bars": e.bar_count,
                    "last_H": e.last_H,
                    "h_thresh": PAIRS[p]["h_thresh"],
                    "mid": books[p].mid,
                }
            logger.info(f"STATUS: {json.dumps(status)}")

    threading.Thread(target=status_loop, daemon=True).start()

    # WebSocket loop
    WS_URL = "wss://futures.kraken.com/ws/v1"
    reconnect_delay = 1

    while RUNNING:
        try:
            ws = websocket.WebSocketApp(
                WS_URL,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
            )
            ws.run_forever(ping_interval=30)
        except Exception as e:
            logger.error(f"WS error: {e}")

        if RUNNING:
            logger.info(f"Reconnecting in {reconnect_delay}s...")
            time.sleep(min(reconnect_delay, 60))
            reconnect_delay = min(reconnect_delay * 2, 60)

    logger.info("Shutting down...")
    if mgr.has_position():
        logger.warning(f"OPEN POSITION at shutdown: {mgr.position['pair']} "
                       f"entry={mgr.position['entry_price']}")
    logger.info(f"Session: {len(mgr.trades)} trades, daily PnL={mgr.daily_pnl:.1f}bps")


if __name__ == "__main__":
    main()
