# Multi-Exchange Entropy Trader

Fork of the single-exchange entropy trader (`entropy_live_multi.py`) that
adds real-time L2 orderbook collection from multiple exchanges and uses
cross-exchange lead-lag signals to trade on Kraken Futures.

## Thesis

The exchange with the deepest liquidity and lowest latency (Binance for crypto)
tends to lead price discovery on other venues by 100-500ms. When Binance's
order-flow entropy collapses before Kraken's does — and leaders agree on
direction via imbalance — we can front-run the catch-up move on Kraken.

## Files

| File | Purpose |
|---|---|
| `collectors.py` | L2 WebSocket clients for Binance, Bybit, Coinbase, Kraken Futures |
| `mx_entropy.py` | Per-exchange rolling 27-state Markov entropy engine |
| `cross_exchange_signal.py` | Aggregates state and fires cross-exchange signals |
| `entropy_live_mx.py` | Live trader: orchestrates collectors + signal + execution |

## Exchanges & symbols

Trade venue is **Kraken Futures** (`PF_XBTUSD`, `PF_ETHUSD`). Leaders are
Binance (`BTCUSDT`, `ETHUSDT`), Bybit (`BTCUSDT`, `ETHUSDT`), Coinbase
(`BTC-USD`, `ETH-USD`). All connections are free public WebSockets — no
API keys needed except for Kraken trading.

## Signal logic

```
check_signal:
  1. Kraken spread <= 20 bps
  2. Kraken trailing 5-min return in [20, 80] bps
  3. At least 1 leader exchange has entropy < its threshold
  4. >= 60% of triggered leaders agree on direction via imbalance
  5. Cooldown 60s since last signal
  -> Trade direction = majority vote of leader imbalance signs
```

## Differences from single-exchange bot

| Aspect | `entropy_live_multi.py` | `entropy_live_mx.py` |
|---|---|---|
| Signal source | Kraken only | 3 leader exchanges |
| Direction rule | Kraken imbalance | Leader imbalance consensus |
| Latency edge | None | Front-run Kraken's catch-up |
| Data deps | Kraken WS | Kraken + Binance + Bybit + Coinbase |

## Run

```bash
# Foreground
python algo/multi_exchange/entropy_live_mx.py

# Detached (Windows)
start /B pythonw algo/multi_exchange/entropy_live_mx.py

# Logs
algo/multi_exchange/logs/mx_live.log

# Orderbook data written to (for later offline backtesting)
algo/multi_exchange/data/ob_<exchange>_<symbol>_<YYYYMMDD>.jsonl
```

## Thresholds

Per-exchange `H` thresholds are in `H_THRESHOLDS` in `entropy_live_mx.py`.
Starting values are conservative (Kaggle-fitted):

| Exchange | H threshold |
|---|---|
| Binance | 0.35 |
| Bybit | 0.40 |
| Coinbase | 0.42 |
| Kraken Futures | 0.35 |

These should be refit per-exchange once enough live data is collected
(several days of JSONL snapshots in `data/`).

## Backtest with collected data

The collector writes JSONL files in the same format as the Pi collector,
so existing backtest code (`algo/run_ob_optimization.py`) works directly
on this data for each exchange.

## Next steps (TODO)

- [ ] Backtest on historical Binance+Kraken synchronized orderbook (e.g. Tardis)
  to measure actual lead-lag and calibrate `min_leaders_triggered` vs `direction_agreement`
- [ ] Add latency measurement per collector to weight leader contributions
- [ ] Evaluate feature: relative H (cross-exchange H divergence) as direction signal
- [ ] Extend to more pairs (SOLUSDT, XRPUSDT) — all code is pair-generic
