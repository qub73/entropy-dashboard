# Entropy Trader — Caveats & Next Steps

**As of:** 2026-04-15
**Live track record:** 8 trades over ~2 days, 38% WR, **−1,069 bps @10x cumulative (−$304 USD, −17% on equity)**

The first trade was a +20% winner that made me overconfident. Five consecutive losers since have wiped out those gains and then some. This document is what I should have foregrounded earlier.

---

## 1. Data Quality Caveats

### The orderbook data isn't what the paper used

- **Singha (2025) used tick-by-tick data** — every trade event with full timestamps. We use 1-minute resampled bars from a Pi WebSocket collector, with imbalance/spread/mid recomputed from snapshots. Resampling collapses microstructure information that drives the original signal.
- **The 27-state Markov chain has 729 transitions to estimate**, but a 30-bar window only provides 29 transitions. The transition matrix is overwhelmingly Laplace-smoothed, not data-driven. This means the "entropy" measure is more about regime classification than true predictability.
- **Kaggle training data is from Binance** (2023-2024). Live data is from Kraken Futures (2026). Different exchange microstructure, different liquidity, different latency, different spread structure. Cross-venue transfer is not guaranteed.
- **Spread on Binance was effectively 0 bps** in the Kaggle data (mean=0.00 in the loader output). Kraken Futures spreads run 3-20 bps. The signal trained on a different microstructure regime than where it trades.

### State estimation is subtly path-dependent

- Entropy quintile boundaries (volume terciles, spread terciles) are computed on a **rolling window**. A regime shift changes the boundaries, which changes the state assignment, which changes the transition matrix, which changes the entropy. The signal can drift without any "real" change in market behavior.

---

## 2. Live Test Duration

### 8 trades is statistically meaningless

- Backtest claimed +171% with 32 trades over 21 days. Live: −17% with 8 trades over 2 days.
- Standard error on a 50% binomial WR estimate from 8 trials is **±18 percentage points**. The 38% live WR is well within noise of the 56% backtest WR.
- The strategy is designed for **fat-tail capture** — 9 of 32 backtest trades were full TPs (+200 bps each). Those 9 trades produced ~+1,800 bps unleveraged. The other 23 net to roughly −600 bps. **You need to actually live through the TP-clusters or the math doesn't work.**
- We've had zero TPs live so far. 0 of 8 = an unsustainable losing streak that probably reverses, OR a sign that the live regime is fundamentally different from the backtest period.
- **Sample size required for 95% confidence the live edge matches backtest:** roughly 100-200 trades. At 1-2 trades/day that's 2-3 months of live operation.

### Regime-specific edge

- The Pi-data backtest (Feb-Mar 2026) was a **specific market regime** — mid-volatility crypto with high overnight order flow concentration. April 2026 looks lower-vol, higher chop.
- The "entropy collapse" signal fires when one side of the book gets aggressive. In a dead market, false signals fire on minor imbalances that don't follow through.

---

## 3. Overfitting Risk

### What we did right
- Fit thresholds on Kaggle (out-of-sample exchange + time period)
- Tested on Pi data only after fitting
- Held out a separate test slice for validation
- Used walk-forward in places

### What we did wrong
- **Tested ~190 parameter combinations** in `stop_sweep.py` and picked the best Calmar. With 190 trials and 21-day test window, Bonferroni correction implies the "winner" needs to be ~3 standard deviations better than baseline to be statistically significant. The +171% baseline winner was probably a few percentage points above the median of the search — well within selection bias range.
- **Picked SL=65 / TP=200 specifically because it won the search**, not because it has independent theoretical justification. The 3:1 reward/risk ratio looks principled but emerged from the optimizer.
- **Cross-tested ETH using BTC's threshold approach** — same fitting procedure on ETH gave SL=50 / TP=200. Two-pair concordance reduces overfit risk somewhat but both fits used the same Pi test period.
- **Direction rule (use imbalance sign) was confirmed retrospectively** as 62% accurate on Pi. But that's the IN-SAMPLE accuracy on the data we then tested everything else against. True out-of-sample direction accuracy is unknown.
- **Live results suggest some overfit:** 38% WR vs 56% backtest WR is the kind of drop characteristic of selection bias.

---

## 4. The Single Most Important Next Test

**Run a true forward test on completely fresh data** — accumulate 4-6 weeks of new Pi orderbook data (April-May 2026) and re-evaluate the same fixed parameters (`SL=65 TP=200 TO=240`) on it before changing anything.

### Why this matters more than anything else

Every other improvement (multi-exchange, HMM filter, adaptive exits, TimesFM confirmation) gets validated against the SAME 21-day Pi test set. If that test set is a lucky regime, all our optimizations are overfitted to it. Adding more cleverness without fresh validation just amplifies the overfit.

### Concrete test plan (~1 week of dev, 6 weeks of waiting)

```
Week 0:
  - Stop modifying signal/exit logic in production
  - Keep collecting Pi orderbook data (already happening)
  - Reduce live position size to 0.001 BTC (~$73 notional) to bleed less while we wait
  - Or pause live trading entirely; keep collector running

Weeks 1-6: Pure data collection
  - Pi collects 24/7 BTC + ETH orderbook snapshots
  - Bot continues at minimal size for live signal log

Week 7: True out-of-sample evaluation
  - Run full backtest on the new 6-week period
  - Use the EXACT SAME thresholds fit on Kaggle (no re-fitting on the new data)
  - Compare: does +171% / 56% WR replicate?
  - If yes, we have a real edge — can scale up
  - If no, what failed: signal frequency, direction accuracy, payoff structure?
```

### Specific metrics that must hold

To call the strategy validated, these need to survive on the fresh 6-week sample:

1. **Win rate ≥ 45%** (backtest 56%, allowing 11pp degradation)
2. **Mean trade PnL > 0** unleveraged (after costs)
3. **Cumulative PnL > 0** at the configured leverage
4. **Direction accuracy ≥ 55%** on entries
5. **At least 30 trades** to make any of this statistically meaningful

If WR < 35% or fewer than 15 trades fire over 6 weeks, the signal is dead in the current regime and needs a structural rethink — not parameter tweaking.

### What I'd build immediately to support this

A **shadow-mode logger** that records every signal candidate (even rejected ones) with all the features used to evaluate it. Right now the bot only logs trades that fired. We need:
- Every bar where H < threshold (regardless of other filters)
- The full filter outcome (which conditions blocked entry)
- The actual subsequent price path for 5/10/30/60 bars
- Whether the imbalance-direction was correct in retrospect

This dataset is what lets us do a real attribution analysis after the 6-week window: "Did the signal correctly identify the move? Did the direction rule fail? Did the SL/TP structure fail? Did transaction costs dominate?"

I can build this shadow logger today — it's an additive write to a JSONL file inside the existing on_snapshot callback, ~50 lines of code. Want me to ship it before reducing position size?
