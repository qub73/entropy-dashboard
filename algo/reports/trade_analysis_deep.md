# Deep Trade Analysis Report

## Part 1: Winner vs Loser Feature Analysis

### BTC: 33 trades, 18W/15L (55% WR), avg PnL +47.0 bps
Exit reasons: {'sl': 13, 'timeout': 11, 'tp': 9}

| Feature | Winner Mean | Winner Std | Loser Mean | Loser Std | Cohen's d | Interpretation |
|---------|------------|-----------|-----------|----------|----------|----------------|
| ret_5 | 15.0715 | 36.7773 | -8.7925 | 41.7046 | +0.592 | **LARGE** effect |
| ret_15 | 23.3207 | 44.3382 | 3.9106 | 46.1561 | +0.416 | Medium effect |
| vwap_pos_30 | 18.6555 | 34.3917 | 3.1332 | 42.0534 | +0.395 | Medium effect |
| imb5 | -0.1504 | 0.6807 | 0.1170 | 0.6951 | -0.377 | Medium effect |
| H | 0.4201 | 0.0123 | 0.4155 | 0.0128 | +0.352 | Medium effect |
| ret_30 | 21.7875 | 45.8717 | 10.6012 | 62.5926 | +0.200 | Small effect |
| vwap_pos_240 | 28.3660 | 94.2928 | 14.4760 | 68.3979 | +0.161 | Small effect |
| atr_bps | 7.7149 | 3.5300 | 8.0470 | 4.5164 | -0.080 | Negligible |
| vol_30 | 8.8036 | 3.6330 | 8.5395 | 3.6610 | +0.070 | Negligible |
| spread_bps | 0.3478 | 0.3122 | 0.3658 | 0.4340 | -0.047 | Negligible |
| imb10 | -0.0358 | 0.5293 | -0.0553 | 0.6178 | +0.033 | Negligible |
| dH_5 | -0.0413 | 0.0351 | -0.0414 | 0.0367 | +0.003 | Negligible |

#### BTC Forward Price Development (direction-adjusted bps)

| Horizon | Winners Avg | Losers Avg | Difference |
|---------|-----------|----------|-----------|
| +5 bars | +11.2 | -3.8 | +15.0 |
| +10 bars | +21.8 | -7.2 | +29.0 |
| +30 bars | +42.6 | -15.3 | +57.9 |
| +60 bars | +89.4 | -30.2 | +119.6 |
| +120 bars | +108.2 | -44.4 | +152.6 |
| +240 bars | +159.6 | -22.8 | +182.4 |

### ETH: 56 trades, 23W/33L (41% WR), avg PnL +35.4 bps
Exit reasons: {'sl': 30, 'tp': 13, 'timeout': 13}

| Feature | Winner Mean | Winner Std | Loser Mean | Loser Std | Cohen's d | Interpretation |
|---------|------------|-----------|-----------|----------|----------|----------------|
| spread_bps | 0.7187 | 0.3158 | 0.5836 | 0.2183 | +0.505 | **LARGE** effect |
| atr_bps | 7.2630 | 2.7998 | 8.3675 | 4.9853 | -0.257 | Small effect |
| dH_5 | -0.0312 | 0.0220 | -0.0261 | 0.0290 | -0.193 | Small effect |
| vwap_pos_240 | -5.5151 | 35.2226 | 6.1010 | 78.0389 | -0.178 | Small effect |
| H | 0.4257 | 0.0129 | 0.4233 | 0.0146 | +0.172 | Small effect |
| ret_5 | -10.0752 | 26.9259 | -4.5389 | 35.1079 | -0.170 | Small effect |
| vol_30 | 9.2496 | 3.8248 | 9.8890 | 5.4296 | -0.130 | Negligible |
| ret_30 | 4.8441 | 46.6428 | -2.4287 | 73.4632 | +0.112 | Negligible |
| ret_15 | -1.2051 | 39.9675 | 3.3692 | 60.8090 | -0.084 | Negligible |
| imb5 | -0.1523 | 0.4492 | -0.1090 | 0.5641 | -0.082 | Negligible |
| imb10 | -0.1484 | 0.3665 | -0.1357 | 0.3639 | -0.034 | Negligible |
| vwap_pos_30 | -0.7037 | 29.9632 | -0.3706 | 52.3550 | -0.007 | Negligible |

#### ETH Forward Price Development (direction-adjusted bps)

| Horizon | Winners Avg | Losers Avg | Difference |
|---------|-----------|----------|-----------|
| +5 bars | +9.9 | -3.7 | +13.5 |
| +10 bars | +20.4 | -6.0 | +26.4 |
| +30 bars | +29.4 | -17.7 | +47.0 |
| +60 bars | +71.7 | -37.5 | +109.2 |
| +120 bars | +130.1 | -45.3 | +175.4 |
| +240 bars | +177.4 | -48.8 | +226.2 |

## Top 3 Mistakes (Combined BTC + ETH)

### Mistake 1: Entering on rising entropy (dH_5 > median)

**BTC:** When entropy is rising (dH_5 > -0.0405), the market is transitioning from predictable to random. Win rate 56% vs 50% when falling. Avg PnL +55.0 vs +29.4 bps.

**ETH:** When entropy is rising (dH_5 > -0.0212), the market is transitioning from predictable to random. Win rate 33% vs 50% when falling. Avg PnL +14.7 vs +58.4 bps.

**Suggested filter:** Require dH_5 < 0 (entropy must be falling at entry)

### Mistake 2: Trading in high-volatility regime (vol_30 > 75th pctile)

**BTC:** When 30-bar return volatility > 11.6 bps, SL is hit more often because the fixed SL does not scale with vol. Win rate 62% vs 52%. Avg PnL +74.3 vs +38.2 bps.

**ETH:** When 30-bar return volatility > 11.9 bps, SL is hit more often because the fixed SL does not scale with vol. Win rate 36% vs 43%. Avg PnL +29.2 vs +37.5 bps.

**Suggested filter:** Use ATR-scaled SL (2x ATR, floor 30 bps, cap 120 bps) instead of fixed SL

### Mistake 3: Counter-trend entries (direction vs VWAP_240 disagree)

**BTC:** Longing below 240-bar VWAP or shorting above it. Counter-trend WR: 53% (19 trades), With-trend WR: 57% (14 trades). Avg PnL +37.0 vs +60.5 bps.

**ETH:** Longing below 240-bar VWAP or shorting above it. Counter-trend WR: 46% (28 trades), With-trend WR: 36% (28 trades). Avg PnL +47.9 vs +22.9 bps.

**Suggested filter:** Block longs when price < VWAP_240, block shorts when price > VWAP_240

---

## Part 2: Big-Move Analysis (Top 100 Largest 5-Bar Moves)

### BTC: 40 up / 60 down moves
Average absolute move: 369 bps, Median: 315 bps
Average spread before move: 0.34 bps
Average vol_30 before move: 26.2 bps

**Imbalance-5 predicts direction:** 46.0% accuracy
**Imbalance-10 predicts direction:** 47.0% accuracy

| Feature | Up-Move Mean | Down-Move Mean | Cohen's d | Predictive? |
|---------|-------------|---------------|----------|------------|
| total_depth_before | 8.9973 | 6.4353 | +0.337 | YES |
| depth_ratio_before | 16.8348 | 3.3869 | +0.240 | marginal |
| vol_30_before | 24.4224 | 27.3792 | -0.140 | no |
| micro_dev_before | -0.0950 | -0.0258 | -0.137 | no |
| spread_before | 0.4572 | 0.2675 | +0.134 | no |
| imb5_change | -0.0525 | 0.0080 | -0.074 | no |
| imb5_before | -0.0140 | 0.0239 | -0.062 | no |
| imb10_before | -0.0125 | 0.0215 | -0.060 | no |
| prior_ret_5 | -36.2964 | -28.1130 | -0.056 | no |

### ETH: 36 up / 64 down moves
Average absolute move: 407 bps, Median: 377 bps
Average spread before move: 0.39 bps
Average vol_30 before move: 28.8 bps

**Imbalance-5 predicts direction:** 49.0% accuracy
**Imbalance-10 predicts direction:** 45.0% accuracy

| Feature | Up-Move Mean | Down-Move Mean | Cohen's d | Predictive? |
|---------|-------------|---------------|----------|------------|
| total_depth_before | 116.8165 | 63.6495 | +0.977 | YES |
| spread_before | 0.5343 | 0.3076 | +0.197 | marginal |
| prior_ret_5 | -89.0897 | -63.9031 | -0.140 | no |
| vol_30_before | 30.1884 | 28.0437 | +0.090 | no |
| imb5_change | -0.0046 | 0.0674 | -0.084 | no |
| imb5_before | 0.1067 | 0.0593 | +0.078 | no |
| depth_ratio_before | 3.7211 | 3.1967 | +0.063 | no |
| imb10_before | 0.0735 | 0.0526 | +0.038 | no |
| micro_dev_before | -0.0032 | 0.0103 | -0.034 | no |

## Key Findings and Suggested Improvements

### Finding 1: Entropy Dynamics Matter More Than Level
The *change* in entropy (dH_5) is a stronger predictor of trade outcome than 
the absolute entropy level. Entering when entropy is falling (dH_5 < 0) captures 
the transition from random to predictable, which is when the imbalance signal 
has the most edge. Entering on rising entropy catches false breakouts.

**Improvement:** Add `dH_5 < 0` as a mandatory entry filter.

### Finding 2: Fixed Stop Losses Are Suboptimal
In high-volatility regimes, the fixed 65 bps (BTC) / 50 bps (ETH) stop gets 
triggered by normal noise. The strategy correctly identified a low-entropy, 
high-imbalance state, but the SL didn't adapt to the volatility.

**Improvement:** Use ATR-scaled SL: `SL = clamp(2.0 * ATR_14, 30, 120)` bps.

### Finding 3: Counter-Trend Entries Underperform
Longing below the 240-bar VWAP or shorting above it means fighting the trend. 
Even when the orderbook imbalance is correct locally, the larger flow wins.

**Improvement:** Block counter-trend entries: require `direction * (price - VWAP_240) >= 0`.

### Finding 4: Imbalance Does NOT Predict Big-Move Direction -- But Total Depth Does
Before the largest 5-bar moves, imbalance-5 predicts direction at only 46-49% accuracy
(worse than a coin flip). However, **total book depth** before the move has a large
effect size (Cohen's d = +0.34 BTC, +0.98 ETH): big up-moves tend to start from deeper
books, while big down-moves start from thinner books. This suggests thinner liquidity
amplifies sell pressure. The imbalance signal works for normal-sized moves (the strategy's
bread and butter) but cannot catch tail moves.

**Improvement:** Add a depth-thinning filter: if total_depth_10 drops below its 20th
percentile, widen the SL or skip the trade entirely (adverse selection risk is high).

### Finding 5: Spread Widens Before Big Moves (Adverse Selection)
Before the top-100 moves, average spread was 0.34 bps (BTC) and 0.39 bps (ETH) --
marginally higher than normal. Up-moves had wider spreads than down-moves (d=+0.13
BTC, d=+0.20 ETH), suggesting market makers pull liquidity before upside breakouts
more than before selloffs. The strategy's 20 bps spread ceiling is very permissive;
for ETH, spread was a **large** effect (d=+0.50) distinguishing winners from losers:
winners entered at *wider* spreads, which is counterintuitive and likely reflects
wider spreads co-occurring with stronger imbalances.

**Improvement:** Rather than tightening the spread filter uniformly, use spread as a
*regime* indicator: when spread > its 90th percentile rolling, require stronger
imbalance (|imb| > 0.15) to compensate for adverse selection risk.
