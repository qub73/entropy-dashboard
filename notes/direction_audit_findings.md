# Phase 1 — Direction audit findings

**Date:** 2026-04-19
**Data:** Pi Kraken Futures ETH L2 orderbook, Feb 18 – Apr 7 2026 (21.3 days, 30,644 × 1-min bars), plus 24 h of Kraken public `/Trades` feed for the trade-flow prototype.
**Report:** `algo/reports/direction_audit.json`, `algo/reports/direction_audit.png`.

---

## Headline

> **Book imbalance is NOT structurally long-biased on Kraken Futures ETH over the 21-day audit window. The 17/17 LONG outcome in the live Apr 13–19 period is a regime artifact, not a signal-polarity artifact.**

If anything, the filter-pass pool is tilted slightly **short** on this window:

| Pool | Long candidates | Short candidates | L/S ratio |
|------|-----------------|------------------|-----------|
| All filter passes except direction | 36 | 49 | 0.73 |
| H < 0.4352 AND \|imb\| > 0.05 | 317 | 399 | 0.79 |

If the imbalance signal were structurally long-biased, we'd expect the backtest (and its parameter-swept result of **17 long / 34 short** reported in earlier reports on the same window) to also go long-heavy. It doesn't.

---

## (a) Is book imbalance biased?

**No.** At every bar over 21 days:

- mean = **−0.007**
- median = **−0.011**
- frac_positive = **0.492**
- p5 / p95 = symmetric about zero
- fraction with \|imb\| > 0.05 = 44% of bars (similar in either direction)

The formula itself is clean (verified in `algo/ob_entropy.py` lines 89–93):

```python
def imbalance(arr_b, arr_a, n):
    bd = depth(arr_b, n)        # sum of size over top-n bid levels
    ad = depth(arr_a, n)        # sum of size over top-n ask levels
    return (bd - ad) / (bd + ad)
```

- Top-5 levels on both sides
- From the same orderbook snapshot
- No asymmetric filtering
- No price-band gating

**Ruled out.** There is no hidden bias in the signal construction on this data window.

---

## (b) Is the entropy threshold polarity-dependent?

**No.** Conditional on imbalance sign over all 21-day bars:

| Subset | Median H | n |
|--------|---------:|--:|
| Bars where imb > 0 | 0.4900 | ~14.8 k |
| Bars where imb < 0 | 0.4905 | ~14.7 k |

Essentially identical. The entropy measure is symmetric with respect to direction. H dropping below 0.4352 happens at roughly equal rate in both polarity regimes:

- Bars with H < threshold **and** imb > 0: 317
- Bars with H < threshold **and** imb < 0: 399

**Ruled out.** Entropy doesn't selectively flag long-leaning states.

---

## (c) Why 17/17 LONG in live, then?

Because **Apr 12–19 was a near-continuous ETH uptrend** (~+1000 bps from 2190 → 2440+). In a sustained trend:

- The book's aggregate depth shifts toward the trend direction — bids sit tight, asks thin out as aggressive buyers lift them
- When entropy drops (state predictability tightens) it tends to do so **at local pullback bottoms** within the uptrend
- So the sign of imbalance at filter-pass moments is almost always positive in a bull regime, even though the *rate* of filter passes isn't biased across all bars

This is consistent with what the per-trade chart review showed: the bot kept entering LONG at the top of shallow dips. The trend kept resuming, sometimes with another leg down first (→ SL hit), sometimes continuing up (→ tp_trail win).

So the 17/17 long isn't the bug. **The regime-mismatch between a mean-reverting entry signal and a trending price path is the bug.** The extended-move filter deployed 2026-04-17 is aimed precisely at this — it would have blocked the worst "chasing a local top" entries.

---

## (d) Trade-flow imbalance prototype

Fetched 19,933 Kraken `/Trades` prints over the last 24 h, bucketed into 5-min windows (289 buckets).

| Metric | Value |
|--------|------:|
| mean 5-min TFI | −0.093 |
| median | −0.059 |
| frac_positive | 0.419 |
| p5 / p95 | −0.73 / +0.34 |

24 h is a small sample (and was a down-leaning day in the live period), but:

- Trade-flow imbalance has a **wider absolute range** than book imbalance (see PNG panel 4) — the tails are more informative.
- It naturally includes the *sign* of aggressor, which book imbalance only approximates.
- It's available in real time via WebSocket (the Pi bot subscribes to `book` but not `trade`; would need a collector change).

**Limitation:** no historical archive. Kraken's `/Trades` REST serves 1,000 trades per page; paginating 21 days of ETH trades at our volume (~1,000 trades/hour avg) would be ~500 requests × 1 s rate-limit = ~8 min per day × 21 days = 3 h of fetch. Doable but haven't done it yet.

**Cannot recommend replacing book imbalance with trade-flow imbalance on the strength of this audit alone** — 24 h is not enough to characterize cross-regime behavior. But TFI is worth shadow-running live and comparing against book imbalance on the same signal bars.

---

## Conclusions and recommendations

1. **Book imbalance is fine** as the direction signal on this data. Do not change the direction rule in live code.

2. **The 17/17 long live outcome is a regime phenomenon**, not a signal bias. Expect the next bear/chop regime to produce a non-trivial short share organically. Already confirmed by the 34-short backtest on Feb–Mar data.

3. **Trade-flow imbalance is worth prototyping in shadow**, not in live. Add a Kraken `trade` feed subscription to the collector (not the live bot), store 5-min buckets alongside the orderbook snapshots, and revisit after we have ≥14 days of both.

4. **Phase 2 should focus on entry filter hardening** (specifically knife-catchers), not direction selection. The MFE<20 losers were the bulk of live damage, not wrong-direction losers.

5. **For Phase 2's 60-day shadow requirement:** the 21-day Pi window is not 60 days. We can either:
   - Accept the 21-day sample with the caveat explicitly written into the report, or
   - Supplement with Kaggle Binance ETH/USDT (cross-venue caveat).
   I'll proceed with "Pi 21d in-sample, Kaggle full-year out-of-sample" labeling unless told otherwise.

---

## Open questions (to `notes/open_questions.md`)

- Does `resample_pi_to_1min` aggregate L2 snapshots into the 1-min bar at start-of-bar, end-of-bar, or as a mean? Matters for the exact value of imbalance at each bar and therefore the filter decisions.
- When multiple signal bars fire on the same minute, priority-mode picks by lowest H. This is fine but could mask a direction-selection bug if both pairs were biased the same way. Now moot since BTC is disabled.
- Trade-flow imbalance needs a historical archive before it can be audited cross-regime. Open question: is a ~$50/mo Tardis subscription justified for 1 month of trial?
