# Open questions during sprint/direction-and-exit-v1

Append-only log of things that look wrong or ambiguous but are
load-bearing; flagged per guardrail 6 without silent refactors.

---

## 2026-04-19 — Phase 0 data gap

**Issue:** 13 of 17 live trade dates (Apr 13/14/15/18/19) have no
orderbook data in the repo. Acceptance criterion "≥14/17 within 15 bps"
is unreachable.

**Resolution:** user approved running engine validation on the 21-day
Pi sample instead (Feb 18 – Mar 11), which confirmed the engine
reproduces its own stored reference within rounding. The live 17
remain a known-outcome replay set rather than a reproduction target.
Paid historical L2 (Tardis/Kaiko) deferred unless later phases exhaust
other explanations.

---

## 2026-04-19 — Phase 1 trade-flow archive

**Issue:** Kraken's `/Trades` REST serves 1,000 trades per page with
~1 req/s rate limit. Paginating 21 days of ETH takes hours; 60 days
takes ~10 h. No local archive.

**Resolution:** trade-flow imbalance prototyped on the last 24 h only.
Cross-regime comparison deferred. Recommended adding a Kraken `trade`
WebSocket subscription to the collector service (separate from the
live bot) to accumulate a forward-looking archive.

---

## 2026-04-19 — Pi resampling semantics

**Issue:** `resample_pi_to_1min` in `kaggle_ob_trainer.py` converts
snapshot-level orderbook records into 1-min bars. It's unclear whether
imbalance on each 1-min bar is:

- snapshot at the end of the minute (last tick), or
- average of snapshots in the minute, or
- snapshot at the start

Matters because the filter decisions (imb > 0.05, etc.) depend on
exactly which snapshot's values we use. Backtest and live bot should
use identical semantics. Not investigating during this sprint because
it's load-bearing and not obviously wrong; flagging per guardrail 6.

---

## 2026-04-20 — Kraken Market Data HF dataset (next-sprint candidate)

**Source:** https://huggingface.co/datasets/Abraxasccs/kraken-market-data

Free MIT dataset with **Kraken-native** L2 orderbook (100 levels), OHLC
(1m/5m/15m/1h/4h/1d), trade prints, and ticker snapshots. Feb 2024 –
Jan 2025. 4.26 GB total. Pairs include ETH/USD and ETH/EUR (spot, NOT
futures -- venue-caveat remains but much milder than Binance).

Unlocks (next sprint, after current 6a/6b sequence):

1. Redo Phase 2/3 ablations with Kraken-spot OOS (currently Binance).
   Verify F3c/timeout_trail/E3 all still pass on the correct venue.
2. H_threshold refit -- was out of scope this sprint. This dataset
   makes it tractable on 11 months of Kraken data.
3. Shadow expectation sample sizes -- 3-6 trades per primary bucket
   today (4 of 6 fallback to direction-only). 11 months of Kraken
   would give 50-150 per bucket; safety-valve z-score becomes meaningful.
4. Trade-flow imbalance archive -- the `trade/` subdir provides the
   historical trade prints Phase 1 flagged as missing for cross-regime
   comparison against book imbalance.

Caveats:
 - Spot vs futures: different spread profile, no funding effects in
   spot, retail-heavier book shape. Milder mismatch than Binance but
   not zero.
 - Temporally disjoint from live trades (Apr 2026) and from Pi archive
   (Feb-Mar 2026). Does not fill the on-host regeneration gap.
 - Archive freezes at Jan 2025; not a live feed. Fine for OOS use.

**Do NOT consume during Stage 6a/6b monitoring.** Observation window
is about live behavior vs deployed observer, not offline re-ablation.
Log and revisit in sprint-v2 scoping.

## 2026-04-19 — E2 partial_tp aggregation bug

**Issue:** in `phase3_exit_ablation.py` the partial_tp fill and the
subsequent full-close were appended as SEPARATE rows to `trades[]`.
PF, WR, and knife-rate summaries iterated these rows one-for-one, so a
position that partialled at +100 bps and then stopped out at −50 bps
contributed two rows: one guaranteed-win (+100) + one loser (−50).
This inflated E2's reported PF (OOS 2.14 → 3.70) and WR. Compound
return is correct because both realized USDs are added to equity.

**Resolution (applied 2026-04-19 in the same patch):** position-level
aggregation. Partial_tp fills no longer append to `trades[]`; they
accumulate into the open position's realized USD and flag
`had_partial=True`. At full close, one row is appended combining both
fills. `pnl_bps` on that row is total USD P&L divided by the *initial*
notional (the notional at entry, before any partial close). `peak_bps`
is the max peak observed across the position's lifetime. Knife-rate
excludes positions with partial_tp fills (peak was at least +100 bps
by definition). `phase3_exit_ablation.json` regenerated. Phase 4 sim
uses the same position-level aggregation from the start.

## 2026-04-19 — F3 variants for Phase 2

User expanded F3 into three variants (per-message after D1):
- F3-a: block LONG if `ema_diff_norm < -0.5` (original spec)
- F3-b: block LONG if `ema_diff_norm < -0.3`
- F3-c: block LONG if `ema_diff_norm <  +0.3` (requires positive alignment)

All symmetric for SHORT. Each tested independently and in combination
with F1/F2/F4. Will implement when Phase 2 starts.

## 2026-04-19 — 60-day shadow data

**Issue:** Sprint phases 2–4 specify a "≥60 days of 1-min ETH data"
shadow. We have:
- Pi orderbook: ~24 days consecutive (Feb 18 – Mar 11) plus 3
  scattered days (Apr 7, 16, 17).
- Kaggle Binance ETH/USDT: ~1 year with L10 book snapshots, but
  cross-venue (different fee structure, different spread profile).

**Resolution:** default plan is "Pi 21-day in-sample + Kaggle full-year
out-of-sample", both labeled clearly in reports. Will flag in every
Phase 2/3/4 JSON output which data window was used.
