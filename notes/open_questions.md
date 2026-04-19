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
