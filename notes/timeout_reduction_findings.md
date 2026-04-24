# Timeout reduction ablation — findings

**Generated:** 2026-04-21 · `algo/reports/timeout_reduction_ablation.json`

## Verdict: **RED**

No timeout variant improves on ≥2 native substrates. Single-substrate
uplift only on Kaggle OOS (cross-venue), same substrate-fit pattern as
F3c in sprint v1.5. Do **not** add timeout-reduction to the sprint v2
candidate list as-is.

## Tl;dr per variant

| Variant | LIVE (17 trades) | Pi IS (44 trades) | Kaggle OOS (130 trades) |
|---|---|---|---|
| T_90  | ret **−11.3pp** (ΔDD −4.6pp, winner −37%) | ret **−75.1pp** (winner −25%) | ret **−13.2pp** (ΔDD +4.2pp) |
| T_120 | ret **−25.5pp** (ΔDD +5.8pp, winner −5%)  | ret **−47.7pp** (winner −10%) | ret **−17.8pp** |
| T_150 | ret **−11.3pp** (ΔDD −1.2pp, winner −26%) | ret **−30.1pp** (winner −12%) | ret **−13.8pp** |
| T_180 | ret **−2.1pp**  (ΔDD 0, winner −6%)        | ret **−6.1pp**  (winner −7%)   | ret **+9.9pp** (ΔDD +2.5pp) |
| T_240 | **+1.30%** baseline                          | **+178.36%** baseline          | **+26.39%** baseline       |

Pi IS: monotonic — the shorter the timeout, the worse compound return.
ΔDD is flat zero on all variants (8.09% throughout) because the Pi
drawdown sequence isn't driven by timeout-eligible trades.

Live 17-trade retrospective: T_180 is the least-bad reduction
(−2.1pp), all others materially worse. The live sample's T_240 sim
baseline (+1.3%) differs from the actual live result (−22%) because
this simulation uses Binance 1m spot OHLC (deep history) as a proxy
for Kraken Futures; slippage, fee differences, and the 10-second vs
1-minute snapshot cadence are not modeled. **The live sim is directional
evidence only**; the Pi IS replay is the authoritative native signal.

Kaggle OOS: T_180 is the only variant showing positive ret uplift
(+9.87pp) — but with increased DD (+2.5pp) and a drop in avg winner
(−6%). Fails the "no DD increase" criterion.

## Acceptance criteria check

| Criterion | T_90 | T_120 | T_150 | T_180 | Verdict |
|---|---|---|---|---|---|
| Positive ret uplift on Pi AND Kraken-native | ❌ | ❌ | ❌ | ❌ (Pi −6pp) | **FAIL** |
| No DD increase on either native substrate | ✓ Pi flat; ? Kraken-native not tested yet* | ❌ +5.8pp LIVE | ❌ −1.2pp LIVE (flat Pi) | ✓ Pi flat | mixed |
| Avg winner drop ≤ 15% | ❌ (−25 to −40%) | ✓ | ❌ (−22 to −26%) | ✓ (−6%) | T_180 only |

\*Kraken-native clean-day substrate replay was deferred — it would need
the `sprint_v15_refit_and_confirm` engine re-parameterized, ~10 min work.
Given Pi IS and LIVE both fail the ret criterion, the Kraken-native
cell is decision-irrelevant: no variant can clear the "≥2 native
substrates" bar without Pi passing, and Pi doesn't.

## Second analysis — bucket cross-tab (LIVE 17 trades)

Bucket = hold-time of the original T_240 trade.

| Bucket | n | T_90 sum_delta | T_120 | T_150 | T_180 |
|---|---:|---:|---:|---:|---:|
| <30m | 3 | 0 (no change, as designed) | 0 | 0 | 0 |
| 30-90m | 3 | 0 | 0 | 0 | 0 |
| 90-180m | 6 | **+45.8 bps** | −20.2 | −49.9 | 0 |
| >180m | 5 | **−192.4 bps** | **−314.7 bps** | **−91.9 bps** | **−26.7 bps** |

Three clean observations:

1. **<30m and 30-90m buckets never change** — trades exit via SL/TP long
   before any reduced timeout fires. Expected, confirms the sim.

2. **90-180m (the "pathological bucket") has mixed results.** Some
   reductions help by converting an SL loss into a timeout exit that
   happens to catch a bounceback (e.g. T_90 saves +77.5 bps on one
   trade). But shortening the post-profit ATR-trail window also turns
   some tp_trail wins into worse trail_stop exits (e.g. T_150 cuts
   trade `a18ae249` by −65.5 bps: tp_trail → trail_stop). Net: T_90
   marginally positive (+46 bps), T_120/T_150 negative (−20 to −50),
   T_180 neutral.

3. **>180m bucket is consistently harmed by every reduction.** This is
   the "winner-truncation" effect: the 5 trades in this bucket were
   specifically the ones that benefited from letting the full 240-bar
   window play out. Sum delta:
   - T_90: **−192.4 bps** across 5 trades
   - T_120: **−314.7 bps** (worst)
   - T_150: −91.9 bps
   - T_180: −26.7 bps (closest to no harm)

The bucket analysis explains the aggregate: shorter timeouts save a
small amount on long-held losers (90-180m) but cost far more on
long-held winners (>180m). Net effect is negative at every reduction
on the live sample. T_180 is close to neutral; every variant shorter
is materially worse.

## Third analysis — substrate replication

**Pi IS (native, authoritative for deployment):** T_240 wins
monotonically across all tested variants. Reducing timeout is
**strictly worse** on Pi IS compound return:

```
  T_90:   +103.28%  (-75.08pp vs T_240)
  T_120:  +130.67%  (-47.69pp)
  T_150:  +148.27%  (-30.09pp)
  T_180:  +172.22%  (-6.14pp)
  T_240:  +178.36%  baseline
```

Avg winner size drops 25% → 7% as timeout lengthens. The ATR trail
window matters: shortening means winners don't have room to run.

**Kaggle OOS (cross-venue):** T_180 shows **+9.87pp** ret uplift but
also +2.5pp DD and −5.5% avg winner. Only Kaggle shows any positive
ret signal at any reduction — the same pattern as F3c's sprint v1.5
result (Kaggle positive, natives negative or neutral).

**LIVE 17-trade retrospective (Binance 1m spot proxy):** T_180 is
−2.1pp vs T_240 in sim. With simulation-to-real slippage unmodeled,
this could be noise; directionally, every reduction is neutral-to-worse.

## Interpretation

The timeout-reduction hypothesis was: "cutting long-held losing trades
earlier would save PnL on the pathological 90-180m bucket without
meaningfully hurting winners." The ablation shows:

- The "save losers" effect is real on some 90-180m trades, but small
- The "cost winners" effect is larger and dominates on every
  **native** substrate
- The 240-bar timeout exists specifically to let winners run into the
  ATR-trail phase; shortening it disables that phase

The only substrate where reduction helps is Kaggle OOS, which:

- Is cross-venue (Binance spot, different microstructure than Kraken
  Futures)
- Disagreed with Pi on F3c in sprint v1.5 for similar reasons
- Should not override native-substrate evidence

## Recommendation

**Do not add timeout reduction to the sprint v2 candidate list.**

T_240 is the correct baseline. The pathological bucket's losses are
better addressed by entry-side filters (which the sprint v1 ablation
already fully explored) or by the `timeout_trail` exit (which 6b PATH
A ships and this ablation did not disable). The existing E3 time-
decayed SL already partially addresses the pathological bucket by
tightening SL at 60 bars if peak < +50 bps — a targeted intervention
rather than a blanket timeout reduction.

Note for sprint v2: if Pi L2 collector has 30+ days by then, re-run
this ablation on the forward-archive Kraken Futures native L2 (not
Kaggle Binance spot). Current result is supported by Pi IS but
ideally confirmed on a fresh native OOS slice before final rejection.

## Caveats

- **LIVE sim uses Binance 1m spot OHLC** as a proxy for Kraken Futures
  (deep history via public API; Kraken spot public OHLC only serves
  ~12h of 1m for older trades). Spot and futures diverge by funding-
  related basis (typically <10 bps for ETH), but intra-trade price
  paths are virtually identical. Slippage and fill timing are idealized.
- **LIVE sim T_240 baseline (+1.30%)** diverges from actual live
  (−22% compound at 10×) due to simulation idealization (no slippage,
  exact SL/TP fills, 1m bars vs 10-sec Pi snapshots). The deltas vs
  T_240 baseline remain meaningful; the absolute level is not.
- **Pi IS DD is flat at 8.09% across all variants** — the DD sequence
  is determined by trades outside the timeout-eligible set. Varying
  timeout doesn't reshape the drawdown path because the drawdown
  comes from fast SL trades earlier in the sequence.
- **Kraken-native clean-day substrate was not replayed for this
  ablation** (decision-irrelevant given native failures on LIVE + Pi).
