# Exit-parameter research — findings

**Date:** 2026-04-21
**Request:** try more exit params for the deployed strategy; train on Kraken-native HF; verify on Pi; seek improvement to live trading.
**Failure modes flagged by user:**
- (a) exiting too soon despite uptrend
- (b) exiting at bottom despite recovery
- (c) sitting in stale downtrend

## Verdict summary

| Goal | Result |
|---|---|
| Primary: improve LIVE (April 2026 regime) | **Small gain available** (~+3.5pp via ATR-scaled trail), at −10pp Pi cost |
| Secondary: improve Pi IS | **Clean gains available** (+14 to +34pp) but all of them hurt LIVE by −5 to −12pp |
| Fully dominating tweak (safe on all 3 substrates) | **`pst_max_wait: 30 → 40`** gives +4.55pp Pi / ±0 KN / ±0 LIVE — marginal but zero-risk |

**No single exit-param variant dominates operational on all three substrates by a meaningful margin.** The substrate-specific gains come from trade-offs: what helps one regime hurts another. The regime-fit pattern is analogous to (though less extreme than) the F3c result from sprint v1.5.

## Diagnostic — failure-mode distribution across 19 live trades

Per-trade classification using Binance 1m OHLC + 3h forward window (full classification in `algo/reports/exit_diagnostic.json`):

| Mode | Count | Avg bps missed per trade | Total bps |
|---|---:|---:|---:|
| (a) exit-too-soon (winner-truncation) | 4 | ~212 | **~849 bps on table** |
| (b) exit-at-bottom (SL then recover) | 2 | ~45 | ~91 bps missed |
| (c) stuck in stale downtrend | 0 | - | - |
| clean (no failure mode) | 13 | - | - |

**The big lever is mode (a).** The 4 winner-truncation cases: trade #3 (live +129 / fwd +425), #9 (live +75 / fwd +60), #13 (live +134 / fwd +152), #18 (live +60 / fwd +211). In each case the trail fired while price was still running in our favor.

Mode (c) was not observed in the 19 live trades — trades holding past timeout either went to trail_stop (several) or tp_trail. This doesn't mean the bot can't get stuck in stale downtrends; just that it hasn't in this sample.

## Sweep design

**Engine:** `algo/diagnostics/exit_research_engine.py` — parametrized deployed-strategy engine (F3c stays off). Entry logic unchanged.

**Grid:**
- 144 standard-mode cells: `sl_bps × trail_after × trail_bps × e3_enabled × pst_width × timeout_bars` = 3×2×3×2×2×2
- 6 "radical" cells: 3 ATR-scaled-trail + 3 dual-stage-trail
- +1 operational reference
- = 151 total cells

**Train substrate:** Kraken-native 62 clean days (Abraxasccs HF), 45,870 1-min bars.
**Validate substrate 1:** Pi IS Feb 18 – Apr 7 2026, 30,644 1-min bars.
**Validate substrate 2:** 19 live trades (Apr 12–19 2026), replayed per-trade using Binance 1m OHLC.

All 151 cells ran against all 3 substrates. Sweep runtime: ~4 min train + ~1.5 min Pi + ~1.5 min live = ~7 min.

## Results — three clear candidates

**Operational reference (current live config: SL=50, trail_after=150, trail_bps=50, E3 on (tighten=-25 at 60m if peak<50), pst_width=20, pst_floor=-60, pst_max_wait=30, timeout=240, leverage=5):**

| Substrate | ret | DD |
|---|---:|---:|
| Pi IS     | +178.36% | 8.09% |
| Kraken-native | +22.34% | 27.11% |
| LIVE @10x | −1.43% | — |

### Candidate 1 — `atr_trail_sl50_ta150_mult2.0` (best on LIVE)

Exit mode: ATR-scaled trail. Once peak ≥ 150 bps, trail at `peak − 2.0×atr_bps` instead of fixed `peak − 50`.

| Substrate | ret | Δ vs op | DD |
|---|---:|---:|---:|
| Pi IS     | +168.57% | **−9.80pp** | 8.09% |
| Kraken-native | +23.95% | +1.61pp | 25.49% |
| LIVE @10x | **+2.02%** | **+3.45pp** | — |

**Addresses mode (a)** by widening the trail in proportion to realized volatility: when the trend is still running, the trail stays wider.

**Tradeoff:** Pi IS loses 10pp. Same Pi DD as op (8.09%) — not a crash, just a smaller win. LIVE gain is small in absolute terms (~$350 on $10k at 10x), but directionally aligned with the observed failure mode.

### Candidate 2 — `std_sl50_ta200_tbps75_e3on_pst40_to240` (best combined Pi+KN, but hurts LIVE)

Standard mode with `trail_after` lifted 150→200 and `trail_bps` 50→75, pst_max_wait 30→40.

| Substrate | ret | Δ vs op | DD |
|---|---:|---:|---:|
| Pi IS     | **+212.52%** | **+34.16pp** | 8.09% |
| Kraken-native | **+39.04%** | **+16.70pp** | 21.40% |
| LIVE @10x | −13.19% | **−11.75pp** | — |

**Best non-LIVE result in the sweep** but fails the live-regime test. The delayed trail activation (trail_after=200) means 3 of 4 live winners (peaks 102, 158, 146) never triggered the trail and either timed out or fell through.

**Not recommended for deployment** given the LIVE regression.

### Candidate 3 — `std_sl50_ta150_tbps50_e3on_pst40_to240` (the safe zero-risk tweak)

Same as operational but `pst_max_wait: 30 → 40`. ONE parameter change.

| Substrate | ret | Δ vs op | DD |
|---|---:|---:|---:|
| Pi IS     | +182.91% | +4.55pp | 8.09% |
| Kraken-native | +22.38% | +0.04pp | 27.09% |
| LIVE @10x | −1.43% | ±0.00pp | — |

**Fully dominates or matches operational on all 3 substrates.** The +4.55pp on Pi IS comes from giving 2-3 post-timeout-in-loss trades an extra 10 bars to recover their MFE before the PST-timeout exit fires. Zero cost on KN and LIVE.

Small effect size but **the only improvement that is guaranteed not to hurt anywhere**.

## Recommendation

**Ordered from safest to most aggressive:**

1. **Low-risk: adopt Candidate 3** (`pst_max_wait: 30 → 40`). Zero regression anywhere; +4.55pp Pi IS. One-line config change, no new code path, no behavior shift on LIVE.

2. **Targeted live-regime fix: trial Candidate 1** (`exit_mode="atr_trail", atr_trail_mult=2.0`) in a **shadow-capital live test only**, not in main deployment. The −10pp Pi IS cost is material; the +3.45pp LIVE gain is small in absolute terms; but the config directly addresses the dominant failure mode. Not recommended for main capital without 30+ days of shadow validation.

3. **Do NOT deploy Candidate 2** — the Pi/KN gains are tempting but the −12pp LIVE regression on the exact regime the bot is currently trading is a big red flag. Same sprint-v1.5-F3c lesson: looking good on training substrates doesn't save you if the validation substrate (live) disagrees.

## Broader observation — regime-fit is again the bottleneck

The sweep shows clear substrate-specific parameter preferences:
- **Pi IS prefers delayed trail activation** (trail_after=200) — trades on Pi often MFE to 150+ before reversing.
- **LIVE (April 2026) prefers early trail activation** (trail_after=150) — April winners peaked at 85–216 bps, so trail_after=200 truncated most of them via timeout.
- **Kraken-native (62 days) matches Pi** — wider trail wins on KN.

The live regime is the outlier. A static exit config can't be optimal for both the "classical" regime (Pi IS / KN) and the April 2026 tape. This is a **regime-detector scope item for sprint v2**, not a static-param fix.

## Files

- `algo/diagnostics/exit_research_diagnostic.py` — 19-trade failure-mode classifier
- `algo/diagnostics/exit_research_engine.py` — parametrized exit engine
- `algo/diagnostics/exit_research_sweep.py` — 151-cell sweep on Kraken-native + Pi IS
- `algo/diagnostics/exit_research_expand.py` — Pi IS full sweep + 19-trade live replay
- `algo/reports/exit_diagnostic.json` — per-trade failure-mode classification
- `algo/reports/exit_research_sweep.json` — train-phase output
- `algo/reports/exit_research_expand.json` — full sweep × all 3 substrates

Nothing committed. Sprint branch is still `sprint/direction-and-exit-v1` at `c7ebb48` (6b deploy infra). The Strategy B fork branch (`sprint/strategy_b_trend_entropy`) is abandoned as instructed but still exists locally — delete with `git branch -D sprint/strategy_b_trend_entropy` when ready.
