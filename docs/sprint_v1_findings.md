# Sprint v1 — Findings

**Branch:** `sprint/direction-and-exit-v1`
**Scope:** entry-filter hardening, exit optimization, sizing/leverage review.
**Trigger:** 17 live ETH trades (Apr 12–19 2026), −22% compounded at 10×.

---

## Final promoted config (carry into Phase 6)

| Component | Value | Source |
|---|---|---|
| Entry gate `F3c` | block LONG if `(ema_fast − ema_slow) / ATR_30 < +0.3` (symmetric SHORT) | Phase 2 promoted |
| Exit `timeout_trail` | timeout-at-loss transitions into `post_signal_trail` (width=20 bps, hard_floor=−60 bps, max_wait=30 bars) | Phase 3 promoted |
| Exit `E3 time_decayed_sl` | at 60 bars held, if `peak_pnl_bps < +50`, tighten SL from −50 to −25 | Phase 4 promoted |
| Leverage | **5×** (was 10×) | Phase 4 promoted |
| Entropy-weighted sizing | disabled | Phase 4 rejected |
| `E1` lower tp_trail | disabled | Phase 3 rejected |
| `E2` partial TP | disabled | Phase 3 rejected (measurement artifact + real compound-return loss) |
| `E4` ATR-initial SL | disabled | Phase 3 rejected |
| `E5` entropy-decay exit | disabled | Phase 3 rejected (mechanical H-drift ambiguity) |

Every other parameter (SL=50, TP=200 on Pi / TP=150 on Kaggle, knife=50 on Pi / 100 on Kaggle, H_threshold=0.4352, 240-min timeout, tp_trail@150/50) unchanged from pre-sprint.

---

## Phase-by-phase

### Phase 0 — engine validation (adapted)

Live-trade reproduction was blocked: 13 of 17 live trade dates had no orderbook data in the repo. Substituted an engine sanity check on the 21-day Feb–Mar Pi window — engine reproduces its own stored reference to within rounding.

Artifact: `algo/reports/phase0_engine_validation.json`.

### Phase 1 — direction audit

Hypothesis: `sign(imbalance)` was structurally long-biased, explaining 17/17 LONG live.

Result: **rejected**. Over 30,644 Pi bars, `imbalance_5` mean −0.007, median −0.011, frac_positive 0.492. Filter-pass candidates tilted short (36 long / 49 short). H conditional on imbalance sign was polarity-symmetric (0.4900 vs 0.4905). The 17/17 live outcome was a regime artifact of the Apr 12–19 uptrend — not a signal bias. No direction-rule change.

Artifacts: `algo/reports/direction_audit.{json,png}`, `notes/direction_audit_findings.md`.

### Phase 1 diagnostics — D1, D2

D1 (regime-gate replay on the 17 live trades): F3's original formulation (`ema_diff_norm < −0.5`) would have blocked 6 of 17, net +1342 bps @10× saved. Decision band: F3 toggleable in Phase 2.

D2 (Feb–Mar baseline long/short split): short side delivered 65 % of USD contribution despite the period being net +467 bps UP. Edge is two-sided and anti-aligned with price direction — genuinely mean-reverting, not trend-ride. Not regime-fit.

D2 per-half appendix: edge holds and strengthens in H2. Long side delivered $10 621 / 40 % WR vs short $29 396 / 60 % WR in the up-moving second half. Robust internal consistency.

Artifacts: `algo/reports/d1_regime_gate_replay.{json,csv}`, `d2_baseline_lshort_split.json`, `d2_per_half_appendix.json`.

### Phase 2 — entry filter ablation

Original v1 run: all four filters (F1/F2/F3/F4) × Kraken-fit live baseline. No config passed the promotion rule. Baseline on Kaggle OOS was **−42 %** — a venue-transfer artifact, not a real failure of the strategy.

v2 run with Kaggle-refit baseline (SL=50, TP=150, knife=100, ext=None, refit on a separate 60-day Kaggle slice with 230-day gap to the eval slice): baseline jumps to +22 % OOS. All filter variants re-evaluated against fair baselines.

**F3c** (entry must have `ema_fast − ema_slow > +0.3 × ATR_30`) was the clearest cross-dataset winner:
- IS PF −6.8 % (noise at n=26)
- **OOS PF +40.7 %** (1.52 → 2.14)
- OOS DD halved (37.7 % → 20.8 %)
- OOS compound return +22 % → +200 %

Promoted as a permanent entry-gate overlay. F1/F2/F4 independently tested but F3c dominated; no additive gain from stacking.

Artifacts: `algo/reports/phase2_kaggle_refit.json`, `phase2_filter_ablation_v2.json`.

### Phase 3 — exit ablation

F3c fixed ON. 9 ablation lines: (a) baseline, (b) timeout_trail, (c) E5-long-only, (d) E1 lower trail, (e) E2 partial TP, (f) E3 time-decay SL, (g) E4 ATR-initial SL, (h) best(c–g)+b, (i) all stacked.

First pass showed E2 (partial TP) with OOS PF 2.14 → 3.70 (+73 %). **Measurement artifact** caught in review: partial_tp fills were appended as separate rows to `trades[]`, inflating PF and WR. Compound return was correct and E2 was actually **worse** on compound: IS +110 % vs baseline +175 %, OOS +125 % vs +200 %. Aggregation patched to roll partial fills into their parent position; Phase 3 JSON regenerated. E2 rejected.

Promoted: **(b) timeout_trail** — small but universal improvement on every axis (IS +183 % / OOS +214 % / OOS DD 19.8 % vs baseline 20.8 %).

Deferred: **(f) E3 time_decayed_sl** — largest single-variant OOS uplift (+257 % OOS, DD 21.4 %). IS cost small and suspected to be small-sample noise; formally evaluated under sizing variation in Phase 4.

Rejected: E1 (worse on all axes), E2 (artifact + real compound loss), E4 (OOS return collapse), E5 long-only (mechanical H-drift bit even after narrowing to longs), (i) all-stacked (knife_rate secondary fail).

Artifacts: `algo/reports/phase3_exit_ablation.json` (post-fix).

### Phase 4 — sizing / leverage

8 cells = leverage {5×, 10×} × entropy-weighted {off, on} × E3 overlay {off, on}. F3c + timeout_trail fixed ON.

Promotion rule identified as structurally incompatible: the +15 % OOS PF gate is unreachable by sizing alone because sizing scales notional uniformly — it does not change which trades win or lose. Every ew_off cell had **identical** PF (3.44 IS / 1.89 OOS); every ew_on cell was also identical internally. PF invariance confirms this exactly. Rule correction filed for future sizing phases: use OOS Sharpe and OOS Calmar as gates for sizing/risk variations, not trade PF.

Promoted: **E3 time_decayed_sl** on independent confirmation across all four sizing variants (−38 pp IS return, +43 pp OOS return, −2.5 pp OOS DD, +0.46 OOS Sharpe — consistent across every cell).

Rejected: **entropy-weighted sizing**. Within every (leverage × E3) pairing, ew_on had lower OOS Calmar than ew_off. The thesis ("size up when H is more confident below threshold") does not replicate.

Carried: **leverage = 5×** into live. Sharpe (4.61 vs 4.51 at 10×) and Calmar are preserved at lower leverage; DD halves (15–16 % vs 28–31 %). Live experience of 22 % DD at 10× respected as real-world stress data.

Artifacts: `algo/reports/phase4_sizing_sim.json`.

---

## Measurement issues discovered and fixed

1. **Venue-transfer bias (Phase 2 v1 → v2).** Applying Kraken-fit params to Kaggle Binance evaluated the venue transfer, not the filter edge. Refit on a separate Kaggle slice produced a fair OOS baseline and unmasked F3c's real edge.

2. **Partial-fill double-count (Phase 3).** E2's PF was inflated by counting partial_tp fills as independent trades in PF/WR aggregations. Compound return was unaffected. Aggregation patched to position-level rollup; E2 correctly rejected post-fix.

3. **PF-invariance gate (Phase 4).** A +15 % OOS PF improvement gate is mechanically unreachable by sizing changes that don't alter which trades win. Rule corrected to OOS Sharpe/Calmar gates for sizing phases.

4. **H mechanical drift (Phase 3 E5 acceptance).** The entropy Markov transition matrix rebuilds every bar; median H crosses the `h_exit_threshold` within ~15 bars on *every* trade regardless of outcome, not just invalidated ones. Absolute-threshold E5 cannot discriminate between drift and thesis invalidation. Rejected; dH-gated E5 flagged as future-sprint candidate.

---

## Phase 6 live-promotion checklist (STAGED)

Promotion is **two-stage**. Stage 6a deploys only the observer
(filter-audit writer + auto-flip monitor infrastructure). Stage 6b
flips the strategy flags after 3–7 days of clean observer data.

### Pre-deploy (manual verification, before any stage)

- [ ] `state/shadow_expectation.json` regenerated against the
      promoted config (run `algo/diagnostics/shadow_expectation_generator.py`)
- [ ] `state/multi_trader_state.json` shows `"position": None` on Pi
- [ ] Current Pi service running cleanly (no active trade to interrupt)
- [ ] Backup current `entropy_live_multi.py` on Pi as `entropy_live_multi.py.pre_sprint_v1`
- [ ] Backup `state/engine_state.json` (optional — regenerates from 30-min restore)

### Stage 6a — observer-only deploy (low risk)

- [ ] Merge `sprint/direction-and-exit-v1` → `main` (no config diff yet)
- [ ] Deploy to Pi — all sprint flags still default **False**, leverage stays **10×**
- [ ] Only behavior change: `state/daily_filter_audit.jsonl` starts appending,
      `state/shadow_expectation.json` is present, and `_check_shadow_drift` is
      invoked on each trade close but in dormant mode (the safety valve file
      hasn't been created; monitor just accumulates z scores to
      `state/live_drift_monitor.json`)
- [ ] Verify for 3–7 days:
      - Audit JSONL file is appending one row per candidate bar
      - Candidate-bar stats (direction split, imbalance distribution) match
        the Phase-4 shadow distributions within rough bounds
      - No unexpected IO, memory, or CPU growth on the Pi
      - `live_drift_monitor.json` shows z-scores within the expected range
        given baseline parameters and current live config
      - Dashboard `direction_audit` panel renders with real data

### Stage 6b — strategy promotion (after clean 6a window)

Config diff to apply in `PAIRS["ETH"]`:

```python
"f3c_enabled": True,
"f3c_ema_fast": 30,
"f3c_ema_slow": 150,
"f3c_atr_window": 30,
"f3c_threshold": 0.3,
"timeout_trail_enabled": True,
"timeout_trail_width_bps": 20,
"timeout_trail_hard_floor_bps": -60,
"timeout_trail_max_wait_bars": 30,
"e3_time_decay_sl_enabled": True,
"e3_tighten_after_bars": 60,
"e3_peak_threshold_bps": 50,
"e3_tightened_sl_bps": 25,
```

And in `SHARED_CONFIG`: `"leverage": 5` (from 10).

- [ ] Deploy to Pi (push + restart)
- [ ] First live trade under new config closes → verify `[LIVE-MONITOR]` line prints
- [ ] First trade's z-score is recorded in `state/live_drift_monitor.json`
- [ ] Delete `state/live_drift_monitor.json` **before** this stage starts so
      the 10-trade rolling window begins fresh under the new config
      (otherwise observer-period z scores carry over, which they shouldn't
      because they were computed against unflipped-config realized PnL)

### Auto-flip safety valve (Phase 6 ships with this)

On each trade close in `close_position`:

1. Look up shadow expectation for the trade's bucket `(direction, UTC session)`
   in `state/shadow_expectation.json`. Falls back to direction-only if the
   primary cell has `count < 5`.
2. Compute `z = (realized_pnl_bps - expected_mean) / expected_std`.
3. Append `z` to the rolling 10-trade window at `state/live_drift_monitor.json`.
4. If the window is full AND `sum(last 10 z) < −2·√10 ≈ −6.32`, trigger:
   - Write `state/safety_valve.json` marker
   - Flip in-memory: `f3c_enabled`, `timeout_trail_enabled`,
     `e3_time_decay_sl_enabled` = False across all pairs
   - Revert `SHARED_CONFIG["leverage"]` to 10
   - Emit a prominent `[LIVE-MONITOR] !!! SAFETY VALVE TRIPPED !!!` log line

On every startup: if `safety_valve.json` exists, re-apply the in-memory
overrides before the WebSocket loop starts. Clearing the valve is a manual
operation (delete the file) — no auto-recovery.

**Design note:** only ADVERSE drift disables. Positive drift (doing better
than shadow expectation) is logged but not acted on. This prevents
over-performance from triggering a protective rollback that wasn't needed.

### First 30 live days (post-6b)

- [ ] Cumulative realized return within 1 σ of shadow expectation
- [ ] DD under 15 % (shadow OOS DD was 14.8 %)
- [ ] Knife-catcher rate ≤ 30 % (baseline was 27–30 %)

### Step-up plan (post clean 30-day window)

- 5× → 7.5× leverage at +30 days if shadow-live alignment holds
- 7.5× → 10× at +60 days on continued alignment
- Each step gated on trailing 10 trades' cumulative z ≥ −6.32 (safety-valve condition not near-tripping)

### Config diff to apply on Phase 6 deploy (reference only — see Stage 6b above)

```python
# PAIRS["ETH"] changes:
"stop_loss_bps": 50,                    # unchanged
"take_profit_bps": 200,                 # unchanged
"timeout_minutes": 240,                 # unchanged
"h_thresh": 0.4352,                     # unchanged
"cooldown_bars_after_loss": 0,          # unchanged
"knife_threshold_bps": 50,              # unchanged
"trailing_atr_mult": 2.0,               # unchanged
"trail_tp_after": 150,                  # unchanged
"trail_tp_bps": 50,                     # unchanged
"extended_move_lookback": 150,          # unchanged
"extended_move_cap_bps": 100,           # unchanged
# NEW (flip from default-off to on):
"f3c_enabled": True,
"f3c_ema_fast": 30,
"f3c_ema_slow": 150,
"f3c_atr_window": 30,
"f3c_threshold": 0.3,
"timeout_trail_enabled": True,
"timeout_trail_width_bps": 20,
"timeout_trail_hard_floor_bps": -60,
"timeout_trail_max_wait_bars": 30,
"e3_time_decay_sl_enabled": True,
"e3_tighten_after_bars": 60,
"e3_peak_threshold_bps": 50,
"e3_tightened_sl_bps": 25,

# SHARED_CONFIG changes:
"leverage": 5,   # was 10
```

### First 7 live days monitoring (automated — ships with 6b)

- [ ] `[LIVE-MONITOR]` messages printing per trade (emitted by
      `_check_shadow_drift`)
- [ ] `state/daily_filter_audit.jsonl` appending (one line per candidate bar)
- [ ] `state/live_drift_monitor.json` z-score window updating per trade
- [ ] Auto-flip safety valve ready: trigger condition is `sum(last 10 z) <
      −2·√10 ≈ −6.32`. If the valve trips, `state/safety_valve.json` is
      written and all sprint-v1 flags are disabled in memory and on
      restart — see "Auto-flip safety valve" section above for full
      behavior.

### First 30 live days

- [ ] Cumulative realized return within one standard deviation of shadow expectation
- [ ] No unexpected knife-catcher spike (compare to backtest baseline 25–30 %)
- [ ] DD under 15 % at 5× (shadow OOS expectation: 14.8 %)

### Step-up plan (post clean 30-day window)

- At +30 live days with shadow-live alignment: 5× → 7.5× leverage
- At +60 live days with continued alignment: 7.5× → 10× leverage
- Each step gated on the previous 10 trades matching shadow expectation within 2 σ

---

## Open questions for next sprint

1. **dH-gated E5.** The original E5 (absolute `H > h_exit_threshold`) failed because H drifts mechanically. A derivative-based gate (`H rising faster than a threshold over a short window`) could in principle distinguish drift from invalidation. Worth prototyping on the same Feb–Mar baseline winners.

2. **Asymmetric short-side E5.** All 8 winner-cuts in the E5 acceptance test were on short trades. A short-only E5 with different thresholds could potentially save the SHORT losers without cutting winners. Interesting since shorts are the majority contributor in Feb–Mar (65 % of USD).

3. **Trade-flow imbalance.** Cross-regime comparison against book imbalance requires a trade-tick archive we currently lack. Options: (a) add a Kraken `trade` WebSocket subscription to the Pi collector (cheap, forward-looking only); (b) subscribe to Tardis/Kaiko for historical trades (≈$50/mo, full cross-regime backfill). Revisit after 14+ days of forward archive.

4. **60-day shadow coverage.** Sprint relied on 21 days of Pi + 60 days of Kaggle (cross-venue). Ideally we want a contiguous 90+ days of Kraken Futures L2 — currently not archived. Pi collector change to persist book snapshots beyond the bot's in-memory deques would be a one-time setup that pays off every future sprint.

5. **Filter regime dependency.** F3c promoted based on Feb–Mar + Kaggle 2023–24. If a new regime arrives, F3c could flip from helper to blocker. Monitor F3c block-rate weekly; flag if it exceeds 60 % of candidates (means most bars look anti-trend, likely a chop/mean-reverting regime where F3c should be loosened).

6. **Phase 4 rule language.** The sprint ran up against an ill-posed promotion rule twice (Phase 2 venue bias, Phase 4 PF invariance). Next sprint should define the promotion rule *per-phase-type* up front — rules appropriate for a filter sweep differ structurally from rules appropriate for a sizing sweep.

---

## What this sprint did NOT touch (out of scope)

- **H_threshold = 0.4352.** Fixed throughout. Separate fitting problem; own sprint later.
- **Signal-generation parameters** (imb_min=0.05, spread_max=20, ret_low=20, ret_high=80). Kept constant.
- **Cross-pair priority rule.** BTC is already disabled; moot for this sprint.
- **Collector infrastructure.** Pi collector still doesn't archive Kraken L2 past the bot's 30-min engine-state window. Flagged in the next-sprint list.

---

## Data coverage caveats

- **Pi orderbook (in-sample):** Feb 18 – Mar 11 contiguous (21 days), plus isolated days Apr 7 / 16 / 17. 13 of 17 live-trade dates have no orderbook coverage, so live trades were never a reproduction target.
- **Kaggle Binance ETH/USDT (out-of-sample):** 2023-10-07 to 2024-10-09. Used last 60 days for eval; first 60 days (with 230-day gap) for refit baseline. Cross-venue, so conclusions transfer modulo different fee/spread structure.
- **Fees assumed:** 5 bps per side (10 bps round-trip) on notional. Matches Kraken Futures taker. Binance fees differ slightly — all OOS numbers are estimates at Kraken fee level.
- **Pi baseline closes spot-checked (2026-04-20):** Cross-validated the Pi 1-min close series against an independent Kraken spot ETH/USD feed (HF dataset `Abraxasccs/kraken-market-data`) over the 16-day overlap Feb 18 – Mar 5 2026. 21,581 overlapping bars; mean |delta| = 2.5 bps, p99 = 11.4 bps, drift flat (−0.008 bps/day), no structural breaks. Data integrity check: **GREEN**.

---

*Generated 2026-04-19 on branch `sprint/direction-and-exit-v1` · Phase 5 deliverable.*
