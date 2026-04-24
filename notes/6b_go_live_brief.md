# 6b go-live brief — PATH A (timeout_trail + E3 + 5×, no F3c)

**Generated:** 2026-04-20
**Status:** ready for review. No deploy. No push. No merge.
**Gate:** awaiting explicit "promote 6b per PATH A" instruction.

---

## A. Path and rationale

**Classification:** CASE A (config mismatch, explanatory) — from
`algo/reports/pi_is_f3c_discrepancy_classification.md`.

**Why PATH A:** two of three native substrates prefer no-F3c.

| Substrate | F3c effect | Source |
|---|---|---|
| Pi IS (Feb 18 – Apr 7 Kraken Futures ETH) | **negative** (−117pp ret, +4.4pp DD under phase-4 engine; −385pp / +10pp under phase-2 engine) | `phase4_sizing_sim.json` cell `lev5x_ewoff_e3on` IS vs today's no-F3c replay |
| Kraken-native OOS (Mar 18 – Apr 15 2026 clean days) | **neutral / slightly negative** (−3.23pp ret, +1.60pp DD, +8.93pp knife rate, PF essentially unchanged) | `sprint_v15_f3c_kraken_native_confirmation.json` |
| Kaggle Binance OOS (60d ETH/USDT) | **positive** (+77pp ret, −24pp DD, +0.70 PF) | `phase4_sizing_sim.json` OOS vs today's no-F3c replay |

Native substrates (Pi, Kraken-native) are the deployment target. Kaggle
Binance is cross-venue, cross-pair (spot vs Futures), different fee
structure (Binance fees ≈ 4bps vs Kraken taker 5bps), different
microstructure. Two-of-three native weighting outweighs one-of-three
cross-venue — and the rule that would have caught this at Phase 2 is
"require same-sign uplift on at least two native substrates before
promoting a filter," which would have correctly rejected F3c.

**What sprint v1 is honestly shipping:**

1. `timeout_trail` — post-signal-trail on loss-timeout (Phase 3 promoted,
   multi-substrate confirmed).
2. `e3_time_decayed_sl` — tighten SL to −25 at 60 bars if peak < +50
   (Phase 3+4 cross-confirmed).
3. `leverage` 10× → 5× — half the risk on the same signal (Phase 4 ).

Total scope: two exit improvements + one risk-reduction parameter. That
is the full evidence-supported sprint v1 delivery. No filter changes.

---

## B. Exact config diff (ready to apply)

File: `algo/entropy_live_multi.py`

**PAIRS["ETH"]** — change exactly these three lines:

```diff
-        "timeout_trail_enabled": False,
+        "timeout_trail_enabled": True,
         "timeout_trail_width_bps": 20,
         "timeout_trail_hard_floor_bps": -60,
         "timeout_trail_max_wait_bars": 30,

-        "e3_time_decay_sl_enabled": False,
+        "e3_time_decay_sl_enabled": True,
         "e3_tighten_after_bars": 60,
         "e3_peak_threshold_bps": 50,
         "e3_tightened_sl_bps": 25,
```

`f3c_enabled` stays **False** (no change — already at target).

**SHARED_CONFIG** — change exactly this line:

```diff
-    "leverage": 10,                   # Phase 6 flips this to 5 when promoted
+    "leverage": 5,                    # Phase 6b promoted 2026-04-XX
```

**Nothing else changes.** No filter toggles, no threshold tweaks, no
cooldown changes, no pair config. Every other field stays exactly as it
sits on the current deployed Pi.

---

## C. Shadow expectation alignment

### C.1 State of `state/shadow_expectation.json`

Verified via `python algo/diagnostics/shadow_expectation_generator.py
--no-f3c` (idempotent regen) + file metadata dump:

```json
{
  "cell_name":      "timeout_trail+E3+5x (revised 6b, no F3c)",
  "f3c_enabled":    false,
  "generated_at":   "2026-04-20T18:52:40.159805+00:00",
  "data_source":    "Feb 18 - Apr 7 2026 Pi Kraken Futures ETH L2",
  "n_trades_total": 44
}
```

This matches PATH A exactly: F3c off, timeout_trail + E3 + 5×. The
generator was re-run fresh immediately before this brief and produced
byte-identical bucket statistics — the file IS correct and IS
reproducible.

### C.2 v1 preservation

`state/shadow_expectation.v1.json` (2895 B) is present, holds the
F3c+timeout_trail+E3+5× cell from 2026-04-19T20:58:28Z (before F3c
RED). Naming-nit flagged: your instruction spec called this
`shadow_expectation.v1_with_f3c.json`; I saved it as
`shadow_expectation.v1.json`. Functionally equivalent (no collision).
Recommend rename before deploy for clarity:
`mv shadow_expectation.v1.json shadow_expectation.v1_with_f3c.json`.

### C.3 Code path trace — which shadow does the live bot read?

`algo/entropy_live_multi.py` line 754:

```python
shadow_file = state_dir / "shadow_expectation.json"
```

The live bot reads **only** `shadow_expectation.json`. The `.v1.*`
archive file is NOT referenced by any code path. It is a passive
backup. Archive naming therefore has zero runtime effect.

The load happens inside `_check_shadow_drift()` which is called from
`close_position()` after every trade — so the shadow is re-read fresh
each time. No startup caching. This means swapping the shadow file
between trades is safe; the next trade picks up the new expectation.

---

## D. Expected 30-day live profile at 5×

Trade outcomes in this strategy are not iid; they cluster by regime.
Section D is organized around regime scenarios, not bootstrap
percentiles. Bootstrap numbers (section D.4 caveat) are a ceiling on
expectation, not a median.

### D.1 Modal expectation (realistic single-regime month)

This is the expectation for any given 30-day window where one regime
dominates. April 2026 live data (Apr 12–15, 8 ETH LONG trades at 10×,
3 wins / 5 losses, net −54 bps) is the reference case. The revised
config is NOT well-tuned to this regime:

- Short-side edge (+70 bps/trade avg on Pi shadow) is dormant when
  the tape runs one-way; only long-side sets up
- Long-side mean in Pi shadow: +57 bps/trade, std 122 bps — wide
  enough that 5–8 trade windows can easily be net-negative
- Trading less frequently on a subset of the signal distribution

**Plausible outcome range at 5×:** compound ret **0% to +50%**, max
DD **12–20%**. Not a loss event in expectation, but well below
bootstrap p50 because the full two-sided signal is not live. This is
the scenario to plan for.

### D.2 Upside scenario (two-sided mean-reverting, Feb–Mar 2026-like)

Pi IS (Feb 18 – Apr 7) is the reference; both long and short sides
fire regularly, outcomes bounce symmetrically. Bootstrap from the 44
Pi trades resampled to 62:

| Metric | p10 | p25 | **p50** | p75 | p90 |
|---|---:|---:|---:|---:|---:|
| compound ret% | +160.1 | +226.8 | **+323.8** | +455.3 | +612.4 |
| max DD % | 7.9 | 9.3 | **11.1** | 14.2 | 16.8 |

P(ret < 0) = 0.0% · P(DD > 20%) = 3.7%

**Caveat (see D.4):** these are iid bootstrap numbers. Read as
"favorable regime ceiling," not expected-value. Realistic upside
target for a two-sided regime month: **+80% to +200% at 5×, DD
10–15%** — substantially below bootstrap p50.

### D.3 Downside scenario (persistent adverse regime, Kaggle p25–p50)

Kaggle OOS at 5× shows a very different distribution, driven by 130
trades across a year of mixed regimes. It approximates what happens
when live conditions deviate from Feb–Mar Pi into different
microstructures. Bootstrap of 130 Kaggle trades resampled to 66:

| Metric | p10 | **p25** | **p50** | p75 | p90 |
|---|---:|---:|---:|---:|---:|
| compound ret% | −25.0 | **−9.8** | **+11.3** | +38.5 | +70.5 |
| max DD % | 15.0 | **18.5** | **23.8** | 30.7 | 37.5 |

P(ret < 0) = 36.5% · P(DD > 20%) = 67.9%

**Downside target for planning: ret −10% to +10%, DD 18–25%.** This
is the quadrant where section E's "mixed / continue" applies and
where the failure thresholds are closest to tripping.

### D.4 IID caveat on bootstrap — read carefully

All bootstrap numbers above assume trade outcomes are independent and
identically distributed. In reality:

- Outcomes cluster by regime (consecutive shorts succeed in
  mean-reverting markets; consecutive longs fail in persistent
  trends)
- A 62-trade iid bootstrap over-samples favorable regime sequences
  as well as unfavorable ones, but the favorable sequences push the
  p50 well above any realistic single-regime expectation
- The bootstrap p50 is therefore an **average over all possible
  regime paths**, including paths that could not plausibly occur
  consecutively in the real market
- **Treat bootstrap p50 as a ceiling, not a median.** The modal
  expectation (D.1) is the realistic base case. The upside (D.2
  bootstrap) is achievable only if the entire 30-day window stays
  in a two-sided mean-reverting regime like Feb–Mar 2026

### D.5 Substrate disagreement

Pi (native Kraken Futures) and Kaggle (Binance spot) disagree on DD by
a factor of ~2. Pi sample is smaller (n=44) but correctly-venued;
Kaggle is larger (n=130) but cross-venue. If the April 2026
microstructure is closer to Pi Feb–Mar than to Kaggle 2024, the Pi
range is more relevant. If April 2026 has drifted into a different
regime, the Kaggle range is the more useful reference. Monitor live
DD against BOTH. If live DD tracks above Pi p90 (17%) for 10+
consecutive trades, that alone is an investigate-not-fail signal,
predating any formal threshold breach.

---

## E. Pre-registered success criteria (30-day window) — tiered

All four conditions in a tier must hold; the highest-tier match wins.

**Success (strong)** — matches or beats the D.2 upside scenario:

- Cumulative ret ≥ **+30%** at day 30
- AND max DD ≤ **15%** at any point
- AND valve trips = 0
- AND knife rate ≤ **30%** (Pi baseline × 1.3 ≈ 29.5%)
- → triggers **sprint v2 "scaling" scope** (section F)

**Success (baseline)** — sprint v1 shipped successfully, modal
expectation met:

- Cumulative ret ≥ **+5%** at day 30
- AND max DD ≤ **20%** at any point
- AND valve trips = 0
- AND knife rate ≤ **35%**
- → sprint v1 is delivered; **sprint v2 refinement is optional**,
   not required

**Mixed / continue** — closer to downside than upside but not a failure:

- Cumulative ret in **[−10%, +5%)**
- AND max DD ≤ **20%**
- AND valve trips = 0
- → **no sprint v2 scope yet.** Continue to 60-day observation with
   the same config before drawing conclusions

**Failure** (any one of the following triggers sprint v2 per F):

- Cumulative ret < **−10%** at day 30
- OR max DD > **25%** at any point (regardless of return)
- OR **2+ valve trips** over the 30-day window
- OR knife rate > **40%** sustained over 20+ consecutive trades
- OR live trade rate outside **[0.8, 4.0] trades/day** for 5+
   consecutive days

**Single valve trip** (within the null 13–15% rate): investigate-not-
fail. Classify via section G rules. Clear-and-continue is allowed if
the trip is diagnosed benign (one outlier trade, known regime spike);
otherwise escalate to failure.

---

## F. Outcome buckets → sprint v2 scope (aligned with E tiers)

Each tier from section E maps to a specific sprint v2 objective. No
scope creep — the tier is mechanical from the evidence.

| Section E outcome | Sprint v2 scope |
|---|---|
| **Success (strong):** ret ≥ +30% AND DD ≤ 15% AND no valve trip AND knife ≤ 30% | **Scaling / sizing.** No new data. Capacity test, leverage ladder (5× → 7× → 10×), cross-pair re-enable check. |
| **Success (baseline):** ret ≥ +5% AND DD ≤ 20% AND no valve trip AND knife ≤ 35% | **Optional refinement.** Sizing tweaks, pair re-enable check on tighter criteria. Not required — continue observation if shadow alignment holds. |
| **Mixed / continue:** ret in [−10%, +5%) AND DD ≤ 20% AND no valve trip | **No sprint v2 yet.** Continue 60-day observation at same config. If 60-day window is also mixed → then open sprint v2 on **H_threshold / imbalance formulation** (requires the Pi L2 collector's forward archive, should have 30+ days by then). |
| **Failure (ret-driven):** ret < −10% at day 30 | **Signal rebuild-or-retire.** Cross-exchange basis from Binance + Coinbase public REST as confirmation filter. Lightweight. Hypothesis: signal needs cross-venue corroboration. |
| **Failure (DD-driven):** max DD > 25% (regardless of ret) | **Regime detector.** Binance funding-rate history (free) + realized-vol classifier on existing bars. No new L2 feeds. Hypothesis: high DD is episodic/regime-switchable; strategy needs a gate for the bad regime. |
| **Failure (behavioral):** knife > 40% sustained over 20+ trades OR trade rate outside [0.8, 4.0]/day for 5+ consecutive days | **Regime detector** (same as DD > 25%). Either signal delivery but hostile execution regime (knife) or candidate-rate drift (frequency). Both point to regime detection. |
| **Failure (valve):** 2+ valve trips over 30d | **Signal-level investigation.** Signal is not tracking shadow at all. Do NOT scope v2 until drill-down complete: every tripping trade's chart + bucket expectations + filter audit. |
| **Single valve trip** (within 13–15% null rate) | **Drift-monitor postmortem.** Read `live_drift_monitor.json` + chart the 10 trades around the trip. Clear-and-continue if diagnosed benign; otherwise escalate into one of the failure buckets above. |

---

## G. Auto-flip valve expected behavior

**Window fills at:** 10 trades. At 2.07 trades/day on Pi, that's
~4.8 live days. First trip possible only after day ~5.

**Threshold:** cumulative sum of last 10 z-scores < −2·√10 ≈ **−6.32**.

**Under null (shadow is correct):**

- Each z ~ N(0, 1) (mean 0, std 1 by construction)
- sum(10 z) ~ N(0, √10) → std ≈ 3.16
- P(single window trips) = P(Z < −2.0) ≈ **2.28%**
- Over 30 days with ~62 trades (~53 sliding windows, ~6 effective
  independent), P(at least one trip) ≈ **13–15%** (moving-average
  effective-independence approximation)
- Expected cum_z under null: ≈ 0 (std ~3.16)

**Under true drift (shadow is wrong, or regime has changed):**

- If live pnl_bps systematically ~1σ below shadow mean: per-window
  cum_z ≈ −10 (−1 × 10) → far below −6.32. Trips within first 10
  trades.
- If live pnl_bps ~0.5σ below: cum_z ≈ −5, below threshold half the
  time. Probably trips within first 20 trades.

**Interpretation rules (pre-registered):**

| Event | Interpretation | Action |
|---|---|---|
| 0 trips over 30d | Null-consistent, shadow is tracking. Success criterion met on this axis. | Continue. |
| 1 trip over 30d | Within the null-rate 13–15%. Likely false positive, but investigate the trigger window. | Investigate: chart-review the 10 trades around trip, check regime. If benign (e.g. one outlier trade), clear valve and continue. If trend, escalate. |
| 2+ trips over 30d | Null probability of 2+ trips ≈ 1–2%. This is a genuine structural signal. | Failure per section E. No clear-and-continue. Scope sprint v2 per F. |

---

## H. Process commitments for this deploy

Per the consolidated plan's process rules:

- **No push without number-review.** Before any push, I will show the
  post-deploy verify checklist output and await explicit "push
  approved." No stealth commits of derived metrics.
- **Dashboard update > 20% number shift requires pre-commit pause.**
  Applies during the 30-day window: if the live cumulative return or
  DD moves >20% between dashboard updates, I will pause for review
  before committing the update.
- **Self-assessment at 30-day window end.** Mandatory. Will list any
  commits where verify-cadence slipped, per the same template as the
  2026-04-21 status audit.
- **Pi L2 collector start.** Not blocking this deploy. Per
  `docs/sprint_v1_findings.md` open-questions item #1, it is the
  highest-priority infrastructure task for sprint v2. Recommend
  starting its standup work during the 30-day 6b observation — not on
  the Pi itself (to not disturb 6b) but on a second archive target
  (VPS or secondary host). Agreement pending.
- **Retroactive flags** from the 2026-04-21 status audit remain
  flagged: commits `3f2c5da` (dashboard +26%) and `3740613` (shadow
  regen) executed with lower verify cadence. No action to amend them
  unless your review decides they should be.

---

## Deploy checklist (NOT RUN YET — for reference at deploy time)

When you say "promote 6b per PATH A" this is the sequence. Not
auto-executed — each step pauses for confirmation.

1. [ ] Rename `state/shadow_expectation.v1.json` →
       `state/shadow_expectation.v1_with_f3c.json` (naming nit fix)
2. [ ] Verify `state/live_drift_monitor.json` is absent on dev and on
       Pi (fresh window for new shadow). Clear if present on Pi.
3. [ ] Verify `state/safety_valve.json` is absent on Pi.
4. [ ] Apply the config diff in section B to
       `algo/entropy_live_multi.py`.
5. [ ] Re-run `tests/test_live_drift_monitor.py` — must be green.
6. [ ] Commit with message
       `phase6b: PATH A promote -- timeout_trail + E3 + 5x; F3c stays off`
7. [ ] Push to main (first push of the session).
8. [ ] On Pi: `git pull && python algo/diagnostics/shadow_expectation_generator.py --no-f3c`
       to regenerate shadow on-host (uses local Pi data path if
       configured; otherwise `scp` from dev).
9. [ ] On Pi: clear `state/live_drift_monitor.json` (delete if present).
10. [ ] On Pi: restart the bot service.
11. [ ] Post-deploy verify:
     - [ ] First startup log line reads
           `f3c_enabled=False timeout_trail_enabled=True e3_time_decay_sl_enabled=True`
           and `leverage=5x`.
     - [ ] First candidate bar audit line appears in
           `state/daily_filter_audit.jsonl` with the revised filter
           gate set.
     - [ ] First closed trade emits a `[LIVE-MONITOR]` line with z
           computed against the revised shadow.
     - [ ] Dashboard 6b pill flips from `current` to `done` (post-
           deploy update, separate commit).

---

## Deliverables produced for this brief

- `notes/6b_go_live_brief.md` (this file)
- `algo/reports/pi_is_f3c_discrepancy_diff.json` (Part 1.3)
- `algo/reports/pi_is_f3c_discrepancy_classification.md` (Part 1.5)
- `state/shadow_expectation.json` — verified matches PATH A, reproducible
- `state/shadow_expectation.v1.json` — F3c-ON archive preserved
- `algo/diagnostics/status_2026_04_21_bootstrap.py` — bootstrap helper
  (read-only; not committed)
- Dashboard tagline + status line update (section below; applied to
  `docs/index.html` but not committed)

---

## Pause gate

Per Part 2 step 2.4: no deploy, no push, no merge. Awaiting your
explicit "promote 6b per PATH A" instruction.
