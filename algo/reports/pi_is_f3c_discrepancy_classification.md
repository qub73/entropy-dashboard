# Pi-IS F3c discrepancy — classification

**Generated:** 2026-04-20 (read-only, no commits)
**Evidence file:** `algo/reports/pi_is_f3c_discrepancy_diff.json`

## Question being answered

The 2026-04-21 status audit (Part A2) replayed the revised-6b cell on Pi
IS with F3c OFF and reported +178.36% / 8.1% DD / 22.7% knife (44 trades).
The Phase 4 `lev5x_ewoff_e3on` cell (same engine with F3c ON) reported
+61.4% / 12.5% DD / 29.6% knife (27 trades). That gives a local F3c
effect of −117pp return / +4.4pp DD / +6.9pp knife rate — F3c net
negative on Pi IS.

Phase 2 v2 promoted F3c based on a +40% PF uplift. The question: can
both be true under the same configuration?

## Classification: **CASE A — Config mismatch, explanatory**

The two runs used different exit stacks AND different leverage, but
the direction of F3c's effect is consistent across both engines. The
Pi-IS negative was visible in Phase 2 v2 all along; it was not what
promoted F3c. What promoted F3c was Kaggle OOS PF uplift.

## Evidence

### 1. Direction of F3c's effect on Pi IS is identical across engines

| Metric | Phase 2 engine<br/>(10×, no E3, no PST) | Phase 4 / status-audit engine<br/>(5×, E3 ON, PST ON) | Agree? |
|---|---:|---:|:---:|
| compound_ret_pct delta (F3c − baseline) | **−385.17** | **−116.97** | ✓ both negative |
| max_dd delta | **+9.98** | **+4.39** | ✓ both mean F3c worse |
| pf delta | **−0.29** | **−0.66** | ✓ both negative |
| knife_rate delta | **+0.075** | **+0.069** | ✓ both mean F3c worse |
| trades delta | **−17** | **−17** | ✓ identical trade-block count |

The −17 identical trade-delta is the decisive tell: **F3c blocks the
exact same 17 Pi IS trades under both engines.** The candidate pool
entering F3c, and the filter stack upstream of F3c (base signal,
dH<0, knife<50, ext<100), is identical across the two runs. The only
thing that changes is what happens to those trades AFTER they're
accepted — which is an exit-stack question, not a filter question.

### 2. Phase 2 v2 Pi IS baseline (no F3c) was ALREADY better than F3c-ON

Directly from `algo/reports/phase2_filter_ablation_v2.json` (extracted
verbatim in the diff JSON):

| Cell | trades | ret | DD | PF | WR | knife |
|---|---:|---:|---:|---:|---:|---:|
| baseline (no F3c) | 43 | **+560.08%** | **13.47%** | **4.20** | 53.5% | **23.3%** |
| F3c | 26 | +174.91% | 23.46% | 3.91 | 53.8% | 30.8% |

Baseline beats F3c on ret, DD, PF, and knife-rate. Only win_rate is
approximately tied. F3c on Pi IS at Phase 2 was already net-negative —
it just was not the decision input.

### 3. Phase 2 v2 promotion was driven by Kaggle OOS, not Pi IS

| Slice | Baseline PF | F3c PF | PF uplift |
|---|---:|---:|---:|
| Pi IS (Feb 18 – Apr 7 Kraken Futures) | 4.20 | 3.91 | **−6.9%** |
| Kaggle OOS (last 60d ETH/USDT Binance) | 1.52 | 2.14 | **+40.6%** |

The promotion rule in Phase 2 v2 required +15% OOS PF uplift, which
F3c cleared on Kaggle. The Pi IS PF *regression* was visible in the
same report but was not part of the promotion rule's decision
criterion.

### 4. The Kraken-native OOS test (sprint v1.5) now adds a third
substrate that disagrees with Kaggle

From `algo/reports/sprint_v15_f3c_kraken_native_confirmation.json`:

| Slice | Baseline PF | F3c PF | PF uplift |
|---|---:|---:|---:|
| Kraken-native OOS (Mar 18 – Apr 15 2026, clean days) | 1.464 | 1.466 | **+0.1%** (neutral) |

Two of three substrates (Pi, Kraken-native) show F3c neutral or
negative. One of three (Kaggle Binance) shows F3c positive. F3c was
substrate-specific.

## Corrected interpretation of the sprint

- **Phase 2 v2 decision was not wrong by its own rule** — the rule was
  "+15% OOS PF uplift," and F3c cleared it on Kaggle. But the rule was
  under-specified. It did not require agreement between IS and OOS
  slices, and it gave Kaggle OOS decisive weight without a substrate-
  generalization check.
- **Sprint v1.5 was not wrong either** — its RED verdict was mechanical
  against the GREEN criteria (positive PF uplift AND reduced DD on
  Kraken-native OOS), neither of which held.
- **The status audit's Pi-IS finding is real but not new evidence.** It
  reproduced what Phase 2 v2's JSON already contained. The audit's
  contribution is surfacing that the Pi-IS baseline was superior and
  making the substrate disagreement impossible to ignore.

## What this means for sprint v1's decisions

- **F3c's Kaggle-OOS uplift is genuine but substrate-specific.** Taking
  it as a global endorsement of F3c was the promotion-rule error,
  not a measurement error.
- **Removing F3c is supported by both native substrates (Pi and
  Kraken).** The revised 6b config (timeout_trail + E3 + 5× only) is
  not a compromise; it is the config both native substrates prefer.
- **timeout_trail + E3 explain some of F3c's apparent uplift.**
  Under the phase-2 engine (no timeout_trail, no E3), F3c on Pi IS cut
  ret from +560% to +175% — an enormous cost. Under the phase-4 engine
  (with timeout_trail + E3), F3c on Pi IS cut ret from +178% to +61% —
  still large but proportionally smaller. The exit improvements reduce
  but do not invert F3c's sign on Pi IS. F3c is not "replaced by" E3
  and timeout_trail on Pi; those are independent improvements that
  happen to pick up some of the trades F3c was (incorrectly) gating.
- **Promotion rules going forward should require same-sign uplift on
  at least two native substrates** (specifically: Pi IS + one OOS) before
  a filter/gate is considered validated. This is stronger than the
  original "+15% OOS PF uplift" rule and would have correctly rejected
  F3c at Phase 2.

## Bottom line (one line)

**CASE A. F3c was never net-positive on Pi IS under either engine; the
Phase 2 promotion was Kaggle-OOS-driven. Revised 6b (no F3c) is the
native-substrate-preferred config.**

---

## Gate

Per the user's Part 1 instructions: pausing after Step 1.5. Do not
proceed to Part 2 until classification is confirmed.

Awaiting explicit confirmation that CASE A classification is accepted.
