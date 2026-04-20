# Sprint v1.5 decision — F3c Kraken-native OOS ablation

**Date (UTC):** 2026-04-20
**Venue under test:** Kraken-native L2 (Abraxasccs/kraken-market-data HF dataset), ETH/USD
**Pair under test:** ETH
**Sprint scope:** Confirm F3c entry-gate's edge on Kraken-native OOS (post Phase 6a observer-only deploy, pre 6b flag flip)

---

## Verdict — **RED**

F3c is **neutral on PF and worse on DD** on the Kraken-native EVAL slice. The block rate was healthy (>25%), so this is not a low-power-sample outcome. The Kaggle OOS signal that F3c improved PF by ~40% on Pi replay does **not** replicate on Kraken-native OOS.

**Recommended action:** Pause 6b. Do NOT flip the F3c flag live. Investigate root cause before any live flag activation. Continue 6a observer-only deploy (flags default False) undisturbed.

---

## Step 0 — F3c block-rate check

Replayed base signal on 61 of 62 clean days (one truncated-day start dropped; 43,983 1-min bars total).

| H threshold | cands | blocked | block rate |
|---|---|---|---|
| calibrated p3 (H=0.4001) | 182 | 102 | **56.04%** |
| calibrated p5 (H=0.4201) | 225 | 127 | 56.44% |
| calibrated p10 (H=0.4392) | 353 | 186 | 52.69% |
| Pi-native (H=0.4352) | 305 | 165 | 54.10% |

Pi baseline reference: 48 / 124 = 38.7% of trades blocked by F3c on the v2 ablation.

**PASS (≥15% threshold).** The clean-day sample exposes F3c to enough in-scope regimes; the OOS question is well-posed. Notably the sample **over**-exposes F3c (block rate 54% vs 39% on Pi) — consistent with the known clean-day bias toward chop (ema_diff_norm near zero → the F3c band captures more candidates).

SHORT-side block rate (71.8%) is materially higher than LONG (44.2%), consistent with clean-day drift being net upward (fewer SHORTs trend-aligned).

---

## Step 1 — temporal split

| slice | range | clean days used | raw snapshots | 1-min bars |
|---|---|---|---|---|
| FIT  | 2025-12-18 .. 2026-02-14 | 29 (one partial-day start dropped) | 25,570 | 21,146 |
| EVAL | 2026-03-18 .. 2026-04-15 | 22 | 21,285 | 17,296 |

Natural ~1-month gap between slices. Both ≥20 clean days, no pause needed.

---

## Step 2 — Kraken-native baseline refit (FIT slice, no F3c)

Grid: SL ∈ {30,40,50,60,70} × TP ∈ {150,200,250,300} × H ∈ {0.40, 0.4352, 0.47, 0.50}, knife=50, ext=100 fixed (Pi-baseline).

Top 5 cells by FIT compound return:

| # | SL | TP | H | ret | DD | PF | trades | WR |
|---|---|---|---|---|---|---|---|---|
| 1 | 50 | 150 | **0.50** | **+2471.18%** | 30.39% | 3.48 | 103 | 40.8% |
| 2 | 50 | 200 | 0.50 | +2471.18% | 30.39% | 3.48 | 103 | 40.8% |
| 3 | 50 | 250 | 0.50 | +2471.18% | 30.39% | 3.48 | 103 | 40.8% |
| 4 | 50 | 300 | 0.50 | +2471.18% | 30.39% | 3.48 | 103 | 40.8% |
| 5 | 60 | 150 | 0.50 | +1724.21% | 37.06% | 3.27 | 92 | 40.2% |

TP 150/200/250/300 tie because all eligible trades exit via the tp_trail path activated at peak_bps≥150 (the engine's hard-coded `trail_after`), never touching the fixed TP level. Selected: **SL=50 / TP=150 / H=0.50**.

Caveat: +2471% FIT is structurally inflated (compound-return on leveraged trades across 29 virtual days of clean-only bars). The figure is useful for selecting parameters but is **not** a prediction of live PnL.

---

## Step 3 — F3c OOS confirmation (EVAL slice, refit params)

Both runs: H=0.50, SL=50, TP=150, knife=50, ext=100.

| metric | baseline (no F3c) | with F3c (+/-0.3) | delta |
|---|---|---|---|
| trades | 111 | 97 | -14 |
| win_rate | 33.3% | 32.0% | -1.3pp |
| compound_ret_pct | **+3.57%** | **+0.34%** | **-3.23pp** |
| max_dd | **40.36%** | **41.96%** | **+1.60pp** |
| PF | 1.464 | 1.466 | +0.0015 |
| knife_catchers | 37 | 41 | +4 |
| **knife_rate** | **33.3%** | **42.3%** | **+8.93pp** |
| trades_long | 57 | 52 | -5 |
| trades_short | 54 | 45 | -9 |

### Assessment vs decision tree

- **GREEN requires** positive PF uplift AND reduced DD. ❌ PF essentially unchanged (+0.0015), DD INCREASED by +1.60pp.
- **YELLOW requires** one of PF uplift or DD reduction, OR borderline block rate (<25%). ❌ Neither metric improved. Block rate was healthy (54%, not borderline).
- **RED**: F3c neutral-or-worse on both metrics, block rate healthy. ✅ **Matches.**

Additionally the **knife rate rose 9pp under F3c**, which is the opposite of what F3c was supposed to do (gate out counter-trend entries). On this OOS slice F3c is actively pruning the *good* surviving entries while leaving knife-catchers untouched — the filter is not acting on the dimension it was designed for.

---

## Step 4 — live-window reconstruction (retrospective only, not a decision input)

ETH live trades with entries in 2026-04-12 .. -15, with Kraken-native L2 feature state reconstructed at entry bar. All 8 logged trades were LONG.

| # | overlap | entry_date | ema_diff_norm | F3c decision | live pnl (bps) | live reason |
|---|---|---|---|---|---|---|
| 1 | partial | 2026-04-12 | -0.411 | **BLOCK** | +30.5 | timeout |
| 2 | wholly | 2026-04-13 | -9.768 | **BLOCK** | -53.9 | sl |
| 3 | wholly | 2026-04-13 | +1.787 | allow | +129.1 | manual_close |
| 4 | wholly | 2026-04-14 | +5.614 | allow | -52.6 | sl |
| 5 | wholly | 2026-04-14 | -3.907 | **BLOCK** | -53.8 | sl |
| 6 | wholly | 2026-04-14 | +3.842 | allow | -64.8 | sl |
| 7 | wholly | 2026-04-14 | +3.842 | allow | -56.8 | sl |
| 8 | wholly | 2026-04-15 | -2.789 | **BLOCK** | +15.2 | timeout |

- Wholly-clean overlap: 7 trades, F3c blocks 3.
- Partial overlap: 1 trade (#1), F3c blocks it.
- Of 3 live winners, F3c would block 2 (#1, #8).
- Of 5 live losers, F3c would block only 2 (#2, #5); the 3 biggest losers (#4, #6, #7) passed F3c.

Net F3c pnl impact on this 8-trade sample: saves 107.7 bps on blocked losers, forgoes 45.7 bps on blocked winners → **+62 bps / 8 trades**, well within noise for a sample this size. More importantly, the block distribution is **not aligned with outcome**: F3c does not separate the losers from the winners in this window.

---

## Volatility-regime bias caveat (required disclosure)

The clean-day subset is not a representative random sample of ETH trading conditions. The Kraken WebSocket book-reconstruction bug (missed volume-zero-means-delete messages during aggressive moves) corrupts L2 state during high-volatility episodes. A day is marked "clean" here when ≥80% of ETH/USD snapshots pass the ask-ordering sanity check — so clean days systematically skew toward **calmer**, **lower-volume** regimes.

Consequences for this sprint:

1. **F3c block-rate over-exposure.** In chop, ema_diff_norm hovers near zero, so the F3c ±0.3 band captures a wider fraction of candidates than it would on mixed days (54% clean-day vs 39% Pi baseline). The sample is biased **toward** rejecting F3c's edge — yet even with this over-exposure, the OOS metrics did not favor F3c.
2. **Baseline EVAL return is low (+3.57%).** Clean days don't contain the volatile trend regimes where F3c's alignment filter would shine. The EVAL signal is dominated by chop outcomes, not the trending-vs-counter-trend regime separation F3c was trained for. This is the **strongest argument against over-interpreting the RED verdict**.
3. **H refit chose 0.50**, well above the Pi-calibrated 0.4352 and well above the live entropy_percentile=3 threshold (which on this slice = 0.4001). The selected H accepts much looser entropy signals, diluting the "organized flow" assumption F3c relies on. A tighter H in the refit might have produced a different F3c result — but the grid we were instructed to use did not sweep below 0.40.
4. **EVAL n=111 trades over 22 virtual days.** Statistical power is modest; confidence intervals on ret% and DD% both cover zero-delta.

These caveats do NOT rescue F3c to GREEN or YELLOW: the decision tree is mechanical (PF and DD vs baseline), and both failed. But they argue that the RED verdict is best read as "F3c's Kaggle-OOS signal does not generalize to this particular Kraken-native clean-day slice" rather than "F3c is broken." A replication attempt on a volatile-regime sample (e.g., days the cleanliness filter rejected for L2 corruption but where we can reconstruct at lower fidelity) might tell a different story.

---

## Recommendation

- **Do not flip 6b's F3c flag live.** The RED criterion is met on the OOS slice we actually tested.
- **Continue Phase 6a observer-only deploy unchanged.** Flags default False, 10× leverage, audit writer active, drift monitor active.
- **Before attempting 6b again:** (a) investigate why F3c increased knife rate — that is the most surprising failure mode; (b) consider re-running the OOS with a tighter H sweep (e.g., H ∈ {0.30, 0.35, 0.40, 0.4352}), because H=0.50 may have diluted F3c's domain of usefulness; (c) consider whether clean-day bias is driving a false negative by reconstructing a volatile-regime OOS set from lower-fidelity data.

---

## Output file index

- `reports/sprint_v15_f3c_block_rate.json` — Step 0 block-rate check
- `reports/sprint_v15_kraken_native_refit.json` — Step 2 FIT grid search
- `reports/sprint_v15_f3c_kraken_native_confirmation.json` — Step 3 EVAL confirmation
- `reports/sprint_v15_live_window_reconstruction.json` — Step 4 live-window reconstruction
- `reports/sprint_v15_decision.md` — this document

Paused. Awaiting your call.
