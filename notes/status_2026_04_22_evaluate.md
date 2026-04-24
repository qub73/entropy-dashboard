# Status evaluation — 2026-04-22 (read-only)

**Purpose:** evaluate-first status gather ahead of deploy decision. No
commits, no pushes, no Pi mutations. Single consolidated report.

---

## Section 1 — Repo state

### Current position

| Field | Value |
|---|---|
| Branch | `sprint/direction-and-exit-v1` |
| HEAD SHA | `c7ebb4830ec9b134ab1ac72a68b097f5acff6d8e` |
| HEAD subject | `phase6b: deploy_6b.py --health-check and lineage append error handling` |
| HEAD committed | 2026-04-21 00:05:24 +0300 (Tuesday evening, Sofia time) |
| HEAD author | qub73 |
| Push status | **Nothing pushed.** No remote for sprint branch. |

### Commits on sprint branch since Tuesday evening

**None.** HEAD is still `c7ebb48`. No new commits authored in the 40+ hours
since the last reported state. All sprint work from 2026-04-21 (exit-param
research, diagnostics, docs) sits in the working tree, uncommitted.

### Uncommitted working tree

177 items in `git status --porcelain`, all untracked (`??`). Bucket
breakdown:

| Bucket | Count | Category |
|---|---:|---|
| `algo/reports/` | 79 | JSON/CSV research outputs (exit, timeout, sprint, diagnostics) |
| `algo/diagnostics/` | 24 | Research scripts (sprint v1.5, exit research, status replays, Strategy B, timeout-reduction, kraken-native loader) |
| `algo/` (top-level .py files) | ~40 | **Pre-sprint legacy scripts that were never tracked** (backtests, signal tools, trainers). These predate the current sprint; they're not new work |
| `notes/` | 4 | `status_2026_04_20.md`, `exit_research_findings.md`, `timeout_reduction_findings.md`, `6b_go_live_brief.md` |
| `algo/multi_exchange/` | 1 dir | Scaffolding (not opened yet) |
| `algo/params/`, `data/`, `reports_out/` | 3 dirs | misc |
| root `.txt` | 2 | `New Text Document.txt` (this prompt), `New Text Document (2).txt` (scratch) |

**Most of the 177 is not actually new work from the last 48h.** About 40 of
the `algo/*.py` files are pre-sprint artifacts that were never added to git
in the first place. The genuinely new items are concentrated in
`algo/diagnostics/` (research scripts from 2026-04-20 and 2026-04-21) and
`algo/reports/` (their JSON outputs).

### Branches since Tuesday

| Branch | Location | Notes |
|---|---|---|
| `sprint/direction-and-exit-v1` | local only | current |
| `sprint/strategy_b_trend_entropy` | local only | abandoned Strategy B fork |
| `main` | local + origin | last synced pre-sprint |
| `multi-exchange` | local + origin | unrelated long-standing branch |
| `upgrades-test` | local only | unrelated |

No new branches since Tuesday. `sprint/strategy_b_trend_entropy` is still
present locally though user instructed it as abandoned.

---

## Section 2 — 6a live state on Pi

### Raw probe output

```
=== ENV ===
ENTROPY_SERVICE_NAME=mr-trading.service
=== SERVICE ===
inactive
=== CLOCK ===
Wed 22 Apr 19:35:53 UTC 2026
utc_offset_seconds=0
=== POSITION ===
position: None
=== AUDIT ===
-rw-r--r-- 1 user user 2004 Apr 21 23:47 /home/user/entropy_trader/state/daily_filter_audit.jsonl
5 /home/user/entropy_trader/state/daily_filter_audit.jsonl
=== VALVE ===
no_valve
=== LAST STATE WRITE ===
2026-04-21 02:06:00.219641529 +0300
=== TRADE HISTORY ===
21 /home/user/entropy_trader/state/trade_history.jsonl
```

Recent trades (tail 5) and audit tail captured; see Section 3.

### What the probe literally shows vs what's actually happening

**`systemctl is-active mr-trading.service` → `inactive`** AND
**`systemctl is-enabled mr-trading.service` → `disabled`** AND
**`journalctl -u mr-trading.service -n 30` → `-- No entries --`**.

Cross-check with process list (`ps -ef | grep entropy`):

```
user  227313  1  0 Apr20  ...  python -u entropy_live_multi.py   (41 CPU-min)
user    1231  1  0 Apr17  ...  python serve_status.py
user  452121  ...              python push_gist.py
```

And active systemd units:

```
entropy-dashboard.service    active  running   Entropy Trader Status Server
entropy-trader.service       active  running   Multi-Pair Entropy Trader
                                               (BTC+ETH, Kraken Futures 10x)
```

And `entropy_multi.log` last lines (Sofia TZ stamps, converts to UTC -3h):

```
2026-04-22 22:32:22,245 INFO STATUS: {"position": null, "trades": 21,
  "daily_pnl": -3136.73, "signals": {"ETH": 4}, "rejected_knife": {"ETH":1},
  "ETH": {"bars": 4651, "last_H": 0.5109, "h_thresh": 0.4352, "mid": 2405.9}}
```

Bot is alive and logging at 22:32 Sofia = **19:32 UTC 2026-04-22**.

### Anomaly 1 — `ENTROPY_SERVICE_NAME` is wrong

The env var `ENTROPY_SERVICE_NAME=mr-trading.service` is set, but
**`mr-trading.service` is NOT the running bot**. The actual live unit is
`entropy-trader.service` (10x BTC+ETH, path
`/home/user/entropy_trader/venv/bin/python -u entropy_live_multi.py`).
`mr-trading.service` is a stale unit file from 2026-02-16, disabled, never
activated.

**Consequence for deploy_6b.py:** the script reads
`ENTROPY_SERVICE_NAME` and runs `systemctl restart mr-trading.service` in
Phase 3. That command would succeed silently (restarting an inactive unit
is a no-op / trivial start-then-fail) **while the real live bot keeps
running on the old config.** The banner-verify step would then fail
because it's tailing the right log but the right log won't show a
post-restart banner. Auto-rollback would fire against a Pi that never
actually applied the config in the first place.

This is a **critical misconfiguration**, not a cosmetic one. Either the
env var needs to change to `entropy-trader.service`, or the script default
needs to change, or both.

### Anomaly 2 — Pi is not a git repo

```
cd /home/user/entropy_trader && git log
  fatal: not a git repository (or any of the parent directories): .git
```

The Pi deployment does not use git. Code sync is manual (scp or rsync or
direct edit). deploy_6b.py's Phase 0/1 checks (`git status --porcelain`,
`git log -1 --format=%s HEAD`) are designed to run **on the deploy
target**. On the Pi, they would fail immediately.

If the design intent was that deploy_6b.py runs on dev (not Pi), then the
whole Phase 3 block (systemctl restart, banner verify, audit verify) does
not apply — those are Pi-side operations. The current script mixes both
roles in one process with no clear seam.

### Anomaly 3 — `daily_filter_audit.jsonl` has only 5 lines total

Over ~3.5 days of 6a observer operation, only 5 audit rows. That matches
the previously reported low rate (market dormancy — H above 0.4352
threshold most of the time). 1 rejection (`blocked_knife` at 2026-04-22
05:27 UTC), 4 passes. Not an anomaly by itself, but it means we have
nearly zero live telemetry on whether the F3c filter would or wouldn't
have blocked anything in the current regime — the filter is simply not
being reached often enough to matter.

### Anomaly 4 — `last_state_write` is 45h stale, but bot is alive

`multi_trader_state.json` last modified 2026-04-21 02:06 Sofia = 2026-04-20
23:06 UTC. Current time 2026-04-22 19:35 UTC → ~44h gap.

This file is only written on position change (open/close). Last trade was
2026-04-20T23:06 UTC. No position opened since. The probe assertion "last
state write within 5 min = bot alive" is **wrong as designed** — the
canonical liveness signal is `entropy_multi.log` STATUS line heartbeat,
not state-file mtime. The previous smoke-check criterion should be updated
to tail the log instead.

### Net Section 2 verdict

Bot is alive. Bot has never been running under the service name the
deploy script expects. Deploy script would miss and then falsely
auto-rollback. This is a **deploy_6b.py bug**, not a Pi bug — the Pi is
doing exactly what it's been doing for weeks.

---

## Section 3 — Post-6a trades summary

6a deploy timestamp: **2026-04-19T22:26 UTC**. Trade history has 21 total,
of which the last 4 have entry-timestamp after 6a:

| # | Entry UTC | Exit UTC | Hold | Dir | Entry $ | Exit $ | Reason | PnL bps (raw) | PnL bps @10× |
|---|---|---|---:|:-:|---:|---:|---|---:|---:|
| 1 | 2026-04-20T00:07 | 2026-04-20T07:06 | 419m | LONG | 2271.16 | 2284.75 | trail_stop | **+59.83** | **+598.25** |
| 2 | 2026-04-20T14:22 | 2026-04-20T14:41 | 19m | LONG | 2314.38 | 2301.66 | sl | −54.98 | −549.76 |
| 3 | 2026-04-20T21:05 | 2026-04-20T21:11 | 6m | LONG | 2342.60 | 2330.40 | sl | −52.06 | −520.58 |
| 4 | 2026-04-20T21:13 | 2026-04-20T23:06 | 113m | LONG | 2323.30 | 2311.18 | sl | −52.15 | −521.46 |

Entry time is derived as `exit - hold_min`, since trade_history stores
only exit time.

### MFE / MAE

Not available from `trade_history.jsonl` directly — that schema is
exit-summary only, not per-trade MFE/MAE. Exact MFE/MAE requires replay
through 1-min OHLC (that's what
`algo/diagnostics/exit_research_diagnostic.py` does; its output covers
trades 1 and 2 from the table above, classifying trade 1 as mode (a) and
trade 2 as clean). Trades 3 and 4 post-date that diagnostic and have not
been replayed yet.

Indirect MFE hints from live reasons:
- **Trade 1** (trail_stop, +59.8bps exit): trail fires when price pulls
  back from peak by 50 bps after peak has cleared trail_after=150. Exit
  at +59.8 implies peak was ≥ ~109.8 bps (and likely higher if trail
  activation was indeed at 150). Mode (a) candidate — matches the
  exit-research diagnostic's classification.
- **Trades 2–4**: stopped out at ~-50 to -55 bps, which is right at
  SL=50 plus slippage. No meaningful MFE before the stop, or the stop
  wouldn't have been reached.

### Aggregates (post-6a only, 4 trades)

| Metric | Value |
|---|---:|
| Trades | 4 |
| Wins | 1 |
| Losses | 3 |
| Win rate | 25.0% |
| Net bps (raw) | **−99.35 bps** |
| Net bps (@10× live) | **−993.55 bps** |
| Avg hold (min) | 139 (skewed by trade 1's 419m hold) |
| Median hold (min) | 66 |

### Outliers / flags

- **Trade 1** is the only winner; its 419-minute hold is notably longer
  than the 6–113 minute losers. That's consistent with the general pattern
  in the earlier 19-trade set: the rare winners that survive the
  early-stop zone can run a long time.
- **Trades 3 and 4** happened 2 minutes apart (21:05 → 21:07 entry / 21:11
  and 23:06 exits). This suggests the bot re-entered within one cooldown
  window. Not inherently wrong (cooldown may have elapsed), but worth
  checking that the cooldown logic is firing as designed — two consecutive
  SL losses within 2 hours is a drawdown pattern worth monitoring.
- **All 4 post-6a trades are LONG.** No shorts. Consistent with the audit
  data (4/5 recorded candidate directions are +1, one is −1 and got
  `blocked_knife`). Either the H+filter combination is only firing on
  longs in this regime, or shorts are being blocked upstream of the audit
  writer. Audit log shows one `blocked_knife` on the short side — so the
  bot did consider shorts but the knife filter blocked at least one.

### Pi daily_pnl tracker

Log shows `daily_pnl: -3136.73` bps (10× leveraged, cumulative since the
tracker started — **not** just post-6a). The −993 bps from 4 post-6a
trades accounts for about 1/3 of that. The rest is pre-6a carry.

---

## Section 4 — Exit research brief

Source: `notes/exit_research_findings.md` (dated 2026-04-21), which
documents the sweep results from
`algo/diagnostics/exit_research_{diagnostic,engine,sweep,expand}.py`.

### Diagnostic framing (not a candidate, context)

19-trade failure-mode classification on pre-6a + first-2-post-6a live
trades:

| Failure mode | Count | Total bps lost |
|---|---:|---:|
| (a) exit-too-soon / winner-truncation | 4 | ~849 bps |
| (b) exit-at-bottom / SL then recover | 2 | ~91 bps |
| (c) stuck in stale downtrend | 0 | — |
| clean (no failure mode) | 13 | — |

Mode (a) is the dominant money-on-table lever. Trades #3, #9, #13, #18
of the original 19 each left 60–425 bps forward-favorable behind after
a tp_trail / trail_stop exit. Mode (c) was not observed in this window.

### Candidates

Reference — operational (current live PATH-A spec, 5×): Pi IS +178.36%,
Kraken-native +22.34%, LIVE @10× −1.43% across 19 trades replayed.

#### Candidate 1 — `atr_trail_sl50_ta150_mult2.0` (ATR-scaled trail)

- **Description:** Once peak ≥ 150 bps, replace fixed `peak − 50` trail
  with `peak − 2.0 × atr_bps`.
- **Hypothesis:** Widening the trail in proportion to realized
  volatility keeps the trail looser when the trend is still running,
  addressing mode (a).
- **Substrate evidence:**

  | Substrate | ret | Δ vs op | DD |
  |---|---:|---:|---:|
  | Pi IS (Feb 18 – Apr 7 2026) | +168.57% | **−9.80pp** | 8.09% |
  | Kraken-native (62 clean days, Abraxasccs HF) | +23.95% | +1.61pp | 25.49% |
  | LIVE replay (19 trades Apr 12–19 2026, Binance 1m OHLC) | +2.02% | **+3.45pp** | — |

  Kaggle OOS: **not run** (exit_research_engine.py did not include the
  Kaggle substrate — sweep was Kraken-native + Pi IS + LIVE only).

- **Sprint v1 corrected promotion rule (same-sign uplift on ≥2 native
  substrates, no >10% degradation):** **FAIL.** Uplift is same-sign
  (+) only on KN and LIVE; Pi IS is negative at −9.8pp (right at the
  10% edge). Two-of-three same-sign is technically sufficient, but the
  single −9.8pp degradation is large enough that the rule's "no >10%
  degradation" clause is triggered on the margin.
- **Recommendation: DEFER** (to post-6b shadow-only trial). The +3.45pp
  LIVE signal is small in absolute terms (~$350 on $10k at 10x) and
  counteracted by a material Pi IS cost. Worth a shadow-capital live
  trial; not worth inclusion in the PATH A main-capital commit.

#### Candidate 2 — `std_sl50_ta200_tbps75_e3on_pst40_to240` (delayed & wider trail)

- **Description:** Standard mode, trail_after 150→200, trail_bps 50→75,
  pst_max_wait 30→40.
- **Hypothesis:** Delay trail activation and widen once active — let
  winners run further before any exit pressure.
- **Substrate evidence:**

  | Substrate | ret | Δ vs op | DD |
  |---|---:|---:|---:|
  | Pi IS | +212.52% | **+34.16pp** | 8.09% |
  | Kraken-native | +39.04% | **+16.70pp** | 21.40% |
  | LIVE replay | −13.19% | **−11.75pp** | — |

- **Promotion rule:** **FAIL.** Two-of-three same-sign uplift is
  satisfied (Pi and KN both strongly positive), but the −11.75pp LIVE
  degradation is **above the 10% threshold** on the exact substrate the
  bot is currently trading. This is a textbook sprint-v1.5-F3c replay:
  training-substrate gains that collapse on the live regime.
- **Recommendation: DROP.** Best non-LIVE sweep result, but the LIVE
  regression is the exact failure mode the sprint v1 corrected rule
  exists to catch.

#### Candidate 3 — `std_sl50_ta150_tbps50_e3on_pst40_to240` (pst_max_wait tweak only)

- **Description:** Operational config **except** `pst_max_wait: 30 →
  40`. ONE parameter changed.
- **Hypothesis:** Giving post-timeout-in-loss trades 10 more bars to
  recover MFE before PST-timeout-exit fires will pick up a few trades
  that recover just past the current 30-bar ceiling on Pi IS, with no
  downside elsewhere.
- **Substrate evidence:**

  | Substrate | ret | Δ vs op | DD |
  |---|---:|---:|---:|
  | Pi IS | +182.91% | **+4.55pp** | 8.09% (same) |
  | Kraken-native | +22.38% | +0.04pp | 27.09% (essentially same) |
  | LIVE replay | −1.43% | **±0.00pp** | — |

- **Promotion rule:** **PASS.** Same-sign (non-negative) uplift on all
  3 substrates; strongest on Pi IS; zero degradation anywhere. Small
  effect size (+4.55pp on Pi IS is modest), but the signal is directionally
  consistent and carries zero downside risk.
- **Recommendation: FOLD_IN to PATH A.** Evidence is clean and cheap to
  adopt.
- **Specific change in `entropy_live_multi.py`:** Search for the
  `post_signal_trail` configuration block (appears in `PAIRS["ETH"]`).
  Change `max_wait_bars` (or `pst_max_wait_bars`, depending on the
  actual key) from `30` to `40`. One line. Will need to grep the file
  to confirm the exact key name — the shadow-expectation generator
  and the exit-research engine both refer to it as
  `pst_max_wait_bars=30`.

### Summary table

| Candidate | Pi IS Δ | KN Δ | LIVE Δ | Rule | Recommendation |
|---|---:|---:|---:|:-:|---|
| 1 — atr_trail mult=2.0 | −9.80 | +1.61 | +3.45 | FAIL | **DEFER** (shadow trial) |
| 2 — ta200+tbps75+pst40 | +34.16 | +16.70 | **−11.75** | FAIL | **DROP** |
| 3 — pst_max_wait 30→40 | +4.55 | +0.04 | ±0.00 | **PASS** | **FOLD_IN** |

### Uncommitted working tree — grouped by purpose

| Purpose | Files / dirs |
|---|---|
| **Exit research** | `algo/diagnostics/exit_research_{diagnostic,engine,sweep,expand}.py`, `algo/reports/exit_research_{sweep,expand}.json`, `algo/reports/exit_diagnostic.json`, `notes/exit_research_findings.md` |
| **Sprint v1.5 diagnostics (close-out)** | `algo/diagnostics/sprint_v15_{refit_and_confirm,revised_6b_shadow,step0_block_rate,step4_live_reconstruction}.py`, companion JSONs in `algo/reports/` |
| **Status audits** | `algo/diagnostics/status_2026_04_21_{bootstrap,replay}.py`, `notes/status_2026_04_20.md` |
| **Timeout reduction ablation (RED, kept for record)** | `algo/diagnostics/timeout_reduction_{ablation,live_only}.py`, `notes/timeout_reduction_findings.md` |
| **Phase 2/3 research from earlier** | `algo/diagnostics/phase{2_filter_ablation,2_filter_ablation_v2,2_kaggle_refit,3_e5_acceptance,3_e5_min_hold_sweep}.py`, outputs |
| **D1/D2 diagnostics** | `algo/diagnostics/d1_regime_gate_replay.py`, `algo/diagnostics/d2_*.py`, outputs |
| **Kraken-native HF loader** | `algo/diagnostics/kraken_hf_{loader,coverage_audit,cleanliness,schema_inspect}.py` |
| **6b go-live brief** | `notes/6b_go_live_brief.md` |
| **Legacy / pre-sprint cruft** | ~40 untracked `algo/*.py` files (backtests, trainers) that predate this sprint and were never committed |
| **Scaffolding / empty** | `algo/multi_exchange/` (queued, not opened) |
| **Scratch** | `New Text Document.txt`, `New Text Document (2).txt`, `algo/status.json`, `algo/params/` |

---

## Section 5 — Deploy readiness honest self-assessment

### Can a clean PATH A commit be authored today?

**In principle yes, in practice no — three structural blockers need a
decision before authoring that commit.**

### Blockers

1. **deploy_6b.py targets the wrong service name on the Pi.** The env
   var `ENTROPY_SERVICE_NAME=mr-trading.service` is set, but the real
   live unit is `entropy-trader.service`. If I author the PATH A commit
   today and the deploy is executed tonight, the script restarts the
   wrong unit, banner-verify times out, auto-rollback fires against a
   system that never actually deployed. This is a **hard blocker** that
   requires a deploy_6b.py patch (fix the default + fix the env var on
   Pi, and probably fix the script to verify service name matches an
   actually-running unit before attempting restart).

2. **Pi has no `.git` directory.** deploy_6b.py was designed assuming
   git-based code sync on the target. Pi code sync is manual (scp /
   rsync / direct edit) with no .git. The script's Phase 0/1 git checks
   cannot run on the Pi. If the intent was "script runs on dev only",
   then Phase 3 operations (systemctl restart, banner verify) that must
   run on the Pi need a different mechanism (ssh execution, or a Pi-side
   companion script). The current design conflates both locations.

3. **The PATH A promote commit has never been authored.** deploy_6b.py's
   `EXPECTED_COMMIT_SUBJECT = "phase6b: PATH A promote -- timeout_trail
   + E3 + 5x; F3c stays off"` refers to a commit that does not exist.
   This is the commit the user would be authorizing tonight. Scope for
   that commit needs to be locked down before authoring, specifically:
   - Does it fold in exit-research Candidate 3 (`pst_max_wait 30→40`)?
     The evidence says yes per sprint v1 corrected rule.
   - Does it reset 6a's 10x leverage to 5x as planned? (Yes per brief,
     but confirm.)
   - Does it otherwise match exactly what shadow_expectation was
     regenerated against? (Shadow says: timeout_trail+E3+5x, no F3c,
     Pi live defaults for SL/TP/knife/ext. The config diff should be
     surgically small.)

### Softer issues (not blockers but should be addressed)

4. **177 untracked items on the sprint branch.** A PATH A commit
   should not inherit these. Best path: stash or explicitly exclude
   everything except the config diff; land research artifacts as a
   separate follow-up commit (or don't commit them at all — they're
   research, not product).

5. **6a is currently at 10x live** (confirmed by `entropy-trader.service`
   description). The PATH A transition is 10x → 5x **and** exit-logic
   changes **and** (optionally) pst_max_wait tweak in a single deploy.
   Compound changes compound risk. Mitigation: accept this because the
   shadow_expectation was regenerated against exactly this compound
   config, not against incremental diffs.

6. **Deploy window pressure.** The 2026-04-22 06:00 Sofia target has
   already passed (it's now 22:35 Sofia). A same-day Wednesday deploy
   is no longer a morning event; it would be a late-evening event, with
   execution happening past midnight Sofia time. Evening deploys on a
   low-volume regime are fine from a market-risk standpoint; they are
   riskier from a human-attention standpoint (harder to babysit if
   anything goes wrong at 01:00 local).

### What I would do if user said "proceed toward deploy today"

In order, each pausing for confirmation:

1. **Fix deploy_6b.py service targeting.** Either change
   `SERVICE_NAME_DEFAULT` to `entropy-trader.service`, or keep the env-var
   mechanism but add a pre-flight check that the named service is
   currently `active` (not just `is-active` but actually running with a
   matching ExecStart path). Update the `/etc/environment` line on the
   Pi to match. Add a test case.

2. **Decide on the Pi-git gap.** Either (a) accept that deploy_6b.py
   runs on dev and delegates Pi operations via ssh + a thin Pi-side
   script, or (b) add a Pi-side `--deploy-target` mode that verifies
   config-hash instead of git-subject. I'd lean (a) because it's closer
   to how the Pi is actually administered.

3. **Lock PATH A scope.** Confirm fold-in of Candidate 3
   (`pst_max_wait 30→40`). Confirm no other deltas from the
   shadow_expectation config. Produce a 1-page "what exactly goes into
   the commit" sheet.

4. **Stash the 177 untracked items.** `git stash -u` them or move
   research files into a separate `sprint/research-v1` branch. Leave
   sprint/direction-and-exit-v1 with only the files that need to
   change for PATH A.

5. **Author the PATH A commit** with subject matching deploy_6b.py's
   `EXPECTED_COMMIT_SUBJECT`. Single focused diff in
   `algo/entropy_live_multi.py`:
   - `PAIRS["ETH"]["leverage"] = 5` (was 10)
   - `PAIRS["ETH"]["f3c_enabled"] = False` (confirm already False)
   - `PAIRS["ETH"]["timeout_trail_enabled"] = True`
   - `PAIRS["ETH"]["e3_time_decay_sl_enabled"] = True`
   - `PAIRS["ETH"]["pst_max_wait_bars"] = 40` (was 30, Candidate 3)
   - exact keys TBD on grep of file.

6. **Run tests + dry-run** on dev. Paste output.

7. **Merge to main.** Tag the commit.

8. **Push to origin/main.**

9. **Deploy execute window** in consultation with user's travel/
   availability for the next ~24h.

### Honest bottom line

The codebase is NOT in a deploy-today-safely state. The surface items
(dirty tree, untracked research, missing PATH A commit) are straightforward
fixes that take 30–60 minutes. The **structural item** — deploy_6b.py
targeting the wrong service name and assuming Pi has git — is the real
blocker. That's half a day of careful editing and testing to get right.

My recommendation: **defer deploy to 2026-04-23 (Thursday) 06:00 Sofia or
later**, and spend the remaining time tonight/tomorrow fixing the deploy
script and authoring the PATH A commit. The user's original "deploy
Wednesday" plan assumed all of this infrastructure was ready. It isn't,
and pushing through is how silent-failure deploys happen.

If the user insists on deploying today despite the above: the
minimum-safe variant is to execute the service-restart manually on the
Pi after manually rsync'ing the config change, bypassing deploy_6b.py
entirely. I would not do that — it discards the whole point of the
atomic-backup / banner-verify / auto-rollback machinery — but it is
possible.

Pausing. User reads and decides.
