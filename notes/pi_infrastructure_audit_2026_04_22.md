# Pi infrastructure audit — 2026-04-22

**Context:** Pre-PATH-A-deploy near-miss. Earlier today we discovered
`ENTROPY_SERVICE_NAME=mr-trading.service` was set on the Pi while the
actual live bot runs under `entropy-trader.service`. Initial read: "script
restarts a no-op unit." Revised read after inspecting the unit file:
`mr-trading.service` is an entirely separate live-trading system
(`hft.mr_live --live`), and a `systemctl restart` would have **started**
it. This audit characterizes every trading-capable asset on the Pi so the
deploy script hardening can be scoped to prevent this class of error.

Read-only audits, no mutations.

---

## Audit 1 — systemd trading/collection units

### Inventory (all unit files matching trad/live/bot/hft/meme/collect)

```
entropy-trader.service         enabled         enabled
kraken-trader.service          disabled        enabled
live-trader.service            disabled        enabled
meme-collector.service         disabled        enabled
mr-trading.service             disabled        enabled
ob-collector.service           disabled        enabled
vfin-trader.service            disabled        enabled
```

### Active state

```
entropy-dashboard.service      active running  Entropy Trader Status Server
entropy-trader.service         active running  Multi-Pair Entropy Trader
                                               (BTC+ETH, Kraken Futures 10x)
```

### Auto-start (state=enabled)

```
entropy-trader.service         enabled enabled
```

**Only `entropy-trader.service` is enabled for auto-start. All others are
disabled.** Good state.

### Per-unit details

| Unit | Active? | Auto-start | WorkingDir | ExecStart | Purpose |
|---|---|---|---|---|---|
| **entropy-trader.service** | **active** | enabled | `/home/user/entropy_trader` | `venv/bin/python -u entropy_live_multi.py` | **Our bot** (Multi-Pair Entropy Trader, BTC+ETH Kraken Futures 10x) |
| kraken-trader.service | inactive | disabled | `/home/user/kraken_PEGX` | `venv/bin/python -m src.live_trader --live` | Kraken PAXG/EUR HMM Live Trader |
| live-trader.service | inactive | disabled | `/home/user` | `hft/venv/bin/python -u /home/user/hft/ml/live_trader.py` | PPO v5c Momentum Live Trader (Kraken Futures) |
| meme-collector.service | inactive | disabled | `/home/user/hft` | `hft/venv/bin/python pi_collector.py --interval 10 --depth 10` | Meme-coin futures orderbook WebSocket collector |
| **mr-trading.service** | inactive | disabled | `/home/user` | `hft/venv/bin/python -u -m hft.mr_live --live` | **MR Live Trading (Shadow Mode description, but uses `--live` flag)** |
| ob-collector.service | inactive | disabled | `/home/user/vFin` | `python3 orderbook_collector.py --interval 10 --levels 50` | PAXG orderbook collector |
| vfin-trader.service | inactive | disabled | `/home/user/vFin` | `python3 run_live.py --live` | vFin PAXG/USD Futures Live Trader |

### Per-unit flags

- **3 live-trading bots** present as unit files besides ours:
  `live-trader.service`, `mr-trading.service`, `vfin-trader.service`.
  Also `kraken-trader.service` is a non-futures HMM trader. All 4 are
  disabled + inactive.
- **2 collectors** (`meme-collector`, `ob-collector`) that would pull
  orderbook data but not place trades. Also disabled + inactive.
- **`mr-trading.service` note:** description says "Shadow Mode" but
  ExecStart passes `--live`. That is either a misleading description
  OR `--live` doesn't actually trade in this codebase. Cannot determine
  without reading `hft/mr_live.py` (large file, ~112KB). Treat as live
  unless proven otherwise.

---

## Audit 2 — Running processes

### Python processes on Pi

```
PID     ETIME          USER  CMD
1231    5-20:01:51     user  /home/user/entropy_trader/venv/bin/python serve_status.py
227313  2-21:48:59     user  /home/user/entropy_trader/venv/bin/python -u entropy_live_multi.py
```

Only two Python processes. Both belong to our entropy_trader project.
`serve_status.py` is the dashboard helper (5d 20h uptime, started
2026-04-17). `entropy_live_multi.py` is the bot (2d 21h 48m uptime,
started ~2026-04-19 21:47 UTC — **~39 minutes before** the recorded 6a
deploy timestamp of 22:26 UTC, meaning 6a likely did not actually
restart the process; it was a config-only change or the 22:26 marker
predates an automatic systemd restart).

### Network sockets

```
ESTAB   192.168.1.15:45256    104.17.186.205:443    python pid=227313
CLOSE-WAIT  192.168.1.15:46696 104.17.186.205:443    python pid=227313
CLOSE-WAIT  192.168.1.15:54548 104.17.185.205:443    python pid=227313
```

Both `104.17.186.205` and `104.17.185.205` are Cloudflare edge IPs —
consistent with Kraken Futures' WebSocket/REST endpoints (Kraken fronts
its API via Cloudflare). No direct connections to Binance, Coinbase,
Bybit, or other exchanges from bot pid 227313. Clean.

Other sockets: SSH (this session), `rpi-connectd`, `tailscaled`. Not
trading-related.

### Open files in /home/user (lsof, awk-filtered for .py / .log)

```
python  /home/user/entropy_trader/logs/entropy_multi.log
python  /home/user/entropy_trader/logs/service.log
```

Only our bot's logs are open. No processes holding files in
`/home/user/hft/`, `/home/user/vFin/`, `/home/user/kraken_PEGX/`, or
any other trading-project dir.

### Audit 2 flag

**`sudo ss -tnp` was run without password (`sudo -n`) and returned
output.** Some of the socket lines show process info (`users:(...)`),
which normally requires root. That means **passwordless sudo is
configured for the `user` account on this Pi.** Not inherently wrong,
but worth noting: if anything other than the user's own scripts ran as
`user` and called `sudo`, it would have root-level capability without
a prompt. This is unrelated to the deploy bug but is a security posture
item.

---

## Audit 3 — /home/user/hft/ inspection

### Top-level

```
drwxrwxr-x  7 user user   4096 Feb 26 18:56 .
-rw-rw-r--  1 user user   4494 Feb 18 17:12 config.py
drwxrwxr-x  3 user user   4096 Feb 14 14:23 data
-rw-rw-r--  1 user user      0 Feb 26 18:56 __init__.py
drwxrwxr-x  2 user user   4096 Feb 17 00:00 logs
drwxrwxr-x  6 user user   4096 Feb 26 18:58 ml
-rw-rw-r--  1 user user 112117 Mar 10 01:42 mr_live.py     <-- 112 KB
-rw-rw-r--  1 user user   4984 Feb 16 19:06 orderbook.py
-rw-rw-r--  1 user user  10628 Feb 18 17:15 pi_collector.py
-rw-rw-r--  1 user user     49 Feb 16 19:06 requirements.txt
-rw-rw-r--  1 user user  15230 Feb 18 17:12 shadow_analyzer.py
-rw-rw-r--  1 user user    567 Feb 16 19:10 test_pi_parse.py
-rw-rw-r--  1 user user   2194 Feb 16 19:11 test_tail.py
drwxrwxr-x  6 user user   4096 Feb 26 18:57 venv
```

`mr_live.py` is substantial (112 KB). Last touched **2026-03-10** — 43
days ago. Not currently being iterated on, but the codebase is intact
and has its own venv.

### Last-modified in the last 90 days

```
/home/user/hft/logs/meme_collector.log
/home/user/hft/logs/mr_equity.csv                <-- equity curve, live tracking
/home/user/hft/logs/mr_trades.csv                <-- trade log
/home/user/hft/logs/mr_trading_stdout.log
/home/user/hft/logs/mr_trading_stderr.log
/home/user/hft/logs/mr_shadow.csv                <-- shadow log
/home/user/hft/logs/meme_collector_service.log
/home/user/hft/logs/mr_live.log
/home/user/hft/logs/activation_report.json
/home/user/hft/config.py
/home/user/hft/ml/crypto_mom_env_v5b.py
/home/user/hft/ml/crypto_mom_env_v5c.py
/home/user/hft/ml/crypto_mom_env_v5.py
/home/user/hft/ml/live_trader.py
/home/user/hft/ml/v5_solutions_sweep.py
/home/user/hft/ml/mom_features_v5.py
/home/user/hft/ml/logs/live/fills_20260226.csv
/home/user/hft/ml/logs/live/fills_20260227.csv
/home/user/hft/ml/logs/live/fills_20260407.csv   <-- live fills, 2026-04-07
/home/user/hft/ml/logs/live/bars_20260226_1659.csv
/home/user/hft/ml/logs/live/bars_20260227_1151.csv
/home/user/hft/ml/logs/live/bars_20260407_0931.csv
/home/user/hft/ml/logs/live/live.log
```

**Key observation:** `ml/logs/live/fills_20260407.csv` — this system was
live-trading and producing **real fills as recently as 2026-04-07**
(about 15 days ago). It has its own trade log (`mr_trades.csv`) and
equity curve (`mr_equity.csv`). These aren't backtests; they're live
records.

The hft system is not dead code — it's a dormant live-trading system
that was running in production 2 weeks ago and is one `systemctl start`
away from running again.

### Config files

```
/home/user/hft/config.py    (only config file found)
```

No `.env`, no `*.yaml`, no `*.yml`. The project uses `config.py`
(dataclass-based) for configuration. Whether API keys are in
`config.py`, in env vars, or in a separate keystore cannot be determined
without reading further — and the prompt explicitly says do NOT paste
key contents. The first 30 lines begin with `from dataclasses import
dataclass, field` — consistent with a dataclass-config pattern.

**Posture assumption for risk assessment:** treat `config.py` as if it
contains credentials. Starting `hft.mr_live --live` would have picked
up whatever credentials are configured and begun trading.

### README / docs

None found. `find ... -name README* -o -name *.md` returned nothing.

---

## Summary — what's on this Pi

This Pi hosts **four separate trading-system codebases**:

| Codebase | Path | Service(s) | Last live |
|---|---|---|---|
| entropy_trader (ours) | `/home/user/entropy_trader` | entropy-trader.service, entropy-dashboard.service | **currently live** |
| hft | `/home/user/hft` | mr-trading.service, live-trader.service, meme-collector.service | **~2026-04-07** (15d ago) |
| vFin | `/home/user/vFin` | vfin-trader.service, ob-collector.service | unknown (not inspected) |
| kraken_PEGX | `/home/user/kraken_PEGX` | kraken-trader.service | unknown (not inspected) |

Only ours is running or configured to auto-start. The other three are
quiescent but intact — configs in place, venvs built, log dirs present,
some with recent fill data. Each is reachable via a one-line
`systemctl start <unit>` command.

---

## Risk assessment

### What we avoided today

If the deploy had gone ahead tonight with the current Pi env var
(`ENTROPY_SERVICE_NAME=mr-trading.service`), deploy_6b.py's Phase 3
`systemctl restart mr-trading.service` would have:

1. **Started** `hft.mr_live --live` — a fully-functional live-trading
   system that had been live as recently as 5 weeks ago, on whatever
   symbols/exchange its `config.py` specifies.
2. Picked up whatever API credentials `hft/config.py` holds.
3. Begun independent trading decisions, running in parallel with our
   ETH bot (still alive under entropy-trader.service since the deploy
   script would never touch that unit).
4. Resulted in **two independent live traders on one Pi**, potentially
   on overlapping instruments, with no awareness of each other.

Banner verification would then have failed (tailing the wrong log file
for the wrong process), triggering auto-rollback that restored our
bot's state but left `mr-trading.service` running. Rollback has no
logic to stop services it didn't start.

### Attack-surface-style enumeration of the same class of error

Any deploy script that uses `systemctl restart $ENTROPY_SERVICE_NAME`
with no cross-validation is one environment-variable mistake away
from starting any of:

- `live-trader.service` → PPO v5c momentum trader on Kraken Futures
- `vfin-trader.service` → vFin PAXG/USD Futures trader
- `kraken-trader.service` → Kraken PAXG/EUR HMM trader
- `mr-trading.service` → hft.mr_live

All four read distinct config files and could begin trading. This isn't
a "defense against malicious input" problem — it's a **"defense against
typo/copy-paste of a stale env var"** problem, which is the literal
failure mode we almost hit.

### Acute risk NOW (post-audit, pre-fix)

**None.** All non-ours services are `disabled` + `inactive`. They
cannot auto-start on reboot; they will not run until a `systemctl
start` or `systemctl enable --now` is issued. The env var is wrong but
the deploy script hasn't been run with `--execute`. The mistake is
currently latent.

### Residual operational risk

- **Passwordless sudo** on the `user` account (flagged in Audit 2). Not
  a deploy-script bug, but it means any mis-invoked script running as
  `user` has unrestricted root capability. Would be wise to lock down,
  but orthogonal to this sprint.
- **Four trading projects with no README in any of them.** Any human
  operator (or future-me) walking onto this Pi has to grep unit files
  to understand what's what. Deploy_6b.py's error messages should be
  verbose enough to compensate for this (e.g., error that names both
  the expected unit *and* the actual running unit, to make
  misconfiguration obvious).

---

## Recommendations

Ordered by urgency.

### Mandatory before Thursday deploy

1. **Fix the env var on the Pi:**
   ```
   sudo sed -i 's/^ENTROPY_SERVICE_NAME=.*/ENTROPY_SERVICE_NAME=entropy-trader.service/' /etc/environment
   ```
   (Or equivalent — current line needs to change from
   `mr-trading.service` to `entropy-trader.service` with a dash.)
   Verify:
   ```
   grep ENTROPY_SERVICE_NAME /etc/environment
   ```

2. **Fix the default in deploy_6b.py:**
   `SERVICE_NAME_DEFAULT = "entropy-trader.service"` (dash, not
   underscore; current value is wrong on both counts).

3. **Harden pre-flight checks in deploy_6b.py:**
   Layered defense so typos / stale env vars abort the script
   instead of executing against the wrong unit. Proposed checks:
   - **P1 (existing):** `systemctl is-active <name>` returns `active`
     → else abort with message listing what's active.
   - **P2 (new):** Parse `systemctl cat <name>`, verify
     `WorkingDirectory=/home/user/entropy_trader` → else abort.
   - **P3 (new):** Parse `systemctl cat <name>`, verify `ExecStart`
     line contains `entropy_live_multi.py` → else abort.
   - **P4 (new):** From `systemctl show -p MainPID <name>`, get the
     PID; read `/proc/<pid>/cmdline`; verify it contains
     `entropy_live_multi.py` → else abort.
     (This catches the "unit file is correct, but someone swapped the
     process" edge case.)

   Each abort message should name both the expected value and the
   observed value, to make Pi-administrator debugging fast.

4. **Tests for P2/P3/P4.** Each check needs a unit test with a fixture
   that simulates the failure mode (unit file with wrong
   WorkingDirectory, wrong ExecStart, wrong MainPID cmdline).

### Defer (not blocking Thursday deploy)

5. **README.md in each of the four project dirs** documenting what
   service name corresponds to what project. Reduces future
   archaeology. 30 minutes of work; no deploy pressure.

6. **Consider `systemctl mask`-ing the unused services**. `mask`
   prevents even manual `systemctl start`. Mitigates against a future
   "I'll just restart the trader" operator gaffe. But this is
   irreversible without `unmask` and might surprise the user if they
   ever want to bring hft back up. **Recommend against unless user
   explicitly wants it.**

7. **Passwordless-sudo review.** Outside this sprint's scope but worth
   logging. The current posture means a runaway user-owned process
   has root effectively for free.

### Explicitly do NOT do

- Do NOT delete or modify unit files for the other projects. They're
  someone's past work and they're currently inert.
- Do NOT disable or enable any service beyond what's already in place.
- Do NOT read `config.py` for credentials. Not needed; the pre-flight
  hardening in (3) solves the problem without touching credentials.

---

## Scope proposal for Step 2 (user approval needed)

Given this audit, the originally-scoped Step 2 (four changes: default,
env-var, one pre-flight check, one test) is insufficient. The audit
surfaced a class of error larger than "wrong service name" — it's "wrong
service, which could be any of six other trading/collection systems."

**Proposed expanded Step 2:**

- Items 1–2 above (env-var fix on Pi, default fix in script).
- Item 3 above: add P1/P2/P3/P4 pre-flight checks (P1 already exists;
  P2/P3/P4 are new).
- Item 4 above: four test cases, one per new check.
- Commit message: `phase6b: deploy_6b.py hardens service-target
  preflight against sibling trading systems`.

Estimated scope: ~120 lines in deploy_6b.py (helper to parse systemctl
output, the four checks, error messages), ~80 lines in
tests/test_deploy_6b.py (four tests + fixtures). Still one focused
commit.

Pausing. User reads and decides whether to accept the expanded Step 2
scope or prefer a different split.
