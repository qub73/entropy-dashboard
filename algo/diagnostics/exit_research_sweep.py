"""Exit-parameter sweep. Train on Kraken-native (62 clean days), validate
the top cells on Pi IS. Reference = operational config.

Operational config (revised 6b, current live spec):
  SL=50, trail_after=150, trail_bps=50, mode=standard,
  pst_width=20, pst_floor=-60, pst_max_wait=30,
  E3 on (tighten=-25 at 60m if peak<50),
  timeout=240, leverage=5, tp=200, knife=50, ext=100, H=0.4352

Focus: improve mode (a) exit-too-soon (849bps left on table across 4
live trades) via wider tp_trail / ATR-scaled / dual-stage.
"""
from __future__ import annotations
import itertools, json, sys, time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent
ALGO = HERE.parent
sys.path.insert(0, str(ALGO)); sys.path.insert(0, str(HERE))

from phase4_sizing_sim import load_pi
from sprint_v15_refit_and_confirm import build_slice as build_kraken
from exit_research_engine import ExitParams, run_exit_variant

OUT = ALGO / "reports" / "exit_research_sweep.json"

# Operational (reference) config
OPERATIONAL = ExitParams(
    sl_bps=50, trail_after=150, trail_bps=50, exit_mode="standard",
    pst_width_bps=20, pst_hard_floor_bps=-60, pst_max_wait_bars=30,
    e3_enabled=True, e3_after_bars=60, e3_peak_threshold_bps=50,
    e3_tightened_sl_bps=25, timeout_bars=240, leverage=5,
)

# Sweep: address mode (a) primarily, (b) secondarily, (c) not observed
STANDARD_SWEEP = {
    "sl_bps":                [50, 60, 70],       # (b) wider SL
    "trail_after":           [150, 200],          # (a) delay trail activation
    "trail_bps":             [50, 75, 100],       # (a) wider trail
    "e3_enabled":            [True, False],
    "pst_width_bps":         [20, 40],            # (c) pst escape room
    "timeout_bars":          [240, 300],
}

# Radical variants (non-standard exit_mode)
RADICAL_CELLS = [
    # ATR-scaled trail (let volatility size the trail)
    ExitParams(sl_bps=50, trail_after=150, exit_mode="atr_trail",
                atr_trail_mult=2.0, timeout_bars=240),
    ExitParams(sl_bps=50, trail_after=150, exit_mode="atr_trail",
                atr_trail_mult=3.0, timeout_bars=240),
    ExitParams(sl_bps=60, trail_after=200, exit_mode="atr_trail",
                atr_trail_mult=2.5, timeout_bars=240),
    # Dual-stage trail: tight until +100, wide after
    ExitParams(sl_bps=50, trail_after=100, exit_mode="dual_stage",
                trail_bps_tight=30, trail_bps_loose=100,
                dual_stage_pivot=150, timeout_bars=240),
    ExitParams(sl_bps=50, trail_after=100, exit_mode="dual_stage",
                trail_bps_tight=40, trail_bps_loose=75,
                dual_stage_pivot=150, timeout_bars=240),
    ExitParams(sl_bps=60, trail_after=100, exit_mode="dual_stage",
                trail_bps_tight=40, trail_bps_loose=120,
                dual_stage_pivot=150, timeout_bars=240),
]


def _iter_standard():
    keys = list(STANDARD_SWEEP.keys())
    for combo in itertools.product(*(STANDARD_SWEEP[k] for k in keys)):
        yield dict(zip(keys, combo))


def _cell_label(p: ExitParams) -> str:
    if p.exit_mode == "atr_trail":
        return (f"atr_trail_sl{p.sl_bps:.0f}_ta{p.trail_after:.0f}"
                f"_mult{p.atr_trail_mult:.1f}")
    if p.exit_mode == "dual_stage":
        return (f"dual_sl{p.sl_bps:.0f}_ta{p.trail_after:.0f}"
                f"_tight{p.trail_bps_tight:.0f}_loose{p.trail_bps_loose:.0f}"
                f"_piv{p.dual_stage_pivot:.0f}")
    return (f"std_sl{p.sl_bps:.0f}_ta{p.trail_after:.0f}"
            f"_tbps{p.trail_bps:.0f}_e3{'on' if p.e3_enabled else 'off'}"
            f"_pst{p.pst_width_bps:.0f}_to{p.timeout_bars}")


def main():
    t0 = time.time()
    # ---- Load both substrates ----
    print("loading Kraken-native (train)...")
    kn_feats, kn_extra, kn_meta = build_kraken(
        "kraken_native_full", ("2025-12-18", "2026-04-15"))
    print(f"  kraken-native: {kn_feats['n']:,} bars ({kn_meta['n_days']} clean days)")
    print("loading Pi IS (validate)...")
    pi_feats, pi_extra = load_pi()
    print(f"  pi_is: {pi_feats['n']:,} bars")

    # ---- Build cells ----
    cells: List[Dict] = []
    # Operational as cell 0
    cells.append({"label": "operational_ref", "params": asdict(OPERATIONAL)})
    for cfg in _iter_standard():
        p = ExitParams(**cfg)
        cells.append({"label": _cell_label(p), "params": asdict(p)})
    for p in RADICAL_CELLS:
        cells.append({"label": _cell_label(p), "params": asdict(p)})
    print(f"\ntotal cells: {len(cells)} "
          f"(1 ref + {len(list(_iter_standard()))} standard + "
          f"{len(RADICAL_CELLS)} radical)")

    # ---- Run on Kraken-native (training) ----
    print(f"\n[train] Kraken-native sweep...")
    t1 = time.time()
    for i, c in enumerate(cells, 1):
        p = ExitParams(**c["params"])
        r = run_exit_variant(kn_feats, kn_extra, p)
        c["kraken_native"] = {k: v for k, v in r.items() if k != "trades"}
        if i % 20 == 0 or i == len(cells):
            print(f"  [{i:>3}/{len(cells)}] elapsed {time.time()-t1:.0f}s")

    # ---- Pick top cells on Kraken-native ----
    # Primary: compound return; secondary: sharpe; tertiary: low DD
    scored = []
    for c in cells:
        m = c["kraken_native"]
        if m.get("n_trades", 0) < 20: continue
        # Rank: weighted combo
        score = (m["compound_ret_pct"] - 0.5 * m["max_dd_pct"]
                 + 5 * m["sharpe_daily"])
        scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    top_labels = [c["label"] for _, c in scored[:8]]
    # Also keep operational_ref for validation
    if "operational_ref" not in top_labels:
        top_labels.insert(0, "operational_ref")
    print(f"\ntop-8 on Kraken-native (by ret - 0.5*DD + 5*sharpe):")
    for s, c in scored[:8]:
        m = c["kraken_native"]
        print(f"  {c['label']:<50}  score={s:+.1f}  "
              f"ret={m['compound_ret_pct']:+7.2f}%  DD={m['max_dd_pct']:5.2f}%  "
              f"sharpe={m['sharpe_daily']:+5.2f}  n={m['n_trades']}")

    # ---- Validate top cells on Pi IS ----
    print(f"\n[validate] Pi IS on {len(top_labels)} top cells...")
    for c in cells:
        if c["label"] not in top_labels: continue
        p = ExitParams(**c["params"])
        r = run_exit_variant(pi_feats, pi_extra, p)
        c["pi_is"] = {k: v for k, v in r.items() if k != "trades"}

    # Operational reference metrics on Pi IS for all deltas
    ref_kn = next(c for c in cells if c["label"] == "operational_ref")["kraken_native"]
    ref_pi = next(c for c in cells if c["label"] == "operational_ref")["pi_is"]

    print(f"\nOperational ref: KN ret={ref_kn['compound_ret_pct']:+.2f}% "
          f"DD={ref_kn['max_dd_pct']:.2f}%  |  PI ret={ref_pi['compound_ret_pct']:+.2f}% "
          f"DD={ref_pi['max_dd_pct']:.2f}%")

    print(f"\ntop cells -- Kraken-native (train) vs Pi IS (validate) vs operational:")
    print(f"  {'label':<50}  {'KN ret':>8} {'KN DD':>6}  {'PI ret':>8} {'PI DD':>6}  "
          f"{'KN delta':>9} {'PI delta':>9}")
    for lbl in top_labels:
        c = next(x for x in cells if x["label"] == lbl)
        kn = c["kraken_native"]; pi = c.get("pi_is", {})
        kd = kn["compound_ret_pct"] - ref_kn["compound_ret_pct"]
        pd_ = (pi.get("compound_ret_pct", 0) - ref_pi["compound_ret_pct"]) if pi else 0
        print(f"  {lbl:<50}  "
              f"{kn['compound_ret_pct']:+7.2f}% {kn['max_dd_pct']:5.2f}%  "
              f"{pi.get('compound_ret_pct', 0):+7.2f}% {pi.get('max_dd_pct', 0):5.2f}%  "
              f"{kd:+8.2f}pp {pd_:+8.2f}pp")

    # ---- Write output ----
    out = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "train_substrate": "kraken_native_62_clean_days",
        "validate_substrate": "pi_is_feb_apr",
        "operational": {"params": asdict(OPERATIONAL),
                          "kraken_native": ref_kn, "pi_is": ref_pi},
        "cells": cells,
        "top_labels_by_train_score": top_labels,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {OUT}  ({out['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
