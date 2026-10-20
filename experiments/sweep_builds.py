"""
Two-tier build search.

Running the full DP once per (Pudge build, Invoker build) pair is far too
expensive: one full run is ~35 minutes and the build space is in the hundreds.
So the search is hierarchical, as laid out in docs/formulation.md §4.4:

  tier 1  SCREEN  - a deliberately small state space (short arena, coarse HP,
                    truncated cooldowns) ranks the candidate builds.  It is
                    *not* accurate in absolute terms - truncating Invoker's
                    combo cooldown from 26 s to 12 s makes him roughly twice as
                    strong - but it preserves the ordering well enough to throw
                    away dominated builds.

  tier 2  REFINE  - the full DP is run only on the surviving Pareto set.

Only tier 1 is implemented here; tier 2 reuses run_full.py with --split.

Usage:
    python3 -m experiments.sweep_builds --tier1
"""

from __future__ import annotations

import argparse
import itertools
import json
import time

import numpy as np

from .full_model import build_shape, terminal_array, make_successor
from .run_full import make_cfg, build_actions, orb_stats, combo_damage
from solvers.npdp import solve_game_np, solve_game_np_mp

#: Invoker orb distributions worth searching.  The level-15 budget is 17 points,
#: max 8 per orb, so these are the extreme and middle points of the simplex.
SPLITS = [(8, 8, 1), (8, 7, 2), (7, 7, 3), (8, 4, 5), (8, 1, 8),
          (6, 6, 5), (4, 8, 5), (1, 8, 8), (5, 5, 7)]

#: which orbs are kept active determines the per-instance passives:
#: Quas -> HP regen, Wex -> move speed + attack speed, Exort -> attack damage
INSTANCE_SETS = [(3, 0, 0), (0, 3, 0), (0, 0, 3), (1, 1, 1)]

# d_max must exceed the ~4.36 m Meat Hook dodge threshold, otherwise
# the hook always lands and the kiting game does not exist at all.
#: Screening is deliberately cheap and therefore biased: truncating the hook
#: to 3.3 s and the kit to 6.3 s buffs both sides, and 5 room buckets coarsen
#: the kiting budget.  It is used only to rank builds, never to report a value.
SCREEN = dict(d_max=8.0, dist_q=1.0, hpq=400.0, cd_cap=11, c2_cap=21,
              room=(0.0, 2.0, 4.0, 6.0, 8.0))


def screen_config(split, instances, blink, slow_item, p_hp, i_hp):
    cfg = build_actions(make_cfg(split=split, instances=instances,
                                 blink=blink, slow_item=slow_item,
                                 p_hp=p_hp, i_hp=i_hp, **SCREEN))
    return cfg


def solve_screen(cfg, sweeps=120):
    shape, meta = build_shape(cfg)
    tv = terminal_array(shape, meta, cfg)
    succ = make_successor(cfg, meta)
    V, info = solve_game_np_mp(shape, (len(cfg["acts1"]), len(cfg["acts2"])),
                               succ, tv, chunk=150_000, nproc=2,
                               max_sweeps=sweeps, min_sweeps=40,
                               tol=1e-5, verbose=False)
    return V, meta, info


def value_at(V, meta, cfg, d_m=1.0):
    """Pudge's win probability with both at full HP, everything ready, mid room."""
    d = min(int(round(d_m / cfg["dist_q"])), meta["ND"] - 1)
    return float(V[d, meta["NH1"] - 1, meta["NH2"] - 1, 0, 0, meta["NR"] // 2])


def tier1(args):
    rows = []
    p_hp, i_hp = args.p_hp, args.i_hp
    t0 = time.time()
    for split, inst in itertools.product(SPLITS, INSTANCE_SETS):
        cfgp = screen_config(split, inst, args.pudge_blink, args.pudge_slow,
                             p_hp, i_hp)
        V, meta, info = solve_screen(cfgp)
        v = value_at(V, meta, cfgp)
        P = cfgp["params"]
        rows.append(dict(split=list(split), instances=list(inst),
                         value=v, i_dps=round(P["i_dps"], 1),
                         i_ms=round(P["i_ms"], 1),
                         combo=round(P["c_phys"] + P["c_pure"] + P["c_magic"]),
                         cycle_s=round(cfgp["snap_cd_steps"] * 0.3, 1),
                         hp=meta["NH2"] - 1))
        print(f"  {split} inst={inst}  V={v:.3f}  i_dps={P['i_dps']:6.1f} "
              f"MS={P['i_ms']:6.1f}  combo={rows[-1]['combo']:5d}", flush=True)
    print(f"\ntier1: {len(rows)} configs in {time.time()-t0:.1f}s")
    with open("results/screen.json", "w") as f:
        json.dump(rows, f, indent=1)

    # Pareto front over (damage, mobility) is what matters for Invoker; for a
    # first cut just report the spread and the best/worst.
    vals = sorted(rows, key=lambda r: -r["value"])
    print("\nbest for Pudge (lowest Invoker value):")
    for r in vals[:4]:
        print(f"   {r['split']} inst={r['instances']} V={r['value']:.3f}")
    print("worst for Pudge:")
    for r in vals[-3:]:
        print(f"   {r['split']} inst={r['instances']} V={r['value']:.3f}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier1", action="store_true")
    ap.add_argument("--p-hp", dest="p_hp", type=float, default=3200.0)
    ap.add_argument("--i-hp", dest="i_hp", type=float, default=None)
    ap.add_argument("--pudge-blink", dest="pudge_blink", action="store_true")
    ap.add_argument("--pudge-slow", dest="pudge_slow", type=float, default=0.0)
    a = ap.parse_args()
    if a.tier1:
        tier1(a)
