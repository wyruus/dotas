"""
End-to-end run: Pudge vs Invoker, level 15, solved by value iteration.

State: s = (d, h1, h2, c1, c2, room2)

    d      distance, grid cells of DIST_Q metres (capped at D_MAX)
    h1/h2  HP in quanta of HP_Q
    c1     Pudge cooldown code (Meat Hook), 4 buckets
    c2     Invoker cooldown code (main nuke), 4 buckets
    room2  Invoker's free space behind him - gates whether a hook is dodgeable

Usage:  python3 -m experiments.run_duel [--quick]
"""

from __future__ import annotations

import math
import sys
import time

from dota2sim.constants import UNITS_PER_METER, TICK, armor_multiplier
from dota2sim.tactical import lateral_reach
from solvers.dp import solve_game

# --------------------------------------------------------------------------
# Matchup parameters (patch 7.41f).  Pudge numbers are verified against the
# official datafeed; see dota2sim/herodata.py.
# --------------------------------------------------------------------------
DIST_Q = 0.5
D_MAX_M = 22.0
HP_Q = 100.0

TICK_S = TICK
HOOK = dict(damage=360.0, dtype="pure", mana=120.0, cooldown=12.0,
            cast_point=0.3, cast_range=1300.0, speed=1600.0, radius=100.0,
            drag_stop=125.0)
DISMEMBER = dict(dps=140.2, dtype="magical", mana=130.0, cooldown=25.0,
                 cast_range=200.0, cast_point=0.3, duration=2.75)
ROT = dict(dps=120.0, radius=250.0, slow=0.32)
BOUND = 24.0          # bound radius, used for all AoE geometry

#: placeholder Invoker burst until the datafeed sweep lands; the runner takes
#: these from data/invoker.json when present
INVOKER_SPELL = dict(damage=500.0, dtype="magical", mana=200.0,
                     cooldown=30.0, cast_range=900.0, radius=200.0,
                     speed=1200.0, cast_point=0.3, delay=0.0)


def hp1_max():  return 3200.0
def hp2_max():  return 2200.0


class Model:
    def __init__(self, speed1=280.0, speed2=300.0, has_blink=True,
                 slow_item=0.0, pudge_armor=5.43, invoker_armor=2.0,
                 pudge_mr=0.2912, invoker_mr=0.30, pudge_dps=70.0,
                 invoker_dps=90.0):
        self.d_max = int(round(D_MAX_M / DIST_Q))
        self.n_h1 = int(hp1_max() / HP_Q) + 1
        self.n_h2 = int(hp2_max() / HP_Q) + 1
        self.n_room = 3
        self.room_m = (0.0, 1.6, 8.0)     # <2.36 m => undodgeable
        self.speed = (speed1, speed2)
        self.has_blink = has_blink
        self.slow_item = slow_item        # ranged slow applied by Pudge, 0..1
        self.pudge_armor = pudge_armor
        self.invoker_armor = invoker_armor
        self.pudge_mr = pudge_mr
        self.invoker_mr = invoker_mr
        self.pudge_dps = pudge_dps
        self.invoker_dps = invoker_dps
        self.n_states = (self.d_max + 1) * self.n_h1 * self.n_h2 * 4 * 4 \
            * self.n_room

    def reach_cells(self, who):
        return max(1, int(round(self.speed[who] / UNITS_PER_METER * TICK
                                / DIST_Q)))

    # ------------------------------------------------------------------
    def hook_hits(self, d_m, room2_m):
        t = HOOK["cast_point"] + (d_m * UNITS_PER_METER) / HOOK["speed"]
        eff = HOOK["radius"] + BOUND
        v = self.speed[1] * (1.0 - self.slow_item)
        return lateral_reach(v, t, room2_m) <= eff

    def magic_after(self, dmg, mr):
        return dmg * (1.0 - mr)

    def phys_after(self, dmg, armor):
        return dmg * armor_multiplier(armor)


# --------------------------------------------------------------------------
# Transition function
# --------------------------------------------------------------------------
def build_transitions(m: Model):
    """Return transitions(state) -> Trans for the DP core."""
    hq, dq = HP_Q, DIST_Q
    n_h1, n_h2, n_d = m.n_h1, m.n_h2, m.d_max
    r2m = m.room_m
    k1, k2 = m.reach_cells(0), m.reach_cells(1)
    mv1 = sorted({-k1, -max(1, k1 // 2), 0})
    mv2 = sorted({0, max(1, k2 // 2), k2})

    def trans(s):
        d, ih1, ih2, c1, c2, ir2 = s
        HP1, HP2 = ih1 * hq, ih2 * hq
        if HP1 <= 0:
            return Trans(0.0, 0, 0, [])
        if HP2 <= 0:
            return Trans(1.0, 0, 0, [])
        d_m = d * dq
        room2 = r2m[ir2]
        hook_ready = c1 == 0
        spell_ready = c2 == 0

        # --- Pudge action list: (dd, combat) ---------------------------
        acts1 = []
        for p in mv1:
            d_after = max(0.0, d_m + p * dq)      # p<0 closes the gap
            acts1.append((p, "move", d_after))
            acts1.append((p, "attack", d_after))
            acts1.append((p, "rot", d_after))
            if hook_ready and d_after * UNITS_PER_METER <= HOOK["cast_range"]:
                acts1.append((p, "hook", d_after))
            if m.has_blink and d_m <= 22.0:
                acts1.append((p, "blink", d_after))
        # --- Invoker action list ---------------------------------------
        acts2 = []
        for q in mv2:
            d_after = max(0.0, d_m + q * dq)      # q>0 opens the gap
            nroom = ir2
            if q > 0 and ir2 > 0:
                nroom = ir2 - 1
            acts2.append((q, "move", d_after, nroom))
            acts2.append((q, "attack", d_after, nroom))
            if spell_ready and d_after * UNITS_PER_METER <= INVOKER_SPELL["cast_range"]:
                acts2.append((q, "spell", d_after, nroom))

        succ = []
        for (p, a1, da1) in acts1:
            row = []
            for (q, a2, da2, nroom) in acts2:
                d_new = max(0.0, min(float(n_d), (da1 + (da2 - d_m)) / dq))
                nd = int(round(d_new))
                nHP1, nHP2 = HP1, HP2
                nc1, nc2 = (0 if hook_ready else 1), (0 if spell_ready else 1)

                # ---- Pudge's action resolves ---------------------------
                if a1 == "attack":
                    nHP2 -= m.phys_after(m.pudge_dps * TICK_S, m.invoker_armor)
                elif a1 == "rot":
                    if da1 * UNITS_PER_METER <= ROT["radius"]:
                        nHP2 -= ROT["dps"] * TICK_S * (1 - m.invoker_mr)
                elif a1 == "hook":
                    if m.hook_hits(da1, room2):
                        nHP2 -= HOOK["damage"]           # pure, ignores MR
                        nd = int(round(HOOK["drag_stop"] / UNITS_PER_METER / dq))
                        nHP2 -= DISMEMBER["dps"] * DISMEMBER["duration"] * (1 - m.invoker_mr)
                        nHP1 = min(hp1_max(), nHP1 + DISMEMBER["dps"] * DISMEMBER["duration"])
                        nc1 = 1
                    else:
                        nc1 = 1                          # whiffed, still on CD
                elif a1 == "blink":
                    if da1 <= 22.0:
                        nHP2 -= HOOK["damage"]
                        nHP2 -= DISMEMBER["dps"] * DISMEMBER["duration"] * (1 - m.invoker_mr)
                        nHP1 = min(hp1_max(), nHP1 + DISMEMBER["dps"] * DISMEMBER["duration"])
                    nc1 = 1

                # ---- Invoker's action resolves -------------------------
                if a2 == "attack":
                    nHP1 -= m.phys_after(m.invoker_dps * TICK_S, m.pudge_armor)
                elif a2 == "spell":
                    nHP1 -= m.magic_after(INVOKER_SPELL["damage"], m.pudge_mr)
                    nc2 = 1
                nc2 = 0 if nc2 == 0 and spell_ready is False else nc2

                nh1 = max(0, min(n_h1 - 1, int(round(nHP1 / hq))))
                nh2 = max(0, min(n_h2 - 1, int(round(nHP2 / hq))))
                succ.append((nd, nh1, nh2, nc1, nc2, nroom))
            succ.append_row = None
            succ_rows = row
            succ.append(succ_rows)

        # cooldown ticking is folded into the bucket transition above
        return Trans(None, len(acts1), len(acts2), succ)

    return trans


def main(quick=False):
    m = Model()
    print(f"state space: {m.n_states:,}")
    states = [(d, a, b, c1, c2, r)
              for d in range(m.d_max + 1)
              for a in range(m.n_h1)
              for b in range(m.n_h2)
              for c1 in range(2)
              for c2 in range(2)
              for r in range(m.n_room)]
    print(f"enumerated : {len(states):,}")
    t0 = time.time()
    res = solve_game(states, build_transitions(m), max_sweeps=60,
                     min_sweeps=25, tol=1e-5, verbose=False)
    print(f"solved in {time.time()-t0:.1f}s, {res.sweeps} sweeps, "
          f"residual {res.residual:.2e}, bracket err {res.max_err:.2e}")
    return m, res


if __name__ == "__main__":
    main("--quick" in sys.argv)
