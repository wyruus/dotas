"""
Pudge vs Invoker, level 15 - the concrete tactical model solved by DP.

State
-----
    s = (d, h1, h2, c1, c2, room1, room2)

    d       distance between the heroes, in grid cells
    h1, h2  HP of Pudge / Invoker, in quanta
    c1, c2  cooldown codes: which abilities are available
    room1   Pudge's free space behind him, bucketed (2 levels)
    room2   Invoker's free space behind him, bucketed (3 levels)

`room2` is the variable through which the arena boundary enters the problem.
A skillshot can be dodged iff

    min(v_target * t_available, room2) > radius + bound_radius

so when Invoker is driven back against a wall his dodge reach collapses and a
Meat Hook that would otherwise be trivially side-stepped becomes unavoidable.
Invoker chooses each tick between holding distance (d falls) and keeping his
room (q = 0) - that trade is the core of the positional game.

Actions
-------
Each hero picks (movement, combat) independently:
    movement   Delta-d in {-k..+k}, k = reach in grid cells
    combat     nothing / attack / one castable ability / toggle

Movement and casting are independent, which is what lets a hero kite while
still casting.

Grid resolution
---------------
A 1 m grid at 0.3 s ticks quantises move speed to multiples of 3.33 m/s
(175 u/s).  At that resolution Pudge's 280 u/s reach (1.60 m) and Phase Boots'
330 u/s reach (1.89 m) BOTH round to 2 cells, so the grid cannot tell them
apart - which would silently delete an entire build option from the
optimisation.  `dist_q` is therefore a parameter (default 0.5 m) and the 1.0 m
specification grid is run as a sensitivity check.
"""

from __future__ import annotations

import math

from dota2sim.constants import UNITS_PER_METER, TICK, armor_multiplier
from dota2sim.tactical import lateral_reach

# combat action ids
A_NONE, A_ATTACK, A_HOOK, A_DISMEMBER, A_ROT_ON, A_ROT_OFF = range(6)
A_MEAT_SHIELD, A_BLINK_HOOK, A_S1, A_S2, A_S3, A_S4 = range(6, 12)

#: representative remaining cooldowns (ticks) per bucket
CD_REP = (0.0, 3.0, 20.0, 60.0)


def enc_cd(ticks: float) -> int:
    if ticks <= 0:
        return 0
    if ticks <= 5:
        return 1
    if ticks <= 30:
        return 2
    return 3


class Trans:
    __slots__ = ("terminal", "n_u", "n_v", "succ")

    def __init__(self, terminal, n_u, n_v, succ):
        self.terminal = terminal
        self.n_u = n_u
        self.n_v = n_v
        self.succ = succ


class Duel:
    """The reduced duel.  All numbers come from `cfg`."""

    def __init__(self, cfg):
        self.c = cfg
        c = cfg
        self.n_d = int(round(c["d_max_m"] / c["dist_q"])) + 1
        self.n_h = (int(c["hp1"] / c["hp_q"]) + 1, int(c["hp2"] / c["hp_q"]) + 1)
        self.n_room = (2, 3)
        # cooldown codes: hero 0 tracks (hook, dismember, blink), hero 1 tracks
        # up to three spells.  Each is a 4-bucket code, combined base-4.
        self.n_cd = (4 ** len(c["track1"]), 4 ** len(c["track2"]))
        self.room_m = (c["room_m1"], c["room_m2"])
        self.speed = (c["speed1"], c["speed2"])

    # -- geometry ------------------------------------------------------
    def reach(self, who: int) -> float:
        return self.speed[who] / UNITS_PER_METER * TICK

    def k(self, who: int) -> int:
        return max(1, int(self.reach(who) / self.c["dist_q"]))

    def room_metres(self, who: int, code: int) -> float:
        return self.room_m[who][code]

    def total_states(self):
        return self.n_d * self.n_h[0] * self.n_h[1] * self.n_cd[0] \
            * self.n_cd[1] * self.n_room[0] * self.n_room[1]

    # -- cooldown helpers ----------------------------------------------
    @staticmethod
    def cd_get(code: int, slot: int) -> float:
        return CD_REP[(code >> (2 * slot)) & 3]

    @staticmethod
    def cd_set(code: int, slot: int, ticks: float) -> int:
        b = enc_cd(ticks)
        return (code & ~(3 << (2 * slot))) | (b << (2 * slot))

    @staticmethod
    def cd_tick(code: int, nslots: int) -> int:
        """Advance every tracked cooldown by one tick."""
        out = 0
        for i in range(nslots):
            t = CD_REP[(code >> (2 * i)) & 3]
            out = Duel.cd_set(out, i, max(0.0, t - 1.0))
        return out

    # -- hit test ------------------------------------------------------
    def hook_hits(self, d_m: float, room2_m: float) -> bool:
        ab = self.c["hook"]
        t_avail = ab["cast_point"] + (d_m * UNITS_PER_METER) / ab["speed"]
        eff = ab["radius"] + self.c["bound_radius"]
        return lateral_reach(self.speed[1], t_avail, room2_m) <= eff

    # -- actions -------------------------------------------------------
    def actions1(self, d, h1, h2, cd, rot, mana1):
        """Return list of (movement_cells, combat_id, payload)."""
        c = self.c
        k = self.k(0)
        moves = [m for m in range(-1, k + 1) if m <= k]
        # closing is toward the opponent => d decreases
        moves = [m for m in range(-k, 1)] + [0]
        moves = sorted(set([-k, -max(1, k // 2), 0, max(1, k // 2), k]))
        moves = [m for m in moves if -k <= m <= k]

        acts = []
        hook_t = self.cd_get(cd, 0)
        dism_t = self.cd_get(cd, 1)
        blink_t = self.cd_get(cd, 2) if len(c["track1"]) > 2 else 999.0

        for mv in moves:
            acts.append((mv, A_NONE, None))
            acts.append((mv, A_ATTACK, None))
            d_after = max(0.0, (d - mv) * c["dist_q"])
            if hook_t <= 0 and mana1 >= c["hook"]["mana"] \
                    and d_after * UNITS_PER_METER <= c["hook"]["cast_range"]:
                acts.append((mv, A_HOOK, None))
            if dism_t <= 0 and mana1 >= c["dismember"]["mana"] \
                    and d_after * UNITS_PER_METER <= c["dismember"]["cast_range"]:
                acts.append((mv, A_DISMEMBER, None))
            if rot:
                acts.append((mv, A_ROT_OFF, None))
            else:
                acts.append((mv, A_ROT_ON, None))
            if blink_t <= 0 and c.get("has_blink"):
                acts.append((mv, A_BLINK_HOOK, None))
        return acts

    def actions2(self, d, h1, h2, cd, mana2):
        c = self.c
        k = self.k(1)
        moves = sorted(set([-max(1, k // 2), 0, max(1, k // 2), k]))
        acts = []
        for mv in moves:
            acts.append((mv, A_NONE, None))
            acts.append((mv, A_ATTACK, None))
            d_after = max(0.0, (d + mv) * c["dist_q"])
            for i, name in enumerate(c["track2"]):
                ab = c["spells"][name]
                if self.cd_get(cd, i) <= 0 and mana2 >= ab["mana"] \
                        and d_after * UNITS_PER_METER <= ab["cast_range"]:
                    acts.append((mv, A_S1 + i, name))
        return acts
