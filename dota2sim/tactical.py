"""
Reduced tactical model of the duel - the state space the DP actually solves.

Geometry reduction (see docs/formulation.md §4.2)
-----------------------------------------------
On an empty, isotropic rectangle the only geometric quantities that enter the
dynamics are

    d    - the distance between the heroes (every range check uses it), and
    room - how much lateral space the *target* has to move perpendicular to the
           engagement axis, which is limited by the arena walls.

`room` is what makes the arena boundary matter.  With perfect reactions
(0 s) and full information, a skillshot can be dodged whenever the target has
enough lateral space:

    dodge succeeds  <=>  min(v_target * t_available, room) > radius + collision

where `t_available` is the total time the target has to react, i.e. the
caster's cast point plus the projectile's flight time.  This is a *derived*
consequence of the user's "perfect reactions" assumption, and it is the single
most important structural fact about the matchup:

    **Meat Hook only reliably connects when the target cannot dodge** -
    because it is cornered against an arena wall, is disabled, or is committed
    to a cast/attack animation.

Numerical check with the real numbers (hook speed 1600 u/s, radius ~100 u, hero
collision 24 u, cast point 0.3 s, target at 300 u/s): the hook can only be
dodged-proof out to

    300*(0.3 + d/1600) <= 124   ->   d <= 181 u = 3.45 m

So a raw max-range hook (1300 u = 24.8 m) against a *free* target always misses.
This is not a bug in the model - it is correct Dota theory, and it is why the
arena boundary is the decisive feature of this problem.

Action abstraction
------------------
Each hero's action is (movement, combat):
    movement  in {-1, 0, +1} grid cells, giving up to 3 options
    combat    in {none, attack, <ability>..., <item active>...}
Actions are pruned by dominance before the stage game is built.
"""

from __future__ import annotations

import math

from .constants import UNITS_PER_METER, TICK

# --------------------------------------------------------------------------
# State layout
# --------------------------------------------------------------------------
S_D = 0        # distance, grid cells of DIST_Q metres
S_H1 = 1       # Pudge HP / HP_Q1
S_H2 = 2       # Invoker HP / HP_Q2
S_C1 = 3       # Pudge readiness code
S_C2 = 4       # Invoker readiness code
S_ROOM = 5     # target's lateral room code
STATE_LEN = 6


class TacticalConfig:
    """Everything the reduced model needs, resolved from the hero/item data."""

    def __init__(self, name1, name2,
                 hp1, hp2, hp_q1, hp_q2,
                 d_max_m=25.0, dist_q=1.0,
                 rdy1=(4, 4), rdy2=(4, 4),       # cooldown bucket counts
                 room_levels=(0.0, 2.0, 6.0, 1e9),   # metres, upper bounds
                 speed1=300.0, speed2=300.0,
                 abil1=None, abil2=None,
                 attack1=None, attack2=None):
        self.names = (name1, name2)
        self.hp = (hp1, hp2)
        self.hp_q = (hp_q1, hp_q2)
        self.d_max_m = d_max_m
        self.dist_q = dist_q
        self.n_d = int(round(d_max_m / dist_q)) + 1
        self.rdy_sizes = (rdy1[0] * rdy1[1], rdy2[0] * rdy2[1])
        self.room_levels = room_levels
        self.n_room = len(room_levels)
        self.speed = (speed1, speed2)
        self.abil = (None, abil1 or {}, abil2 or {})
        self.attack = (None, attack1 or {}, attack2 or {})

    # reach per tick in grid cells
    def reach_cells(self, who):
        v = self.speed[who]                 # units / second
        metres = v / UNITS_PER_METER * TICK
        return max(1, int(round(metres / self.dist_q)))

    def total_states(self):
        return (self.n_d * (int(self.hp[0] / self.hp_q[0]) + 1)
                * (int(self.hp[1] / self.hp_q[1]) + 1)
                * self.rdy_sizes[0] * self.rdy_sizes[1] * self.n_room)


# --------------------------------------------------------------------------
# Dodge geometry
# --------------------------------------------------------------------------
def lateral_reach(speed_upersec: float, t_available: float, room_m: float) -> float:
    """How far the target can displace sideways, in *units*, before impact."""
    reach_u = speed_upersec * t_available
    return min(reach_u, room_m * UNITS_PER_METER)


def can_dodge(abil: dict, target_speed: float, room_m: float,
              caster_cast_point: float, collision: float = 24.0) -> bool:
    """Can a free, un-disabled target walk out of this ability?

    Only applies to abilities that are aimable at a point (skillshots and
    ground-targeted AoE).  Unit-targeted abilities cannot be dodged.
    """
    if not abil.get("aimable", False):
        return False
    speed = abil.get("speed", 0.0)
    travel = abil.get("travel_distance", abil.get("cast_range", 0.0))
    flight = travel / speed if speed > 0 else 0.0
    t_avail = caster_cast_point + flight
    eff_radius = abil.get("radius", 0.0) + collision
    return lateral_reach(target_speed, t_avail, room_m) > eff_radius


def time_to_impact(abil: dict, caster_cast_point: float, d_m: float) -> float:
    """Delay between issuing the order and the effect landing at range d."""
    cp = caster_cast_point
    speed = abil.get("speed", 0.0)
    if speed > 0 and abil.get("aimable", False):
        return cp + (d_m * UNITS_PER_METER) / speed
    return cp + abil.get("delay", 0.0)
