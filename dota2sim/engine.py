"""
Deterministic duel engine.

Design
------
There are two consumers of this module:

1. the DP solver, which needs a *cheap*, hashable transition `step(state, u, v)`
   on a packed integer state, and
2. the validator, which needs a *high fidelity* rollout at 1/30 s.

Both are served by the same semantics; fidelity is a parameter
(`substeps`).  Continuous quantities (movement, projectile travel, damage over
time) are integrated analytically where possible so that the cheap path with
`substeps = 1` is still accurate to well under one grid cell.

State layout
------------
The canonical state is a tuple of ints (see `IDX_*`).  All continuous values
are quantised; the quantisation steps are chosen so that the induced error is
strictly smaller than the smallest meaningful damage/heal event in the matchup
(see docs/formulation.md §4.2).

Reduced geometry
----------------
Per §4.2 the game on an empty isotropic rectangle depends on the geometry only
through
    d   - distance between the heroes (metres)
    r_i - free room to retreat for hero i, measured from p_i away from the
          opponent (metres)
This is exact in the interior and approximate near corners; the corner error
is bounded by `CORNER_ERROR_BOUND_M` and is reported with every result.
"""

from __future__ import annotations

import math
from typing import NamedTuple

from .constants import (
    TICK, UNITS_PER_METER, GRID_M, ARENA_W_M, ARENA_H_M,
    armor_multiplier, attacks_per_second, total_attack_speed, move_speed,
)

# --------------------------------------------------------------------------
# Quantisation
# --------------------------------------------------------------------------
HP_QUANT = 25.0       # HP per state unit
MANA_QUANT = 10.0     # mana per state unit
DIST_QUANT = 1.0      # metres per state unit (user-specified grid)

#: Geometry is reduced; near a corner the reduction loses up to this much
#: information about available retreat room.
CORNER_ERROR_BOUND_M = 1.0

MAX_DIST_M = math.hypot(ARENA_W_M, ARENA_H_M)   # 85.44 m

# --------------------------------------------------------------------------
# State tuple layout
# --------------------------------------------------------------------------
IDX_D = 0      # int, metres, 0..85
IDX_R1 = 1     # int, metres of retreat room for hero 1
IDX_R2 = 2     # int, metres of retreat room for hero 2
IDX_H1 = 3     # int, HP / HP_QUANT
IDX_H2 = 4
IDX_M1 = 5     # int, mana / MANA_QUANT
IDX_M2 = 6
IDX_CD1 = 7    # tuple[int] cooldowns in ticks for hero 1's tracked abilities
IDX_CD2 = 8
IDX_BF1 = 9    # tuple[int] active buff/debuff remaining ticks, hero 1
IDX_BF2 = 10
IDX_MODE = 11  # int bitfield, hero-specific persistent mode (Rot on/off, ...)
STATE_LEN = 12


def make_state(d_m, r1_m, r2_m, h1, h2, m1, m2, cd1, cd2, bf1, bf2, mode=0):
    return (
        int(round(d_m / DIST_QUANT)),
        int(round(r1_m / DIST_QUANT)),
        int(round(r2_m / DIST_QUANT)),
        int(round(h1 / HP_QUANT)),
        int(round(h2 / HP_QUANT)),
        int(round(m1 / MANA_QUANT)),
        int(round(m2 / MANA_QUANT)),
        tuple(cd1), tuple(cd2), tuple(bf1), tuple(bf2), int(mode),
    )


def hp_of(state, which: int) -> float:
    return state[IDX_H1 if which == 1 else IDX_H2] * HP_QUANT


def mana_of(state, which: int) -> float:
    return state[IDX_M1 if which == 1 else IDX_M2] * MANA_QUANT


def dist_of(state) -> float:
    return state[IDX_D] * DIST_QUANT


# --------------------------------------------------------------------------
# Matchup configuration = builds + hero stats, pre-resolved
# --------------------------------------------------------------------------
class UnitProfile(NamedTuple):
    """Fully resolved, build-dependent properties of one hero."""
    name: str
    max_hp: float
    max_mana: float
    hp_regen: float
    mana_regen: float
    armor: float
    magic_resist: float
    attack_damage: float
    agility: float                # contributes 1 AS per point
    attack_speed_flat: float      # flat AS bonuses from items/buffs
    attack_speed_pct: float       # percentage AS multiplier, e.g. 0.45
    base_attack_time: float
    move_speed_base: float
    move_speed_flat_bonus: float
    move_speed_pct_bonus: float
    attack_range: float          # units
    attack_point: float
    turn_rate: float
    collision_size: float
    cast_range_bonus: float
    status_resist: float
    spell_amp: float
    evasion: float
    cooldown_reduction: float


class Matchup:
    """Everything the transition function needs, precomputed.

    `abilities[i]` maps ability name -> dict of resolved numbers; the DP only
    ever looks at the abilities listed in `tracked_cds[i]`.
    """

    def __init__(self, p1: UnitProfile, p2: UnitProfile,
                 abilities1: dict, abilities2: dict,
                 tracked_cds1: list[str], tracked_cds2: list[str]):
        self.p = (None, p1, p2)
        self.abil = (None, abilities1, abilities2)
        self.tracked = (None, tracked_cds1, tracked_cds2)
        self.n_cd1 = len(tracked_cds1)
        self.n_cd2 = len(tracked_cds2)

    # -- convenience -------------------------------------------------------
    def cd_index(self, who: int, name: str) -> int:
        return self.tracked[who].index(name)


# --------------------------------------------------------------------------
# Movement
# --------------------------------------------------------------------------
def effective_move_speed(m: Matchup, who: int, slow: float = 0.0,
                         pct_bonus: float = 0.0, haste: bool = False) -> float:
    p = m.p[who]
    return move_speed(
        p.move_speed_base,
        p.move_speed_flat_bonus,
        p.move_speed_pct_bonus + pct_bonus,
        [slow] if slow > 0 else [],
        haste=haste,
    )


def reach_metres(m: Matchup, who: int, slow: float = 0.0) -> float:
    """How far hero `who` can move in one tick, in metres."""
    v = effective_move_speed(m, who, slow=slow)
    return (v / UNITS_PER_METER) * TICK


# --------------------------------------------------------------------------
# Damage application (continuous part)
# --------------------------------------------------------------------------
def apply_damage(target_hp: float, amount: float, dtype: str,
                 armor: float, magic_resist: float,
                 spell_immune: bool = False,
                 ethereal: bool = False,
                 block: float = 0.0) -> float:
    """Return the new HP after `amount` damage of `dtype`.

    Order of operations follows Dota 2: damage block applies first to physical
    attacks, then armour, then generic reductions.  Ethereal amplifies magic
    damage by 40 % and makes the unit immune to physical.
    """
    if spell_immune and dtype in ("magical", "pure"):
        return target_hp

    if dtype == "physical":
        if ethereal:
            return target_hp
        amount = max(0.0, amount - block)
        amount *= armor_multiplier(armor)
    elif dtype == "magical":
        if ethereal:
            amount *= 1.40
        amount *= (1.0 - magic_resist)
    elif dtype == "pure":
        pass
    elif dtype == "hp_removal":
        pass
    return target_hp - amount


def auto_attack_dps(m: Matchup, who: int, target_armor: float,
                    target_evasion: float = 0.0,
                    target_block: float = 0.0,
                    target_ethereal: bool = False,
                    attacker_disarmed: bool = False) -> float:
    p = m.p[who]
    if attacker_disarmed or target_ethereal:
        return 0.0
    aps = attacks_per_second(
        total_attack_speed(p.agility, p.attack_speed_flat, p.attack_speed_pct),
        p.base_attack_time)
    dmg = p.attack_damage * (1.0 - min(0.99, target_evasion))
    dmg = max(0.0, dmg - target_block)
    return aps * dmg * armor_multiplier(target_armor)


# --------------------------------------------------------------------------
# Ability resolution helpers
# --------------------------------------------------------------------------
_RANGE_KEYS = ("cast_range", "range", "travel_distance", "radius")


def in_range(m: Matchup, who: int, ability: str, d_m: float) -> bool:
    """Is the target at `d_m` metres within `ability`'s cast range?"""
    a = m.abil[who].get(ability)
    if a is None:
        return False
    rng = a.get("cast_range", 0.0)
    rng += m.p[who].cast_range_bonus
    return d_m * UNITS_PER_METER <= rng


def travel_time(ability: dict) -> float:
    """Time for a projectile ability to reach its maximum range."""
    speed = ability.get("speed", 0.0)
    if speed <= 0:
        return 0.0
    return ability.get("travel_distance", ability.get("cast_range", 0.0)) / speed
