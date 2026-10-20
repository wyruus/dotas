"""
Physical constants and unit system for the Dota 2 duel simulator.

Unit system
-----------
Dota 2 inherits the Source engine "hammer unit". The community-accepted
conversion is

    1 unit = 1.905 cm = 0.01905 m      =>      1 m = 52.4934 units

This is the ONLY conversion used in this project. It makes the numbers
physically sane: a 300 move-speed hero travels 5.71 m/s, a 600 attack range
is 11.43 m, a melee hero's 150 range is 2.86 m.

Consequences for the arena the user specified (30 m x 80 m):

    30 m = 1574.8 units      80 m = 4199.5 units

which is a very large fighting area - roughly a quarter of the Dota map.
Blink Dagger (1200 units = 22.86 m) covers 3/4 of the short side.
"""

from __future__ import annotations

import math

# --------------------------------------------------------------------------
# Units
# --------------------------------------------------------------------------
METERS_PER_UNIT = 0.01905
UNITS_PER_METER = 1.0 / METERS_PER_UNIT  # 52.4934...


def m2u(x_m: float) -> float:
    """Meters -> Dota units."""
    return x_m * UNITS_PER_METER


def u2m(x_u: float) -> float:
    """Dota units -> meters."""
    return x_u * METERS_PER_UNIT


# --------------------------------------------------------------------------
# Time
# --------------------------------------------------------------------------
TICK = 0.3                 # DP / simulation timestep, seconds (user-specified)
SERVER_TICK = 1.0 / 30.0    # Dota 2 server tick rate is 30 Hz
TICKS_PER_SERVER_TICK = TICK / SERVER_TICK  # 9 server ticks per sim tick

# --------------------------------------------------------------------------
# Arena (user-specified): rectangle 30 m wide x 80 m long, flat, empty.
# Leaving the rectangle = immediate loss.
# --------------------------------------------------------------------------
ARENA_W_M = 30.0
ARENA_H_M = 80.0
ARENA_W_U = m2u(ARENA_W_M)   # 1574.8
ARENA_H_U = m2u(ARENA_H_M)   # 4199.5

# Grid steps
GRID_M = 1.0                 # spatial discretisation
GRID_U = m2u(GRID_M)         # 52.49 units


# --------------------------------------------------------------------------
# Global combat caps and constants (see docs/formulation.md for sources)
# --------------------------------------------------------------------------
MOVESPEED_MAX = 550.0        # hard cap on movement speed (units/s)
MOVESPEED_MIN = 100.0        # hard floor on movement speed

ATTACK_SPEED_MIN = 20.0       # total attack speed is clamped to [20, 700]
ATTACK_SPEED_MAX = 700.0      # raised from 600 in 7.20
BAT_DEFAULT = 1.7             # base attack time, seconds

# Magic resistance was REWORKED in 7.39.  The classic product 1-(1-a)(1-b)
# is now only the modifier term; the base term scales with intelligence:
#     MR = 1 - [ (1 - negation) * (0.25 + 0.001*INT) ] * prod(1 - modifier_i)
BASE_MAGIC_RESIST = 0.25
MAGIC_RESIST_PER_INT = 0.001
ARMOR_CONST = 0.06            # armor formula constant, base 1 (7.27+)

# There is NO cap on status resistance: sources stack multiplicatively and the
# product approaches but never reaches 100%.
STATUS_RESIST_CAP = 1.0

# HP/mana per point of the primary attribute
HP_PER_STR = 22.0             # changed from 20 to 22 in 7.33
HP_REGEN_PER_STR = 0.1        # see NOTE on base regen below
MANA_PER_INT = 12.0
MANA_REGEN_PER_INT = 0.05
ARMOR_PER_AGI = 1.0 / 6.0     # 1 armor per 6 agility
ATTACKSPEED_PER_AGI = 1.0     # 1 attack speed per point of agility
DMG_PER_PRIMARY = 1.0         # 1 damage per point of primary attribute
DMG_PER_UNIVERSAL = 0.7       # universal heroes get 0.7 per point

# Hero base values
BASE_HP = 120.0               # all heroes have 120 base HP at level 1 (7.33+)
BASE_MANA = 75.0              # all heroes have 75 base mana at level 1
BASE_ATTACK_SPEED = 100.0     # in "attack speed points"

# NOTE - base HP regeneration is disputed between sources:
#   the official hero datafeed gives Pudge 5.00 HP/s at level 1 with 25 STR,
#   which is reproduced by (2.50 base + 0.10/STR) and also by (2.75 + 0.09/STR).
#   The wiki game-data module lists base = 0.25 / 0.09 per STR, which fits
#   neither.  We use the pair that reproduces the datafeed.  See
#   docs/uncertainties.md.
BASE_HP_REGEN = 2.5
BASE_MANA_REGEN = 0.0

# Innate melee damage block: 50% chance to block 16 physical damage per attack.
# The simulator is deterministic, so this enters as an expected value.
MELEE_BLOCK_CHANCE = 0.5
MELEE_BLOCK_AMOUNT = 16.0

# Collision
HERO_COLLISION_SIZE = 27.0    # pathing collision size, units
HERO_BOUND_RADIUS = 24.0      # used for all attack-range and AoE geometry

# Turn rate is expressed in radians per 0.03 s in the game files, but the
# wiki's own conversion table is generated with the 1/30 s server tick.
# Use the server tick (see docs/uncertainties.md).
TURN_RATE_PERIOD = 1.0 / 30.0
#: A unit acts once the target is within 11.5 deg of its facing, so the
#: worst-case turn is 180 - 11.5 = 168.5 degrees, not 180.
ACTION_CONE_DEG = 11.5


def armor_multiplier(armor: float) -> float:
    """Fraction of incoming physical damage that gets through `armor`.

    Exact Dota 2 formula, valid for both signs of armor:
        multiplier = 1 - 0.06*armor / (1 + 0.06*|armor|)
    """
    return 1.0 - (ARMOR_CONST * armor) / (1.0 + ARMOR_CONST * abs(armor))


def magic_resistance(intelligence: float, *modifiers: float,
                     negation: float = 0.0) -> float:
    """Total magic resistance as of 7.39.

        MR = 1 - [ (1 - negation) * (0.25 + 0.001*INT) ] * prod(1 - modifier_i)

    `modifiers` are the usual percentage bonuses (Cloak 20 %, Eternal Shroud
    25 %, ...); they stack multiplicatively.  `negation` is Elder Titan's
    Natural Order, which multiplies only the base term.
    """
    # The datafeed stores armour WITH the agility term (Invoker 1.3333 = -1+14/6)
    # but magic_resistance as exactly 25, so MR does not scale with INT.
    base = BASE_MAGIC_RESIST
    # the base term is itself an *already-inverted* multiplier: with no
    # negation and no INT it must evaluate to 0.75, giving 25% resistance
    mult = 1.0 - (1.0 - negation) * base
    for mod in modifiers:
        mult *= (1.0 - max(0.0, min(1.0, mod)))
    return 1.0 - max(0.0, min(1.0, mult))


def magic_multiplier(total_magic_resist: float) -> float:
    """Fraction of incoming magical damage that gets through."""
    return 1.0 - total_magic_resist


def stack_resist(*resistances: float) -> float:
    """Multiplicative stacking of resistances / reductions.

    total = 1 - prod(1 - r_i)

    Used for magic resistance, evasion, and (with a cap) status resistance.
    """
    p = 1.0
    for r in resistances:
        p *= (1.0 - max(0.0, min(1.0, r)))
    return 1.0 - p


def total_attack_speed(agility: float, flat_bonus: float = 0.0,
                       pct_bonus: float = 0.0) -> float:
    """Total attack speed, clamped to [20, 700].

        AS = (100 + AGI + flat) * (1 + pct)
    """
    raw = (BASE_ATTACK_SPEED + agility + flat_bonus) * (1.0 + pct_bonus)
    return max(ATTACK_SPEED_MIN, min(ATTACK_SPEED_MAX, raw))


def attacks_per_second(total_as: float, bat: float = BAT_DEFAULT) -> float:
    """Exact Dota 2 attack rate:  r = AS / (100 * BAT)."""
    return total_as / (100.0 * bat)


def move_speed(base: float, flat_bonus: float, pct_bonus: float,
               slows: list[float] | None = None,
               haste: float | None = None) -> float:
    """Movement speed with Dota 2's order of operations.

        1. base + flat bonuses (boots group collapses via MAX, see note)
        2. multiply by (1 + sum of ALL percentage changes), bonuses and
           slows summed together - percentage slows STACK ADDITIVELY, there
           is no "strongest slow only" rule (that rule is 7.x-era folklore;
           it applies only to the Boots/Yasha/Wind Lace item groups for
           *bonuses*, not to slows)
        3. clamp to [100, 550] LAST

    Haste does not "beat" the cap: it sets both bounds to the haste value,
    making the unit immune to both bonuses and slows.
    """
    if haste is not None:
        return haste
    pct = pct_bonus - sum(slows or ())
    ms = (base + flat_bonus) * (1.0 + pct)
    return max(MOVESPEED_MIN, min(MOVESPEED_MAX, ms))


def turn_time(angle_rad: float, turn_rate: float) -> float:
    """Seconds to rotate `angle_rad` at `turn_rate`.

    Turn rate is in radians per server tick (1/30 s) in the game files.  A
    unit may act once the target is within an 11.5 deg cone, so a full
    reversal costs 168.5 deg, not 180.
    """
    if turn_rate <= 0:
        return float("inf")
    deg = abs(math.degrees(angle_rad))
    usable = max(0.0, deg - ACTION_CONE_DEG)
    return usable / math.degrees(turn_rate / TURN_RATE_PERIOD)
