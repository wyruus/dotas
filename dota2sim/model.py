"""
Core data model: abilities, items, buffs, hero instances.

Everything is plain Python (no numpy available in this environment), so the
hot paths in the solver avoid object allocation.  This module defines the
*semantic* objects; the solver uses a packed integer encoding built from them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, IntFlag
from typing import Callable, Optional


# ==========================================================================
# Damage
# ==========================================================================
class DamageType(Enum):
    PHYSICAL = "physical"
    MAGICAL = "magical"
    PURE = "pure"
    HP_REMOVAL = "hp_removal"   # ignores everything, even spell immunity


class DispelType(Enum):
    NONE = "none"          # cannot be dispelled
    BASIC = "basic"        # dispelled by basic dispels
    STRONG = "strong"      # dispelled only by strong dispels
    DEBUFF = "debuff"


class Disable(IntFlag):
    """Disable categories, as bit flags so sets can be combined."""
    NONE = 0
    STUN = 1 << 0
    ROOT = 1 << 1
    SILENCE = 1 << 2
    MUTE = 1 << 3          # cannot use items
    DISARM = 1 << 4        # cannot attack
    HEX = 1 << 5
    CYCLONE = 1 << 6       # banished, invulnerable, cannot act
    TAUNT = 1 << 7
    FEAR = 1 << 8
    LEASH = 1 << 9
    BREAK = 1 << 10        # disables passives
    HIDDEN = 1 << 11       # Ghost Walk etc - invisible
    ETHEREAL = 1 << 12     # cannot attack, takes +magic, immune to physical
    SPELL_IMMUNE = 1 << 13
    KNOCKBACK = 1 << 14


#: What each disable forbids.
CANNOT_MOVE = (Disable.STUN | Disable.ROOT | Disable.CYCLONE | Disable.HEX
               | Disable.TAUNT | Disable.FEAR)
CANNOT_TURN = (Disable.STUN | Disable.CYCLONE | Disable.HEX | Disable.TAUNT
               | Disable.FEAR)
CANNOT_ATTACK = (Disable.STUN | Disable.CYCLONE | Disable.HEX | Disable.DISARM
                 | Disable.TAUNT | Disable.FEAR | Disable.ETHEREAL)
CANNOT_CAST = (Disable.STUN | Disable.CYCLONE | Disable.HEX | Disable.SILENCE
               | Disable.TAUNT | Disable.FEAR)
CANNOT_ITEM = (Disable.STUN | Disable.CYCLONE | Disable.HEX | Disable.MUTE
               | Disable.TAUNT | Disable.FEAR)
#: A stunned unit can still be damaged; a cycloned one cannot be attacked.


@dataclass
class DamageInstance:
    amount: float
    dtype: DamageType
    source: str = ""
    #: True if this instance interrupts channelling
    interrupts: bool = True
    #: True if magic-immune targets are unaffected
    blocked_by_spell_immunity: bool = True


# ==========================================================================
# Buffs / debuffs
# ==========================================================================
@dataclass
class Buff:
    """A timed modifier on a unit.

    Modifiers are applied in a fixed order by the engine:
        flat attribute -> percentage attribute -> flat speed -> percentage speed
    """
    name: str
    duration: float
    source: str = ""

    # stat modifiers
    bonus_str: float = 0.0
    bonus_agi: float = 0.0
    bonus_int: float = 0.0
    bonus_hp: float = 0.0
    bonus_mana: float = 0.0
    bonus_armor: float = 0.0
    bonus_magic_resist: float = 0.0
    bonus_attack_damage: float = 0.0
    bonus_attack_speed: float = 0.0
    bonus_move_speed_flat: float = 0.0
    bonus_move_speed_pct: float = 0.0
    move_speed_slow: float = 0.0        # 0..1, strongest-only
    attack_speed_slow: float = 0.0
    damage_amp_magic: float = 0.0       # e.g. Veil / Ethereal
    damage_reduction: float = 0.0       # generic % reduction
    evasion: float = 0.0
    status_resist: float = 0.0
    heal_per_sec: float = 0.0
    damage_per_sec: float = 0.0
    damage_per_sec_type: DamageType = DamageType.MAGICAL

    disables: int = Disable.NONE
    dispellable: DispelType = DispelType.BASIC
    undispellable: bool = False

    def is_active(self) -> bool:
        return self.duration > 0.0


# ==========================================================================
# Abilities
# ==========================================================================
@dataclass
class Ability:
    name: str
    #: cast point (seconds of "cast animation" before the effect fires)
    cast_point: float = 0.0
    #: cast backswing - can be cancelled by the next order, so it only costs
    #: time if the player chooses to wait; the engine lets you cancel it.
    cast_backswing: float = 0.0
    mana_cost: float = 0.0
    cooldown: float = 0.0
    #: maximum cast range in *units*
    cast_range: float = 0.0
    #: if True the ability is cast on self / has no target
    no_target: bool = False
    #: if True the ability can be cast while stunned? (none are)
    channel: float = 0.0
    #: level requirement / level of the ability
    level: int = 1
    #: True if blocked by spell immunity
    blocked_by_spell_immunity: bool = True
    #: True if the caster must face the target to start casting
    requires_facing: bool = False
    description: str = ""


@dataclass
class Projectile:
    """A travelling spell or attack projectile."""
    source_id: int
    target_id: Optional[int]
    pos: tuple[float, float]          # units
    direction: tuple[float, float]    # unit vector
    speed: float                      # units / second
    max_distance: float               # units, 0 = unlimited
    travelled: float = 0.0
    radius: float = 0.0               # units, for line projectiles
    on_hit: Optional[Callable] = None
    on_expire: Optional[Callable] = None
    kind: str = "spell"               # "spell" | "attack"
    #: line projectiles drag everything they hit; unit-target projectiles
    #: home in on the target
    homing: bool = False
    ptype: str = "point"              # "point" | "line" | "unit"


# ==========================================================================
# Items
# ==========================================================================
@dataclass
class Item:
    name: str
    cost: int
    # --- passive stats ---
    bonus_str: float = 0.0
    bonus_agi: float = 0.0
    bonus_int: float = 0.0
    bonus_hp: float = 0.0
    bonus_mana: float = 0.0
    bonus_hp_regen: float = 0.0
    bonus_mana_regen: float = 0.0
    bonus_armor: float = 0.0
    bonus_magic_resist: float = 0.0
    bonus_attack_damage: float = 0.0
    bonus_attack_speed: float = 0.0
    bonus_move_speed_flat: float = 0.0
    bonus_move_speed_pct: float = 0.0
    evasion: float = 0.0
    lifesteal: float = 0.0
    spell_amp: float = 0.0
    status_resist: float = 0.0
    cooldown_reduction: float = 0.0
    cast_range_bonus: float = 0.0
    damage_block: float = 0.0
    # --- active ---
    active_name: Optional[str] = None
    active_cooldown: float = 0.0
    active_mana_cost: float = 0.0
    active_cast_point: float = 0.0
    active_range: float = 0.0
    active_duration: float = 0.0
    active_target: str = "none"     # "none" | "unit" | "point" | "self"
    notes: str = ""


# ==========================================================================
# Hero definition
# ==========================================================================
@dataclass
class HeroDef:
    name: str
    primary: str              # "str" | "agi" | "int" | "universal"
    base_str: float
    str_gain: float
    base_agi: float
    agi_gain: float
    base_int: float
    int_gain: float
    base_move_speed: float
    attack_range: float       # units
    base_attack_time: float = 1.7
    base_attack_damage_min: float = 0.0
    base_attack_damage_max: float = 0.0
    attack_point: float = 0.3
    attack_backswing: float = 0.3
    turn_rate: float = 0.6    # radians per 0.03 s
    collision_size: float = 24.0
    projectile_speed: float = 900.0
    vision_day: float = 1800.0
    vision_night: float = 800.0
    base_armor: float = 0.0
    abilities: dict[str, Ability] = field(default_factory=dict)
    talents: dict[int, list[str]] = field(default_factory=dict)


def attribute_bonuses(hd: HeroDef, level: int, bonus_str=0.0, bonus_agi=0.0,
                      bonus_int=0.0):
    """Return (str, agi, int) totals at `level` including item bonuses.

    Dota 2 attribute gain: value(level) = base + gain*(level-1).
    """
    s = hd.base_str + hd.str_gain * (level - 1) + bonus_str
    a = hd.base_agi + hd.agi_gain * (level - 1) + bonus_agi
    i = hd.base_int + hd.int_gain * (level - 1) + bonus_int
    return s, a, i
