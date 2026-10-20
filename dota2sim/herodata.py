"""
Hero data, pinned to patch 7.41f (current as of 2026-09-29).

Primary source: Valve's official hero datafeed
    https://www.dota2.com/datafeed/herodata?language=english&hero_id=<id>
cross-checked against rdy.gg 7.41f and the community wiki.

NOTE ON SOURCES: dota2.fandom.com is a frozen 7.35-era mirror in this
environment and is WRONG for anything changed by 7.40/7.41 (Pudge's Flesh
Heap/Meat Shield split, the Facets removal, talents).  Every number here comes
from the datafeed unless marked otherwise.
"""

from __future__ import annotations

from .model import Ability, HeroDef, DamageType

PATCH = "7.41f"

# --------------------------------------------------------------------------
# PUDGE - hero_id 14, Strength
# --------------------------------------------------------------------------
PUDGE = HeroDef(
    name="Pudge",
    primary="str",
    base_str=25, str_gain=3.0,
    base_agi=13, agi_gain=1.4,
    base_int=16, int_gain=1.8,
    base_move_speed=280,
    attack_range=175,                 # melee, but not the standard 150
    base_attack_time=1.7,
    base_attack_damage_min=45,        # weapon damage; +STR is added separately
    base_attack_damage_max=51,
    attack_point=0.5,
    attack_backswing=1.17,
    turn_rate=0.7,
    collision_size=27,
    projectile_speed=0.0,             # melee: instant, no projectile
    vision_day=1800, vision_night=800,
    base_armor=0.0,                   # offset 0: armor = AGI/6 exactly
    abilities={
        # ---- Meat Hook -------------------------------------------------
        # PURE damage, pierces debuff immunity.  Damage is applied BEFORE
        # the drag.  The hook tip alone has the 100 radius; the chain body
        # does not latch.
        "meat_hook": Ability(
            name="Meat Hook", cast_point=0.3, cast_backswing=0.59,
            mana_cost=120, cooldown=12.0, cast_range=1300, level=4,
            blocked_by_spell_immunity=False, requires_facing=True,
            description="PURE 360, 1600 u/s, width 100, drags and stuns",
        ),
        # ---- Rot -------------------------------------------------------
        # Toggle.  MAGICAL, does NOT pierce spell immunity.  Damages Pudge
        # himself too, but never lethally.  Toggling does not interrupt
        # channeling, so Rot stays on through Dismember.
        "rot": Ability(
            name="Rot", cast_point=0.0, mana_cost=0, cooldown=0.0,
            cast_range=0, no_target=True, level=4,
            blocked_by_spell_immunity=True,
            description="MAGICAL 120 dps, -32% ms, r=250, 0.2 s ticks",
        ),
        # ---- Meat Shield (formerly Flesh Heap) -------------------------
        "meat_shield": Ability(
            name="Meat Shield", cast_point=0.0, mana_cost=80,
            cooldown=17.0, no_target=True, level=4,
            blocked_by_spell_immunity=False,
            description="blocks 26 per instance for 7 s, undispellable",
        ),
        # ---- Dismember -------------------------------------------------
        # MAGICAL, pierces debuff immunity, STRONG dispel only.  Channel:
        # interrupted by stuns/hexes/cyclones/silences, NOT by damage.
        "dismember": Ability(
            name="Dismember", cast_point=0.3, cast_backswing=0.0,
            mana_cost=130, cooldown=25.0, cast_range=200, level=2,
            channel=2.75, blocked_by_spell_immunity=False,
            description="MAGICAL 100 + 60% STR dps, heals same, 2.75 s",
        ),
    },
    talents={
        10: ["+10% Rot Slow", "+5 Armor"],
        15: ["+150 Meat Hook Damage", "+8% Spell Lifesteal"],
        20: ["-4s Meat Hook Cooldown", "+0.75s Dismember Duration"],
        25: ["1.5x Flesh Heap and Meat Shield", "1.5x Dismember Damage/Heal"],
    },
)

# --------------------------------------------------------------------------
# Resolved ability numbers.  Kept as plain dicts because the solver reads them
# in its inner loop and dict lookup beats attribute access.
# --------------------------------------------------------------------------
PUDGE_ABILITIES_15 = {
    "meat_hook": {
        "damage": 360.0, "damage_talent": 510.0,
        "dtype": "pure",
        "mana": 120.0, "cooldown": 12.0,
        "cast_point": 0.3, "cast_backswing": 0.59,
        "cast_range": 1300.0,
        "travel_distance": 1300.0,
        "speed": 1600.0,
        "radius": 100.0,
        "pierces_immunity": True,
        "drags": True,
        "drag_stop_distance": 125.0,
        "self_disable": 0.528,      # 65% of max travel time: Pudge is locked
        "aimable": True,            # dodgeable: see tactical.can_dodge
        "interrupts_channel": True,
    },
    "rot": {
        "dps": 120.0,
        "dtype": "magical",
        "self_damage": 120.0,       # never lethal
        "radius": 250.0,
        "slow": 0.32,
        "tick": 0.2,
        "linger": 0.5,
        "mana": 0.0, "cooldown": 0.0, "cast_point": 0.0,
        "pierces_immunity": False,
        "toggle": True,
    },
    "meat_shield": {
        "block": 26.0, "duration": 7.0,
        "mana": 80.0, "cooldown": 17.0, "cast_point": 0.0,
        "dispellable": False,
    },
    "dismember": {
        "base_dps": 100.0,
        "str_mult": 0.60,           # level 2 of 3
        "dtype": "magical",
        "heals": True,
        "duration": 2.75,
        "intervals": 6,
        "mana": 130.0, "cooldown": 25.0,
        "cast_range": 200.0, "cast_point": 0.3,
        "pull_rate": 75.0, "pull_min": 125.0,
        "channel": True,
        "pierces_immunity": True,
        "dispel": "strong",
    },
}


def pudge_dismember_dps(str_total: float) -> float:
    """Dismember damage per second at level 15 (ability level 2)."""
    a = PUDGE_ABILITIES_15["dismember"]
    return a["base_dps"] + a["str_mult"] * str_total
