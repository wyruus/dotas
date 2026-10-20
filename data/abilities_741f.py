"""
Ability tables for the forward model, patch 7.41f.

Source: Valve's official hero datafeed cross-checked against Liquipedia, see
data/invoker_7.41f.md and data/items_7.41f.md for the provenance of every
number.

Everything is parameterised by the orb levels so that the Quas/Wex/Exort split
is a search parameter rather than a constant.  Where a value is fixed it is
written as a plain number.

Damage is the RAW value before armour or magic resistance; the model applies
mitigation.  Damage types matter enormously in this matchup and are recorded
exactly: Sun Strike and Meat Hook are PURE and pass through everything except
a few specific immunities, which is why Black King Bar is close to useless
here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

# --------------------------------------------------------------------------
# Invoker
# --------------------------------------------------------------------------
# Orb combinations changed in 7.41 and most intuitions about them are wrong.
# Cold Snap is QQQ, not QQE; Alacrity is WWE; Chaos Meteor is WEE.
INVOKE_COMBOS = {
    "cold_snap":       ("Q", "Q", "Q"),
    "ghost_walk":      ("Q", "Q", "W"),
    "ice_wall":        ("Q", "Q", "E"),
    "emp":             ("W", "W", "W"),
    "tornado":         ("Q", "W", "W"),
    "alacrity":        ("W", "W", "E"),
    "sun_strike":      ("E", "E", "E"),
    "forge_spirit":    ("Q", "E", "E"),
    "chaos_meteor":    ("W", "E", "E"),
    "deafening_blast": ("Q", "W", "E"),
}

#: fixed costs and timings, identical for every orb distribution
INVOKE_FIXED = {
    "cold_snap":       dict(mana=100, cooldown=19.0, cast_point=0.05,
                            cast_range=1000.0, target="unit"),
    "ghost_walk":      dict(mana=175, cooldown=40.0, cast_point=0.0,
                            cast_range=0.0, target="self"),
    "ice_wall":        dict(mana=125, cooldown=23.0, cast_point=0.05,
                            cast_range=0.0, target="point"),
    "emp":             dict(mana=125, cooldown=27.0, cast_point=0.05,
                            cast_range=950.0, target="point"),
    "tornado":         dict(mana=140, cooldown=27.0, cast_point=0.05,
                            cast_range=0.0, target="point"),   # global
    "alacrity":        dict(mana=75, cooldown=15.0, cast_point=0.05,
                            cast_range=700.0, target="unit_ally"),
    "sun_strike":      dict(mana=175, cooldown=23.0, cast_point=0.05,
                            cast_range=0.0, target="point"),   # global
    "forge_spirit":    dict(mana=75, cooldown=27.0, cast_point=0.05,
                            cast_range=0.0, target="self"),
    "chaos_meteor":    dict(mana=200, cooldown=50.0, cast_point=0.05,
                            cast_range=700.0, target="point"),
    "deafening_blast": dict(mana=250, cooldown=36.0, cast_point=0.05,
                            cast_range=1000.0, target="point"),
}

#: Invoke itself - an innate since 7.41, no mana, cooldown falls with total
#: orb levels: 7 - 0.3 * (Q + W + E), floored at 0.  Two spell slots.
INVOKE_INNATE = dict(mana=0.0, base_cooldown=7.0, per_orb=0.3, slots=2)


# --------------------------------------------------------------------------
# Per-orb scalings, written as the game writes them
# --------------------------------------------------------------------------
def quas(level, inst=1, talent=False):
    """Quas passives and the scalings that depend on it."""
    t = 2 if talent else 1
    return dict(
        strength_per_level=level,
        hp_regen_per_instance=level * t * inst,
        cold_snap_duration=2.6 + 0.4 * level,
        cold_snap_damage_per_proc=20 + 8 * level,
        cold_snap_proc_interval=0.83 - 0.03 * level,
        ghost_walk_hp_regen=2 * level,
        ghost_walk_enemy_slow=0.15 + 0.05 * level,
        ice_wall_slow=0.15 + 0.15 * level,
        ice_wall_duration=2 + level,
        tornado_lift=1.0 + 0.2 * level,
        deafening_knockback=1.0 + 0.1 * level,
        forge_hp=200 + 100 * level,
        forge_range=250 + 55 * level,
        forge_duration=18 + 6 * level,
    )


def wex(level, inst=1, talent=False):
    t = 2 if talent else 1
    return dict(
        agility_per_level=level,
        move_speed_pct_per_instance=0.006 * level * t * inst,
        attack_speed_per_instance=level * t * inst,
        emp_mana_burn=25 + 75 * level,
        tornado_damage=50 + 45 * level,
        tornado_distance=1200 + 300 * level,
        alacrity_attack_speed=-2 + 12 * level,
        ghost_walk_self_speed=-0.25 + 0.05 * level,
        ghost_walk_mana_regen=2 * level,
        deafening_disarm=1 + 0.5 * level,
        emp_pull=125.0,
    )


def exort(level, inst=1, talent=False):
    t = 2 if talent else 1
    return dict(
        intelligence_per_level=level,
        damage_per_instance=2 * level * t * inst,
        sun_strike_damage=125 + 50 * level,
        chaos_meteor_damage_per_interval=30 + 20 * level,
        chaos_meteor_burn=5 + 5 * level,
        chaos_meteor_distance=465 + 165 * level,
        ice_wall_dps=24 + 6 * level,
        alacrity_damage=-2 + 12 * level,
        deafening_damage=30 + 40 * level,
        forge_mana=50 + 50 * level,
        forge_damage=10 + 10 * level,
        forge_armor=-1 + level,
    )


# --------------------------------------------------------------------------
# Effects that are not simple damage, recorded so the forward model can
# implement them rather than approximate them away
# --------------------------------------------------------------------------
INVOKE_EFFECTS = {
    "cold_snap": dict(
        damage_type="physical",
        stun_per_proc=0.4,
        # procs whenever the target takes >= 10 damage after reductions
        trigger_threshold=10.0,
        note="unit target, cannot be dodged - the enabler for everything else",
    ),
    "ghost_walk": dict(
        damage_type=None, invisibility=50.0, fade=0.0, aura_linger=2.0,
        aura_radius=450.0,
        note="self buff; also slows enemies in the aura",
    ),
    "ice_wall": dict(
        damage_type="magical", segments=15, spacing=80.0, width=105.0,
        slow_linger=2.0, damage_interval=0.5,
        note="length 1200, damage is a small DoT on top of a large slow",
    ),
    "emp": dict(
        damage_type="magical", radius=675.0, delay=2.9, burn_to_damage=0.60,
        self_restore=0.25,
        note="2.9 s delay makes it dodgeable by anything that can move",
    ),
    "tornado": dict(
        damage_type="magical", radius=200.0, speed=1000.0,
        note="cyclones enemies - full disable, and disjoints nothing",
    ),
    "alacrity": dict(
        damage_type=None, duration=9.0, target="ally_or_self",
    ),
    "sun_strike": dict(
        damage_type="pure", radius=175.0, delay=1.7, vision=400.0,
        vision_duration=4.0,
        note="PURE damage, pierces debuff immunity; split evenly across units "
             "in the radius.  The 1.7 s delay is what makes it land only on a "
             "target that is already stunned or channelling",
    ),
    "forge_spirit": dict(
        damage_type=None, unit_bat=1.35, unit_ms=320.0, unit_mr=0.0,
        unit_armor_reduction_per_stack=0.7, max_stacks=20, stack_duration=5.0,
        note="spirits apply Melting Strike, shredding armour on every hit",
    ),
    "chaos_meteor": dict(
        damage_type="magical", radius=275.0, impact_delay=1.3, speed=300.0,
        main_interval=0.5, burn_interval=1.0, burn_duration=3.0,
        note="delayed ground AoE; dodgeable unless the target is held",
    ),
    "deafening_blast": dict(
        damage_type="magical", first_radius=175.0, end_radius=225.0,
        travel_distance=1000.0, speed=1100.0, knockback_per_interval=6.0,
        #: total knockback distance, ~172 at Quas 1 rising to ~283 at Quas 10
        knockback_distance=lambda Q: 160.0 + 12.3 * Q,
        note="travels outward; disarms along the path",
    ),
}


# --------------------------------------------------------------------------
# Pudge
# --------------------------------------------------------------------------
PUDGE_FIXED = {
    "meat_hook": dict(
        mana=120.0, cooldown=12.0, cast_point=0.3, cast_backswing=0.59,
        cast_range=1300.0, travel_distance=1300.0, speed=1600.0, radius=100.0,
        damage=360.0, damage_type="pure", pierces_immunity=True,
        drag_stop=125.0, self_disable_factor=0.65,
        note="width 100 centred on the hook tip only; the chain does not latch",
    ),
    "rot": dict(
        mana=0.0, cooldown=0.0, cast_point=0.0, radius=250.0,
        damage_per_second=120.0, damage_type="magical", self_damage=120.0,
        slow=0.32, tick=0.2, linger=0.5, toggle=True,
        note="does NOT pierce spell immunity, unlike the rest of Pudge's kit; "
             "self damage is never lethal",
    ),
    "meat_shield": dict(
        mana=80.0, cooldown=17.0, cast_point=0.0, duration=7.0, block=26.0,
        dispellable=False,
    ),
    "dismember": dict(
        mana=130.0, cooldown=25.0, cast_point=0.3, cast_range=200.0,
        base_dps=100.0, str_multiplier=0.60, damage_type="magical",
        duration=2.75, intervals=6, heals=True, channel=True,
        pull_rate=75.0, pull_min=125.0, pierces_immunity=True,
        note="channel: interrupted by stun/hex/cyclone/silence, NOT by damage",
    ),
}


def invoker_ability(name: str, Q: int, W: int, E: int,
                    quas_inst: int = 0, wex_inst: int = 0,
                    exort_inst: int = 0, talent_25: bool = False) -> dict:
    """Fully resolved numbers for one invoked spell."""
    q = quas(Q, quas_inst, talent_25)
    w = wex(W, wex_inst, talent_25)
    e = exort(E, exort_inst, talent_25)
    out = dict(INVOKE_FIXED[name])
    out.update(INVOKE_EFFECTS[name])
    out["combos"] = INVOKE_COMBOS[name]

    if name == "cold_snap":
        out["duration"] = q["cold_snap_duration"]
        out["damage_per_proc"] = q["cold_snap_damage_per_proc"]
        out["proc_interval"] = q["cold_snap_proc_interval"]
    elif name == "ghost_walk":
        out["hp_regen"] = q["ghost_walk_hp_regen"]
        out["enemy_slow"] = q["ghost_walk_enemy_slow"]
        out["self_speed"] = w["ghost_walk_self_speed"]
        out["mana_regen"] = w["ghost_walk_mana_regen"]
    elif name == "ice_wall":
        out["slow"] = q["ice_wall_slow"]
        out["duration"] = q["ice_wall_duration"]
        out["dps"] = e["ice_wall_dps"]
    elif name == "emp":
        out["mana_burn"] = w["emp_mana_burn"]
    elif name == "tornado":
        out["lift"] = q["tornado_lift"]
        out["damage"] = w["tornado_damage"]
        out["travel_distance"] = w["tornado_distance"]
    elif name == "alacrity":
        out["attack_speed"] = w["alacrity_attack_speed"]
        out["damage"] = e["alacrity_damage"]
    elif name == "sun_strike":
        out["damage"] = e["sun_strike_damage"]
    elif name == "forge_spirit":
        out["hp"] = q["forge_hp"]
        out["attack_range"] = q["forge_range"]
        out["duration"] = q["forge_duration"]
        out["mana"] = e["forge_mana"]
        out["damage"] = e["forge_damage"]
        out["armor"] = e["forge_armor"]
        out["armor_reduction_per_stack"] = 0.7 + 0.1 * E
        lo = min(Q, E)
        out["count"] = 1 if lo <= 3 else (2 if lo <= 7 else 3)
    elif name == "chaos_meteor":
        out["damage_per_interval"] = e["chaos_meteor_damage_per_interval"]
        out["burn_dps"] = e["chaos_meteor_burn"]
        out["travel_distance"] = e["chaos_meteor_distance"]
    elif name == "deafening_blast":
        out["damage"] = e["deafening_damage"]
        out["disarm"] = w["deafening_disarm"]
        out["knockback_duration"] = q["deafening_knockback"]
        out["knockback_distance"] = out["knockback_distance"](Q)
    return out


def invoker_kit(Q: int, W: int, E: int, **kw) -> dict:
    return {n: invoker_ability(n, Q, W, E, **kw) for n in INVOKE_COMBOS}
